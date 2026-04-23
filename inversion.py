"""Original work by Devin Cowan, 3D Inversion of TMI Data to Recover a Susceptibility Model ,https://simpeg.xyz/user-tutorials/inv-magnetics-induced-3d/, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Changes made: Adapted code to be more modular, added functionality."""

from forward_simulation import ForwardSimulation

# SimPEG functionality
from simpeg.potential_fields import magnetics
from simpeg.utils import plot2Ddata, model_builder
from simpeg import (
    maps,
    data,
    data_misfit,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion,
)

# discretize functionality
from discretize import TreeMesh
from discretize.utils import mkvc, active_from_xyz

# Common Python functionality
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os


class MagneticsInversion:
    """Class to handle inversion of magnetic data to recover a susceptibility model."""

    def __init__(
        self,
        forward_simulation: ForwardSimulation,
        uncertainty_floor,
        min_iterations=5,
        target_chifact=None,
        inversion_type="LS",
        uncertainty=None,
        invert=True,
    ):
        self.forward_simulation = forward_simulation
        self.uncertainty_floor = uncertainty_floor
        self.min_iterations = min_iterations
        self.target_chifact = target_chifact
        self.inversion_type = inversion_type
        self.uncertainty = uncertainty

        (
            self.data_mis,
            self.regularize,
            self.starting_model,
            self.ref_model,
            self.model_map,
            self.optimize,
            self.uncertainty,
        ) = self.set_up_inversion()
        if invert:
            self.recovered_model, self.inv_problem = self.run_inverse_problem()
        else:
            self.recovered_model = None
            self.inv_problem = None

    def build_depth_taper_reference(
        self,
        chi0=0.1,
        d_const_m=2000.0,
        d_zero_m=10000.0,
    ):
        # Active-cell centers only
        cc = self.forward_simulation.mesh.cell_centers[
            self.forward_simulation.active_cells
        ]

        # Depth below highest topography (m, positive downward)
        depth_m = float(self.forward_simulation.z_topo.max()) - cc[:, 2]

        if d_zero_m <= d_const_m:
            raise ValueError("d_zero_m must be greater than d_const_m")

        # Piecewise: constant above d_const_m, linear taper to zero at d_zero_m
        ref = np.empty_like(depth_m, dtype=float)
        shallow = depth_m <= d_const_m
        deep = depth_m >= d_zero_m
        mid = (~shallow) & (~deep)

        ref[shallow] = chi0
        ref[deep] = 0.0
        ref[mid] = chi0 * (1.0 - (depth_m[mid] - d_const_m) / (d_zero_m - d_const_m))
        return ref

    def build_depth_sensitivity_weights(self, power=3.0, z0=None):
        cc = self.forward_simulation.mesh.cell_centers[
            self.forward_simulation.active_cells
        ]
        depth_m = float(self.forward_simulation.z_topo.max()) - cc[:, 2]

        if z0 is None:
            z0 = 0.5 * float(
                np.min(
                    [
                        self.forward_simulation.mesh_dx,
                        self.forward_simulation.mesh_dy,
                        self.forward_simulation.mesh_dz,
                    ]
                )
            )

        weights = 1.0 / (depth_m + z0) ** power
        weights /= np.max(weights)
        return weights

    def set_up_inversion(self):
        if self.uncertainty is None:
            max_anomaly = np.max(np.abs(self.forward_simulation.dpred))
            floor_uncertainty = self.uncertainty_floor * max_anomaly
            uncertainty = (
                np.ones_like(self.forward_simulation.dpred) * floor_uncertainty
            )
        else:
            uncertainty = self.uncertainty
        data_object = data.Data(
            self.forward_simulation.survey,
            dobs=self.forward_simulation.dpred,
            standard_deviation=uncertainty,
        )
        n_active = int(self.forward_simulation.active_cells.sum())
        model_map = maps.IdentityMap(nP=n_active)
        data_mis = data_misfit.L2DataMisfit(
            data=data_object, simulation=self.forward_simulation.simulation
        )
        ref_model = self.build_depth_taper_reference(
            chi0=0.01,  # choose your shallow prior susceptibility
            d_const_m=2000,  # 500M
            d_zero_m=10000,  # 5KM
        )
        depth_weights = self.build_depth_sensitivity_weights()
        starting_model = 1e-6 * np.ones(n_active)
        if self.inversion_type == "LS":
            regularize = regularization.WeightedLeastSquares(
                self.forward_simulation.mesh,
                active_cells=self.forward_simulation.active_cells,
                reference_model=ref_model,
                reference_model_in_smooth=False,
                weights={"depth": depth_weights},
            )
            max_iterations = 10
        elif self.inversion_type == "IRLS":
            regularize = regularization.Sparse(
                self.forward_simulation.mesh,
                active_cells=self.forward_simulation.active_cells,
                alpha_s=float(
                    np.min(
                        [
                            self.forward_simulation.mesh_dx,
                            self.forward_simulation.mesh_dy,
                            self.forward_simulation.mesh_dz,
                        ]
                    )
                )
                ** (-2.0),
                alpha_x=1,
                alpha_y=1,
                alpha_z=1,
                reference_model=ref_model,
                reference_model_in_smooth=False,
                norms=[0, 1, 1, 1],
                weights={"depth": depth_weights},
            )
            max_iterations = 30
        else:
            raise ValueError("Inversion type not supported")

        self.max_iterations = max_iterations

        optimize = optimization.ProjectedGNCG(
            maxIter=max_iterations, lower=0.0, maxIterLS=15, cg_maxiter=10, cg_rtol=1e-2
        )
        return (
            data_mis,
            regularize,
            starting_model,
            ref_model,
            model_map,
            optimize,
            uncertainty,
        )

    def run_inverse_problem(self):
        inv_problem = inverse_problem.BaseInvProblem(
            self.data_mis, self.regularize, self.optimize
        )
        if self.inversion_type == "LS":
            sensitivity_weights = directives.UpdateSensitivityWeights(
                every_iteration=False
            )
            update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)
            starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=10)
            beta_schedule = directives.BetaSchedule(coolingFactor=2.0, coolingRate=1)
            directives_list = [
                sensitivity_weights,
                update_jacobi,
                starting_beta,
                beta_schedule,
            ]
            if self.target_chifact is not None:
                directives_list.append(
                    directives.TargetMisfit(chifact=self.target_chifact)
                )
        elif self.inversion_type == "IRLS":
            sensitivity_weights_irls = directives.UpdateSensitivityWeights(
                every_iteration=False
            )
            starting_beta_irls = directives.BetaEstimate_ByEig(beta0_ratio=10)
            update_jacobi_irls = directives.UpdatePreconditioner(
                update_every_iteration=True
            )
            update_irls = directives.UpdateIRLS(
                cooling_factor=2,
                f_min_change=1e-4,
                max_irls_iterations=25,
                chifact_start=1.0,
            )

            directives_list = [
                update_irls,
                sensitivity_weights_irls,
                starting_beta_irls,
                update_jacobi_irls,
            ]

        else:
            raise ValueError("Inversion type not supported")

        inv_L2 = inversion.BaseInversion(inv_problem, directives_list)
        recovered_model = inv_L2.run(self.starting_model)
        return recovered_model, inv_problem

    def _build_fixed_beta_optimizer(self):
        return optimization.ProjectedGNCG(
            maxIter=self.max_iterations,
            lower=0.0,
            maxIterLS=15,
            cg_maxiter=10,
            cg_rtol=1e-2,
        )

    def _build_fixed_beta_problem(self, beta):
        inv_problem = inverse_problem.BaseInvProblem(
            self.data_mis,
            self.regularize,
            self._build_fixed_beta_optimizer(),
        )
        inv_problem.beta = float(beta)
        return inv_problem

    def run_fixed_beta_inversion(self, beta, starting_model=None):
        if self.inversion_type != "LS":
            raise ValueError("Tikhonov curves are intended for least-squares inversion")

        inv_problem = self._build_fixed_beta_problem(beta)
        inv_L2 = inversion.BaseInversion(inv_problem, [])
        model0 = self.starting_model if starting_model is None else starting_model
        recovered_model = inv_L2.run(model0)
        return recovered_model, inv_problem

    def compute_tikhonov_curve(
        self,
        beta_values=None,
        beta_min=1e-5,
        beta_max=1e5,
        num_beta=21,
        starting_model=None,
        target_chifact=None,
    ):
        if self.inversion_type != "LS":
            raise ValueError("Tikhonov curves are intended for least-squares inversion")

        if beta_values is None:
            beta_values = np.logspace(
                np.log10(beta_max), np.log10(beta_min), int(num_beta)
            )
        else:
            beta_values = np.asarray(beta_values, dtype=float)

        model0 = self.starting_model if starting_model is None else starting_model
        target_phi_d = self.get_target_misfit_value(target_chifact=target_chifact)
        results = []

        for beta in beta_values:
            recovered_model, inv_problem = self.run_fixed_beta_inversion(
                beta, starting_model=model0
            )
            phi_d = float(self.data_mis(recovered_model))
            phi_m = float(self.regularize(recovered_model))
            results.append(
                {
                    "beta": float(beta),
                    "phi_d": phi_d,
                    "phi_m": phi_m,
                    "phi_total": phi_d + float(beta) * phi_m,
                    "recovered_model": recovered_model,
                    "inv_problem": inv_problem,
                }
            )

        return {
            "beta": np.asarray([item["beta"] for item in results], dtype=float),
            "phi_d": np.asarray([item["phi_d"] for item in results], dtype=float),
            "phi_m": np.asarray([item["phi_m"] for item in results], dtype=float),
            "phi_total": np.asarray(
                [item["phi_total"] for item in results], dtype=float
            ),
            "recovered_models": [item["recovered_model"] for item in results],
            "target_phi_d": target_phi_d,
        }

    def get_target_misfit_value(self, target_chifact=None):
        chifact = self.target_chifact if target_chifact is None else target_chifact
        if chifact is None:
            chifact = 1.0
        return float(self.forward_simulation.dpred.size) * float(chifact)
