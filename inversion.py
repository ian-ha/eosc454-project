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
    ):
        self.forward_simulation = forward_simulation
        self.uncertainty_floor = uncertainty_floor
        self.min_iterations = min_iterations
        self.target_chifact = target_chifact
        (
            self.data_mis,
            self.regularize,
            self.starting_model,
            self.ref_model,
            self.model_map,
            self.optimize,
        ) = self.set_up_inversion()
        self.recovered_model, self.inv_problem = self.run_inverse_problem()

    def set_up_inversion(self):
        max_anomaly = np.max(np.abs(self.forward_simulation.dpred))
        floor_uncertainty = self.uncertainty_floor * max_anomaly
        uncertainty = np.ones_like(self.forward_simulation.dpred) * floor_uncertainty
        data_object = data.Data(
            self.forward_simulation.survey, self.forward_simulation.dpred, uncertainty
        )
        n_active = int(self.forward_simulation.active_cells.sum())
        model_map = maps.IdentityMap(nP=n_active)
        data_mis = data_misfit.L2DataMisfit(
            data=data_object, simulation=self.forward_simulation.simulation
        )
        regularize = regularization.WeightedLeastSquares(
            self.forward_simulation.mesh,
            active_cells=self.forward_simulation.active_cells,
        )
        starting_model = 1e-6 * np.ones(n_active)
        ref_model = np.zeros_like(starting_model)
        optimize = optimization.ProjectedGNCG(
            maxIter=8,
            lower=0.0,
            maxIterLS=25,
            cg_maxiter=40,
            cg_rtol=1e-4,
            tolF=1e-6,
            tolX=1e-4,
            tolG=1e-4,
        )

        # SimPEG 0.25 has no built-in minIter option, so gate convergence by iter.
        min_iter_stopper = {
            "str": "%d : minIter  =     %3d    <= iter          =    %3d",
            "left": lambda M: self.min_iterations,
            "right": lambda M: M.iter,
            "stopType": "optimal",
        }
        optimize.stoppers = [
            optimization.StoppingCriteria.tolerance_f,
            optimization.StoppingCriteria.moving_x,
            optimization.StoppingCriteria.tolerance_g,
            min_iter_stopper,
            optimization.StoppingCriteria.iteration,
        ]
        return data_mis, regularize, starting_model, ref_model, model_map, optimize

    def run_inverse_problem(self):
        inv_problem = inverse_problem.BaseInvProblem(
            self.data_mis, self.regularize, self.optimize
        )
        sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=False)
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
            directives_list.append(directives.TargetMisfit(chifact=self.target_chifact))
        inv_L2 = inversion.BaseInversion(inv_problem, directives_list)
        recovered_model = inv_L2.run(self.starting_model)
        return recovered_model, inv_problem
