"""Original work by Devin Cowan, 3D Forward Simulation of TMI Data, https://simpeg.xyz/user-tutorials/fwd-magnetics-induced-3d/, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Changes made: Adapted code to be more modular, added functionality."""

# SimPEG functionality
from simpeg.potential_fields import magnetics
from simpeg.utils import plot2Ddata, model_builder
from simpeg import maps

# discretize functionality
from discretize import TreeMesh
from discretize.utils import mkvc, active_from_xyz

# Common Python functionality
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import yaml


class ForwardSimulation:
    """Class to handle forward simulation, including physical domain setup, topography generation, and forward modeling. Based on SimPEG's magnetics module for forward modeling https://simpeg.xyz/user-tutorials/fwd-magnetics-induced-3d/"""

    def __init__(self, config_yaml, z_func=None, randomize_model=False):
        with open(config_yaml, "r") as f:

            self.config = yaml.safe_load(f)
        print(self.config)
        domain_params = self.config.get("domain", None)
        self.x_bounds = domain_params.get("x_bounds", None)
        self.y_bounds = domain_params.get("y_bounds", None)
        self.n_cells = domain_params.get("n_cells", None)
        self.z_func = z_func
        self.x_topo, self.y_topo, self.z_topo = self.generate_topography()
        survey_params = self.config.get("survey", None)
        self.survey_x_spacing = survey_params.get("x_spacing", None)
        self.survey_y_spacing = survey_params.get("y_spacing", None)
        self.survey_z_height = survey_params.get("z_height", None)
        self.survey_x_bounds = survey_params.get("x_bounds", None)
        self.survey_y_bounds = survey_params.get("y_bounds", None)

        self.x_survey, self.y_survey, self.z_survey = self.define_survey_locations(
            self.survey_x_spacing,
            self.survey_x_bounds,
            self.survey_y_spacing,
            self.survey_y_bounds,
            self.survey_z_height,
        )
        inducing_field_params = self.config.get("inducing_field", None)
        self.survey = self.define_survey(
            self.x_survey,
            self.y_survey,
            self.z_survey,
            components=survey_params.get("components", "tmi"),
            inclination=inducing_field_params.get("inclination", 90),
            declination=inducing_field_params.get("declination", 0),
            amplitude=inducing_field_params.get("amplitude", 50000),
        )
        mesh_params = self.config.get("mesh", None)
        self.mesh_dx = mesh_params.get("dx", None)
        self.mesh_dy = mesh_params.get("dy", None)
        self.mesh_dz = mesh_params.get("dz", None)
        self.mesh = self.define_mesh(
            self.mesh_dx,
            self.mesh_dy,
            self.mesh_dz,
            mesh_params.get("x_length", None),
            mesh_params.get("y_length", None),
            mesh_params.get("z_length", None),
        )
        self._validate_geometry()
        self.active_cells = self.define_active_cells(self.mesh)
        model_params = self.config.get("model", None)
        self.model = self.define_model(
            self.mesh,
            self.active_cells,
            background_susceptibility=model_params.get(
                "background_susceptibility", 0.5
            ),
            curie_depth=model_params.get("curie_depth", 300),
            curie_susceptibility=model_params.get("curie_susceptibility", 0.05),
            curie_wave_amplitude=model_params.get("curie_wave", {}).get(
                "amplitude", 0.0
            ),
            curie_wave_wavelength=model_params.get("curie_wave", {}).get(
                "wavelength", None
            ),
            curie_wave_phase_degrees=model_params.get("curie_wave", {}).get(
                "phase_degrees", 0.0
            ),
            randomize=randomize_model,
        )
        self.simulation, self.dpred = self.run_forward_simulation()

    def generate_topography(self):
        """Generate the topography of the simulation domain."""
        [x_topo, y_topo] = np.meshgrid(
            np.linspace(-self.x_bounds[0], self.x_bounds[1], self.n_cells[0]),
            np.linspace(-self.y_bounds[0], self.y_bounds[1], self.n_cells[1]),
        )

        if self.z_func is None:
            # flat topography
            z_topo = np.zeros_like(x_topo)
        else:
            z_topo = self.z_func(x_topo, y_topo)

        return x_topo, y_topo, z_topo

    def define_survey_locations(
        self, x_spacing, x_bounds, y_spacing, y_bounds, z_height
    ):
        """Define the survey geometry for the forward simulation.

        :param x_spacing: Spacing between survey points in the x-direction
        :param x_bounds: Tuple defining the bounds of the survey in the x-direction (min, max)
        note: bounds should be smaller than the the topography bounds to minimize edge effects
        :param y_spacing: Spacing between survey points in the y-direction
        :param y_bounds: Tuple defining the bounds of the survey in the y-direction (min, max)
        note: bounds should be smaller than the the topography bounds to minimize edge effects
        :param z_height: Height above sea level at which the survey points are located
        """
        num_x = int((x_bounds[1] - x_bounds[0]) / x_spacing) + 1
        num_y = int((y_bounds[1] - y_bounds[0]) / y_spacing) + 1
        x_survey = np.linspace(x_bounds[0], x_bounds[1], num_x)
        y_survey = np.linspace(y_bounds[0], y_bounds[1], num_y)
        z_survey = np.full(
            (num_y, num_x), z_height
        )  # Survey points at a constant height
        return x_survey, y_survey, z_survey

    def define_survey(
        self,
        x_survey,
        y_survey,
        z_survey,
        components=["tmi"],
        inclination=90,
        declination=0,
        amplitude=50000,
    ):
        """Define the survey object for the forward simulation.
        :param x_survey: 1D array of x-coordinates for survey points
        :param y_survey: 1D array of y-coordinates for survey points
        :param z_survey: 2D array of z-coordinates for survey points (should be the same shape as the meshgrid of x and y)
        :param components: List of components to simulate. Options are from simpeg's magnetics module
        :param inclination: Inclination of the inducing field in degrees (default 90)
        :param declination: Declination of the inducing field in degrees (default 0)
        :param amplitude: Amplitude of the inducing field in nT (default 50000)"""
        # Expand 1D survey axes to a full grid so x, y, z vectors have matching length.
        x_grid, y_grid = np.meshgrid(x_survey, y_survey)
        locations = np.c_[mkvc(x_grid), mkvc(y_grid), mkvc(z_survey)]

        # Use the observation locations and components to define the receivers. To
        # simulate data, the receivers must be defined as a list.
        receiver_list = magnetics.receivers.Point(locations, components=components)
        receiver_list = [receiver_list]

        source_field = magnetics.sources.UniformBackgroundField(
            receiver_list=receiver_list,
            amplitude=amplitude,
            inclination=inclination,
            declination=declination,
        )

        # Define the survey
        survey = magnetics.survey.Survey(source_field)
        return survey

    def define_mesh(self, dx, dy, dz, x_length, y_length, z_length):
        """Define a tree mesh for the forward simulation.
        :param dx: minimum cell size in the x-direction
        :param dy: minimum cell size in the y-direction
        :param dz: minimum cell size in the z-direction
        :param x_length: total length of the mesh in the x-direction
        :param y_length: total length of the mesh in the y-direction
        :param z_length: total length of the mesh in the z-direction
        """
        nbcx = 2 ** int(np.round(np.log(x_length / dx) / np.log(2.0)))
        nbcy = 2 ** int(np.round(np.log(y_length / dy) / np.log(2.0)))
        nbcz = 2 ** int(np.round(np.log(z_length / dz) / np.log(2.0)))
        self.mesh_nbcx = nbcx
        self.mesh_nbcy = nbcy
        self.mesh_nbcz = nbcz
        self.mesh_x_length = dx * nbcx
        self.mesh_y_length = dy * nbcy
        self.mesh_z_length = dz * nbcz
        hx = [(dx, nbcx)]
        hy = [(dy, nbcy)]
        hz = [(dz, nbcz)]
        mesh = TreeMesh([hx, hy, hz], x0="CCN", diagonal_balance=True)
        mesh.origin += np.r_[0.0, 0.0, self.z_topo.max()]
        topo_xyz = np.c_[mkvc(self.x_topo), mkvc(self.y_topo), mkvc(self.z_topo)]
        mesh.refine_surface(topo_xyz, padding_cells_by_level=[2, 2], finalize=False)
        mesh.finalize()
        return mesh

    def _validate_geometry(self):
        """Validate that the survey fits inside the mesh and warn on oversized topo domains."""
        mesh_half_x = self.mesh_x_length / 2.0
        mesh_half_y = self.mesh_y_length / 2.0

        survey_x_min = float(np.min(self.survey_x_bounds))
        survey_x_max = float(np.max(self.survey_x_bounds))
        survey_y_min = float(np.min(self.survey_y_bounds))
        survey_y_max = float(np.max(self.survey_y_bounds))

        survey_issues = []
        if survey_x_min < -mesh_half_x or survey_x_max > mesh_half_x:
            survey_issues.append(
                f"survey x-bounds [{survey_x_min:.1f}, {survey_x_max:.1f}] exceed mesh x-extent [-{mesh_half_x:.1f}, {mesh_half_x:.1f}]"
            )
        if survey_y_min < -mesh_half_y or survey_y_max > mesh_half_y:
            survey_issues.append(
                f"survey y-bounds [{survey_y_min:.1f}, {survey_y_max:.1f}] exceed mesh y-extent [-{mesh_half_y:.1f}, {mesh_half_y:.1f}]"
            )

        if survey_issues:
            raise ValueError(
                "Survey grid is outside the mesh footprint. "
                f"Actual mesh extents after power-of-two rounding are x=[-{mesh_half_x:.1f}, {mesh_half_x:.1f}] m and y=[-{mesh_half_y:.1f}, {mesh_half_y:.1f}] m. "
                + " ".join(survey_issues)
                + ". Increase the mesh lengths or shrink the survey bounds so the receivers sit inside the mesh."
            )

        topo_x_min = -float(self.x_bounds[0])
        topo_x_max = float(self.x_bounds[1])
        topo_y_min = -float(self.y_bounds[0])
        topo_y_max = float(self.y_bounds[1])

        topo_issues = []
        if topo_x_min < -mesh_half_x or topo_x_max > mesh_half_x:
            topo_issues.append(
                f"topography x-span [{topo_x_min:.1f}, {topo_x_max:.1f}] exceeds mesh x-extent [-{mesh_half_x:.1f}, {mesh_half_x:.1f}]"
            )
        if topo_y_min < -mesh_half_y or topo_y_max > mesh_half_y:
            topo_issues.append(
                f"topography y-span [{topo_y_min:.1f}, {topo_y_max:.1f}] exceeds mesh y-extent [-{mesh_half_y:.1f}, {mesh_half_y:.1f}]"
            )

        if topo_issues:
            warnings.warn(
                "Topography domain is larger than the mesh footprint. "
                f"Actual mesh extents are x=[-{mesh_half_x:.1f}, {mesh_half_x:.1f}] m and y=[-{mesh_half_y:.1f}, {mesh_half_y:.1f}] m. "
                + " ".join(topo_issues)
                + ". The forward model may still run, but edge effects and surface refinement may be degraded.",
                RuntimeWarning,
                stacklevel=2,
            )

    def define_active_cells(self, mesh):
        """Define active cells for the forward simulation based on the topography.
        :param mesh: The mesh object for the simulation
        """
        topo_xyz = np.c_[mkvc(self.x_topo), mkvc(self.y_topo), mkvc(self.z_topo)]
        active = active_from_xyz(mesh, topo_xyz)
        return active

    def define_model(
        self,
        mesh,
        active,
        background_susceptibility=0.5,
        curie_depth=300,
        curie_susceptibility=0.05,
        curie_wave_amplitude=0.0,
        curie_wave_wavelength=None,
        curie_wave_phase_degrees=0.0,
        randomize=False,
    ):
        """Define the model for the forward simulation.
        :param mesh: The mesh object for the simulation
        :param active: The active cells for the simulation
        :param background_susceptibility: The susceptibility of the background (default 0.0)
        :param curie_depth: The depth of the Curie point in meters (default 300)
        :param curie_susceptibility: The susceptibility below the curie depth (default 0.01)
        :param curie_wave_amplitude: Sine-wave amplitude for smooth Curie-depth variation in meters
        :param curie_wave_wavelength: Sine-wave wavelength in meters along x (None disables variation)
        :param curie_wave_phase_degrees: Sine-wave phase in degrees
        """
        x_cell_centers = mesh.cell_centers[:, 0]
        z_cell_centers = mesh.cell_centers[:, 2]

        if curie_wave_wavelength is None or curie_wave_wavelength <= 0:
            local_curie_depth = np.full(mesh.nC, curie_depth, dtype=float)
        else:
            phase_radians = np.deg2rad(curie_wave_phase_degrees)
            local_curie_depth = curie_depth + curie_wave_amplitude * np.sin(
                (2.0 * np.pi * x_cell_centers / curie_wave_wavelength) + phase_radians
            )

        # Guard against non-physical negative depths from large amplitudes.
        local_curie_depth = np.maximum(local_curie_depth, 0.0)
        curie_surface = self.z_topo.max() - local_curie_depth

        curie_mask = active & (z_cell_centers < curie_surface)

        if (
            randomize
        ):  # Assign random susceptibility values to create a more realistic model.
            np.random.seed(0)  # for reproducibility
            background_susceptibility = np.random.uniform(
                background_susceptibility * 0.5,
                background_susceptibility,
                size=active.sum(),
            )
            curie_susceptibility = np.random.uniform(
                0, curie_susceptibility, size=curie_mask.sum()
            )

        model = np.zeros(mesh.nC)
        model[active] = background_susceptibility
        # set all cells below the curie depth to have the curie susceptibility
        model[curie_mask] = curie_susceptibility
        return model

    def run_forward_simulation(self):
        """Run the forward simulation to generate synthetic magnetic data."""
        n_active = int(self.active_cells.sum())
        model_map = maps.IdentityMap(nP=n_active)
        active_model = self.model[self.active_cells]

        simulation = magnetics.simulation.Simulation3DIntegral(
            survey=self.survey,
            mesh=self.mesh,
            model_type="scalar",
            chiMap=model_map,
            active_cells=self.active_cells,
            store_sensitivities="forward_only",
            engine="choclo",
        )
        dpred = simulation.dpred(active_model)
        return simulation, dpred
