# Curie Depth Via Aeromagnetic Data Inversion and Spectral Methods
This project seeks to determine the Curie depth using both linear inversion and spectral methods on simulated magnetics data. Curie Depth is the depth at which the temperature of the earth is the Curie temperature for Magnetite (~580 $\degree C$). At this temperature, Magnetite loses its magnetization.


## Installation
First, clone the repository.
Required packages are listed in the `environment.yml` file. To install and activate the conda environment, follow the steps below:

1. Create the Conda environment: `conda env create -f environment.yml`
1. Activate the environment: `conda activate eosc-454-project`






## Running The Code
The code for forward simulation and inversion can be found in `forward_simulation.py` and `inversion.py`, respectively. This code is adapted from SimPEG tutorials on forward simulation (https://simpeg.xyz/user-tutorials/fwd-magnetics-induced-3d/), and inversion (https://simpeg.xyz/user-tutorials/inv-magnetics-induced-3d/) of magnetics data. 

The outputs of many intermediate steps can be seen by running `visualizations.ipynb`. To modify the parameters of the forward simulation, modify the values in `forward_model_params.yml`. Customizability of the inversion parameters have not yet been implemented.

The code for running spectral methods is in `spectral_methods.py`. The forward simulation for the spectral methods code is determined using the `spectral_analysis_forward_model.yml` file. Visualizations for spectral methods can be generated using `sweep_curie_depths.py` to generate data and `plot_curie_depth_from_csv.py` to plot.
