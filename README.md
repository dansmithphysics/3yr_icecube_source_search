testing

# Source Search in 3 Years of Muon Track Data in IceCube

This repository documents the analysis code used to produce the results from the following paper:

Revisiting AGN as the Source of IceCube's Diffuse Neutrino Flux, D. Smith *et al.* e-Print: [2007.12706](https://arxiv.org/abs/2007.12706) [astro-ph.HE]

Please direct all code-related questions to [danielsmith@uchicago.edu](mailto:danielsmith@uchicago.edu).

## Data

The IceCube public data is available from their website, [linked here](https://icecube.wisc.edu/science/data/PS-3years).

The 4LAC catalog is available from Fermi's website, [linked here](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/4LACDR2/). A full description of the catalog is available in [this paper](https://iopscience.iop.org/article/10.3847/1538-4357/ab791e). 

## Example Analysis in Jupyter Notebooks

For ease of use and visualization, I've created the following Jupyter Notebooks that step you through the definition of the likelihood used to find sources (`J00`), pre-processing the data (`J01`), the all-sky source search (`J02`), and the source class search (`J03`). 

### J00_math.ipynb

A brief overview of the math used to define likelihoods and perform the source class search.

### J01_preprocess_data.ipynb

Performs all the pre-processing necessary for the all-sky source search and source class search of neutrino flux from IceCube's muon track data. The majority of the pre-processing is simply moving fixed width text files to numpy pickle files for quicker reading. The more time consuming pre-processing performed is the background PDF calculation and the integration of IceCube's detector effective area over energy and a neutrino flux.

### J02_analyze_all_sky_explained.ipynb

Performs an all-sky astrophysical neutrino source search using three years worth of publicly available IceCube muon track data. For points in the sky in a pre-determined grid, the script finds the best-fit value of number of neutrinos from a source if a source existed at that given point. If there are any real sources in the set, they would have a best-fit likelihood that would deviate from the background / normal distribution. 

### J03_analyze_source_classes_limits_explained.ipynb

Performs an astrophysical neutrino source class search using three years worth of publicly available IceCube muon track data and the 4LAC catalog from the Fermi satellite, a catalog of gamma-ray active galactic nuclei in the universe detected by the satellite. 

## Analysis Scripts

The analysis code is roughly arranged in sequential scripts that prepare data (`A01`), computed the background PDF (`A02`), perform the all-sky source search (`A03`) along with plotting the results (`A03p2`), prepare data for the source class search (`A04p1` and `A04p2`), and finally perform the source class search (`A05`).

All scripts were built in Python 3.9.5.

### IceCubeAnalysis.py

First, we load common libraries and `IceCubeAnalysis`, a custom library with two classes, the first (`SourceSearch`) that handle the creation of the neutrino source likelihood function given a source location to test and the second (`SourceClassSearch`) that handles the loading of 4LAC sources.

### A01_open_and_convert_icecube_data.py

Converts the IceCube data from fixed-width text files to numpy pickle files.

### A02_analyze_background.py

Calculates B_i, the background PDF of the neutrino source search. This is done empirically by scrambling RA of track data in a 6 degree declination angle band in the sky.

### A03_analyze_all_sky_map.py

Performs the all-sky source search. The script breaks the sky into a grid, with step between points defined by `step_size`. For each point, we find the most likely value of astrophysical neutrinos from the source at the given point. Creates a map of the max-likelihood and most-likely number of neutrinos from each point.

### A03p1_submit_all_sky.sbatch

The slurm script to distribute `A03_analyze_all_sky_map.py` to UChicago's computing cluster. 

### A03p2_plot_likelihood_map_allsky.py

Plots the results from the all-sky best-fit source search.
Also prints out the most likely points on the sky.

### A04p1_open_and_convert_4LAC_catalog.py

Opens the 4LAC catalog in the format of a FITS file and converts it to a numpy pickle file.

### A04p2_open_and_convert_icecube_Aeff.py

Loads IceCube detector effective volume, averages it by detector type (number of strings IceCube had deployed), and integrating over energy and the neutrino flux.

### A05_analyze_source_classes_limits.py

For points in the sky from the 4LAC catalog, the function scans over the number of neutrinos in the data from the source class and calculates the likelihood. The number of neutrinos associated to each source in the source class is determined by a weighting, which is described in more detail in the paper. If a statistically significant number of tracks in the data are associated with this source class, the resulting likelihood will peak above 3sigma.

### A05p1_plot_source_classes_limits.py

Plotting the limits, both in energy vs. likelihood and energy vs. flux, based on the outputs from A05_analyze_source_classes_limits.py.