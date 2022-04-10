# Source Search in 3 Years of Muon Track Data in IceCube

This repository documents the analysis code used to produce the results from the following paper:

Revisiting AGN as the Source of IceCube's Diffuse Neutrino Flux, D. Smith *et al.* e-Print: [2007.12706](https://arxiv.org/abs/2007.12706) [astro-ph.HE]

Please direct all code-related questions to [danielsmith@uchicago.edu](mailto:danielsmith@uchicago.edu).

## Data

https://icecube.wisc.edu/science/data/PS-3years

https://arxiv.org/abs/1905.10771

## Example Analysis in Jupyter Notebooks

### J01_preprocess_data.ipynb

### J02_analyze_all_sky_explained.ipynb

### J03_analyze_source_classes_limits_explained.ipynb

## Analysis Scripts

The analysis code is roughly arranged in sequential scripts that prepare data (`A01`), plot results (`A02`), and calculate systematic uncertainties (`A03`) and biases (`A04`) before calculating the bulk attenuation (`A05`) and plotting the figures in the paper (`A06`). A description of each script is below.

All scripts were built in Python 3.9.5.

### IceCubeAnalysis.py

### A01_open_and_convert_icecube_data.py

### A02_analyze_background.py

### A03_analyze_all_sky_map.py

### A03p1_submit_all_sky.sbatch

### A03p2_plot_likelihood_map_allsky.py

### A04p1_open_and_convert_4LAC_catalog.py

### A04p2_open_and_convert_icecube_Aeff.py

### A05_analyze_source_classes_limits.py