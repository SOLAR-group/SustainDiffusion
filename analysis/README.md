# Analysis replication package

This directory contains the code to reproduce the analysis reported in the paper _"SustainDiffusion: Optimising the Social and Environmental Sustainability of Stable Diffusion Models"_.

## Usage

- Baselines approaches are implemented in the `baselines` directory.
- Raw data from our experiments can be found in the `s3_<approach>` folders, where `<approach>` is the baseline used (e.g., `s3_fairness`, `s3_energy`, etc.).
- Run the `analysis.ipynb` and `energy_selection.ipynb` notebooks to reproduce the analysis and energy selection results.
