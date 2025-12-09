[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17831454.svg)](https://doi.org/10.5281/zenodo.17831454)

# SustainDiffusion

This repository contains the replication package for the paper _"SustainDiffusion: Optimising the Social and Environmental Sustainability of Stable Diffusion Models"_ accepted for publication at ICSE 2026 Research Track https://arxiv.org/abs/2507.15663 .

## Repository Structure

- `analysis/`: Contains the code to reproduce the analysis reported in the paper.
- `sustaindiffusion/`: Contains the source code of SustainDiffusion.

Refer to the README in each subdirectory for more details.

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate sdenv
```

### pip

- Install `python=3.10`

```bash
pip install -r requirements.txt
```

## Citation Request

Please cite our paper if you use _"SustainDiffusion"_ in your study:

```bibtex
@inproceedings{sustaindiffusion_2026,
    title = {{SustainDiffusion}: {Optimising} the {Social} and {Environmental} {Sustainability} of {Stable} {Diffusion} {Models}},
    url = {https://arxiv.org/abs/2507.15663},
    author = {d'Aloisio, Giordano and Fadahunsi, Tosin and Choy, Jay and Moussa, Rebecca and Sarro, Federica},
    booktitle={48th IEEE/ACM International Conference on Software Engineering, ICSE 2026},
    year = {2026}
}
```
