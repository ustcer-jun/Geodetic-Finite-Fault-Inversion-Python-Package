# Geodetic Finite-Fault Slip Inversion Framework

A Python framework for finite-fault slip inversion using geodetic observations, including InSAR phase gradients and GNSS displacements.
The package supports joint inversion, flexible data weighting, subsampling, and fault geometry construction.

This project is designed for earthquake source studies, geodetic inversion, and research-oriented applications.

## Introduction

This package provides a finite-fault inversion framework based on geodetic
observations, including GNSS and InSAR data. The code is designed to estimate
fault slip distributions by solving a linear inverse problem constrained by
physical and geometric parameters.

The framework supports:
- Multi-source geodetic data integration
- Flexible fault geometry parameterization
- Forwarding 3D displacement using Okada dislocation theory
- Regularized inversion with optional smoothing constraints

## Requirements
This project supports both **conda** and **pip** based environments.

### Conda (recommended)
To fully reproduce the development environment:

```bash
conda env create -f environment.yml
conda activate geodetic-ffi

### Core dependencies
```text
numpy==2.3.4
xarray==2025.10.1
matplotlib==3.10.7
pyproj==3.7.2
scipy==1.16.2
pygmt==0.17.0
psutil==7.1.1
```
## Jupyter environment
```text
jupyterlab==4.4.10
notebook==7.4.7
ipykernel==7.0.1
matplotlib-inline==0.2.1
```
## Optional: for reproducibility and system dependencies
```text
pygmt requires GMT 6.x installed on the system:
macOS: brew install gmt
Linux: sudo apt install gmt gmt-dcw gmt-gshhg
Conda users: conda install -c conda-forge pygmt gmt
```
## Directory Structure
```text
Geodetic-Finite-Fault-Inversion/
├── example/
│   └── Ridgecrest/
│       ├── input/          # GNSS, InSAR, phase-gradient observations
│       └── model/          # Inversion configuration files
└── src/
    ├── Faults_Construt.py      # Fault geometry construction
    ├── Green_functions.py      # Green’s function calculation
    ├── Forward_modeling.py
    ├── Inversion.py            # Core inversion routines
    ├── Subsample.py            # Data subsampling
    ├── Read_Config.py          # Configuration parser
    ├── main_inv.py             # Main inversion entry
    └── main_inv_ridgecrest.py

The example dataset corresponds to the 2019 Ridgecrest earthquake sequence
and is provided for demonstration and testing purposes only.
```
## Reference:
Yajun Zhang, Xiaohua Xu, Constraining shallow slip deficit with phase gradient data, Geophysical Journal International, Volume 244, Issue 1, January 2026, ggaf427, https://doi.org/10.1093/gji/ggaf427
