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

## Directory Structure

Geodetic-Finite-Fault-Inversion/
├── example/
│   └── Ridgecrest/
│       ├── input/          # GNSS, InSAR, phase-gradient observations
│       └── model/          # Inversion configuration files
└── src/
    ├── Faults_Construt.py  # Fault geometry construction
    ├── Green_functions.py  # Green’s function calculation
    ├── Forward_modeling.py
    ├── Inversion.py       # Core inversion routines
    ├── Subsample.py       # Data subsampling
    ├── Read_Config.py     # Configuration parser
    ├── main_inv.py        # Main inversion entry
    └── main_inv_ridgecrest.py

The example dataset corresponds to the 2019 Ridgecrest earthquake sequence
and is provided for demonstration and testing purposes only.

## Reference:
Yajun Zhang, Xiaohua Xu, Constraining shallow slip deficit with phase gradient data, Geophysical Journal International, Volume 244, Issue 1, January 2026, ggaf427, https://doi.org/10.1093/gji/ggaf427
