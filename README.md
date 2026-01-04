# Geodetic Finite-Fault Slip Inversion Framework

A Python-based finite-fault inversion framework using geodetic observations 
(e.g. InSAR, GNSS) for coseismic and postseismic slip modeling.

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

Geodetic-Finite-Fault-Inversion
├── src/                    # Core inversion codes
│   ├── main_inv.py         # Main inversion entry
│   ├── Green_functions.py  # Green's function construction
│   ├── Inversion.py        # Inversion solver
│   ├── Faults_Construct.py # Fault geometry definition
│   └── utils/              # Supporting modules
│
├── example/                # Example datasets
│   └── Ridgecrest/
│       ├── input/          # Geodetic observations (GNSS / InSAR)
│       └── model/          # Inversion configuration files
│
└── README.md

The example dataset corresponds to the 2019 Ridgecrest earthquake sequence
and is provided for demonstration and testing purposes only.

## Reference:
Yajun Zhang, Xiaohua Xu, Constraining shallow slip deficit with phase gradient data, Geophysical Journal International, Volume 244, Issue 1, January 2026, ggaf427, https://doi.org/10.1093/gji/ggaf427