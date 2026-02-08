**Panobinostat IC50 Reproduction**

Overview

This repository contains a self-contained reproduction study for drug response modelling, focused on panobinostat. The work reproduces (and lightly extends) the methodology of Park et al. (2023) for predicting ln(IC50) from genomic features.

At a high level, the pipeline consists of three layers:

- Data loading and split management (pre-prepared .npz artefacts)
- Predictive modelling (ridge regression, EC-11K vs MC-9K)
- Local explainability (LIME on sensitive cases)

**Repository Structure**

.
├── data/
│   ├── EC11K_Panobinostat.npz
│   ├── MC9K_Panobinostat.npz
│   ├── EC11K_Panobinostat_1.npz
│   ├── EC11K_Panobinostat_2.npz
│   ├── EC11K_Panobinostat_3.npz
│   ├── MC9K_Panobinostat_1.npz
│   ├── MC9K_Panobinostat_2.npz
│   └── MC9K_Panobinostat_3.npz
│
├── compare_and_lime.py
├── run.py
├── create.py
├── outputs/
│   └── (generated artefacts)
└── README.md

**Data artefacts**

EC11K_*: gene expression features

MC9K_*: mutation indicator features

Targets are ln(IC50) values

Split files (`*_1.npz`, `*_2.npz`, `*_3.npz`) index the train-test splits

**Script Overview**

`compare_and_lime.py`: Core functions for ridge regression reproduction and LIME explainability.

`create.py`: Reporting and visualisation functions (json, figures, LaTeX, optional PDF).

`run.py`: Main entry point that loads data, runs the reproduction pipeline, and generates outputs.

**Running the Project**

Run `run.py` from the repo root.

Outputs are written to `./outputs/`

A PDF report and json are generated.