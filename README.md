**Panobinostat IC50 Reproduction**

Overview

This repository contains a self-contained reproduction study for drug response modelling, focused on panobinostat. The work reproduces (and lightly extends) the methodology of [Park et al. (2023)](https://www.nature.com/articles/s41598-023-39179-2) for predicting ln(IC50) from genomic features.

At a high level, the pipeline consists of three layers:

- Data loading and split management (pre-prepared .npz artefacts)
- Predictive modelling (ridge regression, EC-11K vs MC-9K)
- Local explainability (LIME on sensitive cases)

**Repository Structure**

*Note: raw data files are not included in this repository; see Data note below*

. <br>
├── data/ <br>
│   └── (not included; see Data note) <br>
│ <br>
├── compare_and_lime.py <br>
├── run.py <br>
├── create.py <br>
├── outputs/ <br>
│   └── (generated artefacts) <br>
└── README.md <br>

**Data artefacts**

EC11K_*: gene expression features

MC9K_*: mutation indicator features

Targets are ln(IC50) values

Split files (`*_1.npz`, `*_2.npz`, `*_3.npz`) index the train-test splits

**Data note**

The eight `.npz` files used in Park et. al. (2023) and this project were originally obtained from [this repository](https://mega.nz/folder/SXhXTKYK#T1skByytOWobMHa7Hv3j6A). 

As of **8 Feb 2026**, the relevant files appear to be unavailable. 

Filenames in this repository were **slightly renamed for clarity**, while file contents are unchanged.

**Script Overview**

`compare_and_lime.py`: Core functions for ridge regression reproduction and LIME explainability.

`create.py`: Reporting and visualisation functions (json, figures, LaTeX, optional PDF).

`run.py`: Main entry point that loads data, runs the reproduction pipeline, and generates outputs.

**Running the Project**

Run `run.py` from the repo root.

Outputs are written to `./outputs/`

A JSON results file is generated; PDF report generation is optional.