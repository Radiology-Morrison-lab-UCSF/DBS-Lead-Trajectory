# DBS Lead Trajectory

This repository performs image registration and atlas-based feature extraction for deep brain stimulation (DBS) lead trajectory analysis in individuals with Parkinson's disease. The pipeline registers patient postoperative imaging (CT and MRI) to MNI space, transforms a brain atlas into patient space, and identifies atlas-defined brain regions intersected by DBS lead trajectory segmentations. The repository is part of ongoing research in the Radiology-Morrison-lab-UCSF.

## Project Overview
The workflow consists of the following:
1. Registration of postoperative CT or MRI images to MNI space using ANTsPy
2. Transformation of an atlas from MNI space into patient space
3. Extraction of brain regions intersected by DBS lead trajectories
4. Construction of binary feature matrices representing lead-region intersections
5. Statistical modeling using stability-selected LASSO and ridge regression model
6. Visualization of lead trajectories, atlas overlaps, and regression results

## Code Files

### ```ct_registration.py```
A Python code file that performs ANTs image registration of postoperative CT images to MNI space, transforms the atlas into patient space, and extracts binary brain-region intersection vectors from DBS lead segmentations. Returns folder containing all transformed images and excel file containing brain region vectors.

### ```mri_registration.py```
A Python code file containing an image registration pipeline for MRI using ANTs. Returns folder containing all transformed images and excel file containing brain region vectors.

### ```combined_dataset.py```
A Python code file to combine CT image and MR image datasets for downstream analysis and appends covariate data including age, sex, number of leads, DBS target, and baseline MDS-UPDRS score. Outputs LEDD outcomes dataset and MoCA outcomes dataset in excel file format.

### ```ledd_lasso_ridge.py```
A Python code file to perform stability-selected LASSO feature selection and ridge regression modeling for LEDD outcomes. Also runs OLS modesl comparing model containing clnical covariates only and model containing both clinical covariates and selected brain region features. Returns LASSO feature selection bar plot, ridge regression coefficients and 95% confidence intervals plot, and csv files containing OLS results and ridge regression results.

### ```moca_lasso_ridge.py```
A Python code file to perform stability-selected LASSO feature selection and ridge regression modeling for MoCA outcomes. Returns LASSO feature selection bar plot, ridge regression coefficients and 95% confidence intervals plot, and csv files containing ridge regression results.

###  ```plot_lasso_path.py```
A Python code file to generate plots of LASSO coefficient paths and feature selection frequencies across different values of alpha (regularization parameter).

### ```plot_strat_target.py```
A Python code file to generate plots illustrating ridge regression results from dataset stratified by target (GPi vs. STN).

## Data
The included data is derived from real patient data but has been de-identified to comply with ethical guidelines.

### LEDD_dataset.xlsx
An excel file LEDD outcome dataset containing:
- Patient ID
- Percent change in levodopa equivalent daily dose (ΔLEDD)
- Binary brain region features
- DBS target (STN vs. GPi)
- Hemispheres Treated (Bilateral vs. Unilateral)
- %Δ MDS-UPDRS III (percent improvement in Movement Disorder Society Unified Parkinson’s Disease Rating Scale Part III motor scores between OFF- and ON-medication states)
- Sex
- Age

### MoCA_dataset.xlsx
An excel file MoCA outcome dataset containing:
- Patient ID
- MoCA Simple Discrepency Score (SDS)
- Binary brain region features
- DBS target (STN vs. GPi)
- Hemispheres Treated (Bilateral vs. Unilateral)
- %Δ MDS-UPDRS III (percent improvement in Movement Disorder Society Unified Parkinson’s Disease Rating Scale Part III motor scores between OFF- and ON-medication states)
- Sex
- Age

## Atlases
Brain atlas folder containing atlas file (```Atlas_QSM.nii``` and ```Atlas_QSMdgm.nii```) and adjoining ROI text file (```Atlas_QSM.txt``` and ```Atlas_QSMdgm.txt```)for region identification.

# Licenses
**Copyright 2026 UCSF**

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at:  
[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

- All code files in this repository are licensed under the Apache License 2.0.
- `ledd_dataset.xlsx` and `moca_dataset.xlsx` were acquired at the University of California San Francisco and are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
