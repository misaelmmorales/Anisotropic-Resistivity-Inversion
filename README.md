# Anisotropic Resistivity Inversion

Fast and efficient inversion for parallel- and perpendicular-to-bedding plane resistivity from borehole electromagnetic measurements.

## Description

This repository contains Python tools for automated anisotropic resistivity inversion using both gradient-based optimization and physics-guided neural networks (PGNNs). The methods enable efficient formation evaluation and uncertainty quantification from triaxial electromagnetic induction measurements. The implementation includes multiple inversion approaches:

- **Gradient-based inversion**: Classical optimization for resistivity parameter estimation
- **PGNN-based inversion**: Physics-guided neural network approach for fast predictions
- **Data assimilation**: Integration of multiple core data types into the inversion framework
- **Uncertainty quantification**: Probabilistic analysis of inversion results

## Installation

Clone the repository from GitHub:

```bash
git clone https://github.com/misaelmmorales/Anisotropic-Resistivity-Inversion.git
cd Anisotropic-Resistivity-Inversion
```

Then install dependencies. The main file `main.py` imports the necessary libraries and helper functions.

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- scikit-learn
- torch (PyTorch)
- lasio
- scipy

## Key Features

- **Multiple inversion methods**: Gradient-based and PGNN approaches for different use cases
- **Synthetic and field data**: Includes synthetic test cases and real field examples
- **Uncertainty analysis**: Comprehensive uncertainty quantification with Monte Carlo sampling
- **Jupyter notebooks**: Ready-to-use examples for different applications
- **Pre-trained models**: PyTorch models for different data cases included in `models/` folder

## Usage

The repository includes several Jupyter notebooks in the `notebooks/` folder demonstrating different applications:

- `notebooks/main_GradientBased.ipynb` - Gradient-based inversion workflow
- `notebooks/main_PGNN.ipynb` - Physics-guided neural network inversion
- `notebooks/main_DataAssimilation.ipynb` - Data assimilation approach
- `notebooks/main_MultipleDataAssimilation.ipynb` - Multiple data integration
- `notebooks/main_UncertaintyQuantification.ipynb` - Uncertainty quantification analysis
- `notebooks/main_Isotropic.ipynb` - Isotropic inversion baseline
- `notebooks/main_CompareResults.ipynb` - Comparison of inversion methods
- `notebooks/main_make_plots.ipynb` - Figure generation
- `notebooks/demo_GradientBased.ipynb` - Quick-start demonstration

All notebooks import core functions from `main.py`.

## Example Results

<p align="center">
  <img src="https://github.com/misaelmmorales/Anisotropic-Resistivity-Inversion/blob/main/figures/real1-pinn.png" width=850>
</p> 

<p align="center">
  <img src="https://github.com/misaelmmorales/Anisotropic-Resistivity-Inversion/blob/main/figures/uqcsh-all.png" width=850>
  <img src="https://github.com/misaelmmorales/Anisotropic-Resistivity-Inversion/blob/main/figures/uqrss-all.png", width=850>
</p> 

<p align="center">
  <img src="https://github.com/misaelmmorales/Anisotropic-Resistivity-Inversion/blob/main/figures/loss_landscape.gif" width=500>
</p> 

## Project Structure

- `main.py` - Core inversion functions and utilities
- `notebooks/` - Jupyter notebooks for different inversion workflows and demonstrations
- `models/` - Pre-trained PyTorch models
- `results/` - Inversion results and CSV outputs
- `cases/` - Field and synthetic case data (LAS files)
- `uncertainty/` - Uncertainty quantification results
- `datasets/` - Additional data files

## References and Publications

Misael M. Morales, Ali Eghbali, Oriyomi Raheem, Michael Pyrcz, and Carlos Torres-Verdin. (2024). Anisotropic resistivity estimation and uncertainty quantification from borehole triaxial electromagnetic induction measurements: Gradient-based inversion and physics-informed neural network. <em>Computers & Geosciences</em>. https://doi.org/10.1016/j.cageo.2024.105786.

Misael M. Morales, Oriyomi Raheem, Michael Pyrcz, and Carlos Torres-Verdin. (accepted). Anisotropic resistivity inversion and multiple core data assimilation using physics-guided neural network for enhanced petrophysical interpretation and uncertainty quantification. <em>Computers & Geosciences</em>. 
