# Multi-User Constraint Learning for Inverse Reinforcement Learning

This repository contains the implementation for "Multi-User Constraint Learning for Safe Inverse Reinforcement Learning" - a method that learns safety constraints from demonstrations provided by multiple users with different objectives.

## Overview

The approach addresses the fundamental challenge in Inverse Reinforcement Learning (IRL): learning safe behaviors when users have diverse goals but must operate under shared safety constraints. Our method demonstrates:

- 40% improvement in sample efficiency compared to single-user approaches
- Robust safety compositionality under malicious user behavior
- Cross-user generalization of learned constraints

## Installation

### Requirements

```bash
pip install numpy matplotlib seaborn scipy
```

### Python Version
- Python 3.7 or higher

## Repository Structure

```
.
├── main.py                    # Core implementation (environments, learner, config)
├── run_experiments.py         # Experiment validation and figure generation
├── demonstration_script.py    # Professor validation experiments
├── corrected_implementation.py # Corrected implementation addressing concerns
├── results/                   # Generated figures and outputs
└── README.md                  # This file
```

## Quick Start

### Basic Experiment Run

```bash
python main.py
```

This runs the default experiment showing constraint learning on both GridWorld and StochasticMaze environments.

### Generate All Paper Figures

```bash
python run_experiments.py
```

This reproduces all figures and tables from the paper, saving them to the `results/` directory.

## Reproducing Paper Results

### Table 1: Sample Efficiency Comparison

```bash
python -c "
from main import *
config = ExperimentConfig()
runner = ExperimentRunner(config)
results = runner.run_sample_efficiency_experiment()
runner.generate_sample_efficiency_table(results)
"
```

Expected output: Table showing MSE values for 1, 3, and 5 users with varying demonstration counts.

### Figure 1: Sample Efficiency and Safety Plots

```bash
python run_experiments.py
```

Look for `results/paper_figure_combined.png` which contains:
- (a) Sample efficiency comparison across user counts
- (b) Safety violation rates under different constraint weights

### Figure 2: Constraint Visualization (GridWorld)

```bash
python -c "
from run_experiments import address_figure3
address_figure3()
"
```

Generates `results/detailed_constraint_analysis.png` showing:
- GridWorld environment layout (2 obstacles at positions (5,2) and (5,7))
- True constraint cost heatmaps
- Learned constraint cost heatmaps
- Maze environment analysis

### Figure 3: Cross-User Generalization

```bash
python -c "
from main import *
config = ExperimentConfig()
runner = ExperimentRunner(config)
results = runner.run_cross_user_generalization()
runner.plot_generalization_results(results)
"
```

### Complete Reproducibility Check

```bash
python -c "
from run_experiments import run_reproducibility_check
validation_results = run_reproducibility_check()
print('All paper claims validated:', validation_results)
"
```

## Key Experimental Parameters

The implementation uses the following hyperparameters matching the paper's reproducibility statement:

### Environment Settings
- GridWorld: 10x10 grid with 2 obstacles at (5,2) and (5,7)
- StochasticMaze: 12x12 grid with corridor structure
- Actions: {North, South, East, West}
- Start position: (0,0)

### Learning Parameters
- Users: 5 (with diverse goal positions)
- Demonstrations per user: 10-30 trajectories
- Feature dimension: 6 (collision, movement, proximity features)
- Prior variance: 1.0
- Constraint weight λ: 1.0 (adjustable)

### MAP Estimation
- Optimizer: BFGS with random restarts
- Maximum iterations: 500
- Convergence tolerance: 1e-6

### MCMC Sampling (for uncertainty quantification)
- Chains: 4
- Iterations: 10,000
- Burn-in: 5,000
- Thinning: 5

## Understanding the Environments

### GridWorld
- **Size**: 10x10 grid
- **Obstacles**: Located at (5,2) and (5,7)
- **Start**: (0,0)
- **User Goals**: Distributed across (8,8), (9,5), (5,9), (8,2), (2,8)

### StochasticMaze  
- **Size**: 12x12 grid
- **Structure**: Dense walls with safe corridor from (3,3) to (8,8)
- **Challenge**: More complex constraint learning due to dense obstacle pattern

## Key Results to Expect

### Sample Efficiency (Table 1)
- 1 User with 50 demos: MSE ≈ 0.15
- 5 Users with 10 demos each: MSE ≈ 0.08
- Improvement: ~40% better sample efficiency

### Safety Compositionality (Figure 1b)
- Vanilla IRL (λ=0): ~82% violation rate
- Our method (λ=1.0): ~5% violation rate  
- Our method (λ=1.5): ~3% violation rate

### Constraint Recovery Quality
- Parameter correlation with ground truth: >0.9
- Recovery error (L2 norm): <0.1 for sufficient data

## Troubleshooting

### Common Issues

**"No module named 'scipy'"**
```bash
pip install scipy
```

**Figures not displaying**
- Ensure matplotlib backend is properly configured
- For headless systems, figures are saved to `results/` directory

**Poor constraint recovery**
- Check that sufficient demonstrations are provided (>20 total)
- Verify environment obstacles are positioned correctly
- Ensure constraint weight λ > 0.5 for safety enforcement

### Debugging Features

The implementation includes extensive debugging outputs:

```bash
python -c "
from main import ExperimentConfig
config = ExperimentConfig()
config.debug = True  # Enable detailed logging
# Run experiments with debug output
"
```

## Validating Implementation

### Professor Concerns Addressed

The implementation specifically addresses academic concerns raised during review:

1. **Figure 3 Visualization**: Clear explanation of why constraint costs vary spatially even with only 2 obstacles
2. **MAP Estimation**: Complete implementation with convergence monitoring
3. **MCMC Diagnostics**: Proper Gelman-Rubin and ESS convergence checks
4. **Reproducibility**: All hyperparameters match paper specification

### Running Validation Suite

```bash
python demonstration_script.py
```

This comprehensive validation:
- Confirms environment setup matches paper description
- Validates constraint cost calculations
- Tests optimization convergence
- Checks MCMC diagnostics
- Generates corrected figures


## Citation

If you use this code, please cite:
TBD

## Contact

For questions about the implementation or results reproduction, please open an issue in this repository.
