# MOFS-RFGA: Multi-Objective Feature Selection with ReliefF and Genetic Algorithm

## Overview
The MOFS-RFGA (Multi-Objective Feature Selection - ReliefF and Genetic Algorithm) algorithm is a hybrid feature selection technique that combines the advantages of filter and wrapper methods. It uses ReliefF for feature scoring and a genetic algorithm to search for optimal feature subsets, with innovative crossover and mutation operators informed by feature scoring. This approach improves search efficiency and classification performance while significantly reducing the feature set.

## Features
- **Hybrid Approach**: Combines filter (ReliefF) and wrapper (genetic algorithm) methods.
- **Feature Scoring**: Uses ReliefF to prioritize features based on their importance to class distinction.
- **Custom Genetic Operators**: Incorporates feature-scoring-based crossover and mutation for efficient search.
- **Multi-Objective Optimization**: Balances feature reduction and classification performance.

## Requirements
- **Programming Language**: Python
- **Dependencies**: Install the following libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`


