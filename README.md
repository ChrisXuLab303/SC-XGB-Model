# SC-XGB-Model

This repository provides code for intelligent multi-objective scheduling rule extraction using a Spatiotemporal Correction of decision variables with XGBoost (SC-XGB) model, 
along with a baseline BP_Model for comparison.

Reservoir system scheduling often requires intelligent rules that can adapt to complex multi-objective trade-offs. 
This project proposes a data-driven scheduling rule extraction framework based on optimized samples derived from design flood scenarios.
Data availability: https://github.com/ChrisXuLab303/SC-XGB-flood-scheduling-dataset 

# Dataset & Experiment Setup

- Training & Validation: Optimized scheduling samples from the design flood scenarios.
- Testing: Historical flood scenarios to evaluate applicability and generalization of the models.

Testing Set (Historical Flood Events)
| Date       | Return Period |
|------------|----------------|
| 2005-08-21 | 10-year        |
| 2010-06-21 | 20-year        |
| 2016-06-15 | 50-year        |
| 2020-07-06 | 100-year       |

Python is an open-source programming language widely used for scientific computing, data analysis, and machine learning. It is freely available at: https://www.python.org/. 
Install dependencies (e.g. Python 3.8, XGBoost ≥1.3.0, Scikit-learn ≥0.24, NumPy ≥1.19, Pandas ≥1.1), then run SC-XGB_Model.py and BP_Model.py

Notes
Please ensure the input data files are correctly placed or adjusted in the scripts.
Results from both models can be compared on accuracy, applicability, and robustness to flood event scale.

Contact

Maintainer: HUILI WANG

Feel free to reach out for questions or collaboration.
