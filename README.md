# SC-XGB-Model

This repository provides code for intelligent multi-objective scheduling rule extraction using a Spatiotemporal Correction of decision variables with XGBoost (SC-XGB) model, 
along with a baseline XGB_Model, SC-BP_Model, and BP_Model for comparison.

Reservoir system scheduling often requires intelligent rules that can adapt to complex multi-objective trade-offs. 
This project proposes a data-driven scheduling rule extraction framework based on optimized samples derived from design flood scenarios.

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


