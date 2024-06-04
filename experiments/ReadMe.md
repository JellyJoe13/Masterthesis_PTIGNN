# Experiment folder
## Contents:

- dev: Notebooks used in development phase, contains tests on components, etc.
- hyperoptimization: Contains results of hyper parameter optimization and the final tests based on the best found
  configurations.
- stereo_tests: Contains notebooks testing stereo-isomer distinguishing capabilities for ChiENN and PTIGNN
- dataset_analysis.ipynb: Analysis of datasets, i.e. how many elements, how many stereocenters, cis/trans bonds, rings, etc.
- dataset_cis-trans_creation.ipynb: Notebook showcasing the creation of new dataset (cis/trans bond classification)
- dataset_multi-center-diastereoisomers_creation.ipynb: Notebook showcasing creation of new dataset (l/u classification)
- performance_measurement.ipynb: Notebooks comparing speed of ChiENN and PTIGNN on GPU and CPU