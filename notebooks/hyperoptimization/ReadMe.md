# Hyper (Parameter) Optimization

This folder contains the scripts executing hyper parameter optimization and the following final test based on the 3 best
configurations.

## Contents:

- configs: Configuration files which are of either type:
  + Instruction files: Contains information on which configuration files ar to be merged, to which path to store the results and hyper parameter control elements
  + Configuration files: Contains parameter settings which are to be merged to achieve a total configuration
- (ray_temp): Folder which the ray framework may create to store intermediary results (depends on which script is used)
- results: Results from executing the configurations in the config files. Contains both hyper parameter optimization and final test results - and which config produced it.
- results_unmasked_chiral: Folder containing old results before chiral data was masked
- (src): Folder created when running hyper parameter optimization or final tests - stores the downloaded and processed datasets
- .gitignore
- Experiment-plotting.ipynb: Notebook to plot final test results of one configuration
- full_test_scipt.py & full_test_script_linux.py: Scripts to execute the final test
- hyper_opt_script*.py: Different versions of the hyper parameter optimization script
- Hyperoptimization-plotting.ipynb: Notebook to plot hyperoptimization validation metric results
- load_old_results.ipynb: Notebook for loading aborted ray parameter optimization using intermediate results
- run_test_script.py: Alternative final test script - does not use ray