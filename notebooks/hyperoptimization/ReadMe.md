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
- windows_script.bat

## Execution Procedure

First of all the environment described in the top level ReadMe needs to be created. Then open a command line in the current folder and make sure that the environment is activated.

In there you can use the following commands to execute the scripts:

```bash
CUDA_VISIBLE_DEVICES=x python script_name.py "config_path" --verbose -cpu y -gpu z -device d
```
x can be set to integers describing the devices that should be visible, this can be left out if no restriction is to be
imposed. The script name can be freely set, substitute it for example with 'hyper_opt_script_v3.py' or 
'full_test_script.py'. y and z are variables controlling how many cpus and gpus should be used for each (parallel) 
execution. The 'config_path' is the path to the config file to use for the test script, this defines which dataset is to
use and which configurations should be attempted, etc. d can be either cpu or cuda, depending on whether the gpu should
be used or not.

Example call:
```bash
python hyper_opt_script_v3.py "configs/benchmarking/tox21/benchmark_instructions_tox21_ptree_vertex_default_e_linred.yaml" --verbose -device cpu -cpu 5 -gpu 0
```

If parameters are unclear use 
```bash
python script_name.py --help
```

## Scripts

There is an example of a possible windows bat script that is executing multiple python scripts after each other. 
'windows_script.bat' first activates the conda environment, then asks for confirmation and then starts two python
scripts before waiting for user input again. This can be similarly done for Linux, however in this case it is maybe
easier to chain commands using ';', e.g.:
```bash
python script_a.py "path"; python scipt_b.py "path"
```