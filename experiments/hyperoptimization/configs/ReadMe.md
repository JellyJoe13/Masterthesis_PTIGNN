# Config files

This folder contains parameter configurations as well as hyperparameter and final script configs. The following content
information is aimed to provide a more intuitive understanding on the intent of the folders and files:

- benchmarking: Configs supplied for hyper parameter optimization, contains information on which config files to use
    while parameter optimizing (which dataset, which model, which training parameter, etc). Also contains parameter
    information, i.e. which parameters to optimize and with what values.
- chem: parameter config instructions related to special trees and stereotypes, thus parameters related to chemical
    information used to build the permutation trees
- datasets: Settings of datasets, e.g. loading path, settings and task types
- final_test: Configs supplied for final test script execution, similarly to 'benchmarking' contains information 
    on which configs to use and how many of the best configurations to test.
- hyper_param_opt: Contains configs related to hyper parameter optimization, e.g. epoch reduction and dataset subsetting
- models: Contains configs on the models, how to build them and which variation to use
- add_rwse.yaml: adds positional embedding to data
- add_test_set.yaml: includes test dataset split into training and measuring process
- enable_p_elu.yaml: include the ELU function in P type nodes (model related)
- general.yaml: general parameters and settings
- linear_reduction.yaml: modification for GINE/ptree_e approach and uses linear reduction instead of addition of embeddings
- unmasked.yaml: Unmasks chiral data (disables default mask)
- use_separate_inv.yaml: Use separate modules for C and Q and thus not share modules of Z and S
- vertex_mode.yaml: Enables vertex graph mode and thus disables edge graph transformation introduced by ChiENN