import argparse
import os
import pathlib
import sys

import ray
from ray import tune, train
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.stopper import TrialPlateauStopper
sys.path.append("../../")
from ptgnn.runtime_config.config import import_as, export_as
from ptgnn.runtime_config.config_helpers import load_and_merge_default_configs, run_config_adapter


def do_parsing():
    parser = argparse.ArgumentParser(description="Process input config path.")
    parser.add_argument("config_path", metavar="p", type=str, nargs=1, help="path to config file")

    parser.add_argument(
        "--verbose",
        dest='verbose',
        action='store_const',
        const=True,
        default=False,
        help="Whether or not to print auxiliary information."
    )

    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = do_parsing()

    # load config file
    config_path = args.config_path[0]
    benchmark_config = import_as(config_path)

    # check if verbose flag set
    verbose = args.verbose

    # load default config
    default_config = load_and_merge_default_configs(
        benchmark_config['config_files']
    )

    # create absolute data path for dataset as hyperopt changes directory
    default_config['data']['dataset']['root'] = os.path.abspath(
        os.path.join("src", default_config['data']['dataset']['type'])
    )
    if verbose:
        print(default_config)

    # ==================================================================================================================
    # define search space
    def eval_search_space(d):
        for key in d.keys():
            temp = d[key]

            if isinstance(temp, dict):
                eval_search_space(temp)
            elif isinstance(temp, str):
                d[key] = eval(temp)
            else:
                raise Exception("unknown type in search space, only use str as values")
        return d


    search_space = eval_search_space(benchmark_config['search_space'])
    if verbose:
        print("search space: ", search_space)

    # define trainable function for ray
    def trainable_function(config):
        run_config_adapter(
            config,
            default_config=default_config,
            report=True,
            verbose=False,
            device="cuda"
        )
    
    #define loading/creation of dataset
    def create_load_ds(config):
        # import sys
        # sys.path.append("../../")
        from ptgnn.runtime_config.run_config import fetch_loaders

        print("Begin initial dataset loading/creation:")
        # load/create dataset in one process
        _, _, _ = fetch_loaders(config['data'], verbose=True)

        print("Finished initial dataset loading/creation")
        return

    # execute initial loading
    create_load_ds(default_config)

    # ==================================================================================================================
    # fetch score to optimize
    optimization_score = "val_" + default_config['training']['optimization_metric']
    score_mode = default_config['training']['optimization_metric_mode']

    # fetch hyperopt settings
    hyper_settings = benchmark_config['hyper_settings']

    # init ray
    # ray.init(runtime_env={"env_vars": {"RAY_AIR_NEW_OUTPUT": "0"}})

    # build stopper if in config
    stopper = None
    if 'stopper' in hyper_settings:
        stopper = TrialPlateauStopper(
            metric=optimization_score,
            num_results=hyper_settings['stopper']['num_results'],
            metric_threshold=hyper_settings['stopper']['metric_threshold'],
            mode=score_mode,
            grace_period=hyper_settings['scheduler']['grace_period']
        )

    ray.init(runtime_env={
        "working_dir": "../../",
        "py_modules": ["../../ptgnn"]
    })

    # set up and run tuner
    # tuner = tune.Tuner(tune.with_resources(trainable_function, {"cpu": 5, "gpu": 1}),
    tuner = tune.Tuner(tune.with_resources(
        trainable_function, {"cpu": 3, "gpu": 0.2}),
        # trainable=trainable_function,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric=optimization_score,
            mode=score_mode,
            search_alg=HyperOptSearch(
                metric=optimization_score,
                mode=score_mode,
                random_state_seed=13
            ),
            scheduler=ASHAScheduler(
                max_t=default_config['training']['n_max_epochs'],
                grace_period=hyper_settings['scheduler']['grace_period'],
                reduction_factor=hyper_settings['scheduler']['reduction_factor'],
                brackets=hyper_settings['scheduler']['brackets']
            ),
            num_samples=hyper_settings['num_samples'],
            max_concurrent_trials=hyper_settings['max_concurrent_trials'],
        ),
        run_config=train.RunConfig(
            storage_path=os.path.abspath("ray_temp"),
            progress_reporter=CLIReporter(
                metric_columns=[optimization_score],
            ),
            stop=stopper
        )
    )
    results = tuner.fit()

    # ==================================================================================================================
    # store results
    # fetch output path
    output_path = benchmark_config['output_dir']

    # make sure that output_dir exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save general configs
    export_as(default_config, os.path.join(output_path, "general_config.yaml"), save_type='yaml')
    # save results dataframe
    results.get_dataframe().to_csv(os.path.join(output_path, "results.csv"), index=None)

    # for each trial save results
    for result in results:
        # get metrics
        trial_metrics = result.metrics_dataframe

        # get trial id
        trial_id = trial_metrics.trial_id[0]

        # get config
        trial_config = result.config

        # saving
        trial_metrics.to_csv(os.path.join(output_path, f"{trial_id}.csv"), index=None)
        export_as(trial_config, os.path.join(output_path, f"{trial_id}.yaml"), save_type='yaml')
