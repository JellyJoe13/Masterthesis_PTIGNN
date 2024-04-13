import argparse
import os
import sys

import pandas as pd
import ray
from ray import tune, train
from ray.tune import CLIReporter

sys.path.append("../../")
from ptgnn.runtime_config.config import import_as, export_as
from ptgnn.runtime_config.config_helpers import load_and_merge_default_configs, run_config_adapter
from ptgnn.runtime_config.final_test_suggester import FinalSearcher


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

    parser.add_argument(
        "-cpu",
        type=int,
        default=5,
        help="specifies number of cpus to use per trial"
    )

    parser.add_argument(
        "-gpu",
        type=float,
        default=0.2,
        help="specifies number of gpus to use per trial"
    )

    parser.add_argument(
        "-device",
        type=str,
        default="cuda",
        help="specifies which device to use for the trials"
    )

    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = do_parsing()

    # load config file
    config_path = args.config_path[0]
    test_config = import_as(config_path)

    # check if verbose flag set
    verbose = args.verbose

    # load device and general config
    num_cpu = args.cpu
    num_gpu = args.gpu
    device = args.device

    # load default config
    default_config = load_and_merge_default_configs(
        test_config['config_files']
    )

    # create absolute data path for dataset as hyperopt changes directory
    default_config['data']['dataset']['root'] = os.path.abspath(
        os.path.join("src", default_config['data']['dataset']['type'])
    )
    # load results
    loading_dir = os.path.abspath(test_config['loading_dir'])
    results = pd.read_csv(os.path.join(loading_dir, "results.csv"))
    if verbose:
        print(results)

    n_best = test_config['n_best']

    # fetch optimization_metric
    optimization_metric = default_config['training']['optimization_metric']
    optimization_metric_mode = default_config['training']['optimization_metric_mode']

    # select n_best trials
    metric_val_sorted_idx = results[f"val_{optimization_metric}"].argsort()

    # in case metric is an accuracy value, then invert sort order
    if optimization_metric_mode == 'max':
        metric_val_sorted_idx = metric_val_sorted_idx[::-1]

    if verbose:
        print(metric_val_sorted_idx)

    # search_space = ...
    # if verbose:
    #     print("search space: ", search_space)

    # define trainable function for ray
    def trainable_function(config):
        run_config_adapter(
            config,
            default_config=default_config,
            report=True,
            verbose=False,
            device=device
        )
    
    # define loading/creation of dataset
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
    optimization_score = "test_" + default_config['training']['optimization_metric']
    score_mode = default_config['training']['optimization_metric_mode']

    # init final 'searcher'
    searcher = FinalSearcher(
        {
            f"rank{i+1}_{results.iloc[idx].logdir}": import_as(
                os.path.join(os.path.abspath(test_config['loading_dir']), f"{results.iloc[idx].logdir}.yaml"),
                loading_type='yaml'
            )
            for i, idx in enumerate(metric_val_sorted_idx[:n_best])
        }
    )

    # init ray
    ray.init(runtime_env={
        "working_dir": "../../",
        "py_modules": ["../../ptgnn"]
    })

    # set up and run tuner
    # tuner = tune.Tuner(tune.with_resources(trainable_function, {"cpu": 5, "gpu": 1}),
    tuner = tune.Tuner(tune.with_resources(
        trainable_function, {"cpu": num_cpu, "gpu": num_gpu}),
        # trainable=trainable_function,
        param_space=None,
        tune_config=tune.TuneConfig(
            metric=optimization_score,
            mode=score_mode,
            num_samples=n_best,
            max_concurrent_trials=n_best,
            search_alg=searcher,
        ),
        run_config=train.RunConfig(
            storage_path=os.path.abspath("ray_temp"),
            progress_reporter=CLIReporter(
                metric_columns=[optimization_score],
            ),
        )
    )
    results = tuner.fit()

    # ==================================================================================================================
    # store results
    # define output directory
    output_dir = os.path.join(loading_dir, "final")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # save general configs
    export_as(default_config, os.path.join(output_dir, "general_config.yaml"), save_type='yaml')
    # save results dataframe
    results.get_dataframe().to_csv(os.path.join(output_dir, "results.csv"), index=None)

    # for each trial save results
    for result in results:
        # get metrics
        trial_metrics = result.metrics_dataframe

        # get trial id
        trial_id = searcher.mapping[trial_metrics.trial_id[0]]

        # get config
        trial_config = result.config

        # saving
        trial_metrics.to_csv(os.path.join(output_dir, f"{trial_id}.csv"), index=None)
        export_as(trial_config, os.path.join(output_dir, f"{trial_id}.yaml"), save_type='yaml')
