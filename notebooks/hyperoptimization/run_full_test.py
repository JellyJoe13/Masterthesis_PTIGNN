import argparse
import os

import pandas as pd

from ptgnn.runtime_config.config import import_as, priority_merge_config, export_as
from ptgnn.runtime_config.config_helpers import load_and_merge_default_configs
from ptgnn.runtime_config.run_config import run_config


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


if __name__ == "__main__":
    # parse argument:
    # - config_file_path
    # - verbose
    # - ...
    args = do_parsing()

    # load config file
    config_path = args.config_path[0]
    test_config = import_as(config_path)

    # check if verbose flag set
    verbose = args.verbose

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

    # define output directory
    output_dir = os.path.join(loading_dir, "final")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # iterate over few selected indices and run experiment
    for i, idx in enumerate(metric_val_sorted_idx[:n_best]):

        print(f"Starting rank {i + 1}")

        # get trial identifier
        trial_identifier = results.iloc[idx].logdir

        # load trail config
        trial_config = import_as(
            os.path.join(os.path.abspath(test_config['loading_dir']), f"{trial_identifier}.yaml"),
            loading_type='yaml'
        )

        # deal with issue that model.modules may have the key as a str, not int
        if 'model' in trial_config and 'modules' in trial_config['model']:
            new_dict = {
                int(key): value
                for key, value in trial_config['model']['modules'].items()
            }
            trial_config['model']['modules'] = new_dict

        # load config
        local_config = priority_merge_config(
            trial_config,
            default_config
        )

        if verbose:
            print(local_config)

        # run config with test
        full_trial_df = run_config(
            local_config,
            report=False,
            verbose=verbose
        )

        # define output name
        output_name = f"rank{i+1}_{trial_identifier}_"

        # save config
        export_as(
            local_config,
            path=os.path.join(output_dir, output_name + "config.yaml"),
            save_type='yaml'
        )
        # save results
        full_trial_df.to_csv(
            os.path.join(output_dir, output_name + "results.csv"),
            index=None
        )
