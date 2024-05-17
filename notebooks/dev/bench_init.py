import argparse
import os

import sys
from functools import partial

sys.path.append("../../")

from ptgnn.runtime_config.config import import_as
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
        os.path.join("../hyperoptimization/src", default_config['data']['dataset']['type'])
    )
    if verbose:
        print(default_config)

    test_config = {
        'training': {
            'n_max_epochs': 2 # test 10
        },
        'model': {
            'modules': {
                1: {
                    'times': 10,
                    'parameter': {
                        'dropout': 0.0
                    }
                }
            }
        }
    }

    test_fn = partial(
        run_config_adapter,
        default_config=default_config,
        report=False,
        verbose=True,
        device='cuda'
    )
    print(test_fn(test_config))
