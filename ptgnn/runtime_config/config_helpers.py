# use with partial function
import typing

from ptgnn.runtime_config.config import priority_merge_config, import_as
from ptgnn.runtime_config.run_config import run_config


def run_config_adapter(
        hyper_opt_config: typing.Dict[str, typing.Any],
        default_config: typing.Dict[str, typing.Any] = {},
        report: bool = False,
        verbose: bool = False,
        device: str = None
):
    """
    Function that runs the original function ``run_config`` with parameters from the hyperparameter optimization
    framework and the default parameters. This is meant to work so that the hyper parameter parameters overwrite the
    default parameters defined in the second dict.

    :param hyper_opt_config: Dictionary containing the parameters set by the hyperparameter optimization framework
    :type hyper_opt_config: typing.Dict[str, typing.Any]
    :param default_config: Dictionary containing the default parameter which are to use if not overwritten by the hyper
        parameter optimization parameters
    :type default_config: typing.Dict[str, typing.Any]
    :param report: Whether or not the hyperparameter optimization is active - controlls whether the metrics are to be
        reported using ``ray.train.report(...)``
    :type report: bool
    :param verbose: Whether or not progress is reported
    :type verbose: bool
    :param device: Specifies device to run on. If ``None`` then cuda is used if available, else cpu. If specified then
        this device is used.
    :type device: str
    :return: Returns the output of the ``run_config(...)`` function.
    """
    # merge configs with priority to hyperopt config
    merged_config = priority_merge_config(hyper_opt_config, default_config, in_place=False)

    return run_config(config_dict=merged_config, report=report, verbose=verbose, device=device)


def load_and_merge_default_configs(
        file_path_lists: typing.List[str]
) -> typing.Dict[str, typing.Any]:
    """
    Function that loads stored configs from json or yaml files which are then merged in order of the specified file
    paths.

    :param file_path_lists: List of file paths which are to be loaded to a dictionary
    :type file_path_lists: typing.List[str]
    :return: Config dictionaries loaded from the files
    :rtype: typing.Dict[str, typing.Any]
    """
    # initial dict
    config_dict = {}

    # iterate over file paths, load and merge
    for file_path in file_path_lists:

        # load config
        config = import_as(file_path)

        # merge config
        config_dict = priority_merge_config(config_dict, config, in_place=True)

    return config_dict
