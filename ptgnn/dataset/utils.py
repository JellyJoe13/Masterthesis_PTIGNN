import typing


def dict_to_storage_path(
        parameter_dictionary: typing.Dict[str, typing.Any]
) -> str:
    """
    Converts parameters (passed through config to a function or class, most likely a dataset class) to a string. This
    string is then used to generate a name/storage path related to the config passed to the function.

    :param parameter_dictionary: Dictionary containing the parameter settings passed to a function or class
    :type parameter_dictionary: typing.Dict[str, typing.Any]
    :return: Parameters converted to a string which can be used e.g. in a file path
    :rtype: str
    """
    current_folder = ""
    for key, value in parameter_dictionary.items():
        current_folder += f"_{key}-{value}"

    return current_folder
