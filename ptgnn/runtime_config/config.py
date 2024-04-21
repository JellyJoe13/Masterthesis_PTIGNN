import copy

import yaml
import json


def export_as(
        dict_to_save: dict,
        path: str,
        save_type: str = "json"
):
    """
    Export config dictionary to a file.

    :param dict_to_save: config dict to save
    :type dict_to_save: dict
    :param path: path to which to save to
    :type path: str
    :param save_type: file type, either ``"json"`` or ``"yaml"``
    :type save_type: str
    :return: Nothing.
    """
    if save_type == 'json':
        with open(path, "w") as f:
            json.dump(dict_to_save, f, indent=4)
    elif save_type == "yaml" or save_type == "yml":
        with open(path, "w") as f:
            yaml.dump(dict_to_save, f)
    else:
        raise NotImplementedError(f"{save_type} is not a valid type for dict serialization")


def import_as(
        path: str,
        loading_type: str = None
):
    """
    Import config dict from file (path)

    :param path: Path from which to load from
    :type path: str
    :param loading_type: type of file to load. If not provided will be inferred from file ending
    :type loading_type: str
    :return: loaded config dict
    :rtype: dict
    """
    # infer type if not provided
    if loading_type is None:
        if ".yaml" in path or ".yml" in path:
            loading_type = "yaml"
        elif ".json" in path:
            loading_type = "json"
        else:
            raise NotImplementedError("No loading type provided and the path does not provide type hints. Please "
                                      "specify a type")

    # actual loading part:
    if loading_type == "json":
        with open(path, "r") as f:
            return json.load(f)

    elif loading_type == "yaml" or loading_type == "yml":
        with open(path, "r") as f:
            return yaml.safe_load(f)


def priority_merge_config(
        dict1: dict,
        dict2: dict,
        in_place: bool = True
) -> dict:
    """
    Merges two config dicts. Priority in this case means that the first dicts configs take precedence over the second
    dict. This means that if both dicts have different values for an entry, the first will be prioritized.

    :param dict1: first dict (higher priority)
    :type dict1: dict
    :param dict2: second dict (lower priority)
    :type dict2: dict
    :param in_place: Whether or not the contents should be merged into the first dictionary
    :type in_place: bool
    :return: Merged dict
    :rtype: dict
    """
    if in_place:
        merged_dict = dict1
    else:
        merged_dict = copy.deepcopy(dict1)

    # iterate over second dict and add entries not present
    for key, value in dict2.items():
        if key not in merged_dict:
            merged_dict[key] = copy.deepcopy(value)
        else:
            value_original = merged_dict[key]

            # evaluate conditions
            is_left_dict = type(value_original) == dict
            is_right_dict = type(value) == dict
            is_dict_left_only = (not is_left_dict) and is_right_dict
            is_dict_rigth_only = is_left_dict and (not is_right_dict)
            if is_dict_rigth_only or is_dict_left_only:
                raise ValueError(f"overwriting dict with single value or vice versa in the field {key}. Check input.")
            elif is_left_dict and is_right_dict:
                merged_dict[key] = priority_merge_config(merged_dict[key], value)
            else:
                continue

    return merged_dict


def optional_fetch(
        config_dict: dict,
        key: str
) -> dict:
    """
    Optional fetch from config dict. Else empty dictionarity is returned.

    :param config_dict: config dict from which to optionally fetch
    :type config_dict: dict
    :param key: key which is to be fetched
    :type key: str
    :return: fetched value/dict
    :rtype: dict
    """
    if key in config_dict:
        return config_dict[key]
    else:
        return {}

