import yaml
import json


def export_as(
        dict_to_save: dict,
        path: str,
        save_type: str = "json"
):
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
