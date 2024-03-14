def dict_to_storage_path(d) -> str:
    current_folder = ""
    for key, value in d.items():
        current_folder += f"_{key}-{value}"

    return current_folder
