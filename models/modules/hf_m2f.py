from transformers import is_safetensors_available
from transformers.modeling_utils import load_state_dict
from transformers.utils import cached_file


def handle_m2f_swinb_citysem(model_name, config):
    # used to rename m2f swinB cityScapes weights that don't match the current structure
    # (probably transformers version missmatch when they were saved)

    commit_hash = getattr(config, "_commit_hash", None)

    cached_file_kwargs = {
        "cache_dir": None,
        "force_download": False,
        "proxies": None,
        "resume_download": None,
        "local_files_only": False,
        "token": None,
        "user_agent": {
            "file_type": "model",
            "framework": "pytorch",
            "from_auto_class": False,
        },
        "revision": "main",
        "subfolder": "",
        "_raise_exceptions_for_gated_repo": False,
        "_raise_exceptions_for_missing_entries": False,
        "_commit_hash": commit_hash,
    }

    if is_safetensors_available():
        filename = "model.safetensors"
    else:
        filename = "pytorch_model.bin"

    resolved_archive_file = cached_file(model_name, filename, **cached_file_kwargs)

    state_dict = load_state_dict(resolved_archive_file, weights_only=True)

    modified = {}

    # edit
    print(f"Renaming for {model_name}...")
    for name, val in state_dict.items():
        if "encoder.model" in name:
            new_key = name.replace("encoder.model.", "encoder.")
            modified[new_key] = val
        else:
            modified[name] = val

    return modified
