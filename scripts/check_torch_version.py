"""
Ensures currently installed torch version is the newest allowed.
"""

from typing import Tuple, cast


def main():
    current_torch_version = _get_current_installed_torch_version()
    latest_torch_version = _get_latest_torch_version()
    torch_version_upper_limit = _get_torch_version_upper_limit()

    if current_torch_version < latest_torch_version < torch_version_upper_limit:
        raise RuntimeError(
            f"current torch version {current_torch_version} is behind "
            f"latest allowed torch version {latest_torch_version}"
        )

    print("All good!")


def _get_current_installed_torch_version() -> Tuple[str, str, str]:
    import torch

    version = tuple(torch.version.__version__.split("."))
    assert len(version) == 3, f"Bad parsed version '{version}'"
    return cast(Tuple[str, str, str], version)


def _get_latest_torch_version() -> Tuple[str, str, str]:
    import requests

    r = requests.get("https://api.github.com/repos/pytorch/pytorch/tags")
    assert r.ok
    for tag_data in r.json():
        tag = tag_data["name"]
        if tag.startswith("v") and "-rc" not in tag:
            # Tag should look like "vX.Y.Z"
            version = tuple(tag[1:].split("."))
            assert len(version) == 3, f"Bad parsed version '{version}'"
            break
    else:
        raise RuntimeError("could not find latest stable release tag")
    return cast(Tuple[str, str, str], version)


def _get_torch_version_upper_limit() -> Tuple[str, str, str]:
    with open("setup.py") as f:
        for line in f:
            # The torch version line should look like:
            #   "torch>=X.Y.Z,<X.V.0",
            if '"torch>=' in line:
                version = tuple(line.split('"')[1].split("<")[1].strip().split("."))
                assert len(version) == 3, f"Bad parsed version '{version}'"
                break
        else:
            raise RuntimeError("could not find torch version spec in setup.py")
    return cast(Tuple[str, str, str], version)


if __name__ == "__main__":
    main()
