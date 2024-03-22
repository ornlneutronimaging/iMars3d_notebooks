from pathlib import Path


def find_first_real_dir(start_dir="./"):
    """return the first existing folder from the tree up"""
    if start_dir.exists():
        return start_dir

    dir = start_dir
    while not Path(dir).exists():
        dir = Path(dir).parent

    return dir
