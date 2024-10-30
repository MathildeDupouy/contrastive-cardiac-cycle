from collections.abc import Mapping
import json

def get_centered_range(new_duration, old_width, old_duration):
    """
    Get a range of index ad a duration to be the closest of new_duration
    while keepping a parity similar to old_width to keep the middle pixels iin the middle
    """
    width_raw = old_width / old_duration * new_duration
    middle_index = int(old_width / 2)
    odd_old_width = old_width%2
    # If rounding down is not even, rounding up
    new_width = int(width_raw) + 1 * (int(width_raw)%2 != odd_old_width)
    duration = new_width * old_duration / old_width # real vignette duration
    start = middle_index - (new_width // 2)
    end = middle_index + (new_width // 2) + odd_old_width
    return start, end, duration

def get_closest_multiple(n, x):
    """
    Get the closest integer of n that is a multiple of x.
    Arguments:
    ----------
        n: int
        number to approximate

        x: int
        divider of the result
    Returns:
    --------
        int: result
    """
    res = n + x / 2
    res = res - (res % x)
    return res

def update_dict(old_dict, new_dict):
    for k, v in new_dict.items():
        if isinstance(v, Mapping):
            # print("key", k, "value is a dict")
            old_dict[k] = update_dict(old_dict.get(k, {}), v)
        elif isinstance(v, list) and k in old_dict.keys():
            # print("key", k, "value is a list")
            old_dict[k] += v
        else:
            # print("key", k, "value is created/replaced")
            old_dict[k] = v
    return old_dict

def update_saved_dict(file_path, new_dict):
    """
    Load a dict from a json file and add/replace keys with values from new_dict
    """
    if file_path.exists():
        with file_path.open("r") as f:
            new_dict_old = json.load(f)
        new_dict = update_dict(new_dict_old, new_dict)
    with file_path.open("w") as f:
        json.dump(new_dict, f, indent=4)
