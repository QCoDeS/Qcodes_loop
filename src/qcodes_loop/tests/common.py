from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def strip_qc(
    d: dict[str, Any], keys: Sequence[str] = ("instrument", "__class__")
) -> dict[str, Any]:
    # depending on how you run the tests, __module__ can either
    # have qcodes on the front or not. Just strip it off.
    for key in keys:
        if key in d:
            d[key] = d[key].replace("qcodes.tests.", "tests.")
    return d


def compare_dictionaries(
    dict_1: Mapping[Any, Any],
    dict_2: Mapping[Any, Any],
    dict_1_name: str | None = "d1",
    dict_2_name: str | None = "d2",
    path: str = "",
) -> tuple[bool, str]:
    """
    Compare two dictionaries recursively to find non matching elements.

    Args:
        dict_1: First dictionary to compare.
        dict_2: Second dictionary to compare.
        dict_1_name: Optional name of the first dictionary used in the
                     differences string.
        dict_2_name: Optional name of the second dictionary used in the
                     differences string.
    Returns:
        Tuple: Are the dicts equal and the difference rendered as
               a string.

    """
    err = ""
    key_err = ""
    value_err = ""
    old_path = path
    for k in dict_1.keys():
        path = old_path + "[%s]" % k
        if k not in dict_2.keys():
            key_err += f"Key {dict_1_name}{path} not in {dict_2_name}\n"
        else:
            if isinstance(dict_1[k], dict) and isinstance(dict_2[k], dict):
                err += compare_dictionaries(
                    dict_1[k], dict_2[k], dict_1_name, dict_2_name, path
                )[1]
            else:
                match = dict_1[k] == dict_2[k]

                # if values are equal-length numpy arrays, the result of
                # "==" is a bool array, so we need to 'all' it.
                # In any other case "==" returns a bool
                # TODO(alexcjohnson): actually, if *one* is a numpy array
                # and the other is another sequence with the same entries,
                # this will compare them as equal. Do we want this, or should
                # we require exact type match?
                if hasattr(match, "all"):
                    match = match.all()

                if not match:
                    value_err += (
                        'Value of "{}{}" ("{}", type"{}") not same as\n'
                        '  "{}{}" ("{}", type"{}")\n\n'
                    ).format(
                        dict_1_name,
                        path,
                        dict_1[k],
                        type(dict_1[k]),
                        dict_2_name,
                        path,
                        dict_2[k],
                        type(dict_2[k]),
                    )

    for k in dict_2.keys():
        path = old_path + f"[{k}]"
        if k not in dict_1.keys():
            key_err += f"Key {dict_2_name}{path} not in {dict_1_name}\n"

    dict_differences = key_err + value_err + err
    if len(dict_differences) == 0:
        dicts_equal = True
    else:
        dicts_equal = False
    return dicts_equal, dict_differences
