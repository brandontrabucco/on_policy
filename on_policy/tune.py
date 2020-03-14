import itertools
import copy


def merge_dicts(a,
                b,
                path=None,
                ignore_duplicate=False):
    """Recursively merge to dictionaries that may have different keys
    by creating a new merged dictionary

    Arguments:

    a: dict
        a dictionary that may have nested dictionaries inside
        not required to have the same keys as b
    b: dict
        a dictionary that may have nested dictionaries inside
        not required to have the same keys as a
    path: list
        a list of the dictionary keys encountered while traversing
        a nested structure of dictionaries
    ignore_duplicate: bool
        ignore if keys are duplicated in both a and b while
        the values are different

    Returns:

    result: dict
        a dictionary that contains all elements in a and all elements
        in b including recursively contained dicts"""

    if path is None:
        path = []

    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key],
                            b[key],
                            path + [str(key)],
                            ignore_duplicate=ignore_duplicate)

            elif a[key] == b[key]:
                print("Same value for key: {}".format(key))

            else:
                dup = '.'.join(path + [str(key)])

                if ignore_duplicate:
                    print("duplicate key ignored: {}".format(dup))

                else:
                    raise Exception('Duplicate keys at {}'.format(dup))

        else:
            a[key] = b[key]

    return a


def grid_search(variant,
                **search_space):
    """Creates a list of variants for performing as hyper parameter sweep
    using the provided search space grid

    Arguments:

    variant: dict
        a dictionary of hyper parameters that control the
        rl algorithm
    search_space: dict
        a dictionary containing lists of search space points to
        perform a grid search over

    Returns:

    variants: list
        a list of dictionary variants of the same structure as variant
        but different combinations of elements in search_space"""

    named_h = []
    for name, values in search_space.items():
        named_h.append([(name, v) for v in values])

    return [merge_dicts(dict(h),
                        copy.deepcopy(variant),
                        ignore_duplicate=True)
            for h in itertools.product(*named_h)]
