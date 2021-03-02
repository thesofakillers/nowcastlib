"""
Uncategorised utilities.
"""


def find_closest_factor(dividend, target):
    """
    Given a `dividend`, finds the number `x` closest to a `target` number such that
    `dividend % x == 0`

    Parameters
    ----------
    dividend : int
        The number whose factors we are interested in
    target : int
        The number we'd ideally wish to divide `dividend` by

    Returns
    -------
    int
        The closest number to `target` that factors `dividend`. May also simply be
        `target` if it is already a factor.
    """
    # ensure parameters are valid for our problem
    assert (
        dividend / 2 >= target
    ), "Can't factor with a number greater than the dividend"
    for i in range(dividend):
        if (dividend % (target + i)) == 0:
            return target + i
        elif (dividend % (target - i)) == 0:
            return target - i
    return None
