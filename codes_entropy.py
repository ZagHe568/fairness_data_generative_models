import numpy as np
from numpy import (log, asarray, inf)
from scipy.stats import entropy


def my_entr(pk):  # real signature unknown; NOTE: unreliably restored from __doc__
    """
    entr(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

    entr(x)

    Elementwise function for computing entropy.

    .. math:: \text{entr}(x) = \begin{cases} - x \log(x) & x > 0  \\ 0 & x = 0 \\ -\infty & \text{otherwise} \end{cases}

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    res : ndarray
        The value of the elementwise entropy function at the given points `x`.

    See Also
    --------
    kl_div, rel_entr

    Notes
    -----
    This function is concave.

    .. versionadded:: 0.15.0
    """
    n_classes = np.count_nonzero(pk)
    if n_classes <= 1:
        return np.zeros(pk.size)
    res = np.array(list(map(lambda p: - p * log(p) if p > 0 else (-inf if p < 0 else 0), pk))) # this is natural log in base e
    return res


def my_rel_entr(pk, qk):
    """
    Elementwise function for computing relative entropy.
    :param pk: ndarray distribution 1
    :param qk: ndarray distribution 1
    :return: ndarray Relative entropy of the inputs
    """
    n_classes = min(np.count_nonzero(pk), np.count_nonzero(qk))
    if n_classes <= 1:
        return np.zeros(pk.size)
    res = np.array(list(map(lambda pq: pq[0] * (log(pq[0]) - log(pq[1])) if pq[0] > 0 and pq[1] > 0 else (
        0 if pq[0] == 0 and pq[1] >= 0 else inf), zip(pk, qk))))
    return res


def my_entropy(pk, qk=None, base=None, axis=0):
    """Calculate the entropy of a distribution for given probability values.

    If only probabilities `pk` are given, the entropy is calculated as
    ``S = -sum(pk * log(pk), axis=axis)``.

    If `qk` is not None, then compute the Kullback-Leibler divergence
    ``S = sum(pk * log(pk / qk), axis=axis)``.

    This routine will normalize `pk` and `qk` if they don't sum to 1.

    Parameters
    ----------
    pk : sequence
        Defines the (discrete) distribution. ``pk[i]`` is the (possibly
        unnormalized) probability of event ``i``.
    qk : sequence, optional
        Sequence against which the relative entropy is computed. Should be in
        the same format as `pk`.
    base : float, optional
        The logarithmic base to use, defaults to ``e`` (natural logarithm).
    axis: int, optional
        The axis along which the entropy is calculated. Default is 0.

    Returns
    -------
    S : float
        The calculated entropy.

    Examples
    --------
    #>>> from scipy.stats import entropy

    Bernoulli trial with different p.
    The outcome of a fair coin is the most uncertain:

    #>>> entropy([1/2, 1/2], base=2)
    1.0

    The outcome of a biased coin is less uncertain:

    #>>> entropy([9/10, 1/10], base=2)
    0.46899559358928117

    Relative entropy:

    #>>> entropy([1/2, 1/2], qk=[9/10, 1/10])
    0.5108256237659907

    """
    pk = asarray(pk)
    pk = 1.0 * pk / np.sum(pk, axis=axis, keepdims=True)
    if qk is None:
        vec = my_entr(pk)
    else:
        qk = asarray(qk)
        if qk.shape != pk.shape:
            raise ValueError("qk and pk must have same shape.")
        qk = 1.0 * qk / np.sum(qk, axis=axis, keepdims=True)
        vec = my_rel_entr(pk, qk)
    S = np.sum(vec, axis=axis)
    if base is not None:
        S /= log(base)
    return S


def test():
    print("[1/2, 1/2]")
    their_entropy = entropy([1 / 2, 1 / 2], base=2)
    our_entropy = my_entropy([1 / 2, 1 / 2], base=2)
    print("scipy.stats.entropy: {}, our entropy: {}".format(their_entropy, our_entropy))

    print(" [9 / 10, 1 / 10]")
    their_entropy = entropy([9 / 10, 1 / 10], base=2)
    our_entropy = my_entropy([9 / 10, 1 / 10], base=2)
    print("scipy.stats.entropy: {}, our entropy: {}".format(their_entropy, our_entropy))

    print("[3 / 10, 1 / 10, 6 / 10]")
    their_entropy = entropy([3 / 10, 1 / 10, 6 / 10], base=2)
    our_entropy = my_entropy([3 / 10, 1 / 10, 6 / 10], base=2)
    print("scipy.stats.entropy: {}, our entropy: {}".format(their_entropy, our_entropy))

    print("[1/2, 1/2], qk=[9/10, 1/10]")
    their_entropy = entropy([1/2, 1/2], qk=[9/10, 1/10])
    our_entropy = my_entropy([1/2, 1/2], qk=[9/10, 1/10])
    print("scipy.stats.entropy: {}, our entropy: {}".format(their_entropy, our_entropy))




if __name__ == "__main__":
    test()
