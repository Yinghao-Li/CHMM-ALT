import torch
import numpy as np
from typing import Optional, Union


def log_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    a : m \times n
    b : n \times p

    output : m \times p matrix

    Normally, a matrix multiplication
    computes out_{i,j} = sum_k A_{i,k} \times B_{k,j}

    A log domain matrix multiplication
    computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}

    This is needed for numerical stability when A and B are probability matrices.
    """
    a1 = a.unsqueeze(-1)
    b1 = b.unsqueeze(-3)
    return (a1 + b1).logsumexp(-2)


def log_maxmul(a, b):
    a1 = a.unsqueeze(-1)
    b1 = b.unsqueeze(-3)
    return (a1 + b1).max(-2)


# noinspection PyTypeChecker
def logsumexp(x, dim=None, keepdim=False):
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim=dim, keepdim=True)
    x = torch.where(
        (xm == np.inf) | (xm == -np.inf),
        xm,
        xm + torch.logsumexp(x - xm, dim=dim, keepdim=True)
    )
    return x if keepdim else x.squeeze(dim)


def validate_prob(x, dim=-1):
    if (x <= 0).any():
        prob = normalize(x, dim=dim)
    elif (x.sum(dim=dim) != 1).any():
        prob = x / x.sum(dim=dim, keepdim=True)
    else:
        prob = x
    return prob


def normalize(x, dim=-1, epsilon=1e-6):
    result = x - x.min(dim=dim, keepdim=True)[0] + epsilon
    result = result / result.sum(dim=dim, keepdim=True)
    return result


def entropy(p: torch.Tensor, dim: Optional[int] = -1):
    """
    Calculate entropy

    Parameters
    ----------
    p: probabilities
    dim: dimension

    Returns
    -------
    entropy
    """
    h = torch.sum(-p * torch.log(p), dim=dim)
    return h


def prob_scaling(p: Union[float, torch.Tensor, np.ndarray],
                 r: Optional[float] = 0.5,
                 e: Optional[float] = 2,
                 n: Optional[float] = 2):
    """
    scale the probabilities: pushing the probability values to extreme

    Parameters
    ----------
    p: input probabilities
    r: split point: the point that separates the "pushing up" and "pushing down" operations
    e: tier 1 inverse exponential term
    n: tier 2 exponential term

    Returns
    -------
    type(p)
    """
    p_ = p ** (1 / e)

    pu = p_ > r
    pd = p_ <= r

    ru = pu * (-(1 / (1 - r)) ** (n - 1) * (1 - p_) ** n + 1)
    rd = pd * ((1 / r) ** (n - 1) * p_ ** n)

    return ru + rd


def entity_emiss_diag(x):
    """
    emission prior of entity to itself

    Parameters
    ----------
    x

    Returns
    -------

    """
    return x


def entity_emiss_o(x, n_lbs, tp, exp_term=2):
    """
    The function that calculates the emission prior of entity labels to the non-entity label 'O'
    according to the diagonal values of the emission prior

    Parameters
    ----------
    x: diagonal values
    n_lbs: number of entity labels (2e+1)
    tp: turning point
    exp_term: the exponential term that controls the slope of the function

    Returns
    -------
    non-diagonal emission priors
    """
    # separating piecewise function
    low = x < tp
    high = x >= tp

    # parameters for the first piece
    a = (2 - n_lbs) / ((exp_term - 1) * tp ** exp_term - exp_term * tp ** (exp_term - 1))
    b = 1 - n_lbs
    # parameter for the second piece
    f_tp = a * tp ** exp_term + b * tp + 1
    c = f_tp / (tp - 1)

    # piecewise result
    y = low * (a * x ** exp_term + b * x + 1) + high * (c * x - c)
    return y


def entity_emiss_nondiag(x, n_lbs, tp, exp_term=2):
    """
    emission prior of entity to other entities

    Parameters
    ----------
    x: diagonal values
    n_lbs: number of entity labels (2e+1)
    tp: turning point
    exp_term: the exponential term that controls the slope of the function

    Returns
    -------

    """
    return (1 - entity_emiss_diag(x) - entity_emiss_o(x, n_lbs, tp, exp_term)) / (n_lbs - 2)
