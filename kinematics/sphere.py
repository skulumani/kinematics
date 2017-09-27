"""n-Sphere operations

This module holds some useful functions for dealing with elements of n-spheres.
Most usually we tend to deal with the 1-sphere and the 2-sphere.

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

def rand(n, seed=9):
    """Random vector from the n-Sphere

    This function will return a random vector which is an element of the n-Sphere.
    The n-Sphere is defined as a vector in R^n with a norm of one. 

    Basically, we'll find a random vector in R^n and normalize it. 
    This is somewhat like mapping a cube to a sphere but hopefully it doesn't cause any
    issues in the future.

    Parameters
    ----------
    None

    Returns
    -------
    rvec 
        Random (n+1,) numpy vector with a norm of 1

    """
    rs = np.random.RandomState(seed)
    rvec = rs.rand(n+1)
    rvec = rvec / np.linalg.norm(rvec)
    return rvec

def tan_rand(q, seed=9):
    """Find a random vector in the tangent space of the n sphere

    This function will find a random orthogonal vector to q.

    Parameters
    ----------
    q
        (n+1,) array which is in the n-sphere

    Returns
    -------
    qd
        (n+1,) array which is orthogonal to n-sphere and also random

    """
    # probably need a check in case we get a parallel vector
    rs = np.random.RandomState(seed)
    rvec = rs.rand(q.shape[0])

    qd = np.cross(rvec, q)
    qd = qd / np.linalg.norm(qd)

    while np.dot(q, qd) > 1e-6:
        rvec = rs.rand(q.shape[0])
        qd = np.cross(rvec, q)
        qd = qd / np.linalg.norm(qd)

    return qd
