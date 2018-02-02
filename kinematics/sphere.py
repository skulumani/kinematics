"""n-Sphere operations

This module holds some useful functions for dealing with elements of n-spheres.
Most usually we tend to deal with the 1-sphere and the 2-sphere.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from kinematics import attitude

def rand(n, **kwargs):
    """Random vector from the n-Sphere

    This function will return a random vector which is an element of the n-Sphere.
    The n-Sphere is defined as a vector in R^n+1 with a norm of one. 

    Basically, we'll find a random vector in R^n+1 and normalize it. 
    This uses the method of Marsaglia 1972.

    Parameters
    ----------
    None

    Returns
    -------
    rvec 
        Random (n+1,) numpy vector with a norm of 1

    """
    rvec = np.random.randn(3)
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

def perturb_vec(q, cone_half_angle=2):
    r"""Perturb a vector randomly

    qp = perturb_vec(q, cone_half_angle=2)

    Parameters
    ----------
    q : (n,) numpy array
        Vector to perturb
    cone_half_angle : float
        Maximum angle to perturb the vector in degrees

    Returns
    -------
    perturbed : (n,) numpy array
        Perturbed numpy array

    Author
    ------
    Shankar Kulumani		GWU		skulumani@gwu.edu

    References
    ----------

    .. [1] https://stackoverflow.com/questions/2659257/perturb-vector-by-some-angle

    """
    rand_vec = tan_rand(q)
    cross_vector = attitude.unit_vector(np.cross(q, rand_vec))

    s = np.random.uniform(0, 1, 1)
    r = np.random.uniform(0, 1, 1)

    h = np.cos(np.deg2rad(cone_half_angle))

    phi = 2 * np.pi * s
    z = h + ( 1- h) * r
    sinT = np.sqrt(1 - z**2)
    x = np.cos(phi) * sinT
    y = np.sin(phi) * sinT

    perturbed = rand_vec * x + cross_vector * y + q * z
    
    return perturbed
