"""This will test our functions on the n-sphere

"""
from __future__ import absolute_import, division, print_function, unicode_literals
from kinematics import sphere
import numpy as np

def test_two_sphere_random_vector_norm():
    vec = sphere.rand(2)
    np.testing.assert_almost_equal(np.linalg.norm(vec), 1)

def test_two_sphere_tangent_vector():
    vec = sphere.rand(2)
    vecd = sphere.tan_rand(vec, 9)

    np.testing.assert_almost_equal(np.dot(vec, vecd), 0)
