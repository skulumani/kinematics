"""This will test our functions on the n-sphere

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from kinematics import sphere

def test_two_sphere_random_vector_norm():
    vec = sphere.rand(2)
    np.testing.assert_almost_equal(np.linalg.norm(vec), 1)

def test_two_sphere_tangent_vector():
    vec = sphere.rand(2)
    vecd = sphere.tan_rand(vec, 9)

    np.testing.assert_almost_equal(np.dot(vec, vecd), 0)

class TestPerturbedVector():

    q = sphere.rand(2)
    half_angle = 5
    qp = sphere.perturb_vec(q, half_angle)

    def test_angle(self):
        """Ensure the angle is always less than half_angle
        """
        angle = np.arccos(np.dot(self.q, self.qp)) * 180 / np.pi
        np.testing.assert_array_less(angle, self.half_angle)

    def test_norm(self):
        np.testing.assert_allclose(np.linalg.norm(self.qp), 1)
    
