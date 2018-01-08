"""
    Copyright (C) 2017  GWU Flight Dynamics and Control Laboratory 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from kinematics import attitude


class TestHatAndVeeMap():
    x = np.random.rand(3)
    y = np.random.rand(3)

    def test_hat_map_zero(self):
        np.testing.assert_allclose(attitude.hat_map(self.x).dot(self.x), np.zeros(3))

    def test_hat_map_skew_symmetric(self):
        np.testing.assert_allclose(attitude.hat_map(self.x).T, -attitude.hat_map(self.x))

    def test_hat_vee_map_inverse(self):
        np.testing.assert_allclose(attitude.vee_map(attitude.hat_map(self.x)), self.x)

    def test_hat_map_cross_product(self):
        np.testing.assert_allclose(attitude.hat_map(self.x).dot(self.y), np.cross(self.x, self.y))
        np.testing.assert_allclose(attitude.hat_map(self.x).dot(self.y), -attitude.hat_map(self.y).dot(self.x))      


class TestEulerRot():
    angle = (2*np.pi - 0) * np.random.rand(1) + 0

    def test_rot1_orthogonal(self):
        mat = attitude.rot1(self.angle)
        np.testing.assert_array_almost_equal(mat.T.dot(mat), np.eye(3, 3))

    def test_rot2_orthogonal(self):
        mat = attitude.rot2(self.angle)
        np.testing.assert_array_almost_equal(mat.T.dot(mat), np.eye(3, 3))

    def test_rot3_orthogonal(self):
        mat = attitude.rot3(self.angle)
        np.testing.assert_array_almost_equal(mat.T.dot(mat), np.eye(3, 3))

    def test_rot1_determinant_1(self):
        mat = attitude.rot1(self.angle)
        np.testing.assert_allclose(np.linalg.det(mat), 1)

    def test_rot2_determinant_1(self):
        mat = attitude.rot2(self.angle)
        np.testing.assert_allclose(np.linalg.det(mat), 1)

    def test_rot3_determinant_1(self):
        mat = attitude.rot3(self.angle)
        np.testing.assert_allclose(np.linalg.det(mat), 1)

    def test_rot1_orthogonal_row(self):
        mat = attitude.rot1(self.angle, 'r')
        np.testing.assert_array_almost_equal(mat.T.dot(mat), np.eye(3, 3))

    def test_rot2_orthogonal_row(self):
        mat = attitude.rot2(self.angle, 'r')
        np.testing.assert_array_almost_equal(mat.T.dot(mat), np.eye(3, 3))

    def test_rot3_orthogonal_row(self):
        mat = attitude.rot3(self.angle, 'r')
        np.testing.assert_array_almost_equal(mat.T.dot(mat), np.eye(3, 3))

    def test_rot1_determinant_1_row(self):
        mat = attitude.rot1(self.angle, 'r')
        np.testing.assert_allclose(np.linalg.det(mat), 1)

    def test_rot2_determinant_1_row(self):
        mat = attitude.rot2(self.angle, 'r')
        np.testing.assert_allclose(np.linalg.det(mat), 1)

    def test_rot3_determinant_1_row(self):
        mat = attitude.rot3(self.angle, 'r')
        np.testing.assert_allclose(np.linalg.det(mat), 1)



class TestEulerRot90_column():
    angle = np.pi/2
    b1 = np.array([1, 0, 0])
    b2 = np.array([0, 1, 0])
    b3 = np.array([0, 0, 1])

    R1 = attitude.rot1(angle, 'c')
    R2 = attitude.rot2(angle, 'c')
    R3 = attitude.rot3(angle, 'c')
    
    R1_row = attitude.rot1(angle, 'r')
    R2_row = attitude.rot2(angle, 'r')
    R3_row = attitude.rot3(angle, 'r')
    
    def test_rot1_transpose(self):
        np.testing.assert_allclose(self.R1_row, self.R1.T)

    def test_rot2_transpose(self):
        np.testing.assert_allclose(self.R2_row, self.R2.T)

    def test_rot3_transpose(self):
        np.testing.assert_allclose(self.R3_row, self.R3.T)

    def test_rot1_90_b1(self):
        np.testing.assert_array_almost_equal(self.R1.dot(self.b1), self.b1)

    def test_rot1_90_b2(self):
        np.testing.assert_array_almost_equal(self.R1.dot(self.b2), self.b3)

    def test_rot1_90_b3(self):
        np.testing.assert_array_almost_equal(self.R1.dot(self.b3), -self.b2)

    def test_rot2_90_b1(self):
        np.testing.assert_array_almost_equal(self.R2.dot(self.b1), -self.b3)

    def test_rot2_90_b2(self):
        np.testing.assert_array_almost_equal(self.R2.dot(self.b2), self.b2)

    def test_rot2_90_b3(self):
        np.testing.assert_array_almost_equal(self.R2.dot(self.b3), self.b1)

    def test_rot3_90_b1(self):
        np.testing.assert_array_almost_equal(self.R3.dot(self.b1), self.b2)

    def test_rot3_90_b2(self):
        np.testing.assert_array_almost_equal(self.R3.dot(self.b2), -self.b1)

    def test_rot3_90_b3(self):
        np.testing.assert_array_almost_equal(self.R3.dot(self.b3), self.b3)
    
class TestEulerRotInvalid():
    R1 = attitude.rot1(0, 'x')
    R2 = attitude.rot2(0, 'x')
    R3 = attitude.rot3(0, 'x')
    invalid = 1
    def test_rot1_invalid(self):
        np.testing.assert_allclose(self.R1, self.invalid)

    def test_rot2_invalid(self):
        np.testing.assert_allclose(self.R2, self.invalid)

    def test_rot3_invalid(self):
        np.testing.assert_allclose(self.R3, self.invalid)

class TestExpMap():
    angle = (np.pi - 0) * np.random.rand(1) + 0
    axis = np.array([1, 0, 0])
    R = attitude.rot1(angle)

    def test_axisangletodcm(self):
        np.testing.assert_array_almost_equal(attitude.rot1(self.angle), attitude.axisangletodcm(self.angle, self.axis))

    def test_dcmtoaxisangle(self):
        angle, axis = attitude.dcmtoaxisangle(self.R)
        np.testing.assert_array_almost_equal(angle, self.angle)
        np.testing.assert_array_almost_equal(axis, self.axis)

class TestQuaternion():
    angle = (2*np.pi - 0) * np.random.rand(1) + 0
    dcm = attitude.rot1(angle)
    quat = attitude.dcmtoquat(dcm)

    dcm_identity = np.eye(3,3)
    quat_identity = attitude.dcmtoquat(dcm_identity)

    def test_dcmtoquaternion_unit_norm(self):
        np.testing.assert_almost_equal(np.linalg.norm(self.quat), 1)

    def test_quaternion_back_to_rotation_matrix(self):
        np.testing.assert_array_almost_equal(attitude.quattodcm(self.quat),
                                            self.dcm)

    def test_identity_quaternion_scalar_one(self):
        np.testing.assert_equal(self.quat_identity[-1], 1)
    
    def test_identity_quaternion_vector_zero(self):
        np.testing.assert_array_almost_equal(self.quat_identity[0:3], np.zeros(3))

class TestNormalize():
    
    def test_lower_limit_circle(self):
        expected = -180 
        actual = attitude.normalize(-180, -180, 180)
        np.testing.assert_almost_equal(actual, expected)

    def test_upper_limit_circle(self):
        expected = -180
        actual = attitude.normalize(180, -180, 180)
        np.testing.assert_almost_equal(actual, expected)

    def test_past_upper_limit_circle(self):
        expected = -179
        actual = attitude.normalize(181, -180, 180)
        np.testing.assert_almost_equal(actual, expected)

    def test_past_lower_limit_circle(self):
        expected = 179
        actual = attitude.normalize(-181, -180, 180)
        np.testing.assert_almost_equal(actual, expected)
    
    def test_vectorized_interior(self):
        angles = np.linspace(10, 50, 10)
        expected = angles
        actual = attitude.normalize(angles, 0, 180)
        np.testing.assert_allclose(actual, expected)

class TestUnitVector():
    
    q = np.array([5, 0, 0])
    qhat_true = np.array([1, 0, 0])
    qhat = attitude.unit_vector(q)

    def test_norm(self):
        np.testing.assert_allclose(np.linalg.norm(self.qhat), 1)

    def test_unit_vector(self):
        np.testing.assert_allclose(self.qhat, self.qhat_true)
