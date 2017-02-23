import numpy as np
from .. import attitude


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
