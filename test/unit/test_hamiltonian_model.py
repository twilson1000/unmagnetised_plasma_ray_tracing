#!/usr/bin/python3

# Standard imports
import logging
import numpy as np
import pytest
import scipy.constants as const

# Local imports
from unmagnetised_plasma_ray_tracer.hamiltonian_model import UnmagnetisedPlasmaHamiltonian
from unmagnetised_plasma_ray_tracer.plasma_model import C2Ramp

logger = logging.getLogger(__name__)

finite_difference_1st_derivative_2nd_order_stencil = ((-1, -1/2), (1, 1/2))
finite_difference_1st_derivative_4th_order_stencil = ((-2, 1/12), (-1, -2/3), (1, 2/3), (2, -1/12))
finite_difference_2nd_derivative_2nd_order_stencil = ((-1, 1.0), (0, -2.0), (1, 1.0))
finite_difference_mixed_2nd_derivative_2nd_order_stencil = ((1, 1, 0.25), (1, -1, -0.25), (-1, 1, -0.25), (-1, -1, 0.25))

class TestDispersionDerivatives:
    @pytest.fixture
    def stix_X_model(self):
        return C2Ramp(0, 0, 2, 1)

    @pytest.fixture
    def dispersion(self, stix_X_model):
        return UnmagnetisedPlasmaHamiltonian(stix_X_model)
    
    test_values = ('X, N2', (
        (0, 1), (0.2, 0.7), (0.7, 1.1), (1.2, 0.4)
    ))

    @pytest.mark.parametrize(*test_values)
    def test_hamiltonian_first_derivative_X(self, dispersion, X, N2):
        expected_value = dispersion.hamiltonian_first_derivative_X(X, N2)

        h = 1e-6
        actual_value = 0
        for step, weight in finite_difference_1st_derivative_2nd_order_stencil:
            actual_value += weight * dispersion.hamiltonian_function(X + step * h, N2)

        actual_value /= h
        assert np.isclose(actual_value, expected_value)

    @pytest.mark.parametrize(*test_values)
    def test_hamiltonian_first_derivative_N2(self, dispersion, X, N2):
        expected_value = dispersion.hamiltonian_first_derivative_N2(X, N2)

        h = 1e-6
        actual_value = 0
        for step, weight in finite_difference_1st_derivative_2nd_order_stencil:
            actual_value += weight * dispersion.hamiltonian_function(X, N2 + step * h)

        actual_value /= h
        assert np.isclose(actual_value, expected_value)

    @pytest.mark.parametrize(*test_values)
    def test_hamiltonian_second_derivative_X(self, dispersion, X, N2):
        expected_value = dispersion.hamiltonian_second_derivative_X(X, N2)

        h = 1e-4
        actual_value = 0
        for step, weight in finite_difference_2nd_derivative_2nd_order_stencil:
            actual_value += weight * dispersion.hamiltonian_function(X + step * h, N2)

        actual_value /= h**2
        assert np.isclose(actual_value, expected_value)

    @pytest.mark.parametrize(*test_values)
    def test_hamiltonian_second_derivative_N2(self, dispersion, X, N2):
        expected_value = dispersion.hamiltonian_second_derivative_N2(X, N2)

        h = 1e-4
        actual_value = 0
        for step, weight in finite_difference_2nd_derivative_2nd_order_stencil:
            actual_value += weight * dispersion.hamiltonian_function(X, N2 + step * h)

        actual_value /= h**2
        assert np.isclose(actual_value, expected_value)

    test_phase_space_positions = ('position, refractive_index', (
        (np.array([0.29, 0.19, 0.16]), np.array([0.06, 0.05, 0.79])),
        (np.array([0.30, 0.64, 0.72]), np.array([0.30, 0.64, 0.72])),
        (np.array([0.30, 0.46, 0.72]), np.array([0.76, 0.83, 0.93])),
        (np.array([0.67, 0.61, 0.99]), np.array([0.05, 0.61, 0.51])),
        (np.array([0.91, 0.93, 0.60]), np.array([0.37, 0.92, 0.02])),
        (np.array([0.38, 0.61, 0.89]), np.array([0.41, 0.86, 0.22])),
    ))

    @pytest.mark.parametrize(*test_phase_space_positions)
    def test_hamiltonian_first_derivative_x(self, dispersion, position, refractive_index):
        expected_value = dispersion.hamiltonian_first_derivative_x(position, refractive_index)

        h = 1e-6
        actual_value = np.zeros(3)
        dummy_x = np.zeros(3)

        for i in range(3):
            dummy_x[:] = position
            for step, weight in finite_difference_1st_derivative_2nd_order_stencil:
                dummy_x[i] = position[i] + step * h
                actual_value[i] += weight * dispersion.hamiltonian(dummy_x, refractive_index)

        actual_value /= h
        assert np.allclose(actual_value, expected_value)

    @pytest.mark.parametrize(*test_phase_space_positions)
    def test_hamiltonian_first_derivative_N(self, dispersion, position, refractive_index):
        expected_value = dispersion.hamiltonian_first_derivative_N(position, refractive_index)

        h = 1e-6
        actual_value = np.zeros(3)
        dummy_N = np.zeros(3)

        for i in range(3):
            dummy_N[:] = refractive_index
            for step, weight in finite_difference_1st_derivative_2nd_order_stencil:
                dummy_N[i] = refractive_index[i] + step * h
                actual_value[i] += weight * dispersion.hamiltonian(position, dummy_N)

        actual_value /= h
        assert np.allclose(actual_value, expected_value)
    
    @pytest.mark.parametrize(*test_phase_space_positions)
    def test_hamiltonian_first_derivative_f(self, dispersion, position, refractive_index):
        expected_value = dispersion.hamiltonian_normalised_first_derivative_f(position, refractive_index)

        h = 1e-6
        actual_value = 0
        X, N2 = dispersion.get_X_N2(position, refractive_index)
        for step, weight in finite_difference_1st_derivative_2nd_order_stencil:
            factor = (1 + step * h)**2
            actual_value += weight * dispersion.hamiltonian_function(X / factor, N2 / factor)

        actual_value /= h
        assert np.isclose(actual_value, expected_value)

    @pytest.mark.parametrize(*test_phase_space_positions)
    def test_hamiltonian_second_derivative_x(self, dispersion, position, refractive_index):
        expected_value = dispersion.hamiltonian_second_derivative_x(position, refractive_index)

        h = 1e-4
        actual_value = np.zeros((3, 3))
        dummy_x = np.zeros(3)

        for i in range(3):
            dummy_x[:] = position
            for step, weight in finite_difference_2nd_derivative_2nd_order_stencil:
                dummy_x[i] = position[i] + step * h
                actual_value[i, i] += weight * dispersion.hamiltonian(dummy_x, refractive_index)

            for j in range(i+1, 3):
                dummy_x[:] = position
                for step1, step2, weight in finite_difference_mixed_2nd_derivative_2nd_order_stencil:
                    dummy_x[i] = position[i] + step1 * h
                    dummy_x[j] = position[j] + step2 * h
                    actual_value[i, j] += weight * dispersion.hamiltonian(dummy_x, refractive_index)

                actual_value[j, i] = actual_value[i, j]

        actual_value /= h**2
        assert np.allclose(actual_value, expected_value)

    @pytest.mark.parametrize(*test_phase_space_positions)
    def test_hamiltonian_second_derivative_N(self, dispersion, position, refractive_index):
        expected_value = dispersion.hamiltonian_second_derivative_N(position, refractive_index)

        h = 1e-4
        actual_value = np.zeros((3, 3))
        dummy_N = np.zeros(3)

        for i in range(3):
            dummy_N[:] = refractive_index
            for step, weight in finite_difference_2nd_derivative_2nd_order_stencil:
                dummy_N[i] = refractive_index[i] + step * h
                actual_value[i, i] += weight * dispersion.hamiltonian(position, dummy_N)

            for j in range(i+1, 3):
                dummy_N[:] = refractive_index
                for step1, step2, weight in finite_difference_mixed_2nd_derivative_2nd_order_stencil:
                    dummy_N[i] = refractive_index[i] + step1 * h
                    dummy_N[j] = refractive_index[j] + step2 * h
                    actual_value[i, j] += weight * dispersion.hamiltonian(position, dummy_N)

                actual_value[j, i] = actual_value[i, j]

        actual_value /= h**2
        assert np.allclose(actual_value, expected_value, atol=1e-7)

    @pytest.mark.parametrize(*test_phase_space_positions)
    def test_hamiltonian_second_derivative_xN(self, dispersion, position, refractive_index):
        expected_value = dispersion.hamiltonian_second_derivative_xN(position, refractive_index)

        h = 1e-4
        actual_value = np.zeros((3, 3))
        dummy_x = np.zeros(3)
        dummy_N = np.zeros(3)

        for i in range(3):
            for j in range(i+1, 3):
                dummy_x[:] = position
                dummy_N[:] = refractive_index
                for step1, step2, weight in finite_difference_mixed_2nd_derivative_2nd_order_stencil:
                    dummy_x[i] = position[i] + step1 * h
                    dummy_N[j] = refractive_index[j] + step2 * h
                    actual_value[i, j] += weight * dispersion.hamiltonian(dummy_x, dummy_N)

        actual_value /= h**2
        assert np.allclose(actual_value, expected_value)

    @pytest.mark.parametrize(*test_phase_space_positions)
    def test_hamiltonian_second_derivative_xf(self, dispersion, position, refractive_index):
        expected_value = dispersion.hamiltonian_normalised_second_derivative_xf(position, refractive_index)

        h = 1e-4
        actual_value = np.zeros(3)
        dummy_x = np.zeros(3)

        for i in range(3):
            dummy_x[:] = position

            for step1, step2, weight in finite_difference_mixed_2nd_derivative_2nd_order_stencil:
                dummy_x[i] = position[i] + step1 * h
                X, N2 = dispersion.get_X_N2(dummy_x, refractive_index)
                factor = (1 + step2*h)**2
                X /= factor
                N2 /= factor
                actual_value[i] += weight * dispersion.hamiltonian_function(X, N2)

        actual_value /= h**2
        assert np.allclose(actual_value, expected_value)

    @pytest.mark.parametrize(*test_phase_space_positions)
    def test_hamiltonian_second_derivative_Nf(self, dispersion, position, refractive_index):
        expected_value = dispersion.hamiltonian_normalised_second_derivative_Nf(position, refractive_index)

        h = 1e-4
        actual_value = np.zeros(3)
        dummy_N = np.zeros(3)

        for i in range(3):
            dummy_N[:] = refractive_index

            for step1, step2, weight in finite_difference_mixed_2nd_derivative_2nd_order_stencil:
                dummy_N[i] = refractive_index[i] + step1 * h
                X, N2 = dispersion.get_X_N2(position, dummy_N)
                factor = (1 + step2*h)**2
                X /= factor
                N2 /= factor
                actual_value[i] += weight * dispersion.hamiltonian_function(X, N2)

        actual_value /= h**2
        assert np.allclose(actual_value, expected_value)

    @pytest.mark.parametrize(*test_phase_space_positions)
    def test_ray_acceleration(self, dispersion, position, refractive_index):
        ''' Test derivative of velocity with respect to time is acceleration. '''
        expected_value = dispersion.normalised_ray_acceleration(position, refractive_index)

        actual_value = np.zeros(6)
        dt = 1e-11 # 1e-6 * 1e-9
        dummy_x, dummy_N = np.zeros(3), np.zeros(3)
        ray_velocity = const.speed_of_light * dispersion.normalised_ray_velocity(position, refractive_index)

        for step, weight in finite_difference_1st_derivative_4th_order_stencil:
            dummy_x[:] = position + step * dt * ray_velocity[:3]
            dummy_N[:] = refractive_index + step * dt * ray_velocity[3:]
            actual_value += weight * dispersion.normalised_ray_velocity(dummy_x, dummy_N)

        actual_value /= dt * const.speed_of_light
        logger.warning(abs(actual_value - expected_value))
        assert np.allclose(actual_value, expected_value)
