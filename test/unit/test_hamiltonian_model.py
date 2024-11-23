#!/usr/bin/python3

# Standard imports
import logging
import numpy as np
import pytest
import scipy.constants as const
from typing import List, Tuple

# Local imports
from unmagnetised_plasma_ray_tracing.density_model import (QuadraticWell,
    DensityModel)
from unmagnetised_plasma_ray_tracing.hamiltonian_model import (
    UnmagnetisedPlasmaHamiltonian, UnmagnetisedPlasmaHamiltonianOptions,
    HamiltonianType, HamiltonianModel, HamiltonianModelOptions)
from unmagnetised_plasma_ray_tracing.parameter import ParameterCache
from unmagnetised_plasma_ray_tracing.wave_model import WaveModel

from unmagnetised_plasma_ray_tracing.numerics import (
    finite_difference_1st_derivative_2nd_order_stencil,
    finite_difference_2nd_derivative_2nd_order_stencil,
    finite_difference_mixed_2nd_derivative_2nd_order_stencil)

logger = logging.getLogger(__name__)

def first_derivative_finite_difference(x_value, x_parameter: ParameterCache,
    y_parameter: ParameterCache, y_function, h: float=1e-6):
    '''
    Calculate first derivative of a function using finite differences.
    '''
    x = np.array(x_value)
    x = x.reshape(x.size)
    dummy_x = np.zeros_like(x, dtype=x_parameter.dtype)

    n = x.size
    dtype = y_parameter.dtype
    derivative = np.zeros((n, *y_parameter.shape), dtype=dtype)

    for i in range(n):
        dummy_x[:] = x
        for step, weight in finite_difference_1st_derivative_2nd_order_stencil:
            dummy_x[i] = x[i] + step * h
            dummy_y = np.array(y_function(dummy_x)).reshape(y_parameter.shape)
            
            logger.warning((dummy_x, dummy_y))
            derivative[..., i] += weight * dummy_y

    return np.squeeze(derivative) / h

def second_derivative_finite_difference(x_value, x_parameter: ParameterCache,
    y_parameter: ParameterCache, y_function, h: float=1e-4):
    '''
    Calculate second derivative of a function using finite differences.
    '''
    x = np.array(x_value)
    x = x.reshape(x.size)
    dummy_x = np.zeros_like(x, dtype=x_parameter.dtype)

    n = x.size
    dtype = y_parameter.dtype
    derivative = np.zeros((n, n, *y_parameter.shape), dtype=dtype)

    for i in range(n):
        dummy_x[:] = x
        for step, weight in finite_difference_2nd_derivative_2nd_order_stencil:
            dummy_x[i] = x[i] + step * h
            dummy_y = np.array(y_function(dummy_x)).reshape(y_parameter.shape)
            
            derivative[..., i, i] += weight * dummy_y

        for j in range(i+1, n):
            dummy_x[:] = x
            for step1, step2, weight in finite_difference_mixed_2nd_derivative_2nd_order_stencil:
                dummy_x[i] = x[i] + step1 * h
                dummy_x[j] = x[j] + step2 * h
                dummy_y = np.array(y_function(dummy_x)).reshape(y_parameter.shape)
                
                derivative[..., i, j] += weight * dummy_y
    
            derivative[..., j, i] = derivative[..., i, j]

    return np.squeeze(derivative) / h**2

def second_mixed_derivative_finite_difference(x_value_1, 
    x_parameter_1: ParameterCache, x_value_2, x_parameter_2: ParameterCache,
    y_parameter: ParameterCache, y_function, h: float=1e-4):
    '''
    Calculate second derivative of a function using finite differences.
    '''
    x1 = np.array(x_value_1)
    x1 = x1.reshape(x1.size)
    dummy_x1 = np.zeros_like(x1, dtype=x_parameter_1.dtype)

    x2 = np.array(x_value_2)
    x2 = x2.reshape(x2.size)
    dummy_x2 = np.zeros_like(x2, dtype=x_parameter_2.dtype)

    dtype = y_parameter.dtype
    derivative = np.zeros((x1.size, x2.size, *y_parameter.shape), dtype=dtype)

    for i in range(x1.size):
        dummy_x1[:] = x_value_1

        for j in range(x2.size):
            dummy_x2[:] = x_value_2
            
            for step1, step2, weight in finite_difference_mixed_2nd_derivative_2nd_order_stencil:
                dummy_x1[i] = x1[i] + step1 * h
                dummy_x2[j] = x2[j] + step2 * h
                dummy_y = (np.array(y_function(dummy_x1, dummy_x2))
                    .reshape(y_parameter.shape))
                
                logger.warning((dummy_x1, dummy_x2, dummy_y))
                derivative[..., i, j] += weight * dummy_y

    return np.squeeze(derivative) / h**2

class TestUnmagnetisedPlasmaHamiltonian:
    dimension = 3
    frequency_ghz = 28.0

    @pytest.fixture
    def wave_model(self):
        wave_model = WaveModel(self.dimension)
        wave_model.frequency.set(self.frequency_ghz)
        wave_model.time.set(0.0)

        return wave_model
    
    @pytest.fixture
    def density_model(self, wave_model):
        density_data_model = QuadraticWell(self.frequency_ghz, np.zeros(3), 1)
        density_model = DensityModel(density_data_model, wave_model,
            self.dimension)
        
        return density_model

    @pytest.fixture
    def model(self, density_model, wave_model):
        options = UnmagnetisedPlasmaHamiltonianOptions()
        hamiltonian_model = UnmagnetisedPlasmaHamiltonian(density_model,
            wave_model, self.dimension, options)
        
        return hamiltonian_model

    def test_clear(self, model):
        for parameter_cache in model.all_caches():
            parameter_cache.cached = True
        
        model.clear()

        for parameter_cache in model.all_caches():
            assert not parameter_cache.cached, f"{parameter_cache.parameter}"

    test_normalised_values = ((0, 1), (0.2, 0.7), (0.7, 1.1), (1.2, 0.4))

    @pytest.mark.parametrize('X, N2', test_normalised_values)
    def test_hamiltonian_derivatives_X(self, model, X, N2):
        '''
        Test derivatives of hamiltonian with respect to normalised density X.
        '''
        model.density_model.normalised_density.set(X)
        model.refractive_index_squared.set(N2)
        value_func = lambda X: model.hamiltonian_function(X, N2).real

        expected_value = model.hamiltonian_first_derivative_X.get()
        actual_value = first_derivative_finite_difference(X,
            model.density_model.normalised_density, model.hamiltonian,
            value_func).real
        logger.warning(abs(actual_value - expected_value))
        assert np.isclose(actual_value, expected_value)

        expected_value = model.hamiltonian_second_derivative_X.get()
        actual_value = second_derivative_finite_difference(X,
            model.density_model.normalised_density, model.hamiltonian,
            value_func).real
        logger.warning(abs(actual_value - expected_value))
        assert np.isclose(actual_value, expected_value)
        
    @pytest.mark.parametrize('X, N2', test_normalised_values)
    def test_hamiltonian_derivatives_N2(self, model, X, N2):
        '''
        Test derivatives of hamiltonian with respect to squared refractive
        index N2.
        '''
        model.density_model.normalised_density.set(X)
        model.refractive_index_squared.set(N2)
        value_func = lambda N2: model.hamiltonian_function(X, N2).real

        expected_value = model.hamiltonian_first_derivative_N2.get()
        actual_value = first_derivative_finite_difference(N2,
            model.refractive_index_squared, model.hamiltonian, value_func).real
        logger.warning(abs(actual_value - expected_value))
        assert np.isclose(actual_value, expected_value)

        expected_value = model.hamiltonian_second_derivative_N2.get()
        actual_value = second_derivative_finite_difference(N2,
            model.refractive_index_squared, model.hamiltonian, value_func).real
        logger.warning(abs(actual_value - expected_value))
        assert np.isclose(actual_value, expected_value)
        
    @pytest.mark.parametrize('X, N2', test_normalised_values)
    def test_hamiltonian_second_derivative_X_N2(self, model, X, N2):
        '''
        Test second mixed derivative of hamiltonian with respect to
        normalised density X and squared refractive index N2.
        '''
        model.density_model.normalised_density.set(X)
        model.refractive_index_squared.set(N2)
        value_func = lambda X, N2: model.hamiltonian_function(X, N2).real

        expected_value = model.hamiltonian_second_derivative_X_N2.get()
        actual_value = second_mixed_derivative_finite_difference(X,
            model.density_model.normalised_density, N2,
            model.refractive_index_squared, model.hamiltonian, 
            value_func).real
        logger.warning(abs(actual_value - expected_value))
        assert np.isclose(actual_value, expected_value)

    test_phase_space_positions = (
        (np.array([0.29, 0.19, 0.16]), np.array([0.06, 0.05, 0.79])),
        (np.array([0.30, 0.64, 0.72]), np.array([0.30, 0.64, 0.72])),
        (np.array([0.30, 0.46, 0.72]), np.array([0.76, 0.83, 0.93])),
        (np.array([0.67, 0.61, 0.99]), np.array([0.05, 0.61, 0.51])),
        (np.array([0.91, 0.93, 0.60]), np.array([0.37, 0.92, 0.02])),
        (np.array([0.38, 0.61, 0.89]), np.array([0.41, 0.86, 0.22])),
    )

    @pytest.mark.parametrize('x, N', test_phase_space_positions)
    def test_hamiltonian_derivatives_x(self, model, x, N):
        model.wave_model.position.set(x)
        model.wave_model.refractive_index.set(N)

        def value_func(x):
            model.density_model.clear()

            model.wave_model.position.set(x)
            X = model.density_model.normalised_density.get()
            N2 = model.refractive_index_squared.get()

            return model.hamiltonian_function(X, N2).real

        expected_value = model.hamiltonian_first_derivative_x.get()
        actual_value = first_derivative_finite_difference(x,
            model.wave_model.position, model.hamiltonian,
            value_func).real
        logger.warning(abs(actual_value - expected_value))
        assert np.allclose(actual_value, expected_value)

        expected_value = model.hamiltonian_second_derivative_x.get()
        actual_value = second_derivative_finite_difference(x,
            model.wave_model.position, model.hamiltonian,
            value_func).real
        logger.warning(abs(actual_value - expected_value))
        assert np.allclose(actual_value, expected_value, atol=1e-7)

    @pytest.mark.parametrize('x, N', test_phase_space_positions)
    def test_hamiltonian_derivatives_N(self, model, x, N):
        model.wave_model.position.set(x)
        model.wave_model.refractive_index.set(N)

        def value_func(N):
            model.refractive_index_squared.clear()

            model.wave_model.refractive_index.set(N)
            X = model.density_model.normalised_density.get()
            N2 = model.refractive_index_squared.get()

            return model.hamiltonian_function(X, N2).real

        expected_value = model.hamiltonian_first_derivative_N.get()
        actual_value = first_derivative_finite_difference(N,
            model.wave_model.refractive_index, model.hamiltonian,
            value_func).real
        logger.warning(abs(actual_value - expected_value))
        assert np.allclose(actual_value, expected_value)

        expected_value = model.hamiltonian_second_derivative_N.get()
        actual_value = second_derivative_finite_difference(N,
            model.wave_model.refractive_index, model.hamiltonian,
            value_func).real
        logger.warning(abs(actual_value - expected_value))
        assert np.allclose(actual_value, expected_value, atol=1e-7)
       
    @pytest.mark.parametrize('x, N', test_phase_space_positions)
    def test_hamiltonian_second_derivative_xN(self, model, x, N):
        '''
        Test second mixed derivative of hamiltonian with respect to
        normalised density X and squared refractive index N2.
        '''
        model.wave_model.position.set(x)
        model.wave_model.refractive_index.set(N)

        def value_func(x, N):
            model.density_model.clear()
            model.refractive_index_squared.clear()

            model.wave_model.position.set(x)
            model.wave_model.refractive_index.set(N)

            X = model.density_model.normalised_density.get()
            N2 = model.refractive_index_squared.get()

            return model.hamiltonian_function(X, N2).real

        expected_value = model.hamiltonian_second_derivative_xN.get()
        actual_value = second_mixed_derivative_finite_difference(x,
            model.wave_model.position, N, model.wave_model.refractive_index,
            model.hamiltonian, value_func).real
        logger.warning(abs(actual_value - expected_value))
        assert np.allclose(actual_value, expected_value, atol=1e-7)

    @pytest.mark.parametrize('x, N', test_phase_space_positions)
    def test_hamiltonian_normalised_first_derivative_f(self, model, x, N):
        model.wave_model.position.set(x)
        model.wave_model.refractive_index.set(N)

        def value_func(f):
            frequency = model.wave_model.frequency.get().item()
            X = model.density_model.normalised_density.get().item()
            N2 = model.refractive_index_squared.get().item()

            factor = (frequency / f.item())**2
            X *= factor
            N2 *= factor

            return model.hamiltonian_function(X, N2).real
        
        value_func(np.array(self.frequency_ghz))

        expected_value = model.hamiltonian_normalised_first_derivative_f.get()
        actual_value = first_derivative_finite_difference(self.frequency_ghz,
            model.wave_model.frequency, model.hamiltonian, value_func, h=1e-3).real
        
        # Derivative is also multiplied by frequency to normalise.
        actual_value *= self.frequency_ghz
        logger.warning(abs(actual_value - expected_value))
        assert np.allclose(actual_value, expected_value)

    @pytest.mark.parametrize('x, N', test_phase_space_positions)
    def test_hamiltonian_normalised_second_derivative_xf(self, model, x, N):
        '''
        Test second mixed derivative of hamiltonian with respect to
        normalised density X and squared refractive index N2.
        '''
        model.wave_model.position.set(x)
        model.wave_model.refractive_index.set(N)

        def value_func(x, f):
            model.density_model.clear()

            model.wave_model.position.set(x)

            frequency = model.wave_model.frequency.get().item()
            X = model.density_model.normalised_density.get().item()
            N2 = model.refractive_index_squared.get().item()

            factor = (frequency / f.item())**2
            X *= factor
            N2 *= factor

            return model.hamiltonian_function(X, N2).real

        expected_value = model.hamiltonian_normalised_second_derivative_xf.get()
        actual_value = second_mixed_derivative_finite_difference(x,
            model.wave_model.position, self.frequency_ghz,
            model.wave_model.frequency, model.hamiltonian, value_func).real
        
        # Derivative is also multiplied by frequency to normalise.
        actual_value *= self.frequency_ghz
        logger.warning(abs(actual_value - expected_value))
        assert np.allclose(actual_value, expected_value)

    @pytest.mark.parametrize('x, N', test_phase_space_positions)
    def test_hamiltonian_normalised_second_derivative_Nf(self, model, x, N):
        '''
        Test second mixed derivative of hamiltonian with respect to
        normalised density X and squared refractive index N2.
        '''
        model.wave_model.position.set(x)
        model.wave_model.refractive_index.set(N)

        def value_func(N, f):
            model.refractive_index_squared.clear()

            model.wave_model.refractive_index.set(N)

            frequency = model.wave_model.frequency.get().item()
            X = model.density_model.normalised_density.get().item()
            N2 = model.refractive_index_squared.get().item()

            factor = (frequency / f.item())**2
            X *= factor
            N2 *= factor

            return model.hamiltonian_function(X, N2).real

        expected_value = model.hamiltonian_normalised_second_derivative_Nf.get()
        actual_value = second_mixed_derivative_finite_difference(N,
            model.wave_model.refractive_index, self.frequency_ghz,
            model.wave_model.frequency, model.hamiltonian, value_func).real
        
        # Derivative is also multiplied by frequency to normalise.
        actual_value *= self.frequency_ghz
        logger.warning(abs(actual_value - expected_value))
        assert np.allclose(actual_value, expected_value)

    @pytest.mark.parametrize('x, N', test_phase_space_positions)
    def test_ray_acceleration(self, model, x, N):
        ''' Test derivative of velocity with respect to time is acceleration. '''
        model.wave_model.position.set(x)
        model.wave_model.refractive_index.set(N)
        
        expected_value = model.normalised_ray_acceleration.get()

        actual_value = np.zeros(model.normalised_ray_acceleration.shape)
        dt = 1e-11 # 1e-6 * 1e-9
        dummy_x, dummy_N = np.zeros_like(x), np.zeros_like(N)
        ray_velocity = const.speed_of_light * model.normalised_ray_velocity.get()

        for step, weight in finite_difference_1st_derivative_2nd_order_stencil:
            model.clear()
            model.density_model.clear()

            dummy_x[:] = x + step * dt * ray_velocity[:3]
            dummy_N[:] = N + step * dt * ray_velocity[3:]
            model.wave_model.position.set(dummy_x)
            model.wave_model.refractive_index.set(dummy_N)

            actual_value += weight * model.normalised_ray_velocity.get()

        actual_value /= dt * const.speed_of_light
        logger.warning(abs(actual_value - expected_value))
        assert np.allclose(actual_value, expected_value, atol=5e-5)

class TestHamiltonianModel:
    dimension = 3
    frequency_ghz = 28.0

    @pytest.fixture
    def wave_model(self):
        wave_model = WaveModel(self.dimension)
        wave_model.frequency.set(self.frequency_ghz)
        wave_model.time.set(0.0)

        return wave_model

    @pytest.fixture
    def density_model(self, wave_model):
        density_data_model = QuadraticWell(self.frequency_ghz, np.zeros(3), 1)
        density_model = DensityModel(density_data_model, wave_model,
            self.dimension)
        
        return density_model

    @pytest.fixture
    def model(self, density_model, wave_model):
        options = HamiltonianModelOptions()
        return HamiltonianModel(density_model, wave_model, self.dimension,
            options)
    
    def test_model_types(self, model):
        assert isinstance(model[HamiltonianType.COLD_UNMAGNETISED],
            UnmagnetisedPlasmaHamiltonian)
        