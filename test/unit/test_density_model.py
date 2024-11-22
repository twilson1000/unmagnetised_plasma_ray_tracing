#!/usr/bin/python3

# Standard imports
import logging
import numpy as np
import pytest
from typing import Tuple

# Local imports
from unmagnetised_plasma_ray_tracing.density_model import C2Ramp, QuadraticWell
from unmagnetised_plasma_ray_tracing.numerics import (
    finite_difference_1st_derivative_2nd_order_stencil,
    finite_difference_2nd_derivative_2nd_order_stencil,
    finite_difference_mixed_2nd_derivative_2nd_order_stencil)

logger = logging.getLogger(__name__)

def first_derivative_finite_difference(position, value_function,
    value_shape: Tuple[int], h: float=1e-6):
    '''
    Calculate first derivative of a function using finite differences.
    '''
    n = position.size
    derivative = np.zeros((n, *value_shape))

    dummy_x = np.zeros_like(position)
    for i in range(n):
        dummy_x[:] = position
        for step, weight in finite_difference_1st_derivative_2nd_order_stencil:
            dummy_x[i] = position[i] + step * h
            derivative[..., i] += weight * value_function(dummy_x)

    return derivative / h

def second_derivative_finite_difference(position, value_function,
    value_shape: Tuple[int], h: float=1e-4):
    '''
    Calculate second derivative of a function using finite differences.
    '''
    n = position.size
    derivative = np.zeros((n, n, *value_shape))

    dummy_x = np.zeros_like(position)
    for i in range(n):
        dummy_x[:] = position
        for step, weight in finite_difference_2nd_derivative_2nd_order_stencil:
            dummy_x[i] = position[i] + step * h
            derivative[..., i, i] += weight * value_function(dummy_x)

        for j in range(i+1, n):
            dummy_x[:] = position
            for step1, step2, weight in finite_difference_mixed_2nd_derivative_2nd_order_stencil:
                dummy_x[i] = position[i] + step1 * h
                dummy_x[j] = position[j] + step2 * h
                derivative[..., i, j] += weight * value_function(dummy_x)
    
            derivative[..., j, i] = derivative[..., i, j]

    return derivative / h**2

class DensityModelPresets:
    test_positions = (np.array([1.0, 0.0, 0.0]), np.array([1.3, -0.3, 1.3]),
        np.array([-1.12, 1.77, -2.89]), np.array([-1.06, -0.59, -2.80]))

    @pytest.mark.parametrize('position', test_positions)
    def test_derivatives(self, model, position):
        expected_first_derivative = model.normalised_density_first_derivative(position)
        actual_first_derivative = first_derivative_finite_difference(position,
            model.normalised_density, ())
        
        logger.warning(abs(actual_first_derivative - expected_first_derivative))
        assert np.allclose(actual_first_derivative, expected_first_derivative)

        expected_second_derivative = model.normalised_density_second_derivative(position)
        actual_second_derivative = second_derivative_finite_difference(position,
            model.normalised_density, ())
        
        logger.warning(abs(actual_second_derivative - expected_second_derivative))
        assert np.allclose(actual_second_derivative, expected_second_derivative)

class TestC2Ramp(DensityModelPresets):
    @pytest.fixture
    def model(self):
        return C2Ramp(28.0, 0.0, 0.1, 1.3, 0.5)

class TestQuadraticWell(DensityModelPresets):
    @pytest.fixture
    def model(self):
        return QuadraticWell(28.0, [0, 0, 0], 0.5)
