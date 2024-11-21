#!/usr/bin/python3

# Standard imports
import abc
import logging
import numpy as np

# Local imports

logger = logging.getLogger(__name__)

class DensityModel(abc.ABC):
    ''' Model providing electron density data in Cartesian coordinates. '''
    __slots__ = ()

    @abc.abstractmethod
    def density(self, position_cartesian):
        '''
        Return electron density at given cartesian position [m^-3].
        '''
    
    @abc.abstractmethod
    def density_first_derivative(self, position_cartesian):
        '''
        Return first derivative of electron density with respect to position
        at given cartesian position [m^-3.m^-1].
        '''
    
    @abc.abstractmethod
    def density_second_derivative(self, position_cartesian):
        '''
        Return second derivative of electron density with respect to position
        at given cartesian position [m^-3.m^-2].
        '''

class C2Ramp(DensityModel):
    '''
    Density ramps from y0 to y1 in x direction only such that the density
    profile has C2 smoothness.
    '''
    __slots__ = ("x0", "y0", "dy", "Ln_inverse")

    def __init__(self, x0: float, y0: float, y1: float, Ln_inverse: float):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.dy = float(y1 - y0)
        self.Ln_inverse = float(Ln_inverse)

    def normalise_position(self, position):
        return np.clip((position[0] - self.x0) * self.Ln_inverse, 0, 1)

    def density(self, position):
        x = self.normalise_position(position)
        return self.y0 + self.dy * (10 - 15*x + 6*x**2) * x**3

    def density_first_derivative(self, position):
        x = self.normalise_position(position)
        dy_dx = np.zeros_like(position)
        dy_dx[0] = 30 * self.dy * x**2 * (1 - x)**2
        return dy_dx

    def density_second_derivative(self, position):
        x = self.normalise_position(position)
        n = len(position)
        d2y_dx2 = np.zeros((n, n))
        d2y_dx2[0, 0] = 60 * self.dy * x * (1 - x) * (1 - 2*x)
        return d2y_dx2

class QuadraticWell(DensityModel):
    '''
    Density is a quadratic well with the well bottom parallel to the y axis.
    '''
    __slots__ = ("origin", "Ln_inverse")

    def __init__(self, origin: float, Ln_inverse: float):
        self.origin = float(origin)
        self.Ln_inverse = float(Ln_inverse)

    def perpendicular_position(self, position):
        return (dx**2 + dz**2)**0.5

    def density(self, position):
        dx, dz = position[0] - self.origin[0], position[2] - self.origin[2]
        return self.Ln_inverse**2 * (dx**2 + dz**2)

    def density_first_derivative(self, position):
        dy_dx = np.zeros_like(position)
        dy_dx[0] = 2 * position[0]
        dy_dx[2] = 2 * position[2]

        return dy_dx

    def density_second_derivative(self, position):
        n = len(position)
        d2y_dx2 = np.zeros((n, n))
        d2y_dx2[0, 0] = 2
        d2y_dx2[2, 2] = 2

        return self.Ln_inverse**2 * d2y_dx2
