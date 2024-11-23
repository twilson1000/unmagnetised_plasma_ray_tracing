#!/usr/bin/python3

# Standard imports
import abc
import logging
import numpy as np
import numpy.typing as npt

# Local imports
from .model import ModelBase
from .parameter import Parameter, ParameterCache
from .wave_model import WaveModel

logger = logging.getLogger(__name__)

class DensityDataModel(abc.ABC):
    ''' Model providing electron density data in Cartesian coordinates. '''
    __slots__ = ("__weakref__",)

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

class DensityModel(ModelBase):
    __slots__ = ("__weakref__", "model", "wave_model", "density",
        "density_first_derivative", "density_second_derivative",
        "critical_density", "normalised_density",
        "normalised_density_first_derivative",
        "normalised_density_second_derivative")
    
    root = "plasma"

    _density = Parameter.scalar("electron_density", "Electron density",
        "m^-3", root)
    _density_first_derivative = Parameter.covector(
        "electron_density_first_derivative", ("First derivative of electron"
        " density with respect to space"), "m^-3.m^-1", root)
    _density_second_derivative = Parameter.rank_02_tensor(
        "electron_density_second_derivative", ("Second derivative of electron "
        " density with respect to space"), "m^-3.m^-2", root)
    _critical_density = Parameter.scalar("critical_density", ("Critical "
        "density where wave frequency equals electron plasma frequency"),
        "GHz", root)
    _normalised_density = Parameter.scalar("normalised_density", 
        "Density divided by critical density", "", root)
    _normalised_density_first_derivative = Parameter.covector(
        "normalised_density_first_derivative",
        "First derivative of normalised density", "m^-1", root)
    _normalised_density_second_derivative = Parameter.rank_02_tensor(
        "normalised_density_second_derivative",
        "Second derivative of normalised density", "m^-2", root)

    def __init__(self, model: DensityDataModel, wave_model: WaveModel,
        dimension: int):
        '''
        
        '''
        self.model = model
        self.wave_model = wave_model

        self.density = ParameterCache.with_callback(self._density, dimension,
            self.set_density, bound_method=True)
        
        self.density_first_derivative = ParameterCache.with_callback(
            self._density_first_derivative, dimension,
            self.set_density_first_derivative, bound_method=True)
        
        self.density_second_derivative = ParameterCache.with_callback(
            self._density_second_derivative, dimension,
            self.set_density_second_derivative, bound_method=True)
    
        self.critical_density = ParameterCache.with_callback(
            self._critical_density, dimension, self.set_critical_density,
            bound_method=True)
        
        self.normalised_density = ParameterCache.with_callback(
            self._normalised_density, dimension, self.set_normalised_density,
            bound_method=True)
        
        self.normalised_density_first_derivative = ParameterCache.with_callback(
            self._normalised_density_first_derivative, dimension,
            self.set_normalised_density_first_derivative, bound_method=True)
        
        self.normalised_density_second_derivative = ParameterCache.with_callback(
            self._normalised_density_second_derivative, dimension,
            self.set_normalised_density_second_derivative, bound_method=True)

    def set_density(self):
        '''
        
        '''
        position = self.wave_model.position.get()
        density = self.model.density(position)
        self.density.set(density)
    
    def set_density_first_derivative(self):
        '''
        
        '''
        position = self.wave_model.position.get()
        density_first_derivative = self.model.density_first_derivative(position)
        self.density_first_derivative.set(density_first_derivative)
    
    def set_density_second_derivative(self):
        '''
        
        '''
        position = self.wave_model.position.get()
        density_second_derivative = self.model.density_second_derivative(position)
        self.density_second_derivative.set(density_second_derivative)

    def set_critical_density(self):
        '''
        
        '''
        frequency_ghz = self.wave_model.frequency.get()
        critical_density_per_m3 = 1e18 * (frequency_ghz / 9)**2
        self.critical_density.set(critical_density_per_m3)
    
    def set_normalised_density(self):
        '''
        
        '''
        critical_density_per_m3 = self.critical_density.get()
        density_per_m3 = self.density.get()
        normalised_density = density_per_m3 / critical_density_per_m3
        self.normalised_density.set(normalised_density)

    def set_normalised_density_first_derivative(self):
        '''
        
        '''
        critical_density_per_m3 = self.critical_density.get()
        density_first_derivative_per_m4 = (
            self.density_first_derivative.get())
        normalised_density_first_derivative = (
            density_first_derivative_per_m4 / critical_density_per_m3)
        self.normalised_density_first_derivative.set(
            normalised_density_first_derivative)

    def set_normalised_density_second_derivative(self):
        '''
        
        '''
        critical_density_per_m3 = self.critical_density.get()
        density_second_derivative_per_m5 = (
            self.density_second_derivative.get())
        normalised_density_second_derivative = (
            density_second_derivative_per_m5 / critical_density_per_m3)
        self.normalised_density_second_derivative.set(
            normalised_density_second_derivative)
    
class NormalisedDensityDataModel(DensityDataModel):
    __slots__ = ("critical_density",)

    def __init__(self, frequency_ghz: float):
        self.critical_density = 1e18 * (float(frequency_ghz) / 9)**2
    
    @abc.abstractmethod
    def normalised_density(self, position_cartesian):
        '''
        Return normalised electron density at given cartesian position [].
        '''
    
    @abc.abstractmethod
    def normalised_density_first_derivative(self, position_cartesian):
        '''
        Return first derivative of normalised electron density with respect to
        position at given cartesian position [m^-1].
        '''

    @abc.abstractmethod
    def normalised_density_second_derivative(self, position_cartesian):
        '''
        Return second derivative of normalised electron density with respect to
        position at given cartesian position [m^-2].
        '''
    
    def density(self, position_cartesian):
        return self.critical_density * self.normalised_density(position_cartesian)
    
    def density_first_derivative(self, position_cartesian):
        return self.critical_density * self.normalised_density_first_derivative(position_cartesian)

    def density_second_derivative(self, position_cartesian):
        return self.critical_density * self.normalised_density_second_derivative(position_cartesian)

class Vacuum(DensityDataModel):
    '''
    Vacuum, density is zero everywhere.
    '''
    def __init__(self) -> None:
        super().__init__()
    
    def density(self, position_cartesian):
        return 0.0
    
    def density_first_derivative(self, position_cartesian):
        return np.zeros(self.density_first_derivative.shape)
    
    def density_second_derivative(self, position_cartesian):
        return np.zeros(self.density_second_derivative.shape)

class C2Ramp(NormalisedDensityDataModel):
    '''
    Density ramps from y0 to y1 in x direction only such that the density
    profile has C2 smoothness.
    '''
    __slots__ = ("x0", "y0", "dy", "Ln_inverse")

    def __init__(self, frequency_ghz: float, x0: float, y0: float, y1: float,
        Ln_inverse: float):
        super().__init__(frequency_ghz)
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.dy = float(y1 - y0)
        self.Ln_inverse = float(Ln_inverse)

    def normalise_position(self, position):
        return np.clip((position[0] - self.x0) * self.Ln_inverse, 0, 1)

    def normalised_density(self, position):
        x = self.normalise_position(position)
        return self.y0 + self.dy * (10 - 15*x + 6*x**2) * x**3

    def normalised_density_first_derivative(self, position):
        x = self.normalise_position(position)
        dy_dx = np.zeros_like(position)
        dy_dx[0] = 30 * self.Ln_inverse * self.dy * x**2 * (1 - x)**2
        return dy_dx

    def normalised_density_second_derivative(self, position):
        x = self.normalise_position(position)
        n = len(position)
        d2y_dx2 = np.zeros((n, n))
        d2y_dx2[0, 0] = 60 * self.Ln_inverse**2 * self.dy * x * (1 - x) * (1 - 2*x)
        return d2y_dx2

class QuadraticChannel(NormalisedDensityDataModel):
    '''
    Density is a quadratic well with the well bottom parallel to the y axis.
    '''
    __slots__ = ("origin", "Ln_inverse")

    def __init__(self, frequency_ghz: float, origin: npt.NDArray[float],
        Ln_inverse: float):
        super().__init__(frequency_ghz)
        self.origin = np.array(origin)
        self.Ln_inverse = float(Ln_inverse)

    def normalised_density(self, position):
        dx, dz = position[0] - self.origin[0], position[2] - self.origin[2]
        return self.Ln_inverse**2 * (dx**2 + dz**2)

    def normalised_density_first_derivative(self, position):
        dy_dx = np.zeros_like(position)
        dy_dx[0] = 2 * position[0]
        dy_dx[2] = 2 * position[2]

        return self.Ln_inverse**2 * dy_dx

    def normalised_density_second_derivative(self, position):
        n = len(position)
        d2y_dx2 = np.zeros((n, n))
        d2y_dx2[0, 0] = 2
        d2y_dx2[2, 2] = 2

        return self.Ln_inverse**2 * d2y_dx2

class QuadraticWell(NormalisedDensityDataModel):
    '''
    Density is a quadratic well centred at origin.
    '''
    __slots__ = ("origin", "Ln_inverse")

    def __init__(self, frequency_ghz: float, origin: npt.NDArray[float],
        Ln_inverse: float):
        super().__init__(frequency_ghz)
        self.origin = np.array(origin)
        self.Ln_inverse = float(Ln_inverse)

    def normalised_density(self, position):
        dx = position - self.origin
        return self.Ln_inverse**2 * sum(dx**2)

    def normalised_density_first_derivative(self, position):
        dy_dx = np.zeros_like(position)
        dy_dx[:] = 2 * position

        return self.Ln_inverse**2 * dy_dx

    def normalised_density_second_derivative(self, position):
        n = len(position)
        d2y_dx2 = np.zeros((n, n))
        d2y_dx2[0, 0] = 2
        d2y_dx2[1, 1] = 2
        d2y_dx2[2, 2] = 2

        return self.Ln_inverse**2 * d2y_dx2
