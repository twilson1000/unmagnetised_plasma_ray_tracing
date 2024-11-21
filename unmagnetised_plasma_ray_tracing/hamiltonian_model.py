#!/usr/bin/python3

# Standard imports
import abc
from enum import Enum, unique
import logging
import numpy as np
from typing import Dict

# Local imports
from .density_model import DensityModel
from .parameter import Parameter, ParameterCache
from .wave_model import WaveModel

logger = logging.getLogger(__name__)

class HamiltonianBase(abc.ABC):
    __slots__ = ("hamiltonian", "dispersion_tensor", "normalised_ray_velocity",
        "_normalised_ray_acceleration")

    _hamiltonian = Parameter.scalar("hamiltonian", "Ray tracing hamiltonian",
        "", complex=True)
    _dispersion_tensor = Parameter.rank_11_tensor("dispersion_tensor",
        "Dispersion tensor for electromagnetic wave", "", complex=True)
    _normalised_ray_velocity = Parameter.vector("normalised_ray_velocity",
        ("First derivative of ray position in phase space (x, N) with respect "
        "to time normalised to the speed of light."), "")
    _normalised_ray_acceleration = Parameter.vector("normalised_ray_velocity",
        ("Second derivative of ray position in phase space (x, N) with respect "
        "to time normalised to the speed of light squared."), "")


    def __init__(self, dimension: int) -> None:
        self.hamiltonian = ParameterCache(self._hamiltonian, dimension)
        self.hamiltonian.register_cache_miss_callback(
            self.set_hamiltonian, bound_method=True)
        
        self.dispersion_tensor = ParameterCache.rank_11_tensor(
            self._dispersion_tensor, dimension)
        self.dispersion_tensor.register_cache_miss_callback(
            self.set_dispersion_tensor)
        
        self.normalised_ray_velocity = ParameterCache(
            self._normalised_ray_velocity, 2 * dimension)
        self.normalised_ray_velocity.register_cache_miss_callback(
            self.set_normalised_ray_velocity)
        
        self.normalised_ray_acceleration = ParameterCache(
            self._normalised_ray_acceleration, 2 * dimension)
        self.normalised_ray_acceleration.register_cache_miss_callback(
            self.set_normalised_ray_acceleration)
        

    @abc.abstractmethod
    def set_hamiltonian(self):
        '''
        
        '''

    @abc.abstractmethod
    def set_dispersion_tensor(self):
        '''
        
        '''

    @abc.abstractmethod    
    def set_normalised_ray_velocity(self):
        '''
        
        '''

    @abc.abstractmethod
    def set_normalised_ray_acceleration(self):
        '''
        
        '''

class UnmagnetisedPlasmaHamiltonian(HamiltonianBase):
    __slots__ = ("reduced_dispersion", "wave_model",
        "refractive_index_squared", "hamiltonian_first_derivative_X",
        "hamiltonian_first_derivative_N2", "hamiltonian_first_derivative_x",
        "hamiltonian_first_derivative_N", "hamiltonian_first_derivative_f",
        "hamiltonian_second_derivative_X", "hamiltonian_second_derivative_N2",
        "hamiltonian_second_derivative_X_N2", "hamiltonian_second_derivative_x",
        "hamiltonian_second_derivative_N", "hamiltonian_second_derivative_xN",
        )
    
    _refractive_index_squared = Parameter.scalar("refractive_index_squared",
        "Square of magnitude of refractive index covector", "")
    
    _hamiltonian_first_derivative_X = Parameter.scalar(
        "hamiltonian_first_derivative_X", ("First derivative of real part "
        "of hamiltonian with respect to normalised density X."), "")
    _hamiltonian_first_derivative_N2 = Parameter.scalar(
        "hamiltonian_first_derivative_N2", ("First derivative of real part "
        "of hamiltonian with respect to squared refractive index."), "")

    _hamiltonian_first_derivative_x = Parameter.covector(
        "hamiltonian_first_derivative_x", ("First derivative of real "
        "part of hamiltonian with respect to space."), "m^-1")
    _hamiltonian_first_derivative_N = Parameter.covector(
        "hamiltonian_first_derivative_N", ("First derivative of real "
        "part of hamiltonian with respect to refractive index."), "m")
    _hamiltonian_normalised_first_derivative_f = Parameter.covector(
        "hamiltonian_first_derivative_f", ("First derivative of real "
        "part of hamiltonian with respect to frequency."), "ns")
    
    _hamiltonian_second_derivative_X = Parameter.scalar(
        "hamiltonian_second_derivative_X", ("Second derivative of real part "
        "of hamiltonian with respect to normalised density X."), "")
    _hamiltonian_second_derivative_N2 = Parameter.scalar(
        "hamiltonian_second_derivative_N2", ("Second derivative of real part "
        "of hamiltonian with respect to squared refractive index."), "")
    _hamiltonian_second_derivative_X_N2 = Parameter.scalar(
        "hamiltonian_second_derivative_X_N2", ("Second mixed derivative of "
        "real part of hamiltonian with respect to normalised density and "
        "squared refractive index."), "")
    
    _hamiltonian_second_derivative_x = Parameter.covector(
        "hamiltonian_second_derivative_x", ("Second derivative of real "
        "part of hamiltonian with respect to space."), "m^-2")
    _hamiltonian_second_derivative_N = Parameter.covector(
        "hamiltonian_second_derivative_N", ("Second derivative of real "
        "part of hamiltonian with respect to refractive index."), "m^2")
    _hamiltonian_second_derivative_xN = Parameter.covector(
        "hamiltonian_second_derivative_xN", ("Second mixed derivative of real "
        "part of hamiltonian with respect to position and refractive index."),
        "m^2")

    _set_hamiltonian_normalised_second_derivative_xf = Parameter.covector(
        "hamiltonian_normalised_second_derivative_xf", ())

    _set_hamiltonian_normalised_second_derivative_Nf = Parameter.vector(
        "hamiltonian_normalised_second_derivative_Nf", ())
    

    def __init__(self, density_model: DensityModel, wave_model: WaveModel,
        dimension: int, reduced_dispersion: bool=True):
        '''
        
        '''
        # Other data models we draw data from.
        self.density_model = density_model
        self.wave_model = wave_model

        # Options for hamiltonian calculation.
        self.reduced_dispersion = reduced_dispersion

        # Parameter caches.
        self.refractive_index_squared = ParameterCache(
            self._refractive_index_squared, dimension)
        self.refractive_index_squared.register_cache_miss_callback(
            self.get_refractive_index_squared, bound_method=True)
        
        self.hamiltonian_first_derivative_X = ParameterCache(
            self._hamiltonian_first_derivative_X, dimension)
        self.hamiltonian_first_derivative_X.register_cache_miss_callback(
            self.set_hamiltonian_first_derivative_X, bound_method=True)

        self.hamiltonian_first_derivative_N2 = ParameterCache(
            self._hamiltonian_first_derivative_N2, dimension)
        self.hamiltonian_first_derivative_N2.register_cache_miss_callback(
            self.set_hamiltonian_first_derivative_N2, bound_method=True)
        
        self.hamiltonian_first_derivative_x = ParameterCache(
            self._hamiltonian_first_derivative_x, dimension)
        self.hamiltonian_first_derivative_x.register_cache_miss_callback(
            self.set_hamiltonian_first_derivative_x, bound_method=True)
        
        self.hamiltonian_first_derivative_N = ParameterCache(
            self._hamiltonian_first_derivative_N, dimension)
        self.hamiltonian_first_derivative_N.register_cache_miss_callback(
            self.set_hamiltonian_first_derivative_N, bound_method=True)

        self.hamiltonian_normalised_first_derivative_f = ParameterCache(
            self._hamiltonian_normalised_first_derivative_f, dimension)
        self.hamiltonian_normalised_first_derivative_f.register_cache_miss_callback(
            self.set_hamiltonian_normalised_first_derivative_f, bound_method=True)

        self.hamiltonian_second_derivative_X = ParameterCache(
            self.hamiltonian_second_derivative_X, dimension)
        self.hamiltonian_second_derivative_X.register_cache_miss_callback(
            self.set_hamiltonian_second_derivative_X, bound_method=True)

        self.hamiltonian_second_derivative_N2 = ParameterCache(
            self.hamiltonian_second_derivative_N2, dimension)
        self.hamiltonian_second_derivative_N2.register_cache_miss_callback(
            self.set_hamiltonian_second_derivative_N2, bound_method=True)

        self.hamiltonian_second_derivative_X_N2 = ParameterCache(
            self.hamiltonian_second_derivative_X_N2, dimension)
        self.hamiltonian_second_derivative_X_N2.register_cache_miss_callback(
            self.set_hamiltonian_second_derivative_X_N2, bound_method=True)



        self.hamiltonian_second_derivative_x = ParameterCache(
            self.hamiltonian_second_derivative_x, dimension)
        self.hamiltonian_second_derivative_x.register_cache_miss_callback(
            self.set_hamiltonian_second_derivative_x, bound_method=True)

        self.hamiltonian_second_derivative_N = ParameterCache(
            self.hamiltonian_second_derivative_N, dimension)
        self.hamiltonian_second_derivative_N.register_cache_miss_callback(
            self.set_hamiltonian_second_derivative_N, bound_method=True)

        self.hamiltonian_second_derivative_xN = ParameterCache(
            self.hamiltonian_second_derivative_xN, dimension)
        self.hamiltonian_second_derivative_xN.register_cache_miss_callback(
            self.set_hamiltonian_second_derivative_xN, bound_method=True)

    def set_hamiltonian(self):
        '''
        Return complex valued ray tracing Hamiltonian.
        '''
        X = self.wave_model.normalised_density.get()
        N2 = self.refractive_index_squared.get()

        if self.reduced_dispersion:
            hamiltonian = 1 - X - N2
        else:
            hamiltonian = (1 - X) * (1 - X - N2)**2
        
        self.hamiltonian.set(hamiltonian)

    def set_hamiltonian_first_derivative_X(self):
        '''
        Return first derivative of real part of Hamiltonian with respect to
        normalised density X.
        '''
        X = self.wave_model.normalised_density.get()
        N2 = self.refractive_index_squared.get()

        if self.reduced_dispersion:
            return -1
        else:
            return -(1 - X - N2) * (3 - 3*X - N2)

    def set_hamiltonian_first_derivative_N2(self):
        '''
        Return first derivative of real part of Hamiltonian with respect to
        squared refractive index.
        '''
        X = self.wave_model.normalised_density.get()
        N2 = self.refractive_index_squared.get()

        if self.reduced_dispersion:
            return -1
        else:
            return -2 * (1 - X) * (1 - X - N2)
    
    def set_hamiltonian_normalised_first_derivative_f(self):
        '''
        Return first derivative of real part of Hamiltonian with respect to
        frequency multiplied by frequency i.e. f * dH/df.
        '''
        X = self.wave_model.normalised_density.get()
        N2 = self.refractive_index_squared.get()

        if self.reduced_dispersion:
            return 2 * (X + N2)
        else:
            return 2 * (1 - X - N2) * (3 * X * (1 - X - N2) + 2 * N2)

    def set_hamiltonian_first_derivative_x(self):
        '''
        Return first derivative of real part of Hamiltonian with respect to
        space.
        '''
        dD_dX = self.hamiltonian_first_derivative_X.get()
        dX_dx = self.density_model.normalised_density_first_derivative.get()

        if self.reduced_dispersion:
            return dD_dX * dX_dx
        else:
            raise NotImplementedError()

    def set_hamiltonian_first_derivative_N(self):
        '''
        Return first derivative of real part of Hamiltonian with respect to
        refractive index.
        '''
        N = self.wave_model.refractive_index.get()
        dD_dN2 = self.hamiltonian_first_derivative_N2.get()

        if self.reduced_dispersion:
            return 2 * N * dD_dN2
        else:
            raise NotImplementedError()

    def set_hamiltonian_second_derivative_X(self):
        '''
        Return second derivative of real part of Hamiltonian with respect to
        normalised density X.
        '''
        if self.reduced_dispersion:
            return 0.0
        else:
            raise NotImplementedError()

    def set_hamiltonian_second_derivative_N2(self):
        '''
        Return second derivative of real part of Hamiltonian with respect to
        squared refractive index.
        '''
        if self.reduced_dispersion:
            return 0.0
        else:
            raise NotImplementedError()

    def set_hamiltonian_second_derivative_X_N2(self):
        '''
        Return second mixed derivative of real part of Hamiltonian with respect
        to normalised density X and squared refractive index.
        '''
        if self.reduced_dispersion:
            return 0.0
        else:
            raise NotImplementedError()

    def set_hamiltonian_second_derivative_x(self):
        '''
        Return second derivative of real part of Hamiltonian with respect to
        position.
        '''
        if self.reduced_dispersion:
            d2X_dx2 = self.density_model.normalised_density_second_derivative.get()
            return -d2X_dx2
        else:
            raise NotImplementedError()

    def set_hamiltonian_second_derivative_N(self):
        '''
        Return second derivative of real part of Hamiltonian with respect to
        refractive index.
        '''
        if self.reduced_dispersion:
            return -2 * np.identity(self.wave_model.refractive_index.shape[0])
        else:
            raise NotImplementedError()

    def set_hamiltonian_second_derivative_xN(self):
        '''
        Return second mixed derivative of real part of Hamiltonian with respect
        to position and refractive index.
        '''
        if self.reduced_dispersion:
            return np.zeros(self.hamiltonian_second_derivative_xN.shape)
        else:
            raise NotImplementedError()

    def set_hamiltonian_normalised_second_derivative_xf(self):
        '''
        Return normalised second mixed derivative of real part of Hamiltonian
        with respect to frequency and position i.e. f * (d^2 H / dx df).
        '''
        if self.reduced_dispersion:
            dX_dx = self.density_model.normalised_density_first_derivative.get()
            return 2 * dX_dx
        else:
            raise NotImplementedError()

    def set_hamiltonian_normalised_second_derivative_Nf(self):
        '''
        Return normalised second mixed derivative of real part of Hamiltonian
        with respect to frequency and refractive index i.e. f * (d^2 H / dN df).
        '''
        if self.reduced_dispersion:
            N = self.wave_model.refractive_index.get()
            return 4 * N
        else:
            raise NotImplementedError()

    def normalised_ray_velocity(self):
        '''
        Return velocity of ray in phase space (x, N) normalised to the speed
        of light.
        '''
        if self.reduced_dispersion:
            dD_dx = self.hamiltonian_first_derivative_x.get()
            dD_dN = self.hamiltonian_first_derivative_N.get()
            f_dD_df = self.hamiltonian_normalised_first_derivative_f.get()

            normalised_velocity = np.zeros(self.normalised_ray_velocity.shape)
            n = len(normalised_velocity) // 2

            normalised_velocity[:n] = -dD_dN / f_dD_df
            normalised_velocity[n:] = dD_dx / f_dD_df

            return normalised_velocity
        else:
            raise NotImplementedError()

    def normalised_ray_acceleration(self):
        '''
        Return acceleration of ray in phase space (x, N) normalised to the
        speed of light squared.
        '''
        if self.reduced_dispersion:
            dD_dx = self.hamiltonian_first_derivative_x.get()
            dD_dN = self.hamiltonian_first_derivative_N.get()
            f_dD_df = self.hamiltonian_normalised_first_derivative_f.get()
            d2D_dx2 = self.hamiltonian_second_derivative_x.get()
            d2D_dN2 = self.hamiltonian_second_derivative_N.get()

            normalised_acceleration = np.zeros(self.normalised_ray_acceleration.shape)
            n = len(normalised_acceleration) // 2
            normalised_acceleration[:n] = -np.einsum('ij,j', d2D_dN2, dD_dx) / f_dD_df**2
            normalised_acceleration[n:] = -np.einsum('ij,j', d2D_dx2, dD_dN) / f_dD_df**2
            # normalised_acceleration[:3] = -2 * dX_dx / f_dD_df**2
            # normalised_acceleration[3:] = -2 * np.dot(d2X_dx2, refractive_index) / f_dD_df**2

            return normalised_acceleration
        else:
            raise NotImplementedError()

@unique
class HamiltonianType(Enum):
    COLD_UNMAGNETISED = 0

class HamiltonianModels:
    __slots__ = ("hamiltonian_model_caches", )

    def __init__(self):
        '''

        '''
        self.hamiltonian_model_caches: Dict[HamiltonianType, HamiltonianBase] = {}

