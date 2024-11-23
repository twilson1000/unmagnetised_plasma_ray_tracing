#!/usr/bin/python3

# Standard imports
import abc
from enum import Enum, unique
import logging
import numpy as np
from typing import Dict

# Local imports
from .density_model import DensityModel
from .model import ModelBase
from .parameter import Parameter, ParameterCache
from .wave_model import WaveModel

logger = logging.getLogger(__name__)

class HamiltonianBase(abc.ABC, ModelBase):
    __slots__ = ("hamiltonian", "dispersion_tensor",
        "hamiltonian_first_derivative_x", "hamiltonian_first_derivative_N",
        "hamiltonian_normalised_first_derivative_f",
        "hamiltonian_first_derivative_z", "hamiltonian_second_derivative_x",
        "hamiltonian_second_derivative_xN",
        "hamiltonian_normalised_second_derivative_xf",
        "hamiltonian_normalised_second_derivative_Nf",
        "hamiltonian_second_derivative_z", "normalised_ray_velocity",
        "normalised_ray_acceleration")
    
    root = "hamiltonian"

    _hamiltonian = Parameter.scalar("hamiltonian", "Ray tracing hamiltonian",
        "", root, complex=True)
    _dispersion_tensor = Parameter.rank_11_tensor("dispersion_tensor",
        "Dispersion tensor for electromagnetic wave", "", root, complex=True)
    
    _hamiltonian_first_derivative_x = Parameter.covector(
        "hamiltonian_first_derivative_x", ("First derivative of real "
        "part of hamiltonian with respect to space."), "m^-1", root)
    _hamiltonian_first_derivative_N = Parameter.covector(
        "hamiltonian_first_derivative_N", ("First derivative of real "
        "part of hamiltonian with respect to refractive index."), "m", root)
    _hamiltonian_normalised_first_derivative_f = Parameter.scalar(
        "hamiltonian_first_derivative_f", ("First derivative of real "
        "part of hamiltonian with respect to frequency."), "ns", root)
    _hamiltonian_first_derivative_z = Parameter(
        "hamiltonian_first_derivative_z", ("First derivative of real "
        "part of hamiltonian with respect to phase space position (x, N).")
        , "m", root, 1)

    _hamiltonian_second_derivative_x = Parameter.rank_02_tensor(
        "hamiltonian_second_derivative_x", ("Second derivative of real "
        "part of hamiltonian with respect to space."), "m^-2", root)
    _hamiltonian_second_derivative_N = Parameter.rank_20_tensor(
        "hamiltonian_second_derivative_N", ("Second derivative of real "
        "part of hamiltonian with respect to refractive index."), "m^2", root)
    _hamiltonian_second_derivative_xN = Parameter.rank_11_tensor(
        "hamiltonian_second_derivative_xN", ("Second mixed derivative of real "
        "part of hamiltonian with respect to position and refractive index."),
        "", root)
    _hamiltonian_normalised_second_derivative_xf = Parameter.covector(
        "hamiltonian_normalised_second_derivative_xf", ("Second mixed "
        "derivative of hamiltonian with respect to position and frequency."),
        "ns.m^-1", root)
    _hamiltonian_normalised_second_derivative_Nf = Parameter.vector(
        "hamiltonian_normalised_second_derivative_Nf", ("Second mixed "
        "derivative of hamiltonian with respect to refractive index and "
        "frequency."), "ns.m", root)
    _hamiltonian_second_derivative_z = Parameter(
        "hamiltonian_second_derivative_z", ("Second derivative of real "
        "part of hamiltonian with respect to phase space position (x, N).")
        , "m", root, 2)
    
    _normalised_ray_velocity = Parameter.vector("normalised_ray_velocity",
        ("First derivative of ray position in phase space (x, N) with respect "
        "to time normalised to the speed of light."), "", root)
    _normalised_ray_acceleration = Parameter.vector("normalised_ray_velocity",
        ("Second derivative of ray position in phase space (x, N) with respect "
        "to time normalised to the speed of light squared."), "", root)
    

    def __init__(self, dimension: int) -> None:
        self.hamiltonian = ParameterCache.with_callback(self._hamiltonian,
            dimension, self.set_hamiltonian, bound_method=True)
        
        self.dispersion_tensor = ParameterCache.with_callback(
            self._dispersion_tensor, dimension, self.set_dispersion_tensor,
            bound_method=True)
        
        self.hamiltonian_first_derivative_x = ParameterCache.with_callback(
            self._hamiltonian_first_derivative_x, dimension,
            self.set_hamiltonian_first_derivative_x, bound_method=True)
        
        self.hamiltonian_first_derivative_N = ParameterCache.with_callback(
            self._hamiltonian_first_derivative_N, dimension,
            self.set_hamiltonian_first_derivative_N, bound_method=True)

        self.hamiltonian_normalised_first_derivative_f = ParameterCache.with_callback(
            self._hamiltonian_normalised_first_derivative_f, dimension,
            self.set_hamiltonian_normalised_first_derivative_f,
            bound_method=True)
        
        self.hamiltonian_first_derivative_z = ParameterCache.with_callback(
            self._hamiltonian_first_derivative_z, 2 * dimension,
            self.set_hamiltonian_first_derivative_z, bound_method=True)
        
        self.hamiltonian_normalised_second_derivative_xf = ParameterCache.with_callback(
            self._hamiltonian_normalised_second_derivative_xf, dimension,
            self.set_hamiltonian_normalised_second_derivative_xf,
            bound_method=True)
        
        self.hamiltonian_normalised_second_derivative_Nf = ParameterCache.with_callback(
            self._hamiltonian_normalised_second_derivative_Nf, dimension,
            self.set_hamiltonian_normalised_second_derivative_Nf,
            bound_method=True)

        self.hamiltonian_second_derivative_x = ParameterCache.with_callback(
            self._hamiltonian_second_derivative_x, dimension,
            self.set_hamiltonian_second_derivative_x, bound_method=True)

        self.hamiltonian_second_derivative_N = ParameterCache.with_callback(
            self._hamiltonian_second_derivative_N, dimension,
            self.set_hamiltonian_second_derivative_N, bound_method=True)

        self.hamiltonian_second_derivative_xN = ParameterCache.with_callback(
            self._hamiltonian_second_derivative_xN, dimension,
            self.set_hamiltonian_second_derivative_xN, bound_method=True)
        
        self.hamiltonian_second_derivative_z = ParameterCache.with_callback(
            self._hamiltonian_second_derivative_z, 2 * dimension,
            self.set_hamiltonian_second_derivative_z, bound_method=True)
        
        self.normalised_ray_velocity = ParameterCache.with_callback(
            self._normalised_ray_velocity, 2 * dimension,
            self.set_normalised_ray_velocity, bound_method=True)
        
        self.normalised_ray_acceleration = ParameterCache.with_callback(
            self._normalised_ray_acceleration, 2 * dimension,
            self.set_normalised_ray_acceleration, bound_method=True)
        
    @abc.abstractmethod
    def set_hamiltonian(self):
        '''
        Set ray tracing Hamiltonian.
        '''

    @abc.abstractmethod
    def set_dispersion_tensor(self):
        '''
        Set dispersion tensor for electromagnetic wave.
        '''
    
    
    @abc.abstractmethod
    def set_hamiltonian_first_derivative_x(self):
        '''
        Set first derivative of real part of Hamiltonian with respect to
        position.
        '''
    
    @abc.abstractmethod
    def set_hamiltonian_first_derivative_N(self):
        '''
        Set first derivative of real part of Hamiltonian with respect to
        refractive index.
        '''
    
    @abc.abstractmethod
    def set_hamiltonian_normalised_first_derivative_f(self):
        '''
        Set first derivative of real part of Hamiltonian with respect to
        frequency multiplied by frequency i.e. f * dH/df.
        '''
    
    def set_hamiltonian_first_derivative_z(self):
        '''
        Set first derivative of real part of Hamiltonian with respect to
        phase space position (x, N).
        '''
        dH_dx = self.hamiltonian_first_derivative_x.get()
        dH_dN = self.hamiltonian_first_derivative_N.get()
        
        dH_dz = np.empty(self.hamiltonian_first_derivative_z.shape)
        n = dH_dz.shape[0] // 2
        dH_dz[:n] = dH_dx
        dH_dz[n:] = dH_dN

        self.hamiltonian_first_derivative_z.set(dH_dz)

    @abc.abstractmethod
    def set_hamiltonian_second_derivative_x(self):
        '''
        Set second derivative of real part of Hamiltonian with respect to
        position.
        '''
    
    @abc.abstractmethod
    def set_hamiltonian_second_derivative_N(self):
        '''
        Set second derivative of real part of Hamiltonian with respect to
        refractive index.
        '''
    
    @abc.abstractmethod
    def set_hamiltonian_second_derivative_xN(self):
        '''
        Set second mixed derivative of real part of Hamiltonian with respect
        to position and refractive index i.e. d2H_dx_dN[i, j] = d^2H / dx^i dN_j
        '''

    @abc.abstractmethod
    def set_hamiltonian_normalised_second_derivative_xf(self):
        '''
        Set normalised second mixed derivative of real part of Hamiltonian
        with respect to frequency and position i.e. f * (d^2 H / dx df).
        '''
    
    @abc.abstractmethod
    def set_hamiltonian_normalised_second_derivative_Nf(self):
        '''
        Set normalised second mixed derivative of real part of Hamiltonian
        with respect to frequency and refractive index i.e. f * (d^2 H / dN df).
        '''

    def set_hamiltonian_second_derivative_z(self):
        '''
        Set second derivative of real part of Hamiltonian with respect to
        phase space position z = (x, N) i.e. d2H_dz2[i, j] = d^2 H / dz^i dz^j.
        '''
        d2H_dx2 = self.hamiltonian_second_derivative_x.get()
        d2H_dx_dN = self.hamiltonian_second_derivative_xN.get()
        d2H_dN2 = self.hamiltonian_second_derivative_N.get()
        
        d2H_dz2 = np.empty(self.hamiltonian_second_derivative_z.shape)
        n = d2H_dz2.shape[0] // 2
        d2H_dz2[:n, :n] = d2H_dx2
        d2H_dz2[:n, n:] = d2H_dx_dN
        d2H_dz2[n:, :n] = d2H_dx_dN.T
        d2H_dz2[n:, n:] = d2H_dN2

        self.hamiltonian_second_derivative_z.set(d2H_dz2)

    @abc.abstractmethod
    def set_normalised_ray_velocity(self):
        '''
        Set velocity of ray on phase space (x, N) normalised to the speed
        of light.
        '''

    @abc.abstractmethod
    def set_normalised_ray_acceleration(self):
        '''
        Set acceleration of ray on phase space (x, N) normalised to the speed
        of light squared.
        '''

class UnmagnetisedPlasmaHamiltonianOptions:
    __slots__ = ("reduced_dispersion",)

    def __init__(self, reduced_dispersion: bool=True) -> None:
        self.reduced_dispersion = reduced_dispersion

class UnmagnetisedPlasmaHamiltonian(HamiltonianBase):
    __slots__ = ("__weakref__", "options", "density_model", "wave_model",
        "refractive_index_squared", "hamiltonian_first_derivative_X",
        "hamiltonian_first_derivative_N2", "hamiltonian_first_derivative_x",
        "hamiltonian_first_derivative_N",
        "hamiltonian_normalised_first_derivative_f",
        "hamiltonian_second_derivative_X", "hamiltonian_second_derivative_N2",
        "hamiltonian_second_derivative_X_N2", "hamiltonian_second_derivative_x",
        "hamiltonian_second_derivative_N", "hamiltonian_second_derivative_xN",
        "hamiltonian_normalised_second_derivative_xf",
        "hamiltonian_normalised_second_derivative_Nf",
        )
    
    root = HamiltonianBase.root #+ "-cold_unmagnetised"
    
    _refractive_index_squared = Parameter.scalar("refractive_index_squared",
        "Square of magnitude of refractive index covector", "", root)
    
    _hamiltonian_first_derivative_X = Parameter.scalar(
        "hamiltonian_first_derivative_X", ("First derivative of real part "
        "of hamiltonian with respect to normalised density X."), "", root)
    _hamiltonian_first_derivative_N2 = Parameter.scalar(
        "hamiltonian_first_derivative_N2", ("First derivative of real part "
        "of hamiltonian with respect to squared refractive index."), "", root)
    
    _hamiltonian_second_derivative_X = Parameter.scalar(
        "hamiltonian_second_derivative_X", ("Second derivative of real part "
        "of hamiltonian with respect to normalised density X."), "", root)
    _hamiltonian_second_derivative_N2 = Parameter.scalar(
        "hamiltonian_second_derivative_N2", ("Second derivative of real part "
        "of hamiltonian with respect to squared refractive index."), "", root)
    _hamiltonian_second_derivative_X_N2 = Parameter.scalar(
        "hamiltonian_second_derivative_X_N2", ("Second mixed derivative of "
        "real part of hamiltonian with respect to normalised density and "
        "squared refractive index."), "", root)
    
    def __init__(self, density_model: DensityModel, wave_model: WaveModel,
        dimension: int, options: UnmagnetisedPlasmaHamiltonianOptions):
        '''
        
        '''
        assert isinstance(density_model, DensityModel)
        assert isinstance(wave_model, WaveModel)
        assert isinstance(options, UnmagnetisedPlasmaHamiltonianOptions)

        # Other data models we draw data from.
        self.density_model = density_model
        self.wave_model = wave_model
        self.options = options

        # Initialised base class to get default parameter caches.
        super().__init__(dimension)

        # Parameter caches.
        self.refractive_index_squared = ParameterCache.with_callback(
            self._refractive_index_squared, dimension,
            self.set_refractive_index_squared, bound_method=True)
        
        self.hamiltonian_first_derivative_X = ParameterCache.with_callback(
            self._hamiltonian_first_derivative_X, dimension,
            self.set_hamiltonian_first_derivative_X, bound_method=True)

        self.hamiltonian_first_derivative_N2 = ParameterCache.with_callback(
            self._hamiltonian_first_derivative_N2, dimension,
            self.set_hamiltonian_first_derivative_N2, bound_method=True)

        self.hamiltonian_second_derivative_X = ParameterCache.with_callback(
            self._hamiltonian_second_derivative_X, dimension,
            self.set_hamiltonian_second_derivative_X, bound_method=True)

        self.hamiltonian_second_derivative_N2 = ParameterCache.with_callback(
            self._hamiltonian_second_derivative_N2, dimension,
            self.set_hamiltonian_second_derivative_N2, bound_method=True)

        self.hamiltonian_second_derivative_X_N2 = ParameterCache.with_callback(
            self._hamiltonian_second_derivative_X_N2, dimension,
            self.set_hamiltonian_second_derivative_X_N2, bound_method=True)

    def set_refractive_index_squared(self):
        '''
        Set squared magnitude of refractive index covector.
        '''
        N = self.wave_model.refractive_index.get()
        N2 = sum(N**2)
        self.refractive_index_squared.set(N2)

    def hamiltonian_function(self, X, N2):
        if self.options.reduced_dispersion:
            return 1 - X - N2
        else:
            return (1 - X) * (1 - X - N2)**2

    def set_hamiltonian(self):
        '''
        Set complex valued ray tracing Hamiltonian.
        '''
        X = self.density_model.normalised_density.get()
        N2 = self.refractive_index_squared.get()
        
        hamiltonian = self.hamiltonian_function(X, N2)
        
        self.hamiltonian.set(hamiltonian)

    def set_dispersion_tensor(self):
        '''
        Set dispersion tensor for electromagnetic wave.
        '''
        X = self.density_model.normalised_density.get()
        N2 = self.refractive_index_squared.get()

        dispersion_tensor = np.zeros(self.dispersion_tensor.shape,
            dtype=self.dispersion_tensor.dtype)
        dispersion_tensor[0, 0] = 1 - X - N2
        dispersion_tensor[1, 1] = dispersion_tensor[0, 0]
        dispersion_tensor[2, 2] = 1 - X
        
        self.dispersion_tensor.set(dispersion_tensor)

    def set_hamiltonian_first_derivative_X(self):
        '''
        Set first derivative of real part of Hamiltonian with respect to
        normalised density X.
        '''
        X = self.density_model.normalised_density.get()
        N2 = self.refractive_index_squared.get()

        if self.options.reduced_dispersion:
            dH_dX = -1
        else:
            dH_dX = -(1 - X - N2) * (3 - 3*X - N2)
        
        self.hamiltonian_first_derivative_X.set(dH_dX)

    def set_hamiltonian_first_derivative_N2(self):
        '''
        Set first derivative of real part of Hamiltonian with respect to
        squared refractive index.
        '''
        X = self.density_model.normalised_density.get()
        N2 = self.refractive_index_squared.get()

        if self.options.reduced_dispersion:
            dH_dN2 = -1
        else:
            dH_dN2 = -2 * (1 - X) * (1 - X - N2)
        
        self.hamiltonian_first_derivative_N2.set(dH_dN2)

    def set_hamiltonian_first_derivative_x(self):
        '''
        Set first derivative of real part of Hamiltonian with respect to
        space.
        '''
        dH_dX = self.hamiltonian_first_derivative_X.get()
        dX_dx = self.density_model.normalised_density_first_derivative.get()

        if self.options.reduced_dispersion:
            dH_dx = dH_dX * dX_dx
        else:
            raise NotImplementedError()

        self.hamiltonian_first_derivative_x.set(dH_dx)

    def set_hamiltonian_first_derivative_N(self):
        '''
        Set first derivative of real part of Hamiltonian with respect to
        refractive index.
        '''
        N = self.wave_model.refractive_index.get()
        dD_dN2 = self.hamiltonian_first_derivative_N2.get()

        logger.info((N, dD_dN2))

        if self.options.reduced_dispersion:
            dH_dN = 2 * N * dD_dN2
        else:
            raise NotImplementedError()
        
        self.hamiltonian_first_derivative_N.set(dH_dN)
    
    def set_hamiltonian_normalised_first_derivative_f(self):
        '''
        Set first derivative of real part of Hamiltonian with respect to
        frequency multiplied by frequency i.e. f * dH/df.
        '''
        X = self.density_model.normalised_density.get()
        N2 = self.refractive_index_squared.get()

        if self.options.reduced_dispersion:
            dH_dX = self.hamiltonian_first_derivative_X.get()
            dH_dN2 = self.hamiltonian_first_derivative_N2.get()

            f_dH_df = 2 * (X + N2)
            # f_dH_df = -2 * (X * dH_dX + N2 * dH_dN2)
        else:
            f_dH_df = 2 * (1 - X - N2) * (3 * X * (1 - X - N2) + 2 * N2)
        
        self.hamiltonian_normalised_first_derivative_f.set(f_dH_df)

    def set_hamiltonian_second_derivative_X(self):
        '''
        Set second derivative of real part of Hamiltonian with respect to
        normalised density X.
        '''
        if self.options.reduced_dispersion:
            d2H_dX2 = 0.0
        else:
            raise NotImplementedError()
        
        self.hamiltonian_second_derivative_X.set(d2H_dX2)

    def set_hamiltonian_second_derivative_N2(self):
        '''
        Set second derivative of real part of Hamiltonian with respect to
        squared refractive index.
        '''
        if self.options.reduced_dispersion:
            d2H_dN22 = 0.0
        else:
            raise NotImplementedError()
        
        self.hamiltonian_second_derivative_N2.set(d2H_dN22)

    def set_hamiltonian_second_derivative_X_N2(self):
        '''
        Set second mixed derivative of real part of Hamiltonian with respect
        to normalised density X and squared refractive index.
        '''
        if self.options.reduced_dispersion:
            d2H_dX_dN2 = 0.0
        else:
            raise NotImplementedError()
        
        self.hamiltonian_second_derivative_X_N2.set(d2H_dX_dN2)

    def set_hamiltonian_second_derivative_x(self):
        '''
        Set second derivative of real part of Hamiltonian with respect to
        position.
        '''
        if self.options.reduced_dispersion:
            d2X_dx2 = self.density_model.normalised_density_second_derivative.get()
            d2H_dx2 = -d2X_dx2
        else:
            raise NotImplementedError()
        
        self.hamiltonian_second_derivative_x.set(d2H_dx2)

    def set_hamiltonian_second_derivative_N(self):
        '''
        Set second derivative of real part of Hamiltonian with respect to
        refractive index.
        '''
        if self.options.reduced_dispersion:
            d2H_dN2 = -2 * np.identity(self.wave_model.refractive_index.shape[0])
        else:
            raise NotImplementedError()
        
        self.hamiltonian_second_derivative_N.set(d2H_dN2)

    def set_hamiltonian_second_derivative_xN(self):
        '''
        Set second mixed derivative of real part of Hamiltonian with respect
        to position and refractive index.
        '''
        if self.options.reduced_dispersion:
            d2H_dx_dN = np.zeros(self.hamiltonian_second_derivative_xN.shape)
        else:
            raise NotImplementedError()
        
        self.hamiltonian_second_derivative_xN.set(d2H_dx_dN)

    def set_hamiltonian_normalised_second_derivative_xf(self):
        '''
        Set normalised second mixed derivative of real part of Hamiltonian
        with respect to frequency and position i.e. f * (d^2 H / dx df).
        '''
        if self.options.reduced_dispersion:
            dX_dx = self.density_model.normalised_density_first_derivative.get()
            d2H_dx_df = 2 * dX_dx
        else:
            raise NotImplementedError()
        
        self.hamiltonian_normalised_second_derivative_xf.set(d2H_dx_df)

    def set_hamiltonian_normalised_second_derivative_Nf(self):
        '''
        Set normalised second mixed derivative of real part of Hamiltonian
        with respect to frequency and refractive index i.e. f * (d^2 H / dN df).
        '''
        if self.options.reduced_dispersion:
            N = self.wave_model.refractive_index.get()
            d2H_dN_df = 4 * N
        else:
            raise NotImplementedError()

        self.hamiltonian_normalised_second_derivative_Nf.set(d2H_dN_df)

    def set_normalised_ray_velocity(self):
        '''
        Set velocity of ray in phase space (x, N) normalised to the speed
        of light.
        '''
        if self.options.reduced_dispersion:
            dD_dx = self.hamiltonian_first_derivative_x.get()
            dD_dN = self.hamiltonian_first_derivative_N.get()
            f_dD_df = self.hamiltonian_normalised_first_derivative_f.get()

            normalised_ray_velocity = np.zeros(self.normalised_ray_velocity.shape)
            n = len(normalised_ray_velocity) // 2

            logger.warning((dD_dx, dD_dN, f_dD_df))

            normalised_ray_velocity[:n] = -dD_dN / f_dD_df
            normalised_ray_velocity[n:] = dD_dx / f_dD_df
        else:
            raise NotImplementedError()
        
        self.normalised_ray_velocity.set(normalised_ray_velocity)

    def set_normalised_ray_acceleration(self):
        '''
        Set acceleration of ray in phase space (x, N) normalised to the
        speed of light squared.
        '''
        if self.options.reduced_dispersion:
            dD_dx = self.hamiltonian_first_derivative_x.get()
            dD_dN = self.hamiltonian_first_derivative_N.get()
            f_dD_df = self.hamiltonian_normalised_first_derivative_f.get()
            d2D_dx2 = self.hamiltonian_second_derivative_x.get()
            d2D_dN2 = self.hamiltonian_second_derivative_N.get()

            normalised_ray_acceleration = np.zeros(self.normalised_ray_acceleration.shape)
            n = len(normalised_ray_acceleration) // 2
            normalised_ray_acceleration[:n] = (
                -np.einsum('ij,j', d2D_dN2, dD_dx) / f_dD_df**2)
            normalised_ray_acceleration[n:] = (
                -np.einsum('ij,j', d2D_dx2, dD_dN) / f_dD_df**2)
            # normalised_acceleration[:3] = -2 * dX_dx / f_dD_df**2
            # normalised_acceleration[3:] = -2 * np.dot(d2X_dx2, refractive_index) / f_dD_df**2
        else:
            raise NotImplementedError()
        
        self.normalised_ray_acceleration.set(normalised_ray_acceleration)

@unique
class HamiltonianType(Enum):
    COLD_UNMAGNETISED = 0

class HamiltonianModelOptions:
    __slots__ = ("cold_unmagnetised",)

    def __init__(self) -> None:
        self.cold_unmagnetised = UnmagnetisedPlasmaHamiltonianOptions()

class HamiltonianModel:
    __slots__ = ("density_model", "wave_model", "dimension", "options", "hamiltonian_type",
        "hamiltonian_model_caches")

    def __init__(self, density_model: DensityModel, wave_model: WaveModel,
        dimension: int, options: HamiltonianModelOptions):
        '''

        '''
        self.density_model = density_model
        self.wave_model = wave_model
        self.dimension = dimension
        self.options = options

        self.hamiltonian_type = HamiltonianType.COLD_UNMAGNETISED
        self.hamiltonian_model_caches: Dict[HamiltonianType, HamiltonianBase] = {}

    def __getitem__(self, hamiltonian_type: HamiltonianType):
        if hamiltonian_type not in self.hamiltonian_model_caches:
            if hamiltonian_type == HamiltonianType.COLD_UNMAGNETISED:
                model = UnmagnetisedPlasmaHamiltonian(self.density_model,
                    self.wave_model, self.dimension,
                    self.options.cold_unmagnetised)
                self.hamiltonian_model_caches[hamiltonian_type] = model
            else:
                raise NotImplementedError(hamiltonian_type)

        return self.hamiltonian_model_caches[hamiltonian_type]

    def clear(self):
        '''
        Clear all ParameterCaches
        '''
        for model in self.hamiltonian_model_caches.values():
            model.clear()

    @property
    def hamiltonian(self):
        '''
        Set ray tracing Hamiltonian.
        '''
        return self[self.hamiltonian_type].hamiltonian

    @property
    def dispersion_tensor(self):
        '''
        Set dispersion tensor for electromagnetic wave.
        '''
        return self[self.hamiltonian_type].dispersion_tensor
    
    @property
    def normalised_ray_velocity(self):
        '''
        Set velocity of ray on phase space (x, N) normalised to the speed
        of light.
        '''
        return self[self.hamiltonian_type].normalised_ray_velocity

    @property
    def normalised_ray_acceleration(self):
        '''
        Set acceleration of ray on phase space (x, N) normalised to the speed
        of light squared.
        '''
        return self[self.hamiltonian_type].normalised_ray_acceleration
