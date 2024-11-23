#!/usr/bin/python3

# Standard imports
import abc
from enum import Enum
import logging
import numpy as np
import numpy.typing as npt
from typing import Dict, Tuple, Union
import weakref

# Local imports
from .density_model import DensityDataModel, DensityModel
from .hamiltonian_model import HamiltonianModel
from .parameter import Parameter, ParameterCache
from .wave_model import WaveModel

logger = logging.getLogger(__name__)

class WaveMode(Enum):
    MIXED = 0
    O = 1
    X = 2

class ElectricFieldPolarisation:
    ''' Helper class for electric field polarisation. '''
    __slots__ = ("complex_polarisation", "force_mode")

    def __init__(self, complex_polarisation: npt.NDArray[complex]):
        
        self.force_mode = WaveMode.MIXED

    @classmethod
    def OrdinaryMode(cls):
        '''
        Force polarisation to be ordinary (O) mode.
        '''
        obj = object.__new__(cls)
        obj.force_mode = WaveMode.O
        return obj

    @classmethod
    def ExtraordinaryMode(cls):
        '''
        Force polarisation to be extraordinary (X) mode.
        '''
        obj = object.__new__(cls)
        obj.force_mode = WaveMode.X
        return obj

class RayInitialConditions:
    __slots__ = ("position", "refractive_index", "frequency", "time", "phase",
        "power", "electric_field_strength", "electric_field_polarisation",
        "launcher_name", "ray_family_tree")

    def __init__(self, position_m: npt.NDArray[float],
        refractive_index: npt.NDArray[float], frequency_ghz: float,
        time_ns: float, phase_rads: float, power_W: float,
        electric_field_strength_V_per_m: float,
        electric_field_polarisation: ElectricFieldPolarisation,
        launcher_name: str, ray_family_tree: Tuple[int]):
        '''
        position_m
            Ray position in space [m].
        refractive_index
            Ray refractive index [].
        frequency_ghz : float
            Ray frequency [GHz].
        time_ns : float
            Time relative to reference point [ns].
        phase_rads : float
            Phase relative to reference point [radians].
        power_W : float
            Ray power [W].
        electric_field_strength_V_per_m : float
            Ray electric field strength [V/m]
        electric_field_polarisation
            Ray electric field polarisation.
        launcher_name : str
            Name of launcher ray was launched from.
        ray_family_tree
            Indicies of parent rays.
        '''
        self.position = position_m
        self.refractive_index = refractive_index
        self.frequency = frequency_ghz
        self.time = time_ns
        self.phase = phase_rads
        self.power = power_W
        self.electric_field_strength = electric_field_strength_V_per_m
        self.electric_field_polarisation = electric_field_polarisation

        self.launcher_name = launcher_name
        self.ray_family_tree = tuple(int(i) for i in ray_family_tree)

class LauncherType(Enum):
    GAUSSIAN_BEAM = 0
    PLANE_WAVE = 1

class Launcher(abc.ABC):
    __slots__ = ("name", "origin_m", "launch_direction", "frequency_ghz",
        "time_ns", "electric_field_polarisation", "parent_index")

    launcher_type = NotImplemented

    def __init__(self, name: str, origin_m: npt.NDArray[float],
        launch_direction: npt.NDArray[float], frequency_ghz: float,
        time_ns: float, 
        electric_field_polarisation: Union[npt.NDArray[float], WaveMode]):
        '''
        name
            Name of the launcher.
        origin
            Origin of the launch point of the launcher.
        launch_direction
            Nominal direction the launcher points in. All rays will be launched
            from a plane perpendicular to this vector passing through the
            origin.
        frequency_ghz
            Frequency of wave.
        time_ns
            Time of launch relative to reference time.
        electric_field_polarisation
            Electric field polarisation of wave.
        '''
        self.name = name
        self.origin_m = np.array(origin_m, dtype=float)
        self.launch_direction = np.array(launch_direction, dtype=float)
        self.frequency_ghz = float(frequency_ghz)
        self.time_ns = float(time_ns)
        
        if electric_field_polarisation == WaveMode.O:
            self.electric_field_polarisation = \
                ElectricFieldPolarisation.OrdinaryMode()
        elif self.electric_field_polarisation == WaveMode.X:
            electric_field_polarisation = \
                ElectricFieldPolarisation.ExtraordinaryMode()
        else:
            electric_field_polarisation = ElectricFieldPolarisation(
                self.electric_field_polarisation)

        magnitude = np.linalg.norm(self.launch_direction)
        if magnitude == 0:
            raise ValueError("launch_direction cannot have non-zero "
                f"magnitude: {self.launch_direction}")
        else:
            self.launch_direction /= magnitude

        self.parent_index: int = 0

    @abc.abstractmethod
    def get_initial_conditions(self):
        '''
        
        '''

    @classmethod
    def from_launch_angles(cls):
        '''
        
        '''

    @classmethod
    def plane_wave(cls):
        '''
        
        '''

    @classmethod
    def gaussian_beam(cls):
        '''
        
        '''

class LauncherGaussianBeam(Launcher):
    ''' Gaussian beam launcher. '''
    __slots__ = ("n_rays_radial", "n_rays_azimuthal", "beam_waist_radius_m",
        "beam_focus_m", "total_power_W", "peak_intensity_W_per_m2",
        "cutoff_radii")

    def __init__(self, name: str, origin_m: npt.NDArray[float],
        launch_direction: npt.NDArray[float], frequency_ghz: float,
        time_ns: float, 
        electric_field_polarisation: Union[npt.NDArray[float], WaveMode],
        n_rays_radial: int, n_rays_azimuthal: int, beam_waist_radius_m: float,
        beam_focus_m: float, total_power_W: float, cutoff_radii: float=3.0):
        '''
        Total rays launched is 1 + n_rays_radial * n_rays_azimuthal
        
        n_rays_radial : int
            Number of radial points to spawn rays from. If set to 1 only a
            central ray is used.
        n_rays_azimuthal : int
            Number of azimuthal points used to spawn rays in first radial ring.
            Only used if n_rays_radial > 1.
        beam_waist_radius_m : float
            1/e radius of the beam waist [m]. Radius from the beam centre at
            which the electric field drops to 1/e the peak value.
        beam_focus_m : float
            Distance along the launch_direction to the beam focal point [m].
            If > 0 the beam is divergent, if < 0 the beam is convergent. If
            0, the beam is launched at the focal point.
        total_power_W
            Total power contained in the beam [W].
        cutoff_radii
            Maximum multiple of beam waist radius used for spawning rays. 
        '''
        super().__init__(name, origin_m, launch_direction, frequency_ghz,
            time_ns, electric_field_polarisation)

        self.n_rays_radial = int(n_rays_radial)
        self.n_rays_azimuthal = int(n_rays_azimuthal)
        self.beam_waist_radius_m = float(beam_waist_radius_m)
        self.beam_focus_m = float(beam_focus_m)
        self.total_power_W = float(total_power_W)
        self.peak_intensity_W_per_m2 = (2 * self.total_power_W
            / (np.pi * self.beam_waist_radius_m**2))
        self.cutoff_radii = float(cutoff_radii)

        # Validate user provided values.
        if self.n_rays_radial <= 0:
            raise ValueError(
                f"n_rays_radial must be > 0: {self.n_rays_radial}")

        if self.n_rays_radial > 1 and self.n_rays_azimuthal <= 0:
            raise ValueError(
                f"n_rays_azimuthal must be > 0: {self.n_rays_azimuthal}")

        if self.beam_waist_radius_m <= 0:
            raise ValueError(
                f"beam_waist_radius_m must be > 0: {self.beam_waist_radius_m}")

        if self.total_power_W <= 0:
            raise ValueError(
                f"total_power_W must be > 0: {self.total_power_W}")

        if self.cutoff_radii <= 0:
            raise ValueError(f"cutoff_radii must be > 0: {self.cutoff_radii}")

    def get_initial_conditions(self):
        '''
        
        '''
        electric_field_strength_V_per_m = 0.0

        if self.n_rays_radial > 1:
            # Radii is width normalised to beam waist radius at focal point.
            radii = np.linspace(0, self.cutoff_radii,
                self.n_rays_radial + 2)[1:-1]
            
            # Power through circle of radius R normal to beam direction at
            # focal point is P0 * (1 - exp(-2 * (R / w0)**2)) where w0 is the
            # beam waist.
            power_fraction = 1 - np.exp(-2 * radii**2)

            # Add central ray.
            self.parent_index += 1
            
            # Central ray contains all the power in the central radial bin.
            bin_power_W = power_fraction[0] * self.total_power_W

            yield RayInitialConditions(self.origin_m, self.launch_direction,
                self.frequency_ghz, self.time_ns, 0.0, bin_power_W,
                electric_field_strength_V_per_m,
                self.electric_field_polarisation, self.name,
                (self.parent_index,))
            
            # A bundle of rays.
            for radial_index in range(1, self.n_rays_radial):
                # Divide power in this radial bin equally between all rays.
                power_W = (power_fraction[0] * self.total_power_W
                    / self.n_rays_azimuthal)

                # Keep the radial distance between azimuthal points the same.
                # This requires scaling the number of azimuthal points linearly
                # with radius.
                n_rays_azimuthal = int(round(self.n_rays_azimuthal
                    * (radii[radial_index] / radii[0])))
                
                for azimuthal_index in range(n_rays_azimuthal):
                    self.parent_index += 1

                    # Position and refractive index are distorted based on
                    # how far we are from the focal point.
                    position = None
                    refractive_index = None
                    raise NotImplementedError()

                    # Split power in radial bin equally between rays.
                    yield RayInitialConditions(position,
                        refractive_index, frequency_ghz, time_ns, 0.0,
                        power_W, electric_field_strength_V_per_m,
                        self.electric_field_polarisation, self.name,
                        (self.parent_index,))
        else:
            self.parent_index += 1

            # Only the central ray.
            yield RayInitialConditions(self.origin_m, self.launch_direction,
                self.frequency_ghz, self.time_ns, 0.0, self.total_power_W,
                electric_field_strength_V_per_m,
                self.electric_field_polarisation, self.name,
                (self.parent_index,))

class LauncherPlaneWave(Launcher):
    ''' Plane wave launcher. '''
    __slots__ = ("intensity_W_per_m2", "width_m", "height_m", "n_rays_width",
        "n_rays_height")

    def __init__(self, name: str, origin_m: npt.NDArray[float],
        launch_direction: npt.NDArray[float], frequency_ghz: float,
        time_ns: float, 
        electric_field_polarisation: Union[npt.NDArray[float], WaveMode],
        intensity_W_per_m2: float, width_m: float, height_m: float,
        n_rays_width: int, n_rays_height: int):
        '''
        name
            Name of launcher
        origin
            Origin of launcher
        launch_direction
            Nominal direction the plane wave propagates in (in vacuum). All
            rays will be launched from a plane perpendicular to this vector
            passing through the origin.
        frequency_ghz
            Frequency of wave.
        time_ns
            Time of launch relative to reference time.
        electric_field_polarisation
            Electric field polarisation of wave.
        intensity_W_per_m2 : float
            Intensity of the plane wave [W.m^2].
        width_m : float
            Width of the section of plane rays are launched from.
        height_m : float
            Height of the section of plane rays are launched from.
        n_rays_width : int
            Number of points in width used to launch rays.
        n_rays_height : int
            Number of points in height used to launch rays.
        '''
        super().__init__(name, origin_m, launch_direction)
        self.intensity_W_per_m2 = float(intensity_W_per_m2)
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.n_rays_width = int(n_rays_width)
        self.n_rays_height = int(n_rays_height)

    def get_initial_conditions(self):
        pass

class RayTrajectory:
    __slots__ = ("max_elements", "caches", "history")

    def __init__(self, max_elements: int):
        '''
        
        '''
        self.max_elements = int(max_elements)
        
        # Hold weak references to the cache objects.
        self.caches: Dict[str, ParameterCache] = weakref.WeakValueDictionary()
        self.history: Dict[str, npt.NDArray] = {}

    def add_parameter_cache(self, cache: ParameterCache):
        '''
        
        '''
        self.caches[cache.parameter.name] = cache
        self.history[cache.parameter.name] = np.zeros(
            (self.max_elements, *cache.shape), dtype=cache.dtype)

    def write_step(self, index: int):
        '''
        
        '''
        for key, cache in self.caches.items():
            self.history[key][index] = cache.get()

'''
input/
  model_data/
  options/

ray_tracing/
    launcher_1/
        number_parent_rays = 2
        number_rays = 6
        origin = [0, 0, 0]
        type = GAUSSIAN_BEAM
        
        ray_1/
            total_rays = 3
            total_generations = 2
            branch_stop_condition = ALL_RAYS_COMPLETE
            stop_condition
        ray_1-1/
            stop_condition
        ray_1-2/
            stop_condition
        ray_1-3/
            stop_condition
        
        ray_2/
            group_stop_condition
        ray_2-1/
   
linear_current_drive/

'''

class Ray:
    '''
    Holds all data required to trace the ray, the trajectory of the ray and
    metadata about the ray propagation.
    '''
    __slots__ = ("density_model", "hamiltonian_model", "wave_model",
        "caches", "history", "family_tree", "name", "launcher_name")

    def __init__(self, initial_conditions: RayInitialConditions,
        density_data_model: DensityDataModel, dimension: int):
        '''
        
        '''
        # Models.
        self.wave_model = WaveModel(dimension)
        self.density_model = DensityModel(density_data_model, self.wave_model,
            dimension)
        self.hamiltonian_model = HamiltonianModel(dimension,
            self.density_model, self.wave_model)

        # Trajectory data. Hold weak references to the cache objects.
        self.caches: Dict[str, ParameterCache] = weakref.WeakValueDictionary()
        self.history: Dict[str, npt.NDArray] = {}

        # Set initial conditions.
        ic = initial_conditions
        self.wave_model.set_phase_space_position(ic.position,
            ic.refractive_index, ic.frequency, ic.time)
        self.wave_model.phase.set(ic.phase)
        self.wave_model.power.set(ic.power)
        self.wave_model.electric_field_polarisation.set(
            ic.electric_field_polarisation)
        self.wave_model.electric_field_strength.set(ic.electric_field_strength)

        # Configure ray metadata.
        self.family_tree = ic.ray_family_tree
        self.name = "ray_" + '-'.join(str(i) for i in self.family_tree)
        self.launcher_name = ic.launcher_name

    def prepare_to_trace(self, max_ray_elements: int):
        '''
        Configure ray for tracing by the integrator.
        '''
        # Trajectory saves data along the ray.
        self.trajectory = RayTrajectory(max_ray_elements)

        for cache in (
            self.wave_model.position,
            self.wave_model.refractive_index,
            self.wave_model.frequency,
            self.wave_model.time,
            self.wave_model.phase,
            self.wave_model.power,
            self.wave_model.electric_field_polarisation,
            # self.wave_model.wave_action,
            self.wave_model.electric_field_strength,
            
            self.density_model.density,
            self.density_model.normalised_density,

            self.hamiltonian_model.hamiltonian,
            self.hamiltonian_model.normalised_ray_velocity,
            self.hamiltonian_model.normalised_ray_acceleration,
        ):
            self.trajectory.add_parameter_cache(cache)

        # Write initial conditions to trajectory.
        self.trajectory.write_step(0)

    def child(self, child_index: int, power_W: float, position_m=None,
        refractive_index=None, phase_rads=None,
        electric_field_polarisation=None):
        '''
        Return a child ray with the state of the current ray if not otherwise
        set. This should only be called after a call to trajectory.write_step.
        '''
        if position_m is None:
            position_m = self.wave_model.position.get()
        if refractive_index is None:
            refractive_index = self.wave_model.refractive_index.get()
        if phase_rads is None:
            phase_rads = self.wave_model.phase.get()
        
        frequency_ghz = self.wave_model.frequency.get()
        time_ns = self.wave_model.time.get()
        electric_field_strength_V_per_m = \
            self.wave_model.electric_field_strength.get()
        electric_field_polarisation = self

        family_tree = (*self.family_tree, child_index)
        initial_conditions = RayInitialConditions(position_m, refractive_index,
            frequency_ghz, time_ns, phase_rads, power_W,
            electric_field_strength_V_per_m, electric_field_polarisation,
            self.launcher_name, family_tree)

