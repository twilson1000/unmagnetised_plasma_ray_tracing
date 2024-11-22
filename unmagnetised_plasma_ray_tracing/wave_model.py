#!/usr/bin/python3

# Standard imports
import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const

# Local imports
from .model import ModelBase
from .parameter import Parameter, ParameterCache

logger = logging.getLogger(__name__)

class WaveModel(ModelBase):
    __slots__ = ("__weakref__", "dimension", "position", "wavevector",
        "refractive_index", "frequency", "time", "vacuum_wavenumber",
        "vacuum_wavelength", )
    
    root = "wave"

    _position = Parameter.vector("position", "Position in space", "m", root)
    _wavevector = Parameter.covector("wavevector", "Wave vector", "m^-1", root)
    _refractive_index = Parameter.covector("refractive_index",
        "Refractive index covector", "", root)
    _refractive_index_magnitude = Parameter.scalar("refractive_index_magnitude",
        "Mangitude of refractive index covector", "", root)
    _frequency = Parameter.scalar("frequency", "Wave frequency", "GHz", root)
    _time = Parameter.scalar("time", "Time", "ns", root)

    _vacuum_wavenumber = Parameter.scalar("vacuum_wavenumber",
        "Vacuum wavenumber k0", "m^-1", root)
    _vacuum_wavelength = Parameter.scalar("vacuum_wavelength",
        "Vacuum wavelength", "m", root)

    def __init__(self, dimension: int) -> None:
        '''
        
        '''
        self.dimension = dimension

        self.position = ParameterCache(self._position, dimension)
        self.wavevector = ParameterCache(self._wavevector, dimension)
        self.refractive_index = ParameterCache(self._refractive_index, dimension)
        self.frequency = ParameterCache(self._frequency, dimension)
        self.time = ParameterCache(self._time, dimension)

        self.vacuum_wavenumber = ParameterCache.with_callback(
            self._vacuum_wavenumber, dimension, self.set_vacuum_wavenumber,
            bound_method=True)
        
        self.vacuum_wavelength = ParameterCache.with_callback(
            self._vacuum_wavelength, dimension, self.set_vacuum_wavelength,
            bound_method=True)
    
    def set_phase_space_position(self, position_m, refractive_index,
        frequency_ghz, time_ns):
        '''
        
        '''
        self.position.set(position_m)
        self.refractive_index.set(refractive_index)
        self.frequency.set(frequency_ghz)
        self.time.set(time_ns)

        k0 = self.vacuum_wavenumber.get()
        self.wavevector.set(k0 * refractive_index)

    def set_vacuum_wavenumber(self):
        '''
        
        '''
        frequency_ghz = self.frequency.get()
        vacuum_wavenumber_per_m = 2e9 * np.pi * frequency_ghz / const.speed_of_light
        self.vacuum_wavenumber.set(vacuum_wavenumber_per_m)

    def set_vacuum_wavelength(self):
        '''
        
        '''
        frequency_ghz = self.frequency.get()
        vacuum_wavelength_m = 1e-9 * const.speed_of_light / frequency_ghz
        self.set_vacuum_wavelength.set(vacuum_wavelength_m)

    