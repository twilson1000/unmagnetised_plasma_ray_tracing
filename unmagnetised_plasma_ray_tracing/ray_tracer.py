#!/usr/bin/python3

# Standard imports
import logging
import numpy as np

# Local imports
from .density_model import DensityDataModel, DensityModel
from .hamiltonian_model import HamiltonianModel
from .integrator import Integrator
from .options import RayTracerOptions
from .ray import Ray, RayInitialConditions, Launcher
from .wave_model import WaveModel

logger = logging.getLogger(__name__)

class RayTracer:
    __slots__ = ("density_data_model", "options", "dimension",
        "integrator", )

    def __init__(self, density_data_model: DensityDataModel,
        options: RayTracerOptions, dimension: int=3):
        '''
        
        '''
        self.density_data_model = density_data_model
        self.options = options
        self.dimension = int(dimension)

        # Integrator for tracing rays.
        self.integrator = Integrator(self.options.integrator)

    def trace_launcher(self, launcher: Launcher):
        '''
        
        '''
        for initial_conditions in launcher.get_initial_conditions():
            ray = Ray(initial_conditions, self.dimension)
            self.integrator.launch_parent_ray(ray)
        

