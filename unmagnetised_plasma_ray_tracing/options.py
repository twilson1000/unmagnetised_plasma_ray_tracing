#!/usr/bin/python3

# Standard imports
import logging
import numpy as np

# Local imports
from .hamiltonian_model import HamiltonianModelOptions
from .integrator import IntegratorOptions

logger = logging.getLogger(__name__)

class RayTracerOptions:
    __slots__ = ("hamiltonian", "integrator")

    def __init__(self):
        '''
        
        '''
        self.hamiltonian = HamiltonianModelOptions()
        self.integrator = IntegratorOptions()
    