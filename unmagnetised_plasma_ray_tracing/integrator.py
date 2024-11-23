#!/usr/bin/python3

# Standard imports
from enum import Enum
import logging
import numpy as np
import queue
from typing import List

# Local imports
from .hamiltonian_model import HamiltonianModel
from .ray import Ray, RayInitialConditions

logger = logging.getLogger(__name__)



class IntegratorOptions:
    __slots__ = ("max_ray_elements", "max_children_per_ray",
        "max_generations_per_ray")

    def __init__(self, max_ray_elements: int=10,
        max_children_per_ray: int=0, max_generations_per_ray: int=0):
        '''
        max_ray_elements : int
            Maximum number of elements for each ray.
        max_children_per_ray : int
            Maximum number of child rays each parent can spawn.
        max_generations_per_ray : int
            Maximum number of generations of children each parent can spawn
            i.e. the parent is generation 0, its children are generation 1,
            the children of the children are generation 2, etc.
        '''
        self.max_ray_elements = int(max_ray_elements)
        self.max_children_per_ray = int(max_children_per_ray)
        self.max_generations_per_ray = int(max_generations_per_ray)

class BranchStopCondition(Enum):
    '''
    Reason why ray tracing terminated for a ray branch.
    '''
    ALL_RAYS_COMPLETE = 0
    EXCEEDED_MAX_CHILDREN = 1
    EXCEEDED_MAX_GENERATIONS = 2

class RayStopCondition(Enum):
    '''
    Reason why ray tracing terminated for a ray.
    '''
    REACHED_MAX_ELEMENTS = 2
    OUT_OF_DOMAIN = 1

class Integrator:
    __slots__ = ("options",)

    def __init__(self, options: IntegratorOptions):
        '''
        
        '''
        self.options = options

    def launch_parent_ray(self, ray: Ray):
        '''
        
        '''
        # Queue holds the parent ray and will hold any children generated
        # by the parent.
        child_ray_queue = queue.Queue()
        child_ray_queue.put(ray)

        # Flag for why tracing of this ray branch was terminated.
        branch_stop_condition = None

        # Trace the parent ray plus a maximum number of children i.e. 1 + max.
        for children_counter in range(1 + self.options.max_children_per_ray):
            if child_ray_queue.qsize() > 0:
                ray = child_ray_queue.get_nowait()

            else:
                branch_stop_condition = BranchStopCondition.ALL_RAYS_COMPLETE
                break
        
        # If no stop condition, check if ray tried spawning too many children.
        if (branch_stop_condition is None
            and children_counter == self.options.max_children_per_ray):
            branch_stop_condition = BranchStopCondition.EXCEEDED_MAX_CHILDREN
        
    def trace(self, ray: Ray) -> List[RayInitialConditions]:
        '''
        Trace a ray.
        '''
        ray.prepare_to_trace(self.options.max_ray_elements)

        # Flag for why the ray trace was terminated.
        ray_stop_condition = None

        # List of additional rays this ray has spawned.
        child_rays = []

        # Start at element 1 as we have the initial condition.
        for element_counter in range(1, self.options.max_ray_elements + 1):
            pass

        # If stop condition not set, check if we exceeded max ray elements.
        if (ray_stop_condition is None
            and element_counter == self.options.max_ray_elements):
            ray_stop_condition = RayStopCondition.REACHED_MAX_ELEMENTS

    def step(self):
        pass
