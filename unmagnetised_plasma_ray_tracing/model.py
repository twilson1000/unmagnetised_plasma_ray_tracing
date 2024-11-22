#!/usr/bin/python3

# Standard imports
import logging
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from .parameter import ParameterCache

logger = logging.getLogger(__name__)

class ModelBase:
    __slots__ = ()

    root = NotImplemented

    def all_caches(self):
        for variable_name in dir(self):
            variable = getattr(self, variable_name)
            if isinstance(variable, ParameterCache):
                yield variable

    def clear(self):
        '''
        Clear all ParameterCaches
        '''
        # Get all classes in the method resolution order (mro). We may inherit
        # so variables will be defined in the __slots__ of super classes.
        for parameter_cache in self.all_caches():
            if parameter_cache.root == self.root:
                parameter_cache.clear()
            