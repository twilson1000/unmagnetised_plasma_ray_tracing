#!/usr/bin/python3

# Standard imports
import logging
import numpy as np
import numpy.typing as npt
from typing import Callable, Tuple
import weakref

# Local imports

logger = logging.getLogger(__name__)

class Parameter:
    __slots__ = ("name", "description", "unit", "shape", "complex")

    def __init__(self, name: str, description: str, unit: str,
        complex: bool=False):
        '''
        name : str
            Name of the parameter.
        description : str
            Description of the parameter.
        unit : str
            Unit the parameter is measured in.
        shape : Tuple[int]
            Dimensions of the parameter e.g. scalars are (),
            vectors are (n,), matricies are (m, n), etc.
        complex : bool, optional
            If the data is complex valued. Default is False. 
        '''
        self.name = str(name)
        self.description = str(description)
        self.unit = str(unit)
        self.complex = complex

    def __str__(self):
        return self.name
        # return f"{self.name}: {self.description} [{self.unit}]"
    
    def __repr__(self):
        return f"Parameter.{self.name}"

    def __hash__(self):
        return hash(str(self))
    
    def value_shape(self, dimension: int) -> Tuple[int]:
        '''
        Shape of parameter value.
        
        Parameters
        ----------
        dimension : int
            Dimension of coordinate system.
        '''
        return tuple([dimension for _ in range(self.ndim)])

    @property
    def dtype(self) -> type:
        if self.complex:
            return complex
        else:
            return float

class ParameterCache:
    ''' Parameter cache which caches values. If the value is missing,
    a user provided function is called to get the value. '''
    __slots__ = ("__weakref__", "parameter", "cached", "value",
        "cache_miss_callback")

    def __init__(self, parameter: Parameter, dimension: int):
        self.parameter = parameter
        self.cache_miss_callback = None
        
        self.cached = False
        self.value = np.zeros(self.parameter.value_shape(dimension),
            dtype=self.parameter.dtype)

    def set(self, value):
        ''' Set value for given coordinate set. '''
        self.value = np.array(value, dtype=self.parameter.dtype)
        self.cached = True

    def get(self):
        ''' Get value for given coordinate set. '''
        if not self.cached:
            self.cache_miss()

        return self.value
    
    def cache_miss(self):
        ''' Requested value is not in cache. '''
        assert self.cache_miss_callback is not None, \
            f"No value set and no cache miss callback: {self.parameter.name}."

        if isinstance(self.cache_miss_callback, weakref.WeakMethod):
            # WeakMethods need this extra call to get the bound method.
            self.cache_miss_callback()()
        else:
            self.cache_miss_callback()

        assert self.cached, \
            f"{self.parameter} cache_miss_callback failed to set value"

    def clear(self):
        ''' Clear all cached values. '''
        self.cached = False

    def register_cache_miss_callback(self, callback: Callable[[], npt.NDArray],
        bound_method: bool=True):
        '''
        Register a callback function which will return a requested value
        which isn't in the cache.

        Parameters
        ----------
        coordinate_set : CoordinateSet
            The coordinate set the callback is valid for.
        '''
        assert self.cache_miss_callback is None, \
            f"Cache miss callback already registered for {self.parameter.name}"
        
        # Use a weak reference to avoid reference cycles.
        if bound_method:
            self.cache_miss_callback = weakref.WeakMethod(callback)
        else:
            self.cache_miss_callback = weakref.proxy(callback)
     