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
    __slots__ = ("name", "description", "unit", "root", "shape", "complex",
        "ndim")

    def __init__(self, name: str, description: str, unit: str, root: str,
        ndim: int, complex: bool=False):
        '''
        name : str
            Name of the parameter.
        description : str
            Description of the parameter.
        unit : str
            Unit the parameter is measured in.
        root : str
            Name of a root group the parameter belongs to.
        shape : Tuple[int]
            Dimensions of the parameter e.g. scalars are (),
            vectors are (n,), matricies are (m, n), etc.
        complex : bool, optional
            If the data is complex valued. Default is False. 
        '''
        self.name = str(name)
        self.description = str(description)
        self.unit = str(unit)
        self.root = str(root)
        self.ndim = int(ndim)
        self.complex = complex

    def __str__(self):
        return self.name
        # return f"{self.name}: {self.description} [{self.unit}]"
    
    def __repr__(self):
        return f"Parameter.{self.name}"
    
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
            return np.complex128
        else:
            return np.float64

    @classmethod
    def scalar(cls, name: str, description: str, unit: str, root: str,
        complex: bool=False):
        return cls(name, description, unit, root, 0, complex=complex)
    
    @classmethod
    def vector(cls, name: str, description: str, unit: str, root: str,
        complex: bool=False):
        return cls(name, description, unit, root, 1, complex=complex)
    
    covector = vector

    @classmethod
    def rank_02_tensor(cls, name: str, description: str, unit: str, root: str,
        complex: bool=False):
        return cls(name, description, unit, root, 2, complex=complex)
    
    rank_11_tensor = rank_02_tensor
    rank_20_tensor = rank_02_tensor

    @classmethod
    def rank_12_tensor(cls, name: str, description: str, unit: str, root: str,
        complex: bool=False):
        return cls(name, description, unit, root, 3, complex=complex)

class ParameterCache:
    ''' Parameter cache which caches values. If the value is missing,
    a user provided function is called to get the value. '''
    __slots__ = ("__weakref__", "parameter", "cached", "shape", "value",
        "cache_miss_callback")

    def __init__(self, parameter: Parameter, dimension: int):
        assert isinstance(parameter, Parameter)

        self.parameter = parameter
        self.cache_miss_callback = None
        
        self.cached = False
        self.shape = self.parameter.value_shape(dimension)
        self.value = np.zeros(self.shape, dtype=self.parameter.dtype)

    @classmethod
    def with_callback(cls, parameter: Parameter, dimension: int, 
        callback: Callable, bound_method: bool=True):
        '''
        Create ParameterCache and add cache miss callback.
        '''
        obj = cls(parameter, dimension)
        obj.register_cache_miss_callback(callback, bound_method=bound_method)
        return obj

    @property
    def dtype(self) -> type:
        return self.parameter.dtype

    @property
    def root(self) -> str:
        return self.parameter.root

    def set(self, value):
        ''' Set value for given coordinate set. '''
        self.value = (np.array(value, dtype=self.parameter.dtype)
            .reshape(self.shape))
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
        bound_method: bool=True, override: bool=False, use_weakref: bool=True):
        '''
        Register a callback function which will return a requested value
        which isn't in the cache.

        Parameters
        ----------
        coordinate_set : CoordinateSet
            The coordinate set the callback is valid for.
        '''
        assert ~(~override and self.cache_miss_callback is None), \
            f"Cache miss callback already registered for {self.parameter.name}"
        
        # Use a weak reference to avoid reference cycles.
        if use_weakref:
            if bound_method:
                self.cache_miss_callback = weakref.WeakMethod(callback)
            else:
                self.cache_miss_callback = weakref.proxy(callback)
        else:
            self.cache_miss_callback = callback
     