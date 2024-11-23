#!/usr/bin/python3

# Standard imports
import logging
import numpy as np

# Local imports
from unmagnetised_plasma_ray_tracing import (Vacuum, RayTracer,
    RayTracerOptions, LauncherGaussianBeam, WaveMode)

logger = logging.getLogger(__name__)

def trace_ray():
    density_data_model = Vacuum()
    ray_tracer_options = RayTracerOptions()
    ray_tracer = RayTracer(density_data_model, ray_tracer_options)
    
    frequency_ghz = 28.0
    launcher = LauncherGaussianBeam("test", np.zeros(3), np.array([1, 0, 0]),
        frequency_ghz, 0.0, WaveMode.O, 1, 0, 1.0, 0.0, 1.0)

    ray_tracer.trace_launcher(launcher)

def main():
    trace_ray()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H-%M-%S"
    )
    main()