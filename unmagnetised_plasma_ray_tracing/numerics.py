#!/usr/bin/python3

# Standard imports
import logging
# Local imports

logger = logging.getLogger(__name__)

finite_difference_1st_derivative_2nd_order_stencil = ((-1, -1/2), (1, 1/2))
finite_difference_2nd_derivative_2nd_order_stencil = ((-1, 1.0), (0, -2.0), (1, 1.0))
finite_difference_mixed_2nd_derivative_2nd_order_stencil = ((1, 1, 0.25), (1, -1, -0.25), (-1, 1, -0.25), (-1, -1, 0.25))
