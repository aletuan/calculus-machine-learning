"""
Calculus and Machine Learning Examples
A collection of examples demonstrating vector operations and their applications in machine learning.
"""

__version__ = "0.1.0"

from .vector_operations import (
    create_vectors,
    vector_operations,
    unit_vector_and_angle,
    vector_projection
)
from .vector_visualization import (
    plot_basic_vectors,
    plot_vector_angle,
    plot_vector_projection
)
from .linear_regression import (
    generate_data,
    fit_linear_regression,
    plot_linear_regression
)

__all__ = [
    'create_vectors',
    'vector_operations',
    'unit_vector_and_angle',
    'vector_projection',
    'plot_basic_vectors',
    'plot_vector_angle',
    'plot_vector_projection',
    'generate_data',
    'fit_linear_regression',
    'plot_linear_regression'
] 