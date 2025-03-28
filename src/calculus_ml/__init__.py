"""
Calculus Machine Learning Package
"""

from .core.vector import (
    add_vectors,
    dot_product,
    vector_norm,
    create_vectors,
    vector_operations,
    unit_vector_and_angle,
    vector_projection
)

from .core.linear_model import find_linear_regression
from .core.cost_function import compute_cost, generate_house_data
from .core.gradient_descent import gradient_descent, compute_gradient

from .visualization.vector_plot import (
    plot_basic_vectors,
    plot_vector_angle,
    plot_vector_projection
)

from .visualization.cost_plot import (
    plot_cost_surface,
    plot_cost_contour,
    plot_model_errors,
    plot_squared_errors
)

from .visualization.gradient_plot import (
    plot_gradient_descent,
    plot_gradient_steps
)

__version__ = '0.1.0'

__all__ = [
    'add_vectors',
    'dot_product',
    'vector_norm',
    'create_vectors',
    'vector_operations',
    'unit_vector_and_angle',
    'vector_projection',
    'find_linear_regression',
    'compute_cost',
    'generate_house_data',
    'gradient_descent',
    'compute_gradient',
    'plot_basic_vectors',
    'plot_vector_angle',
    'plot_vector_projection',
    'plot_cost_surface',
    'plot_cost_contour',
    'plot_model_errors',
    'plot_squared_errors',
    'plot_gradient_descent',
    'plot_gradient_steps'
] 