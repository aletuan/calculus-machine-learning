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

from .visualization.cost_plot import (
    plot_cost_3d,
    plot_cost_contour,
    plot_linear_regression_fit
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
    'plot_cost_3d',
    'plot_cost_contour',
    'plot_linear_regression_fit',
    'plot_gradient_descent',
    'plot_gradient_steps'
] 