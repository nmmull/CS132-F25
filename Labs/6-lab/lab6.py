import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

def dist(x, y):
    """The distance between two vectors.

    Parameters
    ----------
    x: np.ndarray
    y: np.ndarray

    Returns
    -------
    np.float64
        The l2-distance between `x` and `y`

    """
    return 0.0 # TODO


def linear_design_matrix(ind_vars):
    """The design matrix for fitting a linear function

    Parameters
    ----------
    ind_vars: np.ndarray
        Data in which each row represents the independent variables of
        a single data point.

    Return
    ------
    np.ndarray
        The design matrix for fitting a plane to the data in ind_vars,
        i.e., the matrix [ 1s ind_vars ].

    """
    return np.zeros((ind_vars.shape[0], 1)) # TODO

def quadratic_design_matrix(ind_vars):
    """The design matrix for fitting a quadratic function.

    Parameters
    ----------
    ind_vars: np.ndarray
        Data in which each row represents the independent variables of
        a single data point.

    Return
    ------
    np.ndarray
        The design matrix for fitting a quadratic function to the data
        in ind_vars (see the problem description for more details).

    """
    return np.zeros((ind_vars.shape[0], 1)) # TODO


def cubic_design_matrix(ind_vars):
    """The design matrix for fitting a cubic function.

    Parameters
    ----------
    ind_vars: np.ndarray
        Data in which each row represents the independent variables of
        a single data point.

    Return
    ------
    np.ndarray
        The design matrix for fitting a cubic function to the data in
        ind_vars (see the problem description for more details).

    """
    return np.zeros((ind_vars.shape[0], 1)) # TODO

def fit(design_matrix, observations):
    """Determine a model determined by the a design matrix

    Parameters
    ----------
    design_matrix: np.ndarray
        Design matrix.
    observations: np.ndarray
        Observation matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.float64]

        A tuple consisting of (1) the least squares solution to the
        equation:

        design_matrix @ beta == observations

        (2) the predicted values and (3) the distance between the
        predicted values and the observations.

    """
    parameters = np.linalg.lstsq(design_matrix, observations, rcond=None)[0]
    prediction = design_matrix @ parameters
    return (parameters, prediction, dist(prediction, observations))

housing = fetch_california_housing()
ind_vars = housing.data
dep_vars = housing.target

_, y_hat_line, error_line = fit(linear_design_matrix(ind_vars), dep_vars)
_, y_hat_quad, error_quad = fit(quadratic_design_matrix(ind_vars), dep_vars)
_, y_hat_cube, error_cube = fit(cubic_design_matrix(ind_vars), dep_vars)

print("Dataset Info")
print("------------")
print("    - Number of data points: 20640")
print()
print("    - Independent variables: (8 total)")
print()
for line in housing.DESCR.splitlines()[12:20]:
    print(line)
print()
print("    - Dependent variable: median house value (multiples of $100,000)")
print()
print("    - Note: from 1990 census, each row represents a single district")
print()
print("Prediction Error")
print("----------------")
print(f'   linear func. model:   {error_line}')
print(f'quadratic func. model:   {error_quad}')
print(f'    cubic func. model:   {error_cube}')

fig, axs = plt.subplots(1, 3)
data = [y_hat_line, y_hat_quad, y_hat_cube]
for i in range(3):
    axs[i].scatter(dep_vars, data[i])
    axs[i].set_ylim([-4, 8])
    axs[i].plot([0, 5], [0, 5], color='red')
fig.show()
