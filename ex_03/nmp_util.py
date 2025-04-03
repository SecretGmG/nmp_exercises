from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.sparse
import seaborn as sns
sns.set_style()
import sympy
import scipy
from functools import wraps
from typing import Iterable, Literal, Tuple, List
#improve readability of the output of numpy arrays in jupyter notebooks
np.set_printoptions(linewidth=150)


# File i will keep continuously updating during the course, adding usefull functions

## SERIE 1
def get_inliers(data : np.ndarray, f : float= 4.0, iterative : bool = False, robust = True):
    """
    Identifies inliers in a dataset based on the standard deviation.

    Args:
        data (np.ndarray): Input data array.
        f (float): Maximum number of standard deviations allowed for inliers. Default is 4.0.
        iterative (bool): If True, iteratively refines inliers. Default is False.
        robust (bool) : If true use the median and the value at the 68'th percentile of the absolute deviations instead of mean and std. Default is True.

    Returns:
        np.ndarray: Boolean array indicating inliers (True) and outliers (False).
    """
    # start with the entire data as inliers
    inliers = np.ones_like(data, dtype=bool)
    
    while True:
        if robust:
            mean = np.median(data[inliers])
            deviations = np.abs(data[inliers] - mean)
            std = np.percentile(deviations, 68.3)
        else:
            std = data[inliers].std()
            mean = data[inliers].mean()
        
        new_inliers = np.abs(data-mean) < f*std
        
        # stop iterating if specified or if no inliers removed
        if (not iterative) or np.all(new_inliers == inliers):
            inliers = new_inliers
            break
        
        inliers = new_inliers
        
    return inliers

## SERIE 2

def error_propagation_formula(f : sympy.Matrix|Iterable[sympy.Expr], args : List[sympy.Symbol]) -> Tuple[sympy.Expr, sympy.MatrixSymbol]:
    """
    Computes the error propagation formula A * K * A.T for a given expression matrix (vector).

    Args:
        f: Vector-valued expression matrix or an iterable of expressions.
        args: List of symbols for the Jacobian.

    Returns:
        Tuple containing:
            - The formula A * K * A.T.
            - The symbolic covariance matrix K.
    """
    
    if not isinstance(f, sympy.Matrix):
        #assume that f is an iterable of expressions if it is not already a matrix
        #in this case it should be possible to simply convert into a column matrix
        f = sympy.Matrix(list(f))
    
    A = f.jacobian(args)
    K = sympy.MatrixSymbol('K',len(args), len(args))
    return A * K * A.T , K #skript S.12


@wraps(pd.read_csv)
def read_dat(*args, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file with default parameters: index_col=False, sep='\\s+'.

    Args:
        *args: Positional arguments passed to pd.read_csv.
        **kwargs: Keyword arguments passed to pd.read_csv.

    Returns:
        pd.DataFrame: The DataFrame read from the CSV file.
    """
    READ_DATA_KWARGS = {'index_col' : False, 'sep' : r'\s+', 'skiprows' : [0,1]}
    
    assert 'names' in kwargs, "Please provide the column names using the 'names' keyword argument."
    return pd.read_csv(*args, **READ_DATA_KWARGS, **kwargs)




def matrix_quiver(x : np.ndarray, y: np.ndarray, matrices : np.ndarray, shade_determinant = False, label = None, det_label = None):
    """
    Visualizes eigenvectors of matrices using quiver plots.

    Args:
        x (np.ndarray): X-coordinates for the quiver plot.
        y (np.ndarray): Y-coordinates for the quiver plot.
        matrices (np.ndarray): Matrices for which eigenvectors are computed.
        shade_determinant (bool): If True, shades the plot based on the determinant of the matrices.
        label (str): Label for the eigenvectors. optional defaults to None
        det_label (str): Label for the determinant. optional defaults to None
    Returns:
        None
    """
    #remove the arrowheads and set the pivot to mid to vizualize eigenvectors
    EIGEN_VEC_QUIVER_KWARGS = {'headwidth' : 0, 'headlength' : 0, 'headaxislength' : 0, 'pivot' : 'mid'}
    
    eigenvalues, eigenvectors = np.linalg.eig(matrices)
    # from the docs of np.linalg.eig : eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i] !!!

    #scale by the eigenvalues
    scaled_eigenvectors = eigenvectors * eigenvalues[...,None,:]
    determinants = eigenvalues[...,0]*eigenvalues[...,1]
    
    if shade_determinant:
        mesh = plt.pcolormesh(x, y, determinants)
        if not det_label is None:
            plt.colorbar(mesh, label = det_label)

    q1 = plt.quiver(x, y, scaled_eigenvectors[...,0,0], scaled_eigenvectors[...,1,0], **EIGEN_VEC_QUIVER_KWARGS)
    q2 = plt.quiver(x, y, scaled_eigenvectors[...,0,1], scaled_eigenvectors[...,1,1], **EIGEN_VEC_QUIVER_KWARGS)
    
    # needed to ensure proper scaling of the eigenvector arrows
    scale = np.max(q1.scale,q2.scale)
    q1.scale = scale
    q2.scale = scale
    
    if not label is None:
        #add empty scatter plot to add label, is a little hacky but works
        plt.scatter(None,None,marker = r'+',label = label, color = 'black')
        
        
## Serie 3

@dataclass
class LinearModelResult:
    """
    Class to hold parameters for a linear model.
    """
    parameters : np.ndarray
    rse : float # Residual Standard Error = m_0
    normal_matrix : np.ndarray
    residuals : np.ndarray | None = None
    
    @property
    def cofactor_matrix(self) -> np.ndarray:
        return scipy.linalg.inv(self.normal_matrix)

    def __repr__(self):
        return f"LinearModelResult(parameters = \n{self.parameters}\nm_0 = {self.rse})\n normal_matrix =\n {self.normal_matrix}"
    
def poly_design_matrix(x : np.ndarray, degree : int) -> np.ndarray:
    """Computes the design matrix for a polynomial of a given degree.

    Args:
        x (np.ndarray): input values.
        degree (int): degree of the polynomial.

    Returns:
        np.ndarray: Design matrix.
    """
    return np.column_stack([x**i for i in reversed(range(degree+1))])

def design_matrix(f : sympy.Expr, x : np.ndarray, parameters : List[sympy.Symbol], feature_sym = sympy.Symbol) -> np.ndarray:
    """
    Computes the design matrix for a given function and input x.
    
    Args:
        f (sympy.Expr): Expression for the function.
        x (np.ndarray): Input x.
        parameters (List[sympy.Symbol]): List of symbols representing the parameters.
        feature_sym (sympy.Symbol): Symbol representing the feature variable.
    Return:
        np.ndarray: Design matrix.
    """
    column_functions = [sympy.lambdify([feature_sym], sympy.diff(f, a)) for a in parameters]
    
    design_matrix_columns = [np.broadcast_to(column_function(x), x.shape) for column_function in column_functions]
    
    design_matrix = np.column_stack(design_matrix_columns)
    return design_matrix

def normal_equation(A : np.ndarray, y : np.ndarray) -> LinearModelResult:
    """
    Computes the parameters of a linear model using the normal equation method.
    
    Args:
        A (np.ndarray): Design matrix.
        y (np.ndarray): Target variable.
        weights (np.ndarray | float | None) The covariance of the data, if None uses identity default is None
    
    Returns:
        LinearModelResult: Parameters of the linear model.
    """
    b = A.T@y
    N = A.T@A
    
    parameters = np.linalg.solve(N, b)
    residuals =  A@parameters - y
    m_0_sr = (residuals.T@residuals) / (len(y) - len(parameters))
    
    return LinearModelResult(
        parameters,
        m_0_sr**0.5,
        N,
        residuals)

def _get_N_b(x : np.ndarray, y : np.ndarray, degree : int) -> Tuple[np.ndarray, np.ndarray]:
    """ Computes the normal matrix and the vector b directly for polynomial fitting.

    Args:
        x (np.ndarray): input features.
        y (np.ndarray): target variable.
        degree (int): degree of the polynomial.

    Returns:
        Tuple containing:
            - np.ndarray: Normal matrix.
            - np.ndarray: Vector b.
    """
    b = np.ndarray(degree+1)
    for i in range(degree+1):
        b[degree-i] = np.sum(y * x**i)
    
    #precompute sums of powers since the Matrix N reuses them
    N_entries = np.ndarray(degree*2+1)
    for i in range(degree*2+1):
        N_entries[i] = np.sum(x**i)
    
    N = np.ndarray((degree+1, degree+1))
    
    for i in range(degree+1):
        for j in range(degree+1):
            N[degree-i,degree-j] = np.sum(N_entries[i+j])
    return N, b

def poly_fit(x : np.ndarray, y : np.ndarray, degree : int, method : Literal['design matrix','direct'] = 'direct') -> LinearModelResult:
    """
    Fits a polynomial of a given degree to the data using the normal equation method.
    Does not generate a design matrix, saving memory.

    Args:
        x (np.ndarray): input features.
        y (np.ndarray): target variable.
        degree (int): degree of the polynomial to fit.
        method Lietarl: method to use for fitting.
            - 'design matrix': uses the design matrix method.
            - 'direct': uses the direct method.

    Returns:
        LinearModelResult: Parameters of the polynomial fit.
    """
    if method == 'design matrix':
        #use the design matrix method
        A : np.ndarray = poly_design_matrix(x, degree)
        return normal_equation(A, y)
    if method == 'direct':
        N, b = _get_N_b(x, y, degree)
    
        parameters = np.linalg.solve(N, b)
        residuals =  np.polyval(parameters, x) - y
        m_0_sr = (residuals.T @ residuals) / (len(y) - len(parameters))
    
        return LinearModelResult(parameters, m_0_sr**0.5, N, residuals)
    
    raise Exception(f'method {method} not supported')