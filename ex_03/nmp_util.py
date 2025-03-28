import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style()
import sympy
from functools import wraps
from typing import Iterable, Tuple, List

# File i will keep continuously updating during the course, adding usefull functions

## SERIE 1
def get_inliers(data : np.ndarray, f : float= 4.0, iterative : bool = True, robust = True):
    """
    Identifies inliers in a dataset based on the standard deviation.

    Args:
        data (np.ndarray): Input data array.
        f (float): Maximum number of standard deviations allowed for inliers. Default is 4.0.
        iterative (bool): If True, iteratively refines inliers. Default is True.
        robust (bool) : If true use the median and MAD instead of mean and std. Default is True.

    Returns:
        np.ndarray: Boolean array indicating inliers (True) and outliers (False).
    """
    # start with the entire data as inliers
    inliers = np.ones_like(data, dtype=bool)
    
    while True:
        if robust:
            center = np.median(data[inliers])
            sorted = np.sort(np.abs(data[inliers]-center))
            std = sorted[int(len(sorted)*0.683)-1]
        else:
            std = data[inliers].std()
            center = data[inliers].mean()
        
        new_inliers = np.abs(data-center) < f*std
        
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

def poly_design_matrix(degree, features) -> np.ndarray:
    return np.column_stack([features**i for i in range(degree+1)[::-1]])

def design_matrix(f : sympy.Expr, features : np.ndarray, parameters : List[sympy.Symbol], feature_sym = sympy.Symbol) -> np.ndarray:
    """
    Computes the design matrix for a given function and input features.
    
    Args:
        f (sympy.Expr): Expression for the function.
        features (np.ndarray): Input features.
        parameters (List[sympy.Symbol]): List of symbols representing the parameters.
        feature_sym (sympy.Symbol): Symbol representing the feature variable.
    Return:
        np.ndarray: Design matrix.
    """
    column_functions = [sympy.lambdify([feature_sym], sympy.diff(f, a)) for a in parameters]
    
    design_matrix_columns = [np.broadcast_to(column_function(features), features.shape) for column_function in column_functions]
    
    design_matrix = np.column_stack(design_matrix_columns)
    return design_matrix
    
    

def compute_parameters(y : np.ndarray, A : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the parameters of a linear model using the normal equation method.
    
    Args:
        y (np.ndarray): Target variable.
        A (np.ndarray): Design matrix.
    
    Returns:
        Tuple containing:
            - Parameters of the linear model.
            - Mean square error.
            - Inverse of the normal matrix.
    """
    b = A.T@y
    N = A.T@A
    N_inv = np.linalg.inv(N)
    parameters = N_inv@b
    residuals =  A@parameters - y
    m_0_sr = (residuals.T@residuals) / (len(y) - len(parameters))
    return parameters, m_0_sr, N_inv

def poly_fit(degree, y, features) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits a polynomial of a given degree to the data using the normal equation method.
    Does not generate a design matrix, saving memory.

    Args:
        degree (int): degree of the polynomial to fit.
        y (np.ndarray): target variable.
        features (np.ndarray): input features.

    Returns:
        Tuple containing:
            - Parameters of the polynomial.
            - Mean square error.
            - Inverse of the normal matrix.
    """
    
    b = np.array([
            np.sum(x*y**i for x,y in zip(y, features))
        for i in range(degree+1)[::-1]]
        )
    
    #precompute sums of powers since the Matrix N reuses them
    N_entries = [np.sum(x**i for x in features) for i in range(degree*2+1)]
    
    N = np.array([[
            N_entries[i+j]
        for i in range(degree+1)[::-1]] 
        for j in range(degree+1)[::-1]]
        )
    
    N_inv = np.linalg.inv(N)
    parameters = N_inv @ b
    residuals =  np.polyval(parameters, features) - y
    m_0_sr = (residuals.T @ residuals) / (len(y) - len(parameters))
    return parameters, m_0_sr, N_inv