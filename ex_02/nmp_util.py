import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style()
import sympy
from functools import wraps
from typing import Iterable, Tuple, List
from warnings import catch_warnings

# File i will keep continuously updating during the course, adding usefull functions

## SERIE 1
def get_inliers(data : np.ndarray, max_std : float= 4.0, iterative : bool = True):
    """
    Identifies inliers in a dataset based on the standard deviation.

    Args:
        data (np.ndarray): Input data array.
        max_std (float): Maximum number of standard deviations allowed for inliers. Default is 4.0.
        iterative (bool): If True, iteratively refines inliers. Default is True.

    Returns:
        np.ndarray: Boolean array indicating inliers (True) and outliers (False).
    """
    # start with the entire data as inliers
    inliers = np.ones_like(data, dtype=bool)
    
    while True:
        std = data[inliers].std()
        mean = data[inliers].mean()
        
        new_inliers = np.abs(data-mean) < max_std*std
        
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