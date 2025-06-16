import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse
import seaborn as sns
sns.set_style()
import sympy
from functools import wraps
from typing import Iterable, Literal, Tuple, List

#improve readability of the output of numpy arrays in jupyter notebooks
np.set_printoptions(linewidth=150)


# File i will keep continuously updating during the course, adding usefull functions

## SERIE 1
def get_inliers(data : np.ndarray, f : float= 4.0, iterative : bool = True, robust = True):
    """
    Identifies inliers in a dataset based on the standard deviation.

    Args:
        data (np.ndarray): Input data array.
        f (float): Maximum number of standard deviations allowed for inliers. Default is 4.0.
        iterative (bool): If True, iteratively refines inliers. Default is True.
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
            #use np.percentile for faster computation of the 68.3 percentile
            #it uses clever techniques to compute the percentile without sorting the entire array
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
        matrices (np.ndarray): 2x2 Matrices for which eigenvectors are computed.
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
    determinants = np.linalg.det(matrices)
    
    if shade_determinant:
        mesh = plt.pcolormesh(x, y, determinants)
        if det_label is not None:
            plt.colorbar(mesh, label = det_label)

    q1 = plt.quiver(x, y, scaled_eigenvectors[...,0,0], scaled_eigenvectors[...,1,0], **EIGEN_VEC_QUIVER_KWARGS)
    q2 = plt.quiver(x, y, scaled_eigenvectors[...,0,1], scaled_eigenvectors[...,1,1], **EIGEN_VEC_QUIVER_KWARGS)
    
    # needed to ensure proper scaling of the eigenvector arrows
    scale = max(q1.scale,q2.scale)
    q1.scale = scale
    q2.scale = scale
    
    if label is not None:
        #add empty scatter plot to add label, is a little hacky but works
        plt.scatter([],[],marker = r'+',label = label, color = 'black')

# Serie 3
class PolyFunctionalModel:
    """
    Polynomial funcitonal model extend this to a general non linear funcitonal model during the rest of the course
    """
    
    type Method = Literal['direct','design matrix']
    # If direct method is used, the normal matrix and b vector are computed directly from the data.
    # In this case the off diagonal elements of the Weighting matrix are ignored!
    
    degree : int
    method : Method
    
    x : np.ndarray
    y : np.ndarray
    P : scipy.sparse.spmatrix # weighting matrix
    
    A : np.ndarray # design matrix
    
    normal_matrix : np.ndarray
    b : np.ndarray

    parameters : np.ndarray
    m_0 : float

    y_pred : np.ndarray
    
    @property
    def residuals(self) -> np.ndarray:
        return self.y_pred-self.y
    
    @property
    def covariance_matrix(self) -> np.ndarray:
        return self.m_0**2 * np.linalg.inv(self.normal_matrix)
    
    @property
    def dof(self):
        return len(self.x) - len(self.parameters)
    
    def __init__(self, degree : int, method: Method = 'design matrix'):
        self.degree = degree
        self.method = method
    
    def fit(self, x : np.ndarray, y : np.ndarray, weight_matrix: scipy.sparse.spmatrix|None = None):
        
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        
        if weight_matrix is None:
            weight_matrix = scipy.sparse.identity(len(x))
        
        self.P = weight_matrix
            
        if self.method == 'design matrix':
            self.design_matrix_setup()
        if self.method == 'direct':
            self.direct_setup()
        
        self.parameters = np.linalg.solve(self.normal_matrix, self.b)
        self.y_pred = self.evaluate(self.x)
        self.m_0 = np.sqrt((self.residuals.T @ self.P @ self.residuals) / (self.dof))
    
    def direct_setup(self):
        # Ignore off diagonal elements in the direct method
        diagonal = self.P.diagonal()
        
        #precompute sums of powers since the Matrix N reuses them
        N_entries : np.ndarray = np.empty(self.degree*2+1)

        for i in range(self.degree*2+1):
            N_entries[i] = np.sum(diagonal * self.x**i)
        
        
        self.normal_matrix = np.empty((self.degree+1, self.degree+1))
        
        for i in range(self.degree+1):
            for j in range(self.degree+1):
                self.normal_matrix[self.degree-i,self.degree-j] = N_entries[i+j]
        
        self.b = np.empty(self.degree+1)
        
        for i in range(self.degree+1):
            self.b[self.degree-i] = np.sum(diagonal * self.y * self.x**i)
        
    def design_matrix_setup(self):
        self.A = np.column_stack([self.x**i for i in reversed(range(self.degree+1))])
        
        self.normal_matrix = self.A.T @ self.P @ self.A
        self.b = self.A.T @ self.P @ self.y
        
    def evaluate(self, x):
        return np.polyval(self.parameters, x)
    
    def plot(self):
        """
        Plots the data points and the fitted model.
        """
        sns.scatterplot(x = self.x, y = self.y, label='data')
        sns.lineplot(x = self.x, y = self.y_pred, label='fitted model', color = 'red')
    
    def __repr__(self):
        return f"FunctionalModel, Parameters: {self.parameters}, m_0: {self.m_0}, dof: {self.dof}"