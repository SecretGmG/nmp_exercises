import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse
import seaborn as sns
sns.set_style()
import sympy
import scipy, scipy.stats
from functools import wraps
from typing import Callable, Iterable, Literal, Tuple, List
from abc import ABC, abstractmethod

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
    scale = max(q1.scale,q2.scale)
    q1.scale = scale
    q2.scale = scale
    
    if not label is None:
        #add empty scatter plot to add label, is a little hacky but works
        plt.scatter([],[],marker = r'+',label = label, color = 'black')


class FunctionalModel(ABC):
    """
    Abstract base class for functional models.
    At the moment only linear functional models are implemented.
    The idea is to implement a general functional model that can be used for any funcitonal dependence,
    during the rest of the course.
    """
    
    x : np.ndarray
    y : np.ndarray
    P : scipy.sparse.spmatrix # weighting matrix, if None assume uncorrelated residuals
    
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
    
    
    def fit(self, x : np.ndarray, y : np.ndarray, weight_matrix: scipy.sparse.spmatrix|None = None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        
        if weight_matrix is None:
            weight_matrix = scipy.sparse.identity(len(x))
        
        self.P = weight_matrix
        
        self._set_up_normal_equation()
        
        self.parameters = np.linalg.solve(self.normal_matrix, self.b)
        self.y_pred = self.evaluate(self.x)
        self.m_0 = np.sqrt((self.residuals.T @ self.P @ self.residuals) / (self.dof))
    
    @abstractmethod
    def _set_up_normal_equation(self) -> None:
        pass
    
    @abstractmethod
    def evaluate(self, x : np.ndarray) -> np.ndarray:
        """
        Evaluates the model at the given x values.
        """
        pass
    
    def __call__(self, x : np.ndarray):
        return self.evaluate(x)
    
    def chi2test(self, sigma_0 : float = 1, alpha : float = 0.05) -> Tuple[float, float]:
        """
        Computes the chi-squared test statistic.

        Args:
            sigma_0 (float): Expected standard deviation. Default is 1, 
                assuming the residuals were correctly weighted by the inverse covariance matrix of the data.
            alpha (float): Significance level. Default is 0.05.

        Returns:
            Tuple[float, float]:
                - chi2_stat: The calculated test statistic.
                - critical_value: The chi-squared critical value at the given alpha.
        """
        # apply the methods as in the script chapter 4.11
        z = (self.m_0 / sigma_0)**2 * self.dof # Notation from the script
        critical_value = scipy.stats.chi2.ppf(1 - alpha, self.dof) # In the script called x_{1-alpha}
        return z , critical_value
    
    def plot(self):
        """
        Plots the data points and the fitted model.
        """
        sns.scatterplot(x = self.x, y = self.y, label='Data')
        sns.lineplot(x = self.x, y = self.y_pred, label='Fitted Model', color = 'red')
    
class PolyFunctionalModel(FunctionalModel):
    
    type Method = Literal['direct','design matrix']
    # If direct method is used, the normal matrix and b vector are computed directly from the data.
    # In this case the off diagonal elements of the Weighting matrix are ignored!
    
    degree : int
    method : Method
    
    def __init__(self, degree : int, method: Method = 'direct'):
        self.degree = degree
        self.method = method
    
    def _set_up_normal_equation(self):
        if self.method == 'direct':
            # Ignore off diagonal elements in the direct method
            diagonal = self.P.diagonal()
            
            #precompute sums of powers since the Matrix N reuses them
            N_entries : np.ndarray = np.ndarray(self.degree*2+1)
    
            for i in range(self.degree*2+1):
                N_entries[i] = np.sum(diagonal * self.x**i)
            
            
            self.normal_matrix = np.ndarray((self.degree+1, self.degree+1))
            for i in range(self.degree+1):
                for j in range(self.degree+1):
                    self.normal_matrix[self.degree-i,self.degree-j] = np.sum(N_entries[i+j])
            
            self.b = np.ndarray(self.degree+1)
            for i in range(self.degree+1):
                self.b[self.degree-i] = np.sum(diagonal * self.y * self.x**i)
        
        if self.method == 'design matrix':
            A = np.column_stack([self.x**i for i in reversed(range(self.degree+1))])
            self.normal_matrix = A.T @ self.P @ A
            self.b = A.T @ self.P @ self.y
    
    def evaluate(self, x):
        return np.polyval(self.parameters, x)

class SympyFunctionalModel(FunctionalModel):
    function_expr : sympy.Expr
    parameter_symbols : List[sympy.Symbol]
    feature_symbol : sympy.Symbol
    
    lambdified : Callable
    
    design_matrix : np.ndarray
    
    def __init__(self, function_expr : sympy.Expr, parameter_symbols : List[sympy.Symbol], feature_symbol : sympy.Symbol):
        self.function_expr = function_expr
        self.parameter_symbols = parameter_symbols
        self.feature_symbol = feature_symbol
        self.lambdified = sympy.lambdify([*self.parameter_symbols, self.feature_symbol], self.function_expr)
    
    def _set_up_normal_equation(self):
        
        # compute the partial derivative of the function with respect to each parameter
        differentials = [sympy.lambdify([self.feature_symbol], sympy.diff(self.function_expr, a)) for a in self.parameter_symbols]
    
        # the collumns of the design matrix are the partial derivatives of the function with respect to each parameter
        # evaluated at the input x
        # broadcastring the result is necessary because sometimes the lambdified function returns a scalar (if the function is constant)
        A_collumns = [np.broadcast_to(d(self.x), self.x.shape) for d in differentials]
        
        A = np.column_stack(A_collumns)
        
        self.normal_matrix =  A.T @ self.P @ A
        self.b = A.T @ self.P @ self.y
    
    def evaluate(self, x) -> np.ndarray:
        return self.lambdified(*self.parameters, x)