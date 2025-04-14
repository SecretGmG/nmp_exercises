import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse
import seaborn as sns
sns.set_style()
import sympy
from scipy import stats
from functools import wraps
from typing import Callable, Iterable, Literal, Tuple, List
from abc import ABC, abstractmethod
import warnings
from IPython.display import display

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


# Serie 3&4

class FunctionalModel(ABC):
    """
    Abstract base class for functional models.
    The implementation only needs to provide the design matrix and initialize the Parameters
    The reason that so many values are accessible instead of hidden in the funcion is for logging
    and because in this lecture we might be interested in the intermediate results
    """
    
    max_iter : int = 10_000
    epsilon : float = 1e-4
    
    logger : Callable = lambda arg : None
    
    iterations : int = 0
    
    x : np.ndarray
    y : np.ndarray
    P : scipy.sparse.sparray # weighting matrix
    
    A : np.ndarray # design matrix
    
    normal_matrix : np.ndarray
    b : np.ndarray #rhs of normal equation

    parameters : np.ndarray
    
    def set_parameters(self, parameters : np.ndarray) -> None:
        self.parameters = parameters
    
    
    m_0 : float = np.inf

    y_pred : np.ndarray
    
    @property
    def dof(self):
        return len(self.x) - len(self.parameters)
    
    @property
    def residuals(self) -> np.ndarray:
        return self.y_pred-self.y
    
    def cofactor_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.normal_matrix)
    
    def y_pred_cofactors(self) -> np.ndarray:
        return self.A@self.cofactor_matrix()@self.A.T
    
    def y_pred_cofactor_diag(self) -> np.ndarray:
        # efficently compute the diagonal of the cofactor matrix A^t Q A
        return np.einsum('ij,jk,ki -> i', self.A, self.cofactor_matrix(), self.A.T)
    
    
    def fit(self, x : np.ndarray, y : np.ndarray, weight_matrix: scipy.sparse.sparray|None = None):
        """
        Fit the functional model to the observed data using weighted least squares.

        If the maximum number of iterations is reached without convergence,
        a warning is issued.

        The method assumes that `self.evaluate` and `self._set_design_matrix` are implemented
        correctly in a subclass.
        """
        
        # make sure x and y are arrays
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        
        # by default use the identity matrix as weight matrix 
        if weight_matrix is None:
            weight_matrix = scipy.sparse.eye_array(len(x), len(x))
        
        self.P = weight_matrix
        
        # to start the iteration process the model needs to be evaluated at the initial parameters
        self.y_pred = self.evaluate(x)
        
        for i in range(self.max_iter):
            self.iterations = i+1
            
            # First compute the design matrix, this depends on the implementation of the functional model
            self._set_design_matrix()
            
            # Then use the normal equations to compute the change in the parameterss
            self.normal_matrix = self.A.T @ self.P @ self.A
            self.b = self.A.T @ self.P @ (self.y - self.y_pred)
        
            delta_parameters = np.linalg.solve(self.normal_matrix, self.b)
            
            # Update the parameters and assiciated values
            self.parameters += delta_parameters
            self.y_pred = self.evaluate(self.x)
            self.m_0 = np.sqrt((self.residuals.T @ self.P @ self.residuals) / (self.dof))
            
            # Finally call the logger, this allows selective printing and more
            self.logger(self)            
            
            # stop iterating if the change in all the parameters is smaller than epsilon
            if np.all(np.abs(delta_parameters) < self.epsilon):
                break
        
        if self.iterations == self.max_iter:
            warnings.warn("The Functional Model did not converge, make sure you set reasonable initial parameters")
            
    
    @abstractmethod
    def _set_design_matrix(self) -> None:
        """
        sets the design matrix in self.A
        """
        pass
    
    @abstractmethod
    def evaluate(self, x : np.ndarray) -> np.ndarray:
        """
        Evaluates the model at the given x values.
        """
        pass

    def chi2_statistic(self, sigma_0 : float = 1) -> float:
        """
        Computes the chi-squared test statistic.
        """
        # apply the methods as in the script chapter 4.11
        return (self.m_0 / sigma_0)**2 * self.dof
    
    def chi2_critical_value(self, alpha : float = 0.05) -> float:
        """
        Computes the chi-squared critical value, to compare to the test statistic.
        """
        return stats.chi2.ppf(1 - alpha, self.dof)
    
    
    def plot(self, sigma_0 : float = 1):
        """
        Plots the data points and the fitted model.
        """
        plt.errorbar(x = self.x, y = self.y, yerr=sigma_0 * self.P.diagonal()**(-0.5),fmt = '.', label='data')
        
        x_values = np.linspace(self.x.min(), self.x.max(), 200)
        plt.plot(x_values,self.evaluate(x_values), color = 'black', alpha = 0.5)
        
        #use the max of sigma_o and m_0
        sigma = max(sigma_0, self.m_0)
        
        plt.errorbar(self.x, self.y_pred, sigma * self.y_pred_cofactor_diag()**0.5, fmt = '.', color = 'black', label='model prediction')
        plt.legend()
    
    def data_frame(self) -> pd.DataFrame:
        """
        Creates a dataframe containing most values that have the same dimension as x
        """
        return pd.DataFrame({
            'x' : self.x,
            'x_weight' : self.P.diagonal(),
            'y' : self.y,
            'y_pred' : self.y_pred,
            'residuals' : self.residuals,
            'y_cofactors' : self.y_pred_cofactor_diag()
        })
    
    def display_summary(self, sigma_0 : float = 1):
        # display basic information about the model
        display(self)
        display(self.data_frame())
        
        # plot of the fit
        self.plot(sigma_0)
        plt.show()
        
        # cofactor matrix
        plt.title('cofactor matrix of the parameters')
        mat = plt.gca().matshow(self.cofactor_matrix())
        plt.colorbar(mat)
        
        # print the relevant statistics
        print(f'sigma_0 / m_0 = {sigma_0 / self.m_0:.2f}')
        
        print(f'chi2 statistic : {self.chi2_statistic(sigma_0):.2f}')
        print(f'critical values for {self.dof} degrees of freedom')
        alphas = [0.1,0.05,0.01]
        for a in alphas:
            print(f'x(alpha = {a:.2f}) = {self.chi2_critical_value(a):.2f}')    
        
    
    def __repr__(self):
        return f"FunctionalModel, Parameters: {self.parameters}, m_0: {self.m_0}, dof: {self.dof}"
        
    
class PolyFunctionalModel(FunctionalModel):
    
    # In this case the off diagonal elements of the Weighting matrix are ignored!
    degree : int
    
    def __init__(self, degree : int):
        self.degree = degree
        #set initial parameters to zero by default
        self.parameters = np.zeros(degree+1)
    
    def _set_design_matrix(self):
        self.A = np.column_stack([self.x**i for i in reversed(range(self.degree+1))])
    
    def evaluate(self, x):
        return np.polyval(self.parameters, x)

class SympyFunctionalModel(FunctionalModel):
    function_expr : sympy.Expr
    parameter_symbols : List[sympy.Symbol]
    feature_symbol : sympy.Symbol
    
    differential_expressions : List[sympy.Expr]
    differentials : List[Callable]
    lambdified : Callable
    
    design_matrix : np.ndarray
    
    def __init__(self, function_expr : sympy.Expr, parameter_symbols : List[sympy.Symbol], feature_symbol : sympy.Symbol):
        self.function_expr = function_expr
        self.parameter_symbols = parameter_symbols
        self.feature_symbol = feature_symbol
        
        self.lambdified = sympy.lambdify([*self.parameter_symbols, self.feature_symbol], self.function_expr)
        
        # compute the partial derivative of the function with respect to each parameter
        self.differential_expressions = [sympy.diff(self.function_expr, a) for a in self.parameter_symbols]
        self.differentials = [sympy.lambdify([*self.parameter_symbols, self.feature_symbol], diff) for diff in self.differential_expressions]
        
        #set initial parameters to zero by default
        self.parameters = np.zeros(len(parameter_symbols))
    
    def _set_design_matrix(self):
        
        # the collumns of the design matrix are the partial derivatives of the function with respect to each parameter
        # evaluated at the input x, and the current parameters
        # broadcastring the result is necessary because sometimes the lambdified function returns a scalar (if the function is constant)
        A_collumns = [np.broadcast_to(d(*self.parameters, self.x), self.x.shape) for d in self.differentials]
        
        self.A = np.column_stack(A_collumns)
    
    def evaluate(self, x) -> np.ndarray:
        return np.broadcast_to(self.lambdified(*self.parameters, x), x.shape)