import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse import linalg
import sympy
from scipy import stats
from typing import Callable, Dict, Iterable, Tuple, List
from abc import ABC, abstractmethod
from copy import deepcopy
import warnings

import seaborn as sns
sns.set_style()
#improve readability of the output of numpy arrays in jupyter notebooks
np.set_printoptions(linewidth=150)


# File i will keep continuously updating during the course, adding usefull functions

## SERIE 1
def get_inliers(data : np.ndarray, f : float= 4.0, iterative : bool = True, robust = True):
    """
    Returns a boolean mask identifying inliers in `data` based on deviation from mean or median.

    Args:
        data: 1D data array.
        f: Threshold multiplier for deviation.
        iterative: If True, iteratively refines inliers.
        robust: If True, use robust stats (median & percentile).

    Returns:
        Boolean array marking inliers (True).
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

def error_propagation_formula(f : sympy.Matrix|Iterable[sympy.Expr]|sympy.Expr, args : List[sympy.Symbol]) -> Tuple[sympy.Expr, sympy.MatrixSymbol]:
    """
    Computes symbolic error propagation A K A^T and returns it with symbolic covariance K.

    Args:
        f: Vector-valued expression (Matrix, iterable, or expression).
        args: Variables for Jacobian computation.

    Returns:
        (Expression for propagated error, symbolic covariance matrix K).
    """
    
    if not isinstance(f, sympy.Matrix):
        if isinstance(f, sympy.Expr):
            f = [f]
        
        f = sympy.Matrix(list(f))
    
    A = f.jacobian(args)
    K = sympy.MatrixSymbol('K',len(args), len(args))
    return A * K * A.T , K #skript S.12

def propagate_error(f : sympy.Matrix|Iterable[sympy.Expr]|sympy.Expr, args_symbols : Iterable[sympy.Symbol], args : np.ndarray, cov : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagates error through a function using the covariance matrix.
    
    Args:
        f: Vector-valued expression (Matrix, iterable, or expression).
        args_symbols: Variables for Jacobian computation.
        args: Values of the variables along the first dimension.
        cov: Covariance matrix of the variables.
    
    Returns:
        (Result of the function, propagated covariance matrix).
    """
    cov_expr, K = error_propagation_formula(f, args_symbols)
    result = sympy.lambdify(args_symbols, f)(*args)
    result_cov = sympy.lambdify([*args_symbols, K],cov_expr)(*args, cov)
    return (result, result_cov)

def matrix_quiver(x : np.ndarray, y: np.ndarray, matrices : np.ndarray, shade_determinant = False, label = None, det_label = None, c = 'black'):
    """
    Plots eigenvectors of 2x2 matrices as quivers. Can shade background by determinant.

    Args:
        x, y: Grid coordinates.
        matrices: Array of 2x2 matrices.
        shade_determinant: If True, color field by determinant.
        label: Label for quiver vectors.
        det_label: Label for colorbar if shading is enabled.
    """
    #remove the arrowheads and set the pivot to mid to visualize eigenvectors
    EIGEN_VEC_QUIVER_KWARGS = {'headwidth' : 0, 'headlength' : 0, 'headaxislength' : 0, 'pivot' : 'mid'}
    
    eigenvalues, eigenvectors = np.linalg.eig(matrices)
    # from the docs of np.linalg.eig : eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i] !!!

    #scale by the eigenvalues
    scaled_eigenvectors = eigenvectors * eigenvalues[...,None,:]
    determinants = np.linalg.det(matrices)
    
    if shade_determinant:
        mesh = plt.pcolormesh(x, y, determinants)
        plt.colorbar(mesh, label = det_label)

    q1 = plt.quiver(x, y, scaled_eigenvectors[...,0,0], scaled_eigenvectors[...,1,0], **EIGEN_VEC_QUIVER_KWARGS, c = c)
    q2 = plt.quiver(x, y, scaled_eigenvectors[...,0,1], scaled_eigenvectors[...,1,1], **EIGEN_VEC_QUIVER_KWARGS, c = c)
    
    # needed to ensure proper scaling of the eigenvector arrows
    scale = max(q1.scale,q2.scale)
    q1.scale = scale
    q2.scale = scale
    
    if label is not None:
        #add empty scatter plot to add label, is a little hacky but works
        plt.scatter([],[],marker = r'+',label = label, c = c)


# Serie 3&4
class FunctionalModel(ABC):
    """
    Abstract base class for parametric models fitted with weighted least squares.

    Stores all intermediate computations for transparency and debugging.
    """
    
    ### Public fields
    
    # defines break conditions for the iterative process
    max_iter : int = 10_000
    epsilon : float = 1e-4
    
    # used to provide nicer outputs, needs to be set by the implementation
    parameter_symbols : List[sympy.Symbol]
    
    # is called after each iteration, can be used to print or log the current state of the model
    # default is a no-op, but can be set to a function that takes the model as argument
    logger : Callable = lambda *args : None
    
    # defines the initial parameters, can be set before calling fit() to define initial parameters
    # default value is determined by implementation
    parameters : np.ndarray 
    
    ### Readonly fields
    iterations : int = 0
    
    x : np.ndarray #observed x
    y : np.ndarray #observed y
    P : scipy.sparse.dia_matrix # weighting matrix
    
    A : np.ndarray # design matrix
    
    normal_matrix : np.ndarray
    b : np.ndarray #rhs of normal equation
    delta_parameters : np.ndarray # updates of the parameters in iterative least squares
    
    m_0 : float = np.inf

    y_pred : np.ndarray
    
    residuals : np.ndarray
    
        
    @property
    def dof(self):
        return len(self.x) - len(self.parameters)
    
    
    def fit(self, x : np.ndarray, y : np.ndarray, weight_matrix: scipy.sparse.dia_matrix|None = None):
        """
        Fits model to data using iterative weighted least squares.
        """
        
        if self.parameters is None:
            raise Exception('Cannot fit, because self.parameters is None, set your initial parameters!')
        
        # make sure x and y are arrays
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        
        # by default use the identity matrix as weight matrix 
        if weight_matrix is None:
            weight_matrix = scipy.sparse.diags(np.ones(len(x)))
        
        self.P = weight_matrix
        
        # to start the iteration process the model needs to be evaluated at the initial parameters
        self.y_pred = self.eval(x)
        
        for i in range(self.max_iter):
            self.iterations = i+1
            
            # First compute the design matrix, this depends on the implementation of the functional model
            self.A = self.get_design_matrix(x)
            
            # Then use the normal equations to compute the change in the parameters
            self.normal_matrix = self.A.T @ self.P @ self.A
            # use P.dot() since it is more efficient here to compute from right to left (because the intermediate matrix doesn't need to be stored)
            self.b = self.A.T @ self.P.dot(self.y - self.y_pred)

            #use lstsq instead of inv to avoid computing the inverse and handle singular normal matrices
            self.delta_parameters = np.linalg.lstsq(self.normal_matrix, self.b)
            
            # Update the parameters and associated values
            self.parameters = self.parameters + self.delta_parameters
            self.y_pred = self.eval(self.x)
            self.residuals = self.y - self.y_pred
            self.m_0 = np.sqrt((self.residuals.T @ self.P @ self.residuals) / (self.dof))
            
            # Finally call the logger, this allows selective printing and more
            self.logger(self)
            
            # stop iterating if the change in all the parameters is smaller than epsilon
            if np.all(np.abs(self.delta_parameters) < self.epsilon):
                break
        
        if self.iterations == self.max_iter:
            warnings.warn("The Functional Model did not converge, make sure you set reasonable initial parameters")
    
    def parameter_cof(self) -> np.ndarray:
        return np.linalg.inv(self.normal_matrix)
    
    def parameter_corr(self) -> np.ndarray:
        cofactor_matrix = self.parameter_cof()
        diag = cofactor_matrix.diagonal()**0.5
        return cofactor_matrix / np.outer(diag, diag)
    
    def eval_cof(self, x : np.ndarray) -> np.ndarray:
        A = self.get_design_matrix(x)
        return A@self.parameter_cof()@A.T
    
    def eval_stderr(self, x : np.ndarray, sigma) -> np.ndarray:
        """
        Returns only the diagonal of the covariance of the prediction at x. sigma determines the scale of the covariance, default is m_0.
        """
        if sigma is None:
            sigma = self.m_0
        A = self.get_design_matrix(x)
        # if we only care about the diagonal, it is more efficient to compute the sum directly
        return sigma * np.einsum('ij,jk,ki -> i', A, self.parameter_cof(), A.T)**0.5

    def chi2_threshold(self, alpha : float = 0.05) -> float:
        """
        Computes the threshold for chi-squared test, at a given significance alpha.
        This is the mean normalized critical value X^2(dof) / dof
        This value then needs to be compared to m_0^2 / sigma_0^2
        
        NOTE: for smaller alphas you're not "accepting" larger errors, you're demanding stronger evidence to reject the model.
        """
        return stats.chi2.ppf(1 - alpha, self.dof) / self.dof
    
    def plot(self, sigma_0 : float = 1, sigma : float = None, c_data = 'b', c_model = 'black'):
        """
        Plots the data points and the fitted model.
        The data points are shown with error bars, the model is shown as a line with shaded area for the covariance.
        Args:
            sigma_0: the scale of the covariance of the data points, default is 1 (assuming P = Cov^-1).
            sigma: the scale of the covariance of the model parameters, default is max(sigma_0, m_0).
            c_data: color of the data points, default is blue.
            c_model: color of the model, default is black.
        """
        if sigma is None:
            sigma = max(sigma_0, self.m_0)
        
        # this could be made more efficient by using the covariance matrix directly, 
        # but this is more readable. I'll change it later if needed
        y_stderr = sigma_0 * scipy.sparse.linalg.inv(self.P.tocsc()).diagonal()**0.5
        plt.errorbar(x = self.x, y = self.y, yerr=y_stderr,fmt = '.', label='data',color= c_data)
        
        # use sigma to scale the covariance of the predictions
        plt.errorbar(self.x, self.y_pred, self.eval_stderr(self.x, sigma), fmt = '.',color= c_model, label='model prediction')
        
        linspace = np.linspace(self.x.min(), self.x.max(), 200)
        y = self.eval(linspace)
        eval_stderr = self.eval_stderr(linspace, sigma)          
        plt.plot(linspace, y,color= c_model, alpha = 0.5)
        plt.fill_between(linspace, y-eval_stderr, y+eval_stderr,color= c_model, alpha = 0.2)
        plt.legend()
    
    def show_correlation(self):
        """
        Plots the correlation matrix of the parameters.
        """
        matshow = plt.matshow(self.parameter_corr())
        plt.colorbar(matshow, label = 'correlation')
        n_ticks = len(self.parameter_symbols)
        plt.gca().xaxis.set_label_position('top')
        plt.xlabel('parameter')
        plt.ylabel('parameter')
        ticks = [f'${sympy.latex(s)}$' for s in self.parameter_symbols]
        plt.xticks(range(n_ticks), ticks)
        plt.yticks(range(n_ticks), ticks)
    
    def print_parameters_latex(self, precision : int = 3, sigma : float = None):
        """
        Prints the parameters and their uncertainties in LaTeX format.
        sigma is the scale of the covariance, default is m_0.
        Args:
            precision: number of decimal places to print, default is 3.
        """
        if sigma is None:
            sigma = self.m_0
        for s, v, e in zip(self.parameter_symbols, self.parameters, sigma*np.sqrt(np.diag(self.parameter_cof()))):
            print(f"${sympy.latex(s)} = {v:.{precision}f} \\pm {e:.{precision}f}$")
    
    def copy(self):
        """
        Returns a deep copy of the model.
        """
        return deepcopy(self)
    
    @abstractmethod
    def eval(self, x : np.ndarray) -> np.ndarray:
        """
        Evaluates the model at the given x values. Needs Implementation.
        """
        pass
    
    @abstractmethod
    def get_design_matrix(self, x : np.ndarray) -> np.ndarray:
        """
        Computes and returns the design matrix for the inputs x. Needs Implementation.
        """
        pass
        
class PolyFunctionalModel(FunctionalModel):
    degree : int
    
    def __init__(self, degree : int):
        self.degree = degree
        
        #set initial parameters to zero by default
        self.parameters = np.zeros(degree+1)
        #set parameters symbols to a_0, a_1, ..., a_n (reversed order for integration with numpy)
        self.parameter_symbols = [sympy.Symbol(f'a_{i}') for i in reversed(range(degree+1))]
    
    def get_design_matrix(self, x : np.ndarray) -> np.ndarray:
        return np.column_stack([x**i for i in reversed(range(self.degree+1))])
    
    def eval(self, x):
        return np.polyval(self.parameters, x)


class SympyFunctionalModel(FunctionalModel):
    function_expr : sympy.Expr
    feature_symbol : sympy.Symbol
    
    differential_expressions : List[sympy.Expr]
    differentials : List[Callable] # store these for debugging and transparancy
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
    
    def eval(self, x : np.ndarray) -> np.ndarray:
        # asarray because i don't trust myself to not use lists or pd.Series
        x = np.asarray(x)
        return np.broadcast_to(self.lambdified(*self.parameters, x), x.shape)
    
    def get_design_matrix(self, x : np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        # the columns of the design matrix are the partial derivatives of the function with respect to each parameter
        # evaluated at the input x, and the current parameters
        # broadcasting the result is necessary because sometimes the lambdified function returns a scalar (if the function is constant)
        A_columns = [np.broadcast_to(d(*self.parameters, x), x.shape) for d in self.differentials]
        
        return np.column_stack(A_columns)