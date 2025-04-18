import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse
import seaborn as sns
sns.set_style()
import sympy
from scipy import stats
from functools import wraps
from typing import Callable, Dict, Iterable, Literal, Tuple, List
from abc import ABC, abstractmethod
import warnings
from IPython.display import display

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

def error_propagation_formula(f : sympy.Matrix|Iterable[sympy.Expr], args : List[sympy.Symbol]) -> Tuple[sympy.Expr, sympy.MatrixSymbol]:
    """
    Computes symbolic error propagation A K A^T and returns it with symbolic covariance K.

    Args:
        f: Vector-valued expression (Matrix or iterable).
        args: Variables for Jacobian computation.

    Returns:
        (Expression for propagated error, symbolic covariance matrix K).
    """
    
    if not isinstance(f, sympy.Matrix):
        #assume that f is an iterable of expressions if it is not already a matrix
        #in this case it should be possible to simply convert into a column matrix
        f = sympy.Matrix(list(f))
    
    A = f.jacobian(args)
    K = sympy.MatrixSymbol('K',len(args), len(args))
    return A * K * A.T , K #skript S.12

def propagate_error(f : sympy.Matrix|Iterable[sympy.Expr], args_symbols : List[sympy.Symbol], args : np.ndarray, cov : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cov_expr, K = error_propagation_formula(f, args_symbols)
    result = sympy.lambdify(args_symbols, f)(*args)
    result_cov = sympy.lambdify([*args_symbols, K],cov_expr)(*args, cov)
    return (result, result_cov)

def matrix_quiver(x : np.ndarray, y: np.ndarray, matrices : np.ndarray, shade_determinant = False, label = None, det_label = None):
    """
    Plots eigenvectors of 2x2 matrices as quivers. Can shade background by determinant.

    Args:
        x, y: Grid coordinates.
        matrices: Array of 2x2 matrices.
        shade_determinant: If True, color field by determinant.
        label: Label for quiver vectors.
        det_label: Label for colorbar if shading is enabled.
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
# I chose to delete the direct method since it is not noticably more efficient and bloats the code 

class FunctionalModel(ABC):
    """
    Abstract base class for parametric models fitted with weighted least squares.

    Stores all intermediate computations for transparency and debugging.
    
    Mutliple functions accept sigma as argument, for these functions the default is sigma = m_0 tha a posteriory weight.
    If you are only interested in the cofactors, just pass sigma = 1, if you have sigma a priori, pass it excplicitally.
    """
    
    # These parameters can be set expcicitally
    max_iter : int = 10_000
    epsilon : float = 1e-4
    
    parameters : np.ndarray # set this field to define initial parameters before calling fit()
    
    logger : Callable = lambda *args : None
    
    # These are read only and should not be changed from outside the class
    iterations : int = 0
    
    x : np.ndarray #observed x
    y : np.ndarray #observed y
    P : scipy.sparse.dia_array # weighting matrix
    
    A : np.ndarray # design matrix
    
    normal_matrix : np.ndarray
    b : np.ndarray #rhs of normal equation
    delta_parameters : np.ndarray # updates of the parameters in iterative least squares
    
    m_0 : float = np.inf

    y_pred : np.ndarray
    
    @property
    def dof(self):
        return len(self.x) - len(self.parameters)
    
    @property
    def residuals(self) -> np.ndarray:
        # observed minus computed
        return self.y-self.y_pred
    
    
    def fit(self, x : np.ndarray, y : np.ndarray, weight_matrix: scipy.sparse.dia_array|None = None):
        """
        Fits model to data using iterative weighted least squares.
        """
        
        # make sure x and y are arrays
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        
        # by default use the identity matrix as weight matrix 
        if weight_matrix is None:
            weight_matrix = scipy.sparse.eye_array(len(x), len(x))
        
        self.P = weight_matrix
        
        # to start the iteration process the model needs to be evaluated at the initial parameters
        self.y_pred = self.eval(x)
        
        for i in range(self.max_iter):
            self.iterations = i+1
            
            # First compute the design matrix, this depends on the implementation of the functional model
            self.A = self.get_design_matrix(x)
            
            # Then use the normal equations to compute the change in the parameterss
            self.normal_matrix = self.A.T @ self.P @ self.A
            # use P.dot() since it is more efficient here to compute from right to left (because the intermediate matrix doesn't need to be stored)
            self.b = self.A.T @ self.P.dot(self.y - self.y_pred)
        
            self.delta_parameters = np.linalg.solve(self.normal_matrix, self.b)
            
            # Update the parameters and assiciated values
            self.parameters = self.parameters + self.delta_parameters
            self.y_pred = self.eval(self.x)
            self.m_0 = np.sqrt((self.residuals.T @ self.P @ self.residuals) / (self.dof))
            
            # Finally call the logger, this allows selective printing and more
            self.logger(self)
            
            # stop iterating if the change in all the parameters is smaller than epsilon
            if np.all(np.abs(self.delta_parameters) < self.epsilon):
                break
        
        if self.iterations == self.max_iter:
            warnings.warn("The Functional Model did not converge, make sure you set reasonable initial parameters")
    
    def parameter_cov(self, sigma = None) -> np.ndarray:
        """
        Returns covariance matrix of the fitted parameters.
        """
        if sigma is None:
            sigma = self.m_0
        return sigma**2 * np.linalg.inv(self.normal_matrix)
    
    def parametter_corr(self) -> np.ndarray:
        cofactor_matrix = self.parameter_cov(1)
        diag = cofactor_matrix.diagonal()**0.5
        return cofactor_matrix / np.outer(diag, diag)
        
    
    def eval_cov(self, x : np.ndarray, sigma : float = None) -> np.ndarray:
        """
        Returns full covariance of model predictions at x.
        """
        A = self.get_design_matrix(x)
        return A@self.parameter_cov(sigma)@A.T
    
    def eval_cov_diags(self, x : np.ndarray, sigma : float = None) -> np.ndarray:
        """
        Returns only the diagonal of the covariance matrix of the prediction at x.
        """
        A = self.get_design_matrix(x)
        # if we only care about the diagonal, it is more efficient to compute the sum directly
        return np.einsum('ij,jk,ki -> i', A, self.parameter_cov(sigma), A.T)

    
    def chi2_statistic(self, sigma_0 : float = 1) -> float:
        """
        Computes the chi-squared test statistic.
        """
        # apply the methods as in the script chapter 4.11
        return (self.m_0 / sigma_0)**2 * self.dof
    
    def chi2_critical_value(self, alpha : float = 0.05) -> float:
        """
        Computes the chi-squared critical value, at a given significance alpha
        """
        return stats.chi2.ppf(1 - alpha, self.dof)
    
    def plot(self, sigma : float = 1, c_data = 'b', c_model = 'black'):
        """
        Plots the data points and the fitted model.
        """
        if sigma is None:
            sigma = self.m_0
        
        plt.errorbar(x = self.x, y = self.y, yerr=sigma * self.P.diagonal()**(-0.5),fmt = '.', label='data',color= c_data)
        
        #use the max of sigma_0 and m_0
        sigma = max(sigma, self.m_0)
        
        plt.errorbar(self.x, self.y_pred, self.eval_cov_diags(self.x, sigma)**0.5, fmt = '.',color= c_model, label='model prediction')
        
        linspace = np.linspace(self.x.min(), self.x.max(), 200)
        y = self.eval(linspace)
        stderr = self.eval_cov_diags(linspace, sigma)**0.5          
        plt.plot(linspace, y,color= c_model, alpha = 0.5)
        plt.fill_between(linspace, y-stderr, y+stderr,color= c_model, alpha = 0.2)
        plt.legend()
    
    def data_frame(self, sigma = None) -> pd.DataFrame:
        """
        Creates a dataframe containing x, y, weights of x, predicted y, residuals and covariance of y
        """
        return pd.DataFrame({
            'x' : self.x,
            'x_weight' : self.P.diagonal(),
            'y' : self.y,
            'y_pred' : self.y_pred,
            'residuals' : self.residuals,
            'y_cov' : self.eval_cov_diags(self.x, sigma)
        })
    
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
    
    
    def display_summary(self, sigma_0 : float = 1, sigma : float = None):
        # if sigma is not defined, choose the max of sigma_0 and m_0
        if sigma is None:
            sigma = max(sigma_0, self.m_0)
        
        # print m_0 and sigma_0
        print(f'sigma_0 = {sigma_0}')
        print(f'm_0 = {self.m_0}')
        print(f'sigma_0 / m_0 = {sigma_0 / self.m_0:.2f}')
        print(f'sigma = {sigma:.2f}')
        
        # display basic information about the model
        print(f'degrees of freedom f = n-u = {self.dof}')
        print(f'parameters:')
        display(self.parameters)
        print(f'parameter covariance:')
        display(self.parameter_cov(sigma))
        
        display(self.data_frame(sigma))
        
        # plot of the fit
        self.plot(sigma)
        plt.show()
        
        # plot the covariance matrix
        plt.title('covariance matrix of the parameters')
        mat = plt.gca().matshow(self.parameter_cov(sigma))
        plt.colorbar(mat)
        plt.show()
        
        # print the relevant statistics
        print(f'chi2 statistic : {self.chi2_statistic(sigma_0):.2f}')
        print(f'critical values for {self.dof} degrees of freedom')
        alphas = [0.1,0.05,0.01]
        for a in alphas:
            print(f'x(alpha = {a:.2f}) = {self.chi2_critical_value(a):.2f}')
        
    
class PolyFunctionalModel(FunctionalModel):
    
    # In this case the off diagonal elements of the Weighting matrix are ignored!
    degree : int
    
    def __init__(self, degree : int):
        self.degree = degree
        #set initial parameters to zero by default
        self.parameters = np.zeros(degree+1)
    
    def get_design_matrix(self, x : np.ndarray) -> np.ndarray:
        return np.column_stack([x**i for i in reversed(range(self.degree+1))])
    
    def eval(self, x):
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
    
    def eval(self, x : np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.broadcast_to(self.lambdified(*self.parameters, x), x.shape)
    
    def get_design_matrix(self, x : np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        # the collumns of the design matrix are the partial derivatives of the function with respect to each parameter
        # evaluated at the input x, and the current parameters
        # broadcastring the result is necessary because sometimes the lambdified function returns a scalar (if the function is constant)
        A_collumns = [np.broadcast_to(d(*self.parameters, x), x.shape) for d in self.differentials]
        
        return np.column_stack(A_collumns)
    
    def show_correlation(self):
        matshow = plt.matshow(self.parametter_corr())
        plt.colorbar(matshow, label = 'correlation')
        n_ticks = len(self.parameter_symbols)
        ticks = [f'${sympy.latex(s)}$' for s in self.parameter_symbols]
        plt.xlabel('parameter')
        plt.gca().xaxis.set_label_position('top')
        plt.ylabel('parameter')
        plt.xticks(range(n_ticks), ticks)
        plt.yticks(range(n_ticks), ticks)



def fit_inliers(model : FunctionalModel, x : np.ndarray, y : np.ndarray, cofactor_matrix : np.ndarray, get_inliers_args : Dict = None) -> np.ndarray:
    """
    fits a model and iteratively removes outliers via the normalized residuals.
    returns the final inliers
    """
    if get_inliers_args is None:
        get_inliers_args = {}
    
    inliers = np.ones_like(x, bool)
    
    weights = cofactor_matrix.diagonal() ** (-0.5)
    
    while True:
        inlier_cofactor_matrix = cofactor_matrix[inliers,:][:,inliers]
        weight_matrix = scipy.sparse.dia_array(np.linalg.inv(inlier_cofactor_matrix))
        model.fit(x[inliers], y[inliers], weight_matrix)
        new_inliers = get_inliers((model.eval(x) - y) * weights, **get_inliers_args)
        
        if np.all(new_inliers == inliers):
            break
        inliers = new_inliers
    
    return inliers