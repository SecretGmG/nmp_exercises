from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style()
import sympy
import scipy, scipy.linalg, scipy.stats, scipy.sparse
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
    Class to hold relevant parameters for linear models.
    """
    parameters : np.ndarray
    rse : float # Residual Standard Error = m_0
    normal_matrix : np.ndarray
    f : int # degrees of freedom = n - u
    residuals : np.ndarray
    
    @property
    def cofactor_matrix(self) -> np.ndarray:
        return scipy.linalg.inv(self.normal_matrix)

    @property
    def covariance_matrix(self) -> np.ndarray:
        return self.rse**2 * self.cofactor_matrix
    
    def chi2test(self, sigma_0 : float = 1, alpha : float = 0.05) -> bool:
        """
        Computes the chi-squared test statistic.

        Args:
            sigma_0 (float): Expected standard deviation. Default is 1, 
                assuming the Cofactor matrix was equal to the Covariance matrix of the data.
            alpha (float): Significance level. Default is 0.05.

        Returns:
            Tuple:
                - chi2_stat (float): Chi-squared statistic.
                - bool: True if the chi-squared test passes, False otherwise.
        """
        # apply the methods as in the script chapter 4.11
        chi2_stat = (self.rse / sigma_0)**2
        critical_value = scipy.stats.chi2.ppf(1 - alpha, self.f) # In the script called x_{1-alpha}
        return chi2_stat, chi2_stat * self.f < critical_value
    
    
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
    # reversed order to comply with np.polyval
    # since np.polyval expects the coefficients in decreasing order of power
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
    
    # compute the partial derivative of the function with respect to each parameter
    differentials = [sympy.lambdify([feature_sym], sympy.diff(f, a)) for a in parameters]
    
    # the collumns of the design matrix are the partial derivatives of the function with respect to each parameter
    # evaluated at the input x
    design_matrix_columns = [np.broadcast_to(d(x), x.shape) for d in differentials]
    
    design_matrix = np.column_stack(design_matrix_columns)
    return design_matrix

def normal_equation(A : np.ndarray, y : np.ndarray, P : scipy.sparse.spmatrix | None = None) -> LinearModelResult:
    """
    Computes the parameters of a linear model using the normal equation method.
    
    Args:
        A (np.ndarray): Design matrix.
        y (np.ndarray): Target variable.
        P (scipy.sparse.spmatrix | None) The covariance of the data, if None uses identity default is None
    
    Returns:
        LinearModelResult: Parameters of the linear model.
    """
    
    if P == None:
        b = A.T@y
        N = A.T@A
    else:
        b = A.T@P@y
        N = A.T@P@A
    
    parameters = np.linalg.solve(N, b)
    residuals =  A@parameters - y
    
    if P == None:
        m_0_sr = (residuals.T@residuals) / (len(y) - len(parameters))
    else:
        m_0_sr = (residuals.T@P@residuals) / (len(y) - len(parameters))
    
    return LinearModelResult(
        parameters,
        m_0_sr**0.5,
        N,
        len(y) - len(parameters),
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

def poly_fit(x : np.ndarray, y : np.ndarray, degree : int, P : scipy.sparse.spmatrix | None = None, method : Literal['design matrix','direct'] = 'direct') -> LinearModelResult:
    """
    Fits a polynomial of a given degree to the data using the normal equation method.
    If the method 'direct' is used it does not generate a design matrix, saving memory.

    Args:
        x (np.ndarray): input features.
        y (np.ndarray): target variable.
        degree (int): degree of the polynomial to fit.
        P (scipy.sparse.spmatrix | None): The covariance of the data, if None uses identity, default is None.
        method Literal: method to use for fitting.
            - 'design matrix': uses the design matrix method.
            - 'direct': uses the direct method.

    Returns:
        LinearModelResult: Parameters of the polynomial fit.
    """
    
    if method == 'design matrix':
        #use the design matrix method
        A : np.ndarray = poly_design_matrix(x, degree)
        return normal_equation(A, y, P)
    
    if method == 'direct':
        assert P == None, "P not supported for direct method"
        
        N, b = _get_N_b(x, y, degree)
    
        parameters = np.linalg.solve(N, b)
        residuals =  np.polyval(parameters, x) - y
        m_0_sr = (residuals.T @ residuals) / (len(y) - len(parameters))
    
        return LinearModelResult(
            parameters, 
            m_0_sr**0.5, 
            N, 
            len(y) - len(parameters),
            residuals)
    
    raise Exception(f'method {method} not supported')