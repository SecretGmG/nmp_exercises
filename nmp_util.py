"""
Module containing utility functions and classes for the NMP course.
I worked through the course and collected useful functions and classes here.

For examm preparation, i will do the same with old exams, so that i can use them as a reference.
"""
from abc import ABC, abstractmethod
from copy import deepcopy
import warnings
from typing import Callable, Iterable, Tuple, List, Literal
import matplotlib.pyplot as plt
import scipy.sparse
import sympy
from scipy import stats
import numpy as np

#improve readability of the output of numpy arrays in jupyter notebooks
np.set_printoptions(linewidth=150)


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

def mjd_to_datetime(
    mjd: np.ndarray,
    start: np.datetime64 = None
) -> np.ndarray:
    """
    Parses Modified Julian Date (MJD) to datetime64[ns] or timedelta64[ns].

    Args:
        mjd: Array of MJD values (float or int).
        start: Optional custom start date.
               If None, defaults to MJD epoch (1858-11-17).
               If 0 returns timedelta64[ns] from MJD epoch.

    Returns:
        Array of datetime64[ns] or timedelta64[ns] values.
    """
    mjd_ns = (np.asarray(mjd, dtype=np.float64) * 86_400_000_000_000).astype("timedelta64[ns]")

    if start is None:
        start = np.datetime64("1858-11-17T00:00:00", "ns")

    return start + mjd_ns


## SERIE 2

def error_propagation_formula(f : sympy.Matrix|Iterable[sympy.Expr]|sympy.Expr, args : List[sympy.Symbol]) -> Tuple[sympy.Expr, sympy.MatrixSymbol]:
    """
    Computes symbolic error propagation A K A^T and returns it with symbolic covariance K.
    should not be used direkctly most of the time, use propagate_error instead.

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
    # these are sympy expressions, so we need the asteriks * to multiply them
    # this is the same as A @ K @ A.T in numpy, but sympy doesn't support the @ operator
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


## Serie 3&4
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
    parameter_symbols : List[sympy.Symbol] = None

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
    P : scipy.sparse.spmatrix # weighting matrix

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


    def fit(self, x : np.ndarray, y : np.ndarray, weight_matrix: scipy.sparse.spmatrix|None = None):
        """
        Fits model to data using iterative weighted least squares.
        """

        if self.parameters is None:
            raise ValueError('Cannot fit, because self.parameters is None, set your initial parameters!')

        self.x = np.asarray(x)
        self.y = np.asarray(y)

        # by default use the identity matrix as weight matrix
        if weight_matrix is None:
            weight_matrix = scipy.sparse.diags(np.ones(len(x)))
        weight_matrix = scipy.sparse.csr_matrix(weight_matrix)

        self.P = weight_matrix

        # to start the iteration process the model needs to be evaluated at the initial parameters
        self.y_pred = self.eval(x)

        for i in range(self.max_iter):
            self.iterations = i+1

            self.A = self.get_design_matrix(x)

            self.normal_matrix = self.A.T @ self.P @ self.A
            # use P.dot() since it is more efficient here to compute from right to left
            # because the intermediate matrix doesn't need to be stored and the sparse matrix might improve performance
            self.b = self.A.T @ self.P.dot(self.y - self.y_pred)

            # use lstsq instead of inv to avoid computing the inverse and handle singular normal matrix
            self.delta_parameters = np.linalg.lstsq(self.normal_matrix, self.b)[0]

            # Update the parameters and associated values
            self.parameters = self.parameters + self.delta_parameters
            self.y_pred = self.eval(self.x)
            self.residuals = self.y - self.y_pred
            self.m_0 = np.sqrt((self.residuals.T @ self.P @ self.residuals) / (self.dof))

            self.logger(self)

            if np.all(np.abs(self.delta_parameters) < self.epsilon):
                break

        if not np.all(np.abs(self.delta_parameters) < self.epsilon):
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
        Returns the critical value for the reduced chi-squared test at significance level alpha.

        This value is (X^2 / dof), where X^2 is the critical value from the chi-squared distribution with dof = degrees of freedom.

        To test the model, compare the ratio m_0^2 / sigma_0^2 (i.e. the reduced chi-squared statistic) to this threshold.
        If it is greater, the model is rejected at significance level alpha.

        Note: Smaller alpha means stricter evidence is required to reject the model,
        not a tolerance for larger errors.
        """
        return stats.chi2.ppf(1 - alpha, self.dof) / self.dof

    def plot_prediction(self, sigma = None, errorbar = True, kwargs = None):
        """
        Plots the model prediction with error bars.
        """
        if sigma is None:
            sigma = self.m_0
        if kwargs is None:
            kwargs = {'marker' : '.', 'label': 'model prediction', 'color': 'black', 'alpha': 0.5}

        if not errorbar:
            plt.scatter(self.x, self.y_pred, **kwargs)
        else:
            y_stderr = self.eval_stderr(self.x, sigma)
            plt.errorbar(self.x, self.y_pred, y_stderr, linestyle = '', **kwargs)

    def plot_prediction_smooth(self, sigma = None, errorband = True, n_points = 200, plt_kwargs = None, fill_kwargs = None):
        """
        Plots the model prediction as a smooth line with error band.
        """
        if sigma is None:
            sigma = self.m_0
        if plt_kwargs is None:
            plt_kwargs = {'color': 'black', 'label': 'model prediction', 'alpha': 0.5}
        if fill_kwargs is None:
            fill_kwargs = {'color': 'black', 'alpha': 0.2, 'label': 'error band'}

        linspace = np.linspace(self.x.min(), self.x.max(), n_points)
        y = self.eval(linspace)
        plt.plot(linspace, y, **plt_kwargs)

        if errorband:
            eval_stderr = self.eval_stderr(linspace, sigma)
            plt.fill_between(linspace, y-eval_stderr, y+eval_stderr,**fill_kwargs)

    def show_correlation(self, parameter_ticks : bool = None):
        """
        Plots the correlation matrix of the parameters.
        """
        matshow = plt.matshow(self.parameter_corr())
        plt.colorbar(matshow, label = 'correlation')

        # set the ticks to the parameter symbols if they are available
        # and the number of parameters is small enough
        if parameter_ticks is None:
            if self.parameter_symbols is None:
                parameter_ticks = False
            else:
                parameter_ticks = len(self.parameter_symbols) < 10

        if parameter_ticks:
            n_ticks = len(self.parameter_symbols)
            ticks = [f'${sympy.latex(s)}$' for s in self.parameter_symbols]
            plt.xticks(range(n_ticks), ticks)
            plt.yticks(range(n_ticks), ticks)

        plt.gca().xaxis.set_label_position('top')
        plt.xlabel('parameter')
        plt.ylabel('parameter')

    def print_parameters(self, precision : int = 3, sigma : float = None):
        """prints the parameters and their uncertainties in a human-readable format.

        Args:
            precision int: Defaults to 3.
            sigma float: Defaults to None.
        """
        if sigma is None:
            sigma = self.m_0
        errors = sigma*np.sqrt(np.diag(self.parameter_cof()))
        for i, (v, e) in enumerate(zip(self.parameters, errors)):
            if self.parameter_symbols is not None:
                s = self.parameter_symbols[i]
            else:
                s = sympy.Symbol(f'a_{i}')
            print(f"{s} = {v:.{precision}f} ± {e:.{precision}f}")

    def print_parameters_latex(self, precision : int = 3, sigma : float = None):
        """
        Prints the parameters and their uncertainties in LaTeX format.
        sigma is the scale of the covariance, default is m_0.
        Args:
            precision: number of decimal places to print, default is 3.
        """
        if sigma is None:
            sigma = self.m_0
        errors = sigma*np.sqrt(np.diag(self.parameter_cof()))
        for i, (v, e) in enumerate(zip(self.parameters, errors)):
            if self.parameter_symbols is not None:
                s = self.parameter_symbols[i]
            else:
                s = sympy.Symbol(f'a_{i}')
            print(f"${sympy.latex(s)} = \\SI{{{v:.{precision}f}({e:.{precision}f})}}$")

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

        self.parameter_symbols = [sympy.Symbol(f'a_{i}') for i in reversed(range(degree+1))]

        self.max_iter = 1 #this is a linear model, so we only need one iteration to fit the parameters
        self.epsilon = np.inf # we don't need to check for convergence, since we only do one iteration

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

    def __init__(self, function_expr : sympy.Expr, parameter_symbols : List[sympy.Symbol], feature_symbol : sympy.Symbol):
        self.function_expr = function_expr
        self.parameter_symbols = parameter_symbols
        self.feature_symbol = feature_symbol

        self.lambdified = sympy.lambdify([*parameter_symbols, self.feature_symbol], self.function_expr)

        # compute the partial derivative of the function with respect to each parameter
        self.differential_expressions = [sympy.diff(self.function_expr, a) for a in parameter_symbols]
        self.differentials = [sympy.lambdify([*parameter_symbols, self.feature_symbol], diff) for diff in self.differential_expressions]

        #set initial parameters to zero by default
        self.parameters = np.zeros(len(parameter_symbols))

    def eval(self, x : np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.broadcast_to(self.lambdified(*self.parameters, x), x.shape)

    def get_design_matrix(self, x : np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        # the columns of the design matrix are the partial derivatives of the function with respect to each parameter
        # evaluated at the input x, and the current parameters
        # broadcasting the result is necessary because sometimes the lambdified function returns a scalar (if the function is constant)
        A_columns = [np.broadcast_to(d(*self.parameters, x), x.shape) for d in self.differentials]

        return np.column_stack(A_columns)

# Serie 5
def savitzki_golay_filter(y: np.ndarray, deg: int = 1, window: int = None, edge: Literal['pad', 'polynomial', 'none'] = 'polynomial') -> np.ndarray:
    """
    Applies a Savitzki-Golay filter to the input data y.
    Args:
        y: Input data to be filtered.
        deg: Degree of the polynomial to fit, default is 1 (linear).
        window: Size of the window to use for the filter, must be odd. If None, it will be set to deg*2+1.
        edge: How to handle the edges of the data. Options are 'pad', 'polynomial', or 'none'.
            'pad' will pad the data with the edge values,
            'polynomial' will use the polynomial coefficients of the first and last window to compute the start and end of the filtered signal.
            'none' will not handle the edges and return only the valid part of the convolution. in this case the return size will be len(y) - window + 1.
    Returns:
        filtered: The filtered data.
    """

    if window is None:
        window = deg*2+1

    assert window % 2 == 1, "Window size must be odd"

    radius = window // 2

    A = np.array([[j**i for i in range(deg+1)] for j in range(-radius, radius+1)])
    B = np.linalg.inv(A.T @ A) @ A.T

    match edge:
        case 'pad':
            y = np.pad(y, (radius, radius), mode='edge')
            filtered = np.convolve(B[0,:], y, mode='valid')
        case 'polynomial':
            # use the polynomial coefficients to compute the start and end of the filtered signal
            unproblematic_part = np.convolve(B[0,:], y, mode = 'valid')
            start_params = B @ y[:window]
            end_params = B @ y[-window:]
            start = np.polyval(start_params[::-1], np.arange(-radius, 0))
            end = np.polyval(end_params[::-1], np.arange(1, radius+1))
            filtered = np.concatenate((start, unproblematic_part, end))
        case 'none':
            filtered = np.convolve(B[0,:], y, mode = 'valid')

    return filtered


def fft_to_coeffs(fft : np.ndarray, m = None) -> np.ndarray:
    """
    Converts the FFT coefficients to the coefficients of the polynomial.
    The coefficients are returned in the order a_0, a_1, ..., a_m, b_1, b_2, ... b_m
    """

    fft /= len(fft)
    if m is None:
        m = len(fft) // 2
    coeffs = np.zeros(2*m+1)
    coeffs[0] = fft[0].real
    coeffs[1:m+1] = 2*fft[1:m+1].real
    coeffs[m+1:] = -2*fft[1:m+1].imag
    return coeffs

def coeffs_to_amplitude(coeffs : np.ndarray) -> np.ndarray:
    """
    Converts the coefficients of the polynomial to the amplitudes of the Fourier series.
    The coefficients are assumed to be in the order a_0, a_1, ..., a_m, b_1, b_2, ... b_m
    """
    m = len(coeffs) // 2
    assert len(coeffs) == 2*m+1, "coeffs must be of length 2*m+1"
    return np.concatenate((coeffs[0:1], np.sqrt(coeffs[1:m+1]**2 + coeffs[m+1:]**2)))

def amplitude_spectrum_via_numpy(y : np.ndarray, m : int = None, d : float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the amplitude spectrum of the data using numpy's fft.
    The amplitude is normalized by the length of the data, therefore corresponding to the factors of the discrete fourier transform.
    Returns:
        frequencies: Frequencies of the amplitude spectrum.
        amplitudes: Amplitudes of the amplitude spectrum.
    """
    if m is None:
        m = len(y) // 2
    fft = np.fft.fft(y)
    coeffs = fft_to_coeffs(fft, m)
    amplitudes = coeffs_to_amplitude(coeffs)
    frequencies = np.arange(m+1) / (d * len(y))
    return frequencies, amplitudes

def discrete_fourier_transform(y : np.ndarray, m) -> np.ndarray:
    """
    Computes the discrete fourier transform of the data using a design matrix.
    The coefficients are returned in the order a_0, a_1, ..., a_m, b_1, b_2, ... b_m
    """
    x = np.arange(len(y))
    base_frequency = 2*np.pi/len(y)
    # compute the design matrix
    A = np.column_stack(
            [np.cos(i*x*base_frequency) for i in range(0,m+1)]+
            [np.sin(i*x*base_frequency) for i in range(1,m+1)]
        )

    # compute the coefficients
    coeffs = np.linalg.inv(A.T @ A) @ A.T @ y

    return coeffs

class DFT_FunctionalModel(FunctionalModel):
    """
    Functional model for discrete fourier transform.
    """

    def __init__(self, m : int):
        self.m = m
        self.parameters = np.zeros(2*m+1)
        self.parameter_symbols = [sympy.Symbol(f'a_{i}') for i in range(m+1)] + [sympy.Symbol(f'b_{i}') for i in range(1,m+1)]
        self.max_iter = 1 #this is a linear model, so we only need one iteration to fit the parameters
        self.epsilon = np.inf # we don't need to check for convergence, since we only do one iteration

    def eval(self, x : np.ndarray) -> np.ndarray:
        return sum(
            self.parameters[i] * np.cos(i*x*2*np.pi/len(x)) for i in range(self.m+1)
        ) + sum(
            self.parameters[i] * np.sin(i*x*2*np.pi/len(x)) for i in range(self.m+1,2*self.m+1)
        )

    def get_design_matrix(self, x : np.ndarray) -> np.ndarray:
        base_frequency = 2*np.pi/len(x)
        # compute the design matrix
        A = np.column_stack(
            [np.cos(i*x*base_frequency) for i in range(0,self.m+1)]+
            [np.sin(i*x*base_frequency) for i in range(1,self.m+1)]
        )
        return A

    def amplitudes(self) -> np.ndarray:
        """
        Returns the amplitudes of the discrete fourier transform.
        """
        return coeffs_to_amplitude(self.parameters)

    def amplitude_cof(self) -> np.ndarray:
        """
        Returns the cofactor matrix of the amplitudes.
        """
        # compute the covariance matrix of the parameters
        parameter_cof = self.parameter_cof()

        exprs = [self.parameters[0]] + [
            sympy.sqrt(self.parameter_symbols[i]**2 + self.parameter_symbols[self.m+1+i]**2) for i in range(1,self.m)
        ]
        # compute the standard errors of the amplitudes
        return propagate_error(exprs, self.parameter_symbols, self.parameters, parameter_cof)[1]