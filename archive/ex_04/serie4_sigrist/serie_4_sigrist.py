# %%
import numpy as np
import nmp_util
import sympy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from IPython.display import display

# %%
data = pd.read_csv('ex_04/serie4-observationsobs.sec', delimiter = r'\s+', names = ['I', 'z' , 's_z'], skiprows = [0])
display(data)

# %%
sns.set_theme()
plt.xlabel('I [A]')
plt.ylabel('z [1/min]')
plt.errorbar(data['I'], data['z'], data['s_z'], fmt='.', label = 'Data')
plt.legend()
plt.show()

# %% [markdown]
# # Define the functional model with sympy

# %%
a_0, a_1, a_2, sigma_1, sigma_2, mu_1, mu_2, I = sympy.symbols('a_0, a_1, a_2, sigma_1, sigma_2, mu_1, mu_2, I')

with sympy.evaluate(False):
    sqrt2pi = sympy.sqrt(2*sympy.pi)
    f_expr = \
        a_1 / (sqrt2pi * sigma_1) * sympy.exp(-(I-mu_1)**2 / (2 * sigma_1**2)) +\
        a_2 / (sqrt2pi * sigma_2) * sympy.exp(-(I-mu_2)**2 / (2 * sigma_2**2)) + a_0

parameter_symbols = [a_0, a_1, a_2, sigma_1, sigma_2, mu_1, mu_2]

display(f_expr)
print(sympy.latex(f_expr))

# %%
from typing import List

f = nmp_util.SympyFunctionalModel(f_expr, parameter_symbols, I)



models : List[nmp_util.SympyFunctionalModel] = []

def log(model : nmp_util.SympyFunctionalModel):
    print(f'Itaration:{model.iterations:2d}, m_0: {model.m_0:.10f}')
    # store these for a later exercise
    models.append(model.copy())
    
f.logger = log

for p in f.differential_expressions:
    display(p)
    print(sympy.latex(p))

# %%
sigma_0 = 1

cofactor_matrix = np.diag(data['s_z'].to_numpy()**2) / sigma_0**2
weight_matrix = sparse.dia_matrix(np.linalg.inv(cofactor_matrix))
# the default values are given in the exercise, except for a_0, simply chose 0 for a_0 and see if this converges
f.parameters = np.array([0, 50, 50, 0.05, 0.05, 0.3, 0.6])

f.fit(data['I'], data['z'], weight_matrix)

# %%
print(f'chi statistic for alpha = 5%')
print(f.m_0 / sigma_0)
print(f.dof)
print(f'functional model : {f.m_0**2 / sigma_0**2}')
print(f'critical value : {f.chi2_threshold(0.05)}')

# %%
print(f'parameters:')
f.print_parameters_latex()
print(f'parameter correlation matrix')
f.show_correlation()
plt.show()

# %%
f.plot()
plt.xlabel('I [A]')
plt.ylabel('z [1/min]')
plt.show()

# %%
def evaluate(I_val):
    z = f.eval(I_val)
    z_err = f.eval_stderr(I_val, sigma = f.m_0)[0]
    print(f'I = {I_val:.3f}')
    print(f'z = {z:.3f} +- {z_err:.3f}')

evaluate(data['I'].iloc[0])
print()
evaluate(0.12)
print()

factor_expr = a_1 / (sqrt2pi * sigma_1)

print(sympy.latex(factor_expr))

factor, factor_cov = nmp_util.propagate_error(factor_expr, parameter_symbols, f.parameters, f.m_0**2 * f.parameter_cof())
factor_err = factor_cov[0,0]**0.5
print(f'factor = {factor:.3f} +- {factor_err:.3f}')

# %%
true_residuals = f.residuals
linearized_residuals = f.A @ f.parameters - f.y
i = 8

other_model = models[i]

# this checks if the design matrix is a good taylor approximation
linearized_residuals = other_model.residuals - other_model.A @ (f.parameters - other_model.parameters)

# i+1 because we the first iteration is 1, but the arrays are 0-indexed
plt.title('Deviations from the true residual at the last iteration (i = 10)')
plt.scatter(f.x, true_residuals-linearized_residuals, marker='o', label = f'linearized residuals')
plt.scatter(f.x, true_residuals-true_residuals,marker = '.', label = f'true residuals (i = {len(models)})')
plt.scatter(f.x, true_residuals-other_model.residuals,marker ='x', label = f'residuals (i = {i+1})')
plt.xlabel('I [A]')
plt.ylabel('$\\Delta v$ [1/min]')
plt.legend()
plt.show()

# %%
residual_covariance = f.m_0**2 * (cofactor_matrix - f.eval_cof(f.x))

normalized_residuals = f.residuals / np.diag(residual_covariance)**0.5
plt.scatter(data['I'], normalized_residuals, label = 'normalized residuals')

plt.xlabel('I [A]')
plt.ylabel('$v_i / \\sigma_i$')
plt.legend()
print(normalized_residuals.std(ddof=1))

print(f'no outliers detected: {nmp_util.get_inliers(normalized_residuals).all()}')


