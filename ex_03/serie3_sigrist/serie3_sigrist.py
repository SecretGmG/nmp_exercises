# %%
import nmp_util as nmp_util
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# %%
data = pd.read_csv('SeaLevel_exclGIA.txt', names = ['date [y]','rise [mm]'], delimiter = r'\s+')
display(data)

# %%
sns.scatterplot(data = data, x = 'date [y]', y = 'rise [mm]')
plt.show()
print(f'datapoints : {len(data)}')
data = data.loc[nmp_util.get_inliers(data['rise [mm]']) & nmp_util.get_inliers(data['date [y]'])]
sns.scatterplot(data = data, x = 'date [y]', y = 'rise [mm]')
plt.show()
print(f'datapoings: {len(data)}')

# %%
rise = data['rise [mm]'].to_numpy()
date = data['date [y]'].to_numpy()

# %%
zero_order = nmp_util.PolyFunctionalModel(0)

zero_order.fit(date, rise)

plt.title('Sea level rise')
zero_order.plot()
sns.lineplot(x = date, y = zero_order.parameters[0]+zero_order.m_0, linestyle = '--',c='r')
sns.lineplot(x = date, y = zero_order.parameters[0]-zero_order.m_0, linestyle = '--', label = '$x_0\\pm m_0$',c='r')
plt.xlabel('date [y]')
plt.ylabel('rise [mm]')
plt.legend()

print(zero_order)

print(f'mean : {rise.mean()}')
print(f'standard deviation : {rise.std()}')
plt.show()

# %%
first_order = nmp_util.PolyFunctionalModel(1)

first_order.fit(date, rise)

x_center = date.mean()
y_center = rise.mean()

plt.title('Sea level rise')
first_order.plot()
sns.scatterplot(x = [x_center], y = [y_center], label = 'center')
plt.xlabel('date [y]')
plt.ylabel('rise [mm]')
plt.legend()
plt.show()

assert np.isclose(first_order.evaluate(x_center), y_center)

# %%
relative_date = date - date.min()

first_order.fit(relative_date, rise)

display(first_order)

x_center = relative_date.mean()
y_center = rise.mean()

plt.title('Sea level rise since begin of measurement')
first_order.plot()
plt.scatter(x_center, y_center, label = 'center')
plt.xlabel('date [y]')
plt.ylabel('rise [mm]')
plt.legend()
plt.show()

assert np.isclose(first_order.evaluate(x_center), y_center)

# %%
order = 6

poly_model = nmp_util.PolyFunctionalModel(order)

poly_model.fit(relative_date, rise)

display(poly_model)

plt.title('Sea level rise since begin of measurement')
poly_model.plot()
plt.xlabel('date [y]')
plt.ylabel('rise [mm]')
plt.legend()
plt.show()

# %%
m_0_alternative = np.sqrt((np.sum(rise**2) - np.sum(rise*poly_model.y_pred)) / (poly_model.dof))

print(m_0_alternative)
print(poly_model.m_0)

# %%
# compare the methods

np.random.seed(1)
x = np.random.random(1000)
y = np.random.random(1000)
degree = 7

#fit the direct model
direct = nmp_util.PolyFunctionalModel(degree, 'direct')
direct.fit(x, y)

#fit the design matrix model
design_matrix = nmp_util.PolyFunctionalModel(degree, 'design matrix')
design_matrix.fit(x, y)



assert np.allclose(direct.parameters, design_matrix.parameters), 'The methods are not equivalent!'
assert np.allclose(direct.normal_matrix, design_matrix.normal_matrix), 'The methods are not equivalent!'

print('The methods are equivalent!')

# %%
N = 500_000
test_x = np.random.rand(N)
test_y = np.random.rand(N)
order = 10

# %%
# timeit unfortunately only works in notebooks
# %timeit -n 10 -r 10 design_matrix.fit(test_x, test_y)
# %timeit -n 10 -r 10 direct.fit(test_x, test_y)

# %%
# How fast is the poly fit if we use a P matrix?
import scipy.sparse
diag_elements = np.random.rand(N)
P = scipy.sparse.diags(diag_elements)
#%timeit -r 10 design_matrix.fit(test_x, test_y, P)

try:
    P = P.todense()
    #%timeit -r 10 design_matrix.fit(test_x, test_y, P)
except Exception as e:
    print(e)
    print('P is too large to convert to dense matrix.')


