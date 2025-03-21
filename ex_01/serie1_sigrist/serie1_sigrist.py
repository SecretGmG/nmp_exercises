# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
# each value is equally likely, therefore the probability of the outcomes is given by
outcomes = np.array([a+b+2 for a in range(6) for b in range(6)])
values = np.arange(2,13)
probabilities = np.histogram(outcomes, bins=np.arange(2,14), density=True)[0]

# we can now make the table
probabilities_table = pd.DataFrame({'Wert': values, 'Wahrscheinlichkeit': probabilities})
print(probabilities_table.to_latex(index=False))

# plot the probabilities
plt.xticks(values)
plt.xlabel('Wert')
plt.ylabel('Wahrscheinlichkeit')
plt.title('Wahrscheinlichkeitsverteilung der Summe zweier Würfel')
plt.bar(values, np.cumsum(probabilities), label = 'Verteilungsfunktion')
plt.bar(values, probabilities, label = 'Dichtefunktion', width=0.6)
plt.legend()
plt.show()

# calculate the expected value and variance
expected_value = outcomes.mean()
variance = outcomes.var(ddof=0)
print(f'Erwartungswert: {expected_value}')
print(f'Varianz: {variance}')


# %%
data = pd.read_csv('obs_res_LARES.dat', names = ['time', 'residuals'], index_col=False, sep='\s+')

#remove the dashes '---'
data.drop(index = [0,1], inplace=True)
data['time'] = data['time'].astype(float)
data['residuals'] = data['residuals'].astype(float)
print(data.head())

# %%
def get_inliers(data : np.ndarray, max_std : float= 4.0, iterative : bool = True):
    
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

# %%
from astropy.time import Time

time_values = Time(val = data['time'], format = 'mjd')

data['datetime'] = pd.DatetimeIndex(time_values.to_datetime())

for month, month_name in zip([9,10,11], ['September', 'Oktober', 'November']):
    
    data_of_the_month = data.loc[data['datetime'].dt.month == month]
    inliers = get_inliers(data_of_the_month['residuals'])
    filtered_data = data_of_the_month.loc[inliers]
    
    plt.figure()
    plt.title(month_name)
    
    
    plt.scatter(data_of_the_month['datetime'], data_of_the_month['residuals'], label = 'raw data', s=3)
    plt.scatter(filtered_data['datetime'], filtered_data['residuals'], label = 'filtered data', s=3)
    plt.xticks(rotation = 30)
    plt.xlabel('date [date]')
    plt.ylabel('residual [mm]')
    plt.legend()
    plt.grid()
    plt.show()
    print(f'Analyse der daten vom {month_name}:')
    print(f'Mittelwert : {filtered_data["residuals"].mean():.3f}')
    
    # Formel aus skript S.8
    uncertainty = filtered_data["residuals"].std() / np.sqrt(len(filtered_data))
    
    print(f'Unsicherheit des Mittelwerts : {uncertainty:.3f}')


