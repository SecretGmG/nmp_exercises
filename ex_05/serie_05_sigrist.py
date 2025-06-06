# %%
import numpy as np
import pandas as pd
import nmp_util
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# %%
mjd = pd.read_csv('GRCAT07001ACC.sec', delimiter = r'\s+', names = ['mjd'], skiprows = 8, nrows=1)
data_cols = ['R_lin','S_lin','W_lin','R_ang','S_ang','W_ang']
data = pd.read_csv('GRCAT07001ACC.sec', delimiter = r'\s+', names = ['dmjd'] + data_cols, skiprows = 9)
date = nmp_util.mjd_to_datetime(data['dmjd'] + mjd['mjd'][0])
data['date'] = date

# %%
display(data)

# %%
# sample every 10th entry
data_sample = data.loc[::10]
display(data_sample)

# %%
def analize_data(date : np.ndarray, original : np.ndarray, window : int, edge):
    # Ctop the date and data depending on how the edge is handled
    cropped = original[window// 2:-window // 2+1] if edge == 'none' else original
    cropped_date = date[window // 2:-window // 2+1] if edge == 'none' else date

    def plot_amplitude_spectrum(data,label, d = 10, max_period = 30, exponent = 1):
        freq, amplitude = nmp_util.amplitude_spectrum_via_numpy(data, d = d)
        # only use positive frequencies with a period smaller than max_period
        mask = (freq < 1/max_period) & (freq > 0)
        # convert to units of minutes
        plt.plot(1/(60*freq[mask]), amplitude[mask]**exponent, label=label)

    for i in range(3):
        # Low-pass filter

        l = nmp_util.savitzki_golay_filter(original, deg = i, window = window, edge=edge)
        h = cropped - l

        # plot the entire data (filtered and original)
        plt.figure(figsize=(10, 5))
        plt.scatter(date, original, label=f'Original', s=1)
        plt.scatter(cropped_date, h, label=f'High-pass q={i}', s=1)
        plt.scatter(cropped_date, l, label=f'Low-pass q={i}', s=1)
        plt.xlabel('Date')
        plt.ylabel('Acceleration [mm/s$^2$]')
        plt.legend()
        plt.show()
        
        # create zoom for one hour (360 entries)
        s, e = 0, 360
        plt.figure(figsize=(10, 5))
        plt.scatter(date[s:e], original[s:e], label=f'Original', s=1)
        plt.scatter(cropped_date[s:e], h[s:e], label=f'High-pass q={i}', s=1)
        plt.scatter(cropped_date[s:e], l[s:e], label=f'Low-pass q={i}', s=1)
        plt.xlabel('Date')
        plt.ylabel('Acceleration [mm/s$^2$]')
        plt.legend()
        plt.show()
        
        # plot the amplitude spectrum
        
        plt.figure(figsize=(10, 5))
        plot_amplitude_spectrum(original, 'Original')
        plot_amplitude_spectrum(l, f'Low-pass q={i}')
        plot_amplitude_spectrum(h, f'High-pass q={i}')
        plt.xlabel('Period [min]')
        plt.ylabel('Amplitude [mm/s$^2$]')
        plt.xscale('log') 
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        plot_amplitude_spectrum(original, 'Original',  exponent = 2)
        plot_amplitude_spectrum(l, f'Low-pass q={i}',  exponent = 2)
        plot_amplitude_spectrum(h, f'High-pass q={i}', exponent = 2)
        plt.xlabel('Period [min]')
        plt.ylabel('Power [mm^2/s$^4$]')
        plt.xscale('log') 
        plt.legend()
        plt.show()

# %% [markdown]
# # Note:
# the filters for q=0 and q=1 are equivalent, but only because i evaluate at x=0, and the data points are equidistant!

# %%
#analize_data(data_sample['date'], data_sample['S_lin'], window = 61, edge = 'none')

# %%
analize_data(data_sample['date'], data_sample['S_lin'], window = 61, edge = 'polynomial')

# %%
def fourier_analize(data, m, dt, use_functional_model = False):
    # get amplitudes via my own DFT implementation
    if use_functional_model:
        f = nmp_util.DFT_FunctionalModel(m)
        f.fit(np.arange(len(data)), data)
        coeffs = f.parameters
        amplitudes = f.amplitudes()
    else:
        coeffs = nmp_util.discrete_fourier_transform(data, m)
        amplitudes = nmp_util.coeffs_to_amplitude(coeffs)

    # and via numpy
    freqs, amplitudes_np = nmp_util.amplitude_spectrum_via_numpy(data, m, d=dt/60)
    
    m_max = np.argmax(amplitudes)
    print(f'n_max = {m_max}')
    print(f'perid = {1/freqs[m_max]} [min]')
    print(f'amplitude = {amplitudes[m_max]} [mm$^2$/s]')
    print(f'coefficients a = {coeffs[m_max]} [mm/s$^2$]')
    print(f'coefficients b = {coeffs[m_max + m]} [mm/s$^2$]') 
    plt.figure(figsize=(10, 5))
    plt.title(f'Fourier Analysis m = {m}')
    plt.scatter(freqs, amplitudes, label='Amplitudes',s=10, marker='x')
    plt.scatter(freqs, amplitudes_np, label='Amplitudes numpy',s=5)
    plt.xlabel('frequency [1/min]')
    plt.ylabel('amplitude [mm$^2$/s]')
    plt.legend()
    plt.show()

# %%
# analize the data with different m values
fourier_analize(data_sample['S_lin'], 30, dt = 10)
fourier_analize(data_sample['S_lin'], 120, dt = 10)
fourier_analize(data_sample['S_lin'], 720, dt = 10)
fourier_analize(data_sample['S_lin'], 1440, dt = 10)

# %% [markdown]
# # performance test
# 
# Unsurprisingly numpy is way faster. The funcitonal model and direct approach do show a difference, but it's not drastic.

# %%
def time_functional():
    f = nmp_util.DFT_FunctionalModel(720)
    f.fit(np.arange(len(data_sample['S_lin'])), data_sample['S_lin'])
def time_numpy():
    nmp_util.amplitude_spectrum_via_numpy(data_sample['S_lin'], 720, d=10/60)
def time_dft():
    nmp_util.discrete_fourier_transform(data_sample['S_lin'], 720)
%timeit time_functional()
%timeit time_dft()
%timeit time_numpy()


