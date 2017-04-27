import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# computes error for an LRC circuit with L and C in parallel
data = pd.DataFrame.from_csv('data.csv')

R = 99.75
L = .2224e-3
L_ESR = .151
L_PC = 1/((530e3)**2 * L)
C = 10.351e-6
C_ESR = .850

# Exp resonance 3.332 kHz
# Est resonance 3.336 kHz

def theoretical(freq):
    omega = 2 * np.pi * freq
    Z_L = ((L_ESR + 1j * omega * L)**-1 + (1j / (omega * L_PC))**-1)**-1
    Z_C = C_ESR + (-1j/(omega * C))
    Z_LC = (Z_L**-1 + Z_C**-1)**-1
    # Z_LC = (Z_L**-1 + Z_C**-1)**-1
    Z_tot = Z_LC + R
    V_in = 1.0
    V_out = V_in * (Z_LC)/(Z_tot)
    return np.absolute(V_out), np.angle(V_out, deg=True)
data['th_gain'], data['th_phase'] = np.vectorize(theoretical)(data['freq'])
data['th_phase'] = -data['th_phase']
data['th_gain'] = 20 * np.log10(data['th_gain'])

# plt.semilogx(data['freq'], (data['th_gain'] - data['gain'])/data['th_gain'])
# plt.semilogx(data['freq'], data['th_phase'] - data['phase'])
# plt.show()

# freq = np.logspace(np.log10(1e3), np.log10(1e6), 100)
# gain, phase = theoretical(freq)
# gain = 20 * np.log10(gain)

f, axarr = plt.subplots(2, sharex=True)

axarr[0].semilogx(data['freq'], data['th_gain'], label='theoretical')
axarr[0].semilogx(data['freq'], data['gain'], label='experimental')
axarr[0].legend()
axarr[0].grid()
axarr[0].set_ylabel('Gain')
# axarr[0].set_title('Gain')

axarr[1].semilogx(data['freq'], data['th_phase'], label='theoretical')
axarr[1].semilogx(data['freq'], data['phase'], label='experimental')
# axarr[1].legend()
axarr[1].grid()
axarr[1].set_ylabel('Phase [deg]')
axarr[1].set_xlabel('Frequency [Hz]')
# axarr[1].set_title('Phase')

plt.show()