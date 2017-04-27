#!/bin/python2
from optparse import OptionParser
from scipy.signal import butter, lfilter
import scipy.optimize as opt
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import visa
import time

def bandpass(data, width, dt, frequency):
    nyqu_f = 1.0/(2*dt)
    frequency = frequency/nyqu_f
    b, a = butter(9, [frequency * (1.0-width), frequency * (1.0+width)], btype='bandpass')
    return lfilter(b, a, data)

# Carries out a linear least squares fit and converts to polar coordinates
def fit_sine(t, y, freq):
    t = t.astype(np.float64)
    y = y.astype(np.float64)

    X = np.ndarray(shape=(len(y),2))
    X[:,0] = np.sin(2*np.pi*freq*t)
    X[:,1] = np.cos(2*np.pi*freq*t)

    fit = la.solve(np.matmul(X.T,X), np.matmul(X.T,y.T))

    fit = {'mag':la.norm(fit), 'phase':np.arctan2(fit[1], fit[0])}

    return fit

# Sets timebase to get approximately 2 periods
def set_base(channel):
    channel = 'CH'+str(channel)

    # Set trigger to target channel and Get trigger frequency and set timebase to capture at least a full get_wave
    freq = np.float(tek.query(':TRIG:MAIN:EDGE:SOURCE '+channel+';:TRIG:MAIN:FREQ?'))
    mintimebase = 1.0/freq / 4
    tek.write('HORIZONTAL:MAIN:SCALE '+str(mintimebase))

# Set vertical scale
def set_scale(channel):
    channel = 'CH'+str(channel)
    # Set maximum voltage and measure to scale to new voltage
    tek.write(':'+channel+':SCALE 5')
    tek.write(':MEASU:MEAS1:SOURCE '+channel)
    tek.write(':MEASU:MEAS1:TYPE PK2PK')
    time.sleep(1)
    voltage = np.float(tek.query(':MEASU:MEAS1:VAL?'))
    tek.write(':'+channel+':SCALE '+str(voltage/8))

# Get waveform
def get_wave(channel):
    channel_str = 'CH'+str(channel)

    for attempt in range(2):
        # Capture data
        tek.write(':DATA:SOURCE '+channel_str)
        wavescale = np.array(tek.query('WFMPRE:YMULT?;XINC?').split(';')).astype(np.float)
        ymulti = wavescale[0]
        xincr = wavescale[1]
        # print('Capturing '+channel_str)
        data = tek.query_binary_values(':CURVE?', datatype=u'h', container=np.array).astype(np.float)
        # print('Capture Finished.')
        datamax = np.absolute(data).max()
        rangemax = np.iinfo(np.int16).max

        if (datamax < rangemax * 0.5 and np.float(tek.query(':'+channel_str+':SCALE?')) <= 25e-3) or \
           (datamax > rangemax * 0.5 and datamax < rangemax * 0.95):
            break
        set_scale(channel)

    data *= ymulti
    return data, xincr

# Get gain and phase shift at a particular frequency
def get_point(freq):
    gen.write('FREQ '+str(freq))
    time.sleep(1)
    set_base(1)

    data, dt = get_wave(1)
    data = pd.DataFrame.from_records({'y':data, 't':np.arange(len(data)) * dt})
    # plt.plot(data['t'], data['y'])
    x_1 = fit_sine(data['t'], data['y'], freq)

    data, dt = get_wave(2)
    data = pd.DataFrame.from_records({'y':data, 't':np.arange(len(data)) * dt})
    # plt.plot(data['t'], data['y'])
    x_2 = fit_sine(data['t'], data['y'], freq)

    # dB
    gain = 20*np.log10(np.abs(x_2['mag']/x_1['mag']))
    # gain = np.abs(x_2['mag']/x_1['mag'])
    phase =  (x_1['phase']-x_2['phase'] + np.pi) % (2*np.pi) - np.pi
    phase *= 180./np.pi
    # print(x_1, x_2)
    print('Frequency ', freq)
    print('Gain ', gain)
    print('Phase [deg]', phase)
    return {'freq':freq, 'gain':gain, 'phase':phase}


rm = visa.ResourceManager('@py')
print(rm.list_resources())
print('Starting acquisition at '+str(datetime.datetime.now()))

# Open Oscilloscope (Tektronix TDS1002B)
tek = rm.open_resource('USB0::1689::867::C050569::0::INSTR')

tek.timeout = 1000

# Oscilloscope will not respond if it has not finished last set of commands
tek.write(':LOCK ALL')
tek.write(':DATA:ENCDG SRIbinary;WIDTH 2')
time.sleep(.1)
tek.write(':DATA:START 1;STOP 2500')
tek.write(':DISPLAY:PERS OFF')

time.sleep(.1)
tek.write(':ACQUIRE:MODE AVE;NUMAV 128') #Average 4, 16, 64, or 128
# tek.write('ACQUIRE:MODE SAMPLE')
tek.write(':WFMPRE:YOFF 0')

time.sleep(.1)
tek.write('SEL:CH1 ON;CH2 ON')
time.sleep(.1)
# tek.write(':CH1:COUPLING AC;POSITION 0;PROBE 1')
# tek.write(':CH2:COUPLING AC;POSITION 0;PROBE 1')
tek.write(':CH1:COUPLING AC;POSITION 0')
tek.write(':CH2:COUPLING AC;POSITION 0')

time.sleep(.1)
tek.write(':TRIG:MAIN:EDGE:SOURCE CH1;SLOPE RISE;TRIG:MAIN:LEVEL 0')

# Open Function Generator (SRS DS345)
gen = rm.open_resource('ASRL/dev/ttyUSB0::INSTR')
gen.write('*RST')
gen.write('FUNC 0')
time.sleep(.1)
gen.write('AMPL 5VP')
gen.write('OFFS 0')

set_base(1)
set_scale(1)
set_scale(2)
frequencies = np.logspace(np.log10(1e2), np.log10(1e5), 500)[::-1]
# frequencies = np.linspace(20, 20e3, 100)
data = np.vectorize(get_point)(frequencies)
tek.write('UNLOCK ALL')
data = pd.DataFrame.from_records(data)

print('Acquisition ended at '+str(datetime.datetime.now()))

print(data)
data.to_csv('data.csv')

f, axarr = plt.subplots(2, sharex=True)


# axarr[0].plot(data['freq'], data['gain'])
axarr[0].semilogx(data['freq'], data['gain'])
axarr[0].set_ylabel('Gain')
axarr[0].grid()

# axarr[1].plot(data['freq'], data['phase'])
axarr[1].semilogx(data['freq'], data['phase'])
axarr[1].set_ylabel('Phase [deg]')
axarr[1].grid()
axarr[1].set_xlabel('Frequency [Hz]')

plt.savefig('bode.png')
plt.show()