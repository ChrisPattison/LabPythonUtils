#!/bin/python2
import pandas as pd
import numpy as np
import visa
import time

rm = visa.ResourceManager('@py')

# Open Function Generator (SRS DS345)
gen = rm.open_resource('ASRL/dev/ttyUSB1::INSTR')
# Open voltmeter (Fluke 8846A)
voltmeter = rm.open_resource('ASRL/dev/ttyUSB0::INSTR')

def set_frequency(freq):
    gen.write('FREQ '+str(freq))
    time.sleep(.5)

def setup_voltmeter():
    voltmeter.write('*cls;')
    voltmeter.write('conf:volt:ac 1.0;')
    voltmeter.write('trig:sour imm;trig:del 0;trig:coun 1;')
    voltmeter.write('syst:rem;')
    voltmeter.write('samp:coun 10;')

def measure_voltage():
    voltmeter.write(':INIT;')
    for i in range(100):
        if voltmeter.query_ascii_values('*OPC?;'):
            break
        time.sleep(.25)
        if i == 99:
            print('MEASUREMENT TIMED OUT')
    value = voltmeter.query(':FETCH?;')[1:]
    value = [float(v) for v in value.split(',')]
    return np.mean(value), np.std(value)

# setup_voltmeter()

points = 20
# freq = np.exp(np.linspace(np.log(3e3), np.log(50e3), points))
freq = np.exp(np.linspace(np.log(50e3), np.log(100e3), points))
for f in freq:
    set_frequency(f)
    mean, err = measure_voltage()
    print('{:}, {:}, {:}'.format(f, mean, err))