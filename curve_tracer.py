import visa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

rm = visa.ResourceManager('@py')

# HP E3631
supply = rm.open_resource('ASRL/dev/ttyUSB0::INSTR')

#Diode
def measure(voltage):
    supply.write(':VOLT '+'{:04.2f}'.format(voltage))
    time.sleep(1.000)
    return list(supply.query_ascii_values(':MEASURE:CURRENT:DC? P25V'))[0]

points = 60

voltage = np.linspace(0.0, 1.1, points)
vmeasure = np.vectorize(measure)

supply.write(':SYSTEM:REMOTE;:APPLY P25V,DEF,1.0;:OUTPUT ON')

data = pd.DataFrame.from_records({'V':voltage, 'I':vmeasure(voltage)})

supply.write(':OUTPUT OFF;:SYSTEM:LOCAL')
supply.close()
rm.close()

data = data.loc[data['I'] < 0.95]
plt.figure()
plt.plot(data['V'], data['I'], linewidth=2.0)
plt.xlabel('Voltage [V]')
plt.ylabel('Current [I]')
plt.show()
plt.savefig('curve.png')