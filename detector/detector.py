import nidaqmx
import nidaqmx.stream_readers
import nidaqmx.stream_writers
import datetime
import time
import asyncio
import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib
import matplotlib.pyplot as plt
import pathlib
import sys

matplotlib.style.use('classic')

def jackknife(a):
    resampled = np.ndarray([[a[j] for j in range(len(a)) if i!=j] for i in range(len(a))])
    mean = np.mean(resampled, axis=1)


class Detector:
    # 100 V / V
    _gain = -300
    _offset = 0
    _trim = 0

    _epoch = datetime.datetime(1970, 1, 1)
    _DataPoint = namedtuple('DataPoint', ['time', 'counts'])

    def close(self):
        self.set_voltage(0)
        self._counter_task.close()
        self._voltage_set_task.close()

    def __enter__(self):
        self._counter_task = nidaqmx.Task()
        self._counter_task.ci_channels.add_ci_count_edges_chan('Dev1/ctr2')
        self._counter_reader = nidaqmx.stream_readers.CounterReader(self._counter_task.in_stream)
        self._counter_task.start()

        self._voltage_set_task = nidaqmx.Task()
        self._voltage_set_task.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        self._voltage_writer = nidaqmx.stream_writers.AnalogSingleChannelWriter(self._voltage_set_task.out_stream, auto_start=False)
        self._voltage_set_task.start()

        return self

    def set_voltage(self, voltage):
        # voltage = offset + output * gain
        self._voltage_writer.write_one_sample((voltage - self._offset) / self._gain + self._trim)
        print(r'Set {}V'.format(voltage))
        time.sleep(4.0)
        print(r'Done.')

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def log(self, voltage = None, runtime = None, sample_rate = None, count_timeout = None):
        if not sample_rate:
            sample_rate = 1.0
        if not runtime:
            runtime = 1e12
            
        self.set_voltage(voltage)

        runtime = datetime.timedelta(seconds = runtime)
        time_start = datetime.datetime.utcnow()
        count_start = None
        data = []
        try:
            while (datetime.datetime.utcnow() - time_start) < runtime:
                time.sleep(sample_rate)
                count = self._counter_reader.read_one_sample_uint32()
                timestamp = (datetime.datetime.utcnow() - self._epoch).total_seconds()

                data.append(self._DataPoint(time=timestamp, counts=count))
                if count_start:
                    if count_timeout and (count - count_start) >= count_timeout:
                        break
                else:
                    count_start = count

                    
        except KeyboardInterrupt as e:
            print(e)

        data = pd.DataFrame.from_records({'time':[v.time for v in data],'counts':[v.counts for v in data]})
        return data

    def test_tube(self, voltage_points, sample_time = None, count_timeout = None, max_rate = None):
        # voltage_points = sorted(voltage_points, reverse=True)
        voltage_points = sorted(voltage_points)
        if not sample_time:
            sample_time = 60
            
        data = []
        for voltage in voltage_points:
            point = self.log(voltage=voltage, runtime=sample_time, count_timeout=None)
            counts = point['counts'].max() - point['counts'].min()
            runtime = point['time'].max() - point['time'].min()
            data.append({
                'voltage':voltage,
                'freq':counts/runtime})
            # Early exit if no more counts
            # if counts == 0:
            #     break
            if max_rate and counts/runtime > max_rate:
                break
        print(data)
        return pd.DataFrame.from_records(data)

class PostData:
    def __init__(self, dataset):
        if isinstance(dataset, str):
            dataset = self._load_data(dataset)
        else:
            attempt_index = 0
            while True:
                try:
                    output_path = pathlib.Path('data.{}.txt'.format(attempt_index))
                    with output_path.open('x') as data_file:
                        dataset.to_csv(data_file)
                    break
                except FileExistsError:
                    attempt_index += 1
        self._dataset = dataset

    def plot_detector_data(self):
        deltas = self._dataset.diff()

        plt.plot(deltas['time'].cumsum(), deltas['counts']/deltas['time'])
        plt.yscale('log')
        plt.grid()
        plt.xlabel(r'Time [s]')
        plt.ylabel(r'Counts [Hz]')
        plt.savefig('detector_data.pdf')

    def plot_tube_data(self):
        plt.plot(self._dataset['voltage'], self._dataset['freq'])
        plt.yscale('log')
        plt.grid()
        plt.xlim((1e3, 3e3))
        plt.xlabel(r'Voltage [V]')
        plt.ylabel(r'Counts [Hz]')
        plt.savefig('tube_data.pdf')
    
    def _load_data(self, filename):
        with open(filename) as data_file:
            data = pd.read_csv(data_file)
        return data
             
def detector_log(voltage, count_timeout):
    with Detector() as detector:
        dataset = detector.log(voltage, count_timeout = count_timeout)
    post = PostData(dataset)
    post.plot_detector_data()

def test_tube(lower_voltage = 1500, upper_voltage=3000, points = 5):
    with Detector() as detector:
        dataset = detector.test_tube(
            voltage_points = np.linspace(lower_voltage, upper_voltage, points), 
            sample_time=60,
            count_timeout = 1000,
            max_rate=1e3)
    post = PostData(dataset)
    post.plot_tube_data()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('detector.py <voltage> <count_timeout>')
    else:
        detector_log(float(sys.argv[1]), int(float(sys.argv[2])))
