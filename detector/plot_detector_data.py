import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path

matplotlib.style.use('classic')

def bootstrap(a, observable = np.mean, samples = 1000):
    '''Observable must be picked such that the central limit theorem holds'''
    a = np.array(a)
    bootstrapped = [observable(np.random.choice(a, size = len(a))) for i in range(samples)]
    estim = np.mean(bootstrapped)
    # 2-sigma
    err = 2*np.std(bootstrapped) / np.sqrt(len(a))
    return (estim, err)

def bootstrap2(a, observable = np.mean, samples = 1000):
    '''Observable must be picked such that the central limit theorem holds'''
    a = np.array(a)
    bootstrapped = [observable(np.random.choice(a, size = len(a))) for i in range(samples)]
    estim = np.mean(bootstrapped)
    # 2-sigma
    return (estim, estim - np.percentile(bootstrapped, 15.865), np.percentile(bootstrapped, 84.135) - estim)

def plot_data(data_set):
    print(data_set)
    data_set = data_set.sort_values('angle')
    plt.errorbar(data_set['angle'], data_set['rate'], data_set['rate_err'])
    plt.xlabel('Angle')
    plt.yscale('linear')
    plt.ylabel('Counts [Hz]')
    plt.show()

def load_data(log_filename):
    data_log = pd.read_csv(log_filename)
    root = Path(log_filename).parents[0]
    data = []
    for row in data_log.itertuples():
        run_data = pd.read_csv(root / row.data_file)
        run_data = run_data.diff().loc[1:]

        rate_estim, rate_std = bootstrap(run_data['counts'] / run_data['time'])
        data.append({
            'rate':rate_estim,
            'rate_err':rate_std,
            'altitude':row.altitude,
            'azimuth':row.azimuth})
    return pd.DataFrame.from_records(data)

def load_data2(log_filename):
    data_log = pd.read_csv(log_filename)
    root = Path(log_filename).parents[0]
    data = []
    for row in data_log.itertuples():
        run_data = pd.read_csv(root / row.data_file)
        run_data = run_data.diff().loc[1:]

        rate_estim, rate_lower, rate_upper = bootstrap2(run_data['counts'] / run_data['time'])
        data.append({
            'rate':rate_estim,
            'rate_err_upper':rate_upper,
            'rate_err_lower':rate_lower,
            'altitude':row.altitude,
            'azimuth':row.azimuth})
    return pd.DataFrame.from_records(data)

if __name__ == '__main__':
    data = load_data(sys.argv[1])
    plot_data(data)