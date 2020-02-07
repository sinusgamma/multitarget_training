import numpy as np
import os
import sklearn
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt

class SeriesBuilder:
    """Make a time series with trend, seasonality, pattern and noise"""
    def __init__(self, baseline=10, slope=0.1, period=365, amplitude=10, phase=0, noise_level=2, seed=42):
        self.baseline = baseline
        self.slope = slope
        self.period = period
        self.amplitude = amplitude
        self.phase = phase
        self.noise_level = noise_level
        self.seed = seed
       
    def trend(self, time, slope=0):
        return slope * time
        
    def _seasonal_pattern(self, season_time):
        """Just an arbitrary pattern, you can change it if you wish"""
        return np.where(season_time < 0.4,
                        np.cos(season_time * 2 * np.pi),
                        1 / np.exp(3 * season_time))

    def seasonality(self, time, period, amplitude=1, phase=0):
        """Repeats the same pattern at each period"""
        season_time = ((time + phase) % period) / period
        return amplitude * self._seasonal_pattern(season_time)
       
    def white_noise(self, time, noise_level=1, seed=None):
        rnd = np.random.RandomState(seed)
        return rnd.randn(len(time)) * noise_level

    def generate_series(self, time):
        ts = self.baseline + self.trend(time, self.slope) 
        ts += self.seasonality(time, self.period, self.amplitude) 
        ts += self.white_noise(time, self.noise_level, self.seed)
        return ts
    
def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)
    
def lr_loss_chart(finder_history, xmin, xmax, ymin, ymax):  
    # plot the learning rate / loss chart
    # the best learning rates are around the lowest smooth part of the curve
    plt.semilogx(finder_history.history["lr"], finder_history.history["loss"])
    plt.axis([xmin, xmax, ymin, ymax])