import numpy as np
from abc import ABC

from metrics import BaseMetric


class SimpleStatisticsMetric(BaseMetric, ABC):
    def __init__(self, config, initial_buffer_size=1000):
        BaseMetric.__init__(self, config)
        self.initial_buffer_size = initial_buffer_size

        self.statistics_size = 0
        self.statistics = np.zeros(self.initial_buffer_size)

    def add_statistics_entry(self, new_statistics_entry):
        """ Adds a single entry to the statistics. """
        if self.statistics_size < self.statistics.shape[0]:
            # insert
            self.statistics[self.statistics_size:self.statistics_size + 1] = new_statistics_entry
        else:
            # append
            self.statistics = np.append(self.statistics, [new_statistics_entry], axis=0)

        self.statistics_size += 1

    def add_statistics_entries(self, new_statistics_entries):
        """
        Adds multiple statistic entries to the statistics.
        The entries variable is a 1d array.
        """
        if self.statistics_size + new_statistics_entries.shape[0] < self.statistics.shape[0]:
            # insert
            self.statistics[
                self.statistics_size:self.statistics_size + new_statistics_entries.shape[0]] = new_statistics_entries
        elif self.statistics_size < self.statistics.shape[0]:
            # insert and append
            self.statistics[self.statistics_size: self.statistics.shape[0]] = \
                new_statistics_entries[:self.statistics.shape[0] - self.statistics_size]

            self.statistics = np.append(
                self.statistics, new_statistics_entries[self.statistics.shape[0] - self.statistics_size:], axis=0)
        else:
            # append
            self.statistics = np.append(self.statistics, new_statistics_entries, axis=0)

        self.statistics_size += new_statistics_entries.shape[0]

    def clear_statistics(self):
        self.statistics_size = 0

    def reset(self):
        self.statistics_size = 0
        self.statistics = np.zeros(self.initial_buffer_size)

    def __call__(self):
        """ Calculates and returns the current statistics of this metric. """
        return {
            "mean": np.mean(self.statistics[:self.statistics_size]),
            "min": np.min(self.statistics[:self.statistics_size]),
            "max": np.max(self.statistics[:self.statistics_size]),
            "median": np.median(self.statistics[:self.statistics_size]),
            "std": np.std(self.statistics[:self.statistics_size])
        }
