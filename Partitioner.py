import numpy as np
from typing import List
"""Partition samples in the construction of a tree.

This module contains the algorithms for moving sample indices to
the left and right child node given a split determined by the
splitting algorithm in `_splitter.pyx`.

Partitioning is done in a way that is efficient for both dense data,"""

FEATURE_THRESHOLD = 1e-7

# Sort n-element arrays pointed to by feature_values and samples, simultaneously,
# by the values in feature_values. Algorithm: Introsort (Musser, SP&E, 1997).
def sort(feature_values: np.array, samples: np.array, start, end):
    #is end inclusive?
    ind_sorted = np.argsort(feature_values[start: end])
    feature_values[start: end] = feature_values[start+ind_sorted]
    samples[start: end] = samples[start+ind_sorted]


class DensePartitioner:
    #this splitter is agnostic to the splitting strategy (best v.s. random)

    def __init__(self, X: np.array, samples: np.array, feature_values: np.array):
        self.X = X #2d array of float
        self.samples = samples # array of ints which are the sample indices in X, y
        self.feature_values = feature_values # array holding feature values
        self.start = None # start position of the current node
        self.end = None # end position of the current node

    def init_node_split(self, start:int, end: int):
        """Initialize splitter at the beginning of node_split."""
        self.start = start
        self.end = end
        #potentially add n_missing in the future

    def sort_samples_and_feature_values(self, current_feature: int):
        """Simultaneously sort based on the feature_values.

        Missing values are stored at the end of feature_values.
        The number of missing values observed in feature_values is stored
        in self.n_missing.
        """

        # in the future add the logic to handle missing valuues here

        # When there are no missing values, we only need to copy the data into
            # feature_values
        for i in range(self.start, self.end):
            self.feature_values[i] = self.X[self.samples[i], current_feature]
        
        # sort(&feature_values[self.start], &samples[self.start], self.end - self.start - n_missing)
        
        sort(self.feature_values, self.samples, self.start, self.end)


    def find_min_max(self, current_feature: int, min_feature_value_out: List[float], max_feature_value_out: List[float]):
        min_feature_value = self.X[self.samples[self.start], current_feature]
        max_feature_value = min_feature_value_out

        self.feature_values[self.start] = min_feature_value
        for p in range(self.start+1, self.end):
            current_feature_value = self.X[self.samples[p], current_feature]
            self.feature_values[p] = current_feature_value

            if current_feature_value < min_feature_value:
                min_feature_value = current_feature_value
            elif current_feature_value > max_feature_value:
                max_feature_value = current_feature_value

        min_feature_value_out[0] = min_feature_value
        max_feature_value_out[0] = max_feature_value

    def next_p(self, p_prev: int, p: int):
        """Compute the next p_prev and p for iteratiing over feature values.

        The missing values are not included when iterating through the feature values.
        """
        feature_values = self.feature_values
        end = self.end

        # doesn't support missing

        while p[0] + 1 < end and feature_values[p[0]+1] <= feature_values[p[0]] + FEATURE_THRESHOLD:
            p[0] += 1

        p_prev[0] = p[0]

        # By adding 1, we have
        # (feature_values[p] >= end) or (feature_values[p] > feature_values[p - 1])
        p[0] += 1


    #def partition_samples(self, current_threshold: float) -> int:

    def partition_samples_final(self, best_pos: int, best_threshold: float, best_feature: int): # can potentially add number missing here

        p = self.start
        end = self.end-1 # this end is inclusive
        #current_value = None
        partition_end = end
    
        while p < partition_end:
            if self.X[self.samples[p], best_feature] <= best_threshold:
                p += 1
            else:
                self.samples[p], self.samples[partition_end] = self.samples[partition_end], self.samples[p]
                partition_end -= 1
            

    