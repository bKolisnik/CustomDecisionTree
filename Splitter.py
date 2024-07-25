from typing import List
import numpy as np
from Criterion import Criterion
from Partitioner import DensePartitioner

class ParentInfo:
    def __init__(self):
        self.impurity = np.inf # the impurity of the parent
        self.lower_bound = -np.inf
        self.upper_bound = np.inf

def _init_parent_record(record: ParentInfo):
    record.impurity = np.inf

class SplitRecord:

    def __init__():

        self.feature # int
        self.pos # int, split samples array at the given position
        self.threshold # float, threshold to split at
        self.improvement # float, improvement given parent node
        self.impurity_left # float, impurity of left split
        self.impurity_right # float, impurity of righ split
        # potentially extra attributes for missing values

def _init_split(split: SplitRecord, start_pos: int):
    split.impurity_left = [np.inf]
    split.impurity_right = [np.inf]
    split.pos = start_pos
    split.feature = 0
    split.improvement = -np.inf



class Splitter:
    # The splitter searches in the input space for a feature and a threshold
    # to split the samples samples[start:end].
    #
    # The impurity computations are delegated to a criterion object.

    self.criterion # impurity criterion
    self.min_samples_leaf # min samples in a leaf

    #in cython the ::1 in the slice type specification indicates in which dimension the data is contiguous.
    self.samples # array of ints which are the sample indices in X, y
    self.n_samples # X.shape[0]
    self.features # array of ints, feature indices in X
    self.n_features # X.shape[1]
    self.feature_values # array holding feature values

    self.y #array of float for target

    self.start # start position of the current node
    self.end # end position of the current node

    def cinit(self, criterion: Criterion, min_samples_leaf: int): # will be the real init function
        """criterion : Criterion
            The criterion to measure the quality of a split.

        min_samples_leaf : int
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered."""
        self.criterion = criterion
        self.n_samples =  0
        self.n_features = 0
        self.min_samples_leaf = min_samples_leaf

    def init(self, X, y):
        self.cinit()

        n_samples = X.shape[0]
        self.samples = np.empty(n_samples, dtype=np.intp)

        #assume all weights 1 which are all positive so skip block chekcing for positive weights
        self.n_samples = n_samples
        self.n_features = X.shape[1]

        self.feature_values = np.empty(n_samples, dtype=np.float32)
        self.y = y

        
    def node_reset(self, start: int, end: int, weighted_n_node_samples: List[float]) -> int:
        self.start = start
        self.end = end

        self.criterion.init(
            self.y,

            self.weighted_n_samples,
            self.samples,
            start,
            end
        )

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0
        
    def node_split(self, ParentCriterion, split: SplitRecord) -> int:
        pass

    def node_value(self, dest: float):
        """Copy the value of node samples[start:end] into dest."""
        self.criterion.node_value(dest)
        
    def node_impurity(self) -> float:
        """Return the impurity of the current node."""

        return self.criterion.node_impurity()
    

def node_split_best(splitter: Splitter, partitioner: DensePartitioner, criterion: Criterion, split: SplitRecord, parent_record: ParentInfo)
    start = splitter.start
    end = splitter.end
    n_left, n_right = None, None
    samples = splitter.samples
    features = splitter.features
    n_features = splitter.n_features
    feature_values = splitter.feature_values
    min_samples_leaf = splitter.min_samples_leaf

    best_split, current_split = None, None # of type SplitRecord
    current_proxy_improvement = -np.inf
    best_proxy_improvement = -np.inf

    impurity = parent_record.impurity

    f_i = n_features # int
    f_j = None # int

    p = None # int
    p_prev = None # int
    #create the split here?
    _init_split(best_split, end)

    partitioner.init_node_split(start, end)

    for j in range(n_features):
        # current_split is not instantiated yet may need to instantiat
        current_split.feature = features[j]
        partitioner.sort_samples_and_feature_values(current_split.feature)

        #no missing values so we only search once for most optimal split.

        #looking through each unique threshold of our sorted feature array
        p = start
        while p < end:
            partitioner.next_p(p_prev, p)

            if p >= end:
                continue
            n_left = p - start
            n_right = end - p
        
            # Reject if min_samples_leaf is not guaranteed
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue

            current_split.pos = p
            criterion.update(current_split.pos)

            #reject other constraints

            current_proxy_improvement = criterion.proxy_impurity_improvement()

            if current_proxy_improvement > best_proxy_improvement:
                best_proxy_improvement = current_proxy_improvement

                # sum of halves is used to avoid infinite value
                current_split.threshold = (
                    feature_values[p_prev] / 2.0 + feature_values[p] / 2.0
                )

                if (
                    current_split.threshold == feature_values[p] or
                    current_split.threshold == np.inf or
                    current_split.threshold == -np.inf
                ):
                    current_split.threshold = feature_values[p_prev]

                #if there was missing logic it would go here for determining if missing go left or right for test time
                best_split = current_split


    # Reorganize into samples[start:best_split.pos] + samples[best_split.pos:end]
    if best_split.pos < end:
        partitioner.partition_samples_final(
            best_split.pos,
            best_split.threshold,
            best_split.feature,
            best_split.n_missing
        )

    criterion.reset()
    criterion.update(best_split.pos)
    criterion.children_impurity(
        best_split.impurity_left, best_split.impurity_right
    )
    best_split.improvement = criterion.impurity_improvement(
        impurity,
        best_split.impurity_left,
        best_split.impurity_right
    )

    split[0] = best_split #updating the best split object that was passed in?


class BestSplitter(Splitter):

    def init(self, X, y):
        Splitter.init(self, X, y)
        self.partitioner = DensePartitioner(X, self.samples, self.feature_values )

    def node_split(self, parent_record: ParentInfo, split: SplitRecord):
        return node_split_best(self, self.partitioner, self.criterion, split, parent_record)
