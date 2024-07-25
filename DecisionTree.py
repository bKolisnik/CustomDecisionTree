import numpy as np
from Partitioner import DensePartitioner
from Splitter import Splitter, ParentInfo

INFINITY = np.inf
EPSILON = np.finfo('double').eps
_TREE_LEAF = -1
_TREE_UNDEFINED = -2


class Node:

    def __init__(self):
        self.left_child # id of left child of node
        self.right_child # id of right child
        self.feature # feature used for splitting, int
        self.threshold # float threshold for split
        self.impurity # float value of the criterion
        self.n_node_samples # number of samples at the node
        #missing_go_to_left

#no need for ParentInfo class which jsut contains the impuurity of the parent node


# need to have a min_samples_split hp to control min number of samples in each leaf
# need max depth hp

# for missing values we can do the same thing as xgboost. First run it with all missing going to left
# then do a second pass all missing going to right

class RegressionTree:

    def __init__(self):
        #properties
        self.n_features_in
        self.max_depth
        self.node_count # counter for node id
        self.capacity # capacity for tree in terms of nodes
        self.nodes #an array of nodes
        self.value # an array of shape (capacity, 1) array of values
        self.feature_names # defined only when X has feature names that are all strings
        self.is_fitted
        self.n_samples # original number of samples used to train tree

        #some sort of indicator for missing values? Maybe missing values come later

        #methods
        def add_node(self, parent: int, is_left: bool, is_leaf: bool, feature: int, threshold: float, impurity: float, n_node_samples: int):


        def check_is_fitted(self):
        def get_n_leaves(self):

        def fit(self, X, y):
            self.n_samples, self.n_features_in = X.shape

        def predict(self, X)
            
        def apply(self, X)
            #returns the index of the leaf that each sample is predicted as.

        def decision_path(self, X)
            #returns the decision path for each sample as a string
        


class TreeBuilder:

    def build(self, tree: RegressionTree, X, y):
        """Build a decision tree from the training set (X, y)."""
        pass

    def _check_input(self, X, y):
        assert len(X.shape) == 2
        assert len(y.shape) == 1

        return X, y

    def __init__(self):
        self.splitter # best splits possible only, no random splits.
        self.min_samples_leaf
        self.max_depth
        self.min_criterion_gain


class StackRecord:
    def __init__(self, start, end, parent, impurity):


class DepthFirstTreeBuilder(TreeBuilder): # there can be also be a bestfirsttreebuilder
    """Build a decision tree in depth-first fashion."""
    def __init__(self, splitter: Splitter, min_samples_leaf: int, max_depth: int, min_impurity_decrease: float):
        self.splitter = splitter
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    def build(self, tree: RegressionTree, X, y):
        X, y = self._check_input(X, y)

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        splitter = self.splitter
        max_depth = self.max_depth
        min_samples_leaf = self.min_samples_leaf
        min_weight_leaf = self.min_weight_leaf
        min_samples_split = self.min_samples_split
        min_impurity_decrease = self.min_impurity_decrease

        # Recursive partition (without actual recursion)
        splitter.init(X, y)

        start
        end
        depth
        parent
        n_node_samples = splitter.n_samples
        weighted_n_node_samples = [0]
        split = [None]
        node_id

        middle_value
        left_child_min
        left_child_max
        right_child_min
        right_child_max
        is_leaf
        first = 1
        max_depth_seen = -1
        rc = 0

        builder_stack = []

        #create a blank parent record
        parent_record = ParentInfo()

        builder_stack.append({
                "start": 0,
                "end": n_node_samples,
                "depth": 0,
                "parent": _TREE_UNDEFINED,
                "impurity": INFINITY,
                "lower_bound": -INFINITY,
                "upper_bound": INFINITY,
            })
        
        while builder_stack:
            stack_record = builder_stack.pop()

            start = stack_record.start
            end = stack_record.end
            depth = stack_record.depth
            parent = stack_record.parent
            parent_record.impurity = stack_record.impurity
            parent_record.lower_bound = stack_record.lower_bound
            parent_record.upper_bound = stack_record.upper_bound

            n_node_samples = end - start
            splitter.node_reset(start, end, weighted_n_node_samples)

            is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf)
            
            if first:
                parent_record.impurity = splitter.node_impurity()
                first = 0

            # impurity == 0 with tolerance due to rounding errors
            is_leaf = is_leaf or parent_record.impurity <= EPSILON


            if not is_leaf:
                splitter.node_split(
                    parent_record,
                    split,
                )
                # If EPSILON=0 in the below comparison, float precision
                # issues stop splitting, producing trees that are
                # dissimilar to v0.18
                is_leaf = (is_leaf or split.pos >= end or
                            (split.improvement + EPSILON <
                            min_impurity_decrease))
                
            #this function still needs to be implemented.
            node_id = tree._add_node(parent, is_leaf, split.feature,
                                        split.threshold, parent_record.impurity,
                                        n_node_samples, weighted_n_node_samples,
                                        split.missing_go_to_left)
            

            # if node id is too large break
            if node_id == INTPTR_MAX:
                rc = -1
                break

            # Store value for all nodes, to facilitate tree/model
            # inspection and interpretation
            splitter.node_value(tree.value + node_id * tree.value_stride)
            
            if not is_leaf:
                
                # Split on a feature with no monotonicity constraint

                # Current bounds must always be propagated to both children.
                # If a monotonic constraint is active, bounds are used in
                # node value clipping.
                left_child_min = right_child_min = parent_record.lower_bound
                left_child_max = right_child_max = parent_record.upper_bound
                

                # Push right child on stack
                builder_stack.push({
                    "start": split.pos,
                    "end": end,
                    "depth": depth + 1,
                    "parent": node_id,
                    "is_left": 0,
                    "impurity": split.impurity_right,
                    "lower_bound": right_child_min,
                    "upper_bound": right_child_max,
                })

                # Push left child on stack
                builder_stack.push({
                    "start": start,
                    "end": split.pos,
                    "depth": depth + 1,
                    "parent": node_id,
                    "is_left": 1,
                    "impurity": split.impurity_left,
                    "lower_bound": left_child_min,
                    "upper_bound": left_child_max,
                })

            if depth > max_depth_seen:
                max_depth_seen = depth

        if rc >= 0:
            #srhink the tree arrays after they have been created.
            rc = tree._resize_c(tree.node_count)

        if rc >= 0:
            tree.max_depth = max_depth_seen

        if rc == -1:
            raise MemoryError()


# finally need to add the tree object implementation.
#https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_tree.pyx#L670
# then lastly the python base class stuff from BaseDecisionTree and RegressionTree https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_classes.py#L223