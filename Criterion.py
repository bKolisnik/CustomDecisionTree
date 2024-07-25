import numpy as np
from typing import List
class Criterion:

    def __init__(self, y: np.array, sample_indices: np.array, start: int, end: int):
        self.y = y
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.weighted_n_samples = len(y)

    def reset(self):
        pass

    def reverse_reset(self):
        pass

    def update(self, new_pos: int):
        pass

    def node_impurity(self):
        pass

    def children_impurity(self, impurity_left: List[float], impurity_right: List[float]):
        pass

    def node_value(self, dest: List[float]):
        pass

    def impurity_improvement(self, impurity_parent: float, impurity_left: float, impurity_right: float) -> float:

        #weighted_n_samples is supposedly the The total weight of the samples in the entire dataset.
        #self.weighted_n_node_samples is set in derived classes.

        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity_parent - (self.weighted_n_right /
                                    self.weighted_n_node_samples * impurity_right)
                                 - (self.weighted_n_left /
                                    self.weighted_n_node_samples * impurity_left)))

    def proxy_impurity_improvement(self) -> float:
        self.children_impurity(self.impurity_left, self.impurity_right) # these might need to be one element lists

        return (- self.weighted_n_right * self.impurity_right[0]
                - self.weighted_n_left * self.impurity_left[0])



class RegressionCriterion(Criterion):

    def cinit(self, n_samples):
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.sq_sum_total = 0.0

        self.sum_total = 0
        self.sum_left = 0
        self.sum_right = 0

    def init(self, y: np.array, sample_indices: np.array, start: int, end: int):
        """Initialize the criterion.

        This initializes the criterion at node sample_indices[start:end] and children
        sample_indices[start:start] and sample_indices[start:end].
        """
        self.sq_sum_total = 0 # float
        self.sum_total = 0 # array of float (was supposed to be 1 for each output so in our case just 1 element)
        self.sum_left = None
        self.sum_right = None
        self.weighted_n_node_samples = 0.

        self.y = y
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        w = 1

        for p in range(start, end):
            i = sample_indices[p]

            y_i = self.y[i]
            self.sum_total += y_i
            self.sq_sum_total += y_i * y_i

            self.weighted_n_node_samples += w

        self.reset()

    def reset(self):
        """Reset the criterion at pos=start."""
        self.pos = self.start
        _move_sums_regression(
            self,
            self.sum_left,
            self.sum_right,
            self.weighted_n_left,
            self.weighted_n_right
        )
        
    def reverse_reset(self):
        """Reset the criterion at pos=end."""
        self.pos = self.start
        _move_sums_regression(
            self,
            self.sum_right,
            self.sum_left,
            self.weighted_n_right,
            self.weighted_n_left
        )


    def update(self, new_pos: int):
        """Updated statistics by moving sample_indices[pos:new_pos] to the left."""
        sample_indices = self.sample_indices
        pos = self.pos

        end = self.end
        w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]

                self.sum_left = self.y[i]

                self.weighted_n_left += w

        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = sample_indices[p]
                self.sum_left -= self.y[i]

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
    
        self.sum_right = self.sum_total - self.sum_left
        self.pos = new_pos


    def node_impurity(self):
        pass

    def children_impurity(self, impurity_left: List[float], impurity_right: List[float]):
        pass

    def node_value(self, dest: List[float]):
        dest[0] = self.sum_total / self.weighted_n_node_samples


def _move_sums_regression(criterion: RegressionCriterion, sum_1: List[float], sum_2: List[float], weighted_n_1: List[float], weighted_n_2: List[float]):
    sum_1[0] = 0
    sum_2[0] = criterion.sum_total

    weighted_n_1[0] = 0
    weighted_n_2[0] = criterion.weighted_n_node_samples


class MSE(RegressionCriterion):
    """Mean squared error impurity criterion.

        MSE = var_left + var_right
    """

    def node_impurity(self):
        """Evaluate the impurity of the current node.

        Evaluate the MSE criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end]. The smaller the impurity the
        better.
        """
        impurity = self.sq_sum_total / self.weighted_n_node_samples
        impurity -= (self.sum_total / self.weighted_n_node_samples)**2.0

        return impurity

    def proxy_impurity_improvement(self) -> float:
        """Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.

        The MSE proxy is derived from

            sum_{i left}(y_i - y_pred_L)^2 + sum_{i right}(y_i - y_pred_R)^2
            = sum(y_i^2) - n_L * mean_{i left}(y_i)^2 - n_R * mean_{i right}(y_i)^2

        Neglecting constant terms, this gives:

            - 1/n_L * sum_{i left}(y_i)^2 - 1/n_R * sum_{i right}(y_i)^2
        """

        proxy_impurity_left = self.sum_left * self.sum_left
        proxy_impurity_right = self.sum_right * self.sum_right

        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)
    
    def children_impurity(self, impurity_left: List[float], impurity_right: List[float]):
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]).
        """

        sample_indices = self.sample_indices
        pos = self.pos
        start = self.start

        sq_sum_left = 0
        sq_sum_right = 0
        w = 1.0

        for p in range(start, pos):
            i = sample_indices[p]
            y_i = self.y[i]
            sq_sum_left = y_i * y_i

        sq_sum_right = self.sq_sum_total - sq_sum_left

        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        impurity_left[0] -= (self.sum_left / self.weighted_n_left) ** 2.0
        impurity_right[0] -= (self.sum_right / self.weighted_n_right) ** 2.0
