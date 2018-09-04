from __future__ import division
import numpy as np
from munkres import Munkres, print_matrix
import sys
import itertools
import math
from operator import itemgetter
from permanent import permanent as rysers_permanent
from scipy.optimize import linear_sum_assignment, minimize, LinearConstraint
from pymatgen.optimization import linear_assignment
import matplotlib
matplotlib.use('Agg') #prevent error running remotely
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq
import time
from profilehooks import profile
import pickle
import numba as nb
import scipy
import scipy.stats
# from 01matrix_gumbel_sample_permanent import create_diagonal2

# sys.path.insert(0, '/Users/jkuck/research/atlas_AAAI_2017/refactored_multi_model/')
# from permanent_model import compute_gumbel_upper_bound, approx_permanent3
# from boundZ import calculate_gumbel_slack

#'pymatgen' should be fastest, significantly
#pick from ['munkres', 'scipy', 'pymatgen'], 
ASSIGNMENT_SOLVER = 'pymatgen'

# random.seed(0)
SEED=54
# np.random.seed(SEED)
PICK_PARTITION_ORDER = False
USE_1_GUMBEL = True

DEBUG = False
DEBUG1 = False

FIRST_GUMBEL_LARGER = []

#
#
#References:
# [1] K. G. Murty, "Letter to the Editor--An Algorithm for Ranking all the Assignments in Order of
#     Increasing Cost," Oper. Res., vol. 16, no. May 2016, pp. 682-687, 1968.
#
# [2] I. J. Cox and M. L. Miller, "On finding ranked assignments with application to multitarget
#     tracking and motion correspondence," IEEE Trans. Aerosp. Electron. Syst., vol. 31, no. 1, pp.
#     486-489, Jan. 1995.

def sample_log_permanent_with_gumbels(cost_matrix, num_samples=1, min_sum=None):
    '''

    Inputs:
    - cost_matrix: (numpy array)  
    - min_sum: instead of computing the k best assignments, stop computing assignments when the sum 
        of np.exp(-cost) larger than min_sum

    Output:
    - best_assignments: (list of pairs) best_assignments[i][0] is the cost of the ith best
        assignment.  best_assignments[i][1] is the ith best assignment, which is a list of pairs
        where each pair represents an association in the assignment (1's in assignment matrix)
    '''
    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            if sys.maxint < cost_matrix[i][j]:
                cost_matrix[i][j] = sys.maxint
            # assert(sys.maxint > cost_matrix[i][j])

    init_node = Node(cost_matrix, [], [], 0, gumbel_truncation=np.inf)

    #heap, storing the smallest negative gumbel perturbed states, or largest gumbel perturbed states
    #each element is a tuple of (-gumbel_perturbed_state, Node)
    negative_gumbel_perturbed_heap = []
    heapq.heappush(negative_gumbel_perturbed_heap, (-init_node.rand_assoc_gumbel_perturbed_state, init_node)) 
    if PICK_PARTITION_ORDER:
        cur_partition = init_node.partition_pick_order()
    else:        
        cur_partition = init_node.partition()

    # print( "init_node.upper_bound_gumbel_perturbed_state =", init_node.upper_bound_gumbel_perturbed_state)
    # print( "init_node.rand_assoc_gumbel_perturbed_state =", init_node.rand_assoc_gumbel_perturbed_state)
    # print( "init_node.max_gumbel =", init_node.max_gumbel)
    # print( "-init_node.random_association_cost =", -init_node.random_association_cost)
    # print( )




    found_assignment_sum = 0
    number_of_times_partition_called_for_each_sample = []
    samples_of_log_Z = []
    all_sampled_associations = []
    while len(samples_of_log_Z) < num_samples:
        number_of_times_partition_called = 0
        while(True):
            np.set_printoptions(linewidth=300)

            max_gumbel_upper_bound = -negative_gumbel_perturbed_heap[0][0]


            ###JUST TESTING###
            # largest_perturbed_max_state = None
            # for cur_idx, cur_node in enumerate(cur_partition):
            #     if largest_perturbed_max_state == None or cur_node.test_perturbing_max_state > largest_perturbed_max_state:
            #         largest_perturbed_max_state = cur_node.test_perturbing_max_state
            # if largest_perturbed_max_state> max_gumbel_upper_bound:
            #     print "largest_perturbed_max_state larger than max_gumbel_upper_bound :) diff=", largest_perturbed_max_state- max_gumbel_upper_bound, "largest_perturbed_max_state =", largest_perturbed_max_state, "max_gumbel_upper_bound =", max_gumbel_upper_bound, 'max gumbel state:', -negative_gumbel_perturbed_heap[0][1].random_association_cost

            # else:
            
            #     print "largest_perturbed_max_state smaller than max_gumbel_upper_bound :< diff=", largest_perturbed_max_state- max_gumbel_upper_bound, "largest_perturbed_max_state =", largest_perturbed_max_state, "max_gumbel_upper_bound =", max_gumbel_upper_bound, 'max gumbel state:', -negative_gumbel_perturbed_heap[0][1].random_association_cost

            ###DONE JUST TESTING###

            node_idx_with_max_gumbel_ub = None
            for cur_idx, cur_node in enumerate(cur_partition):
                if cur_node.upper_bound_gumbel_perturbed_state > max_gumbel_upper_bound:
                    max_gumbel_upper_bound = cur_node.upper_bound_gumbel_perturbed_state
                    node_idx_with_max_gumbel_ub = cur_idx
            if node_idx_with_max_gumbel_ub is None:
                break #we found the maximum gumbel perturbed state


            # if cur_partition[node_idx_with_max_gumbel_ub].test_perturbing_max_state > cur_partition[node_idx_with_max_gumbel_ub].rand_assoc_gumbel_perturbed_state:
            #     print( "test_perturbing_max_state larger :):):):):):):):):):)")
            # else:
            #     print( "test_perturbing_max_state smaller :<:<:<:<:<:<:<:<:<:<")

            # print( "-cur_partition[node_idx_with_max_gumbel_ub].minimum_cost =", -cur_partition[node_idx_with_max_gumbel_ub].minimum_cost)
            # print( "-cur_partition[node_idx_with_max_gumbel_ub].random_association_cost =", -cur_partition[node_idx_with_max_gumbel_ub].random_association_cost)
            # print( "cur_partition[node_idx_with_max_gumbel_ub].test_single_gumbel =", cur_partition[node_idx_with_max_gumbel_ub].test_single_gumbel)
            # print( "cur_partition[node_idx_with_max_gumbel_ub].max_gumbel =", cur_partition[node_idx_with_max_gumbel_ub].max_gumbel)
            # print( "cur_partition[node_idx_with_max_gumbel_ub].test_perturbing_max_state =", cur_partition[node_idx_with_max_gumbel_ub].test_perturbing_max_state)
           
            # print( "cur_partition[node_idx_with_max_gumbel_ub].upper_bound_gumbel_perturbed_state =", cur_partition[node_idx_with_max_gumbel_ub].upper_bound_gumbel_perturbed_state)
            # print( "cur_partition[node_idx_with_max_gumbel_ub].rand_assoc_gumbel_perturbed_state =", cur_partition[node_idx_with_max_gumbel_ub].rand_assoc_gumbel_perturbed_state)
            # print( )



            heapq.heappush(negative_gumbel_perturbed_heap, (-cur_partition[node_idx_with_max_gumbel_ub].rand_assoc_gumbel_perturbed_state, cur_partition[node_idx_with_max_gumbel_ub])) 

            if cur_partition[node_idx_with_max_gumbel_ub].assignment_count > 1:
                # print "calling partition on node with log assignment count:", math.log(cur_partition[node_idx_with_max_gumbel_ub].assignment_count)
                # print 
                if PICK_PARTITION_ORDER:
                    cur_partition.extend(cur_partition[node_idx_with_max_gumbel_ub].partition_pick_order())
                else:
                    cur_partition.extend(cur_partition[node_idx_with_max_gumbel_ub].partition())

                number_of_times_partition_called += 1

            del cur_partition[node_idx_with_max_gumbel_ub]

            # assignment_sum += np.exp(-best_assignments[-1][0])
            # if assignment_sum > min_sum:
            #     break        
        number_of_times_partition_called_for_each_sample.append(number_of_times_partition_called)

        smallest_gumbel_perturbed_cost = heapq.heappop(negative_gumbel_perturbed_heap)
        sample_of_log_Z = -smallest_gumbel_perturbed_cost[0] - np.euler_gamma
        sampled_associations = tuple(smallest_gumbel_perturbed_cost[1].random_associations)
        # print( "uncorrected sample_of_log_Z:", sample_of_log_Z)
        samples_of_log_Z.append(np.log(np.exp(sample_of_log_Z) + found_assignment_sum)) #Fixed, double check: FIX ME CORRECT FOR FOUND ASSIGNMENTS
        all_sampled_associations.append(sampled_associations)

        # print( "negative_gumbel_perturbed_heap:")
        # print( negative_gumbel_perturbed_heap)
        found_assignment_sum += np.exp(-smallest_gumbel_perturbed_cost[1].random_association_cost)
        # print( "found_assignment_sum =", found_assignment_sum)
    #the total number of assignments is N! or the number of assignments in each of the partitioned
    #nodes plus the number of explicitlitly found assignments in negative_gumbel_perturbed_heap
    total_assignment_count = sum([node.assignment_count for node in cur_partition]) + len(negative_gumbel_perturbed_heap) + num_samples
    assert(total_assignment_count == math.factorial(cost_matrix.shape[0]))

    node_count_plus_heap_size = len(cur_partition) + len(negative_gumbel_perturbed_heap)
    return samples_of_log_Z, node_count_plus_heap_size, number_of_times_partition_called_for_each_sample, all_sampled_associations


def k_best_assign_mult_cost_matrices(k, cost_matrices):
    '''
    Find the k lowest cost assignments for any of the cost matrices.  That is, the lowest cost will
    be the lowest cost assignment with costs specified by ANY of the cost matrices.  This is 
    useful for multiple hypothesis tracking, where we want to find the k lowest costs and have
    k cost matrices.  Rather than solving k k_best_assignment problems, generating k^2 costs, and
    picking the k smallest costs we will instead initialize Murty's algorithm with a set of 
    cost matrices as explained in [2] on pp. 487-488.


    Inputs:
    - k: (integer) 
    - cost_matrices: (list of numpy arrays)  

    Output:
    - best_assignments: (list of triplets) best_assignments[i][0] is the cost of the ith best
        assignment.  best_assignments[i][1] is the ith best assignment, which is a list of pairs
        where each pair represents an association in the assignment (1's in assignment matrix),
        best_assignments[i][2] is the index in the input cost_matrices of the cost matrix used
        for the ith best assignment
    '''
    for cur_cost_matrix in cost_matrices:
        for i in range(cur_cost_matrix.shape[0]):
            for j in range(cur_cost_matrix.shape[1]):
                assert(sys.maxint > cur_cost_matrix[i][j])
    best_assignments = []
    cur_partition = []
    for (idx, cur_cost_matrix) in enumerate(cost_matrices):
        cur_partition.append(Node(cur_cost_matrix, [], [], idx))

    for itr in range(0, k):
        min_cost = sys.maxint
        min_cost_idx = -1
        np.set_printoptions(linewidth=300)

        if DEBUG1:
            print( len(cur_partition), "nodes in partition with remaing cost matrices:")
        for cur_idx, cur_node in enumerate(cur_partition):
            if DEBUG1:
                print( cur_node.minimum_cost)
                print( cur_node.remaining_cost_matrix   )
                print(       )
            if cur_node.minimum_cost < min_cost:
                min_cost = cur_node.minimum_cost
                min_cost_idx = cur_idx
        if DEBUG1:            
            print( '$'*80)

        if DEBUG1:
            debug_total_assign_count = 0
            for cur_idx, cur_node in enumerate(cur_partition):
                rcm = cur_node.remaining_cost_matrix
                assert(rcm.shape[0] == rcm.shape[1])
                rmc_assign_count = 0
                for row in range(rcm.shape[0]):
                    if row == 0: #count excluded cells
                        excluded_cell_count = 0
                        for col in range(rcm.shape[1]):
                            if rcm[(row, col)] == sys.maxint:
                                excluded_cell_count += 1
                        rmc_assign_count = rcm.shape[0] - excluded_cell_count
                    else:
                        excluded_cell_count = 0
                        for col in range(rcm.shape[1]):
                            if rcm[(row, col)] == sys.maxint:
                                excluded_cell_count += 1
                        assert(excluded_cell_count == 0)
                rmc_assign_count *= math.factorial(rcm.shape[0] - 1)
                debug_total_assign_count += rmc_assign_count                        
            print( "we've kept track of assignments:", (itr + debug_total_assign_count))
            print( '&'*80)
            print()
            print()

        best_assignments.append(cur_partition[min_cost_idx].get_min_cost_assignment())
        if DEBUG:
            print( "iter", itr, "-"*80)
            print( "best assignment: ", best_assignments[-1])
        cur_partition.extend(cur_partition[min_cost_idx].partition())

        if DEBUG:
            print( "all assignment costs in partition: ",)
            for node in cur_partition:
                node_min_cost_assign = node.get_min_cost_assignment()
                print( node_min_cost_assign[0],)
            print()
            print( "required_cells: ", cur_partition[min_cost_idx].required_cells)
            print( "min_cost_associations: ", cur_partition[min_cost_idx].min_cost_associations)
            print()

        del cur_partition[min_cost_idx]
    return best_assignments


MINC_UB_BEST = []
EXTENDEND_MINC_UB_BEST = []
FULLY_INDECOMPOSABLE_UB_BEST = []
FULLY_INDECOMPOSABLE_UB_INVALID = []
def mu_func(m, x):
    '''
    Calculate the geometric mean of m equally spaced numbers from 1 to x
    '''
    if m == 1:
        return (x+1)/2 #CHECK ME!!
    geom_mean = 1.0
    for k in range(1, m+1):
        geom_mean *= ((k-1)*x + m - k)/(m-1)
    geom_mean = geom_mean ** (1/m)
    return geom_mean



LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')

@nb.jit
def fast_factorial(n):
    if n > 20:
        raise ValueError
    return LOOKUP_TABLE[n]

gamma_cache = {}
def gamma(k):
    if k == 0:
        return 0
    else:
        assert(k >= 1)
        if k in gamma_cache:
            return gamma_cache[k]
        else:
            return_val = (math.factorial(k))**(1/k)
            # return_val = (fast_factorial(k))**(1/k)
            gamma_cache[k] = return_val
            return return_val

delta_cache = {}
def delta(k):
    if k in delta_cache:
        return delta_cache[k]
    else:
        return_val = gamma(k) - gamma(k-1)
        delta_cache[k] = return_val
        return return_val

NUMBA_GAMMA_CACHE = np.zeros(100, dtype=np.float64)
@nb.jit(["float64(int32, float64[:])"], "(),(n)->()")
# @nb.jit(nb.float64(nb.int32,), nopython=True)
# @nb.vectorize(nb.float64(nb.int32,))
def numba_gamma(k, numba_gamma_cache=NUMBA_GAMMA_CACHE):
    if k == 0:
        return 0
    else:
        assert(k >= 1)
        if numba_gamma_cache[0] == 0:
            for cur_k in range(100):
                return_val = (fast_factorial(cur_k-1))**(1/cur_k-1)
                numba_gamma_cache[cur_k-1] = return_val
        return numba_gamma_cache[k-1]

NUMBA_DELTA_CACHE = np.zeros(100, dtype=np.float64)
@nb.jit(["float64(int32, float64[:])"], "(),(n)->()")
# @nb.jit((nb.int32,))
# @nb.vectorize(nb.float64(nb.int32,))
def numba_delta(k, numba_delta_cache=NUMBA_DELTA_CACHE):
    assert(k < 100)
    if numba_delta_cache[0]: #compute cache
        for cur_k in range(100):
            return_val = numba_gamma(cur_k, numba_gamma_cache=NUMBA_GAMMA_CACHE) - numba_gamma(cur_k-1, numba_gamma_cache=NUMBA_GAMMA_CACHE)
            numba_delta_cache[cur_k-1] = return_val

    return numba_delta_cache[k-1]

# @profile
# @nb.jit
# @nb.guvectorize(["float64(float64[:,:])"], "(n,n)->()")
def minc_extended_UB2(matrix):
    #another bound
    #https://ac-els-cdn-com.stanford.idm.oclc.org/S002437950400299X/1-s2.0-S002437950400299X-main.pdf?_tid=fa4d00ee-39a5-4030-b7c1-28bb5fbc76c0&acdnat=1534454814_a7411b3006e0e092622de35cbf015275
    # equation (6), U^M(A)

    assert(matrix.shape[0] == matrix.shape[1])
    N = matrix.shape[0]

    minc_extended_upper_bound2 = 1.0
    for row in range(N):
        sorted_row = sorted(matrix[row], reverse=True)
        row_sum = 0
        for col in range(N):
            row_sum += sorted_row[col] * delta(col+1)
            # row_sum += sorted_row[col] * numba_delta(col+1)
        minc_extended_upper_bound2 *= row_sum
    return minc_extended_upper_bound2

def minc_extended_UB2_scaled(matrix, col_scale):
    #another bound
    #https://ac-els-cdn-com.stanford.idm.oclc.org/S002437950400299X/1-s2.0-S002437950400299X-main.pdf?_tid=fa4d00ee-39a5-4030-b7c1-28bb5fbc76c0&acdnat=1534454814_a7411b3006e0e092622de35cbf015275
    # equation (6), U^M(A), with column scaling

    assert(matrix.shape[0] == matrix.shape[1])
    N = matrix.shape[0]
    assert(len(col_scale) == N)
    minc_extended_upper_bound2 = 1.0
    for scale in col_scale:
        minc_extended_upper_bound2 /= scale
    for row in range(N):
        assert(matrix[row].shape == col_scale.shape)
        scaled_row = np.multiply(matrix[row], col_scale) #elmentwise multiplication
        sorted_row = sorted(scaled_row, reverse=True)
        row_sum = 0
        for col in range(N):
            row_sum += sorted_row[col] * delta(col+1)
        minc_extended_upper_bound2 *= row_sum
    return minc_extended_upper_bound2

def get_func_of_scale_minc_extended_UB2_scaled(matrix):
    #get the upper bound as a function of a column scaling for a particular matrix
    #another bound
    #https://ac-els-cdn-com.stanford.idm.oclc.org/S002437950400299X/1-s2.0-S002437950400299X-main.pdf?_tid=fa4d00ee-39a5-4030-b7c1-28bb5fbc76c0&acdnat=1534454814_a7411b3006e0e092622de35cbf015275
    # equation (6), U^M(A), with column scaling

    assert(matrix.shape[0] == matrix.shape[1])
    N = matrix.shape[0]

    def func_of_column_scale(col_scale):
        assert(len(col_scale) == N)
        minc_extended_upper_bound2 = 1.0
        for scale in col_scale:
            minc_extended_upper_bound2 /= scale
        for row in range(N):
            assert(matrix[row].shape == col_scale.shape)
            scaled_row = np.multiply(matrix[row], col_scale) #elmentwise multiplication
            sorted_row = sorted(scaled_row, reverse=True)
            row_sum = 0
            for col in range(N):
                row_sum += sorted_row[col] * delta(col+1)
            minc_extended_upper_bound2 *= row_sum
        return minc_extended_upper_bound2

    return func_of_column_scale

def optimized_minc_extened_UB2(matrix):
    #https://ac-els-cdn-com.stanford.idm.oclc.org/S002437950400299X/1-s2.0-S002437950400299X-main.pdf?_tid=fa4d00ee-39a5-4030-b7c1-28bb5fbc76c0&acdnat=1534454814_a7411b3006e0e092622de35cbf015275
    # optimize equation (6), U^M(A) according to equation (9) (that is, rescaling columns)

    assert(matrix.shape[0] == matrix.shape[1])
    N = matrix.shape[0]

    # # rescale by positive numbers, SHOULD BE GREATER THAN 0, NOT >=, is this a probelm??
    x0 = np.ones(N)
    function = get_func_of_scale_minc_extended_UB2_scaled(matrix)
    result = minimize(fun=function, x0=x0, bounds=[(0, np.inf) for i in range(N)])
    optimzed_upper_bound = function(result.x)

    assert((result.x>0).all()), (result.x, optimzed_upper_bound)
    print ("result.x =", result.x, "optimzed_upper_bound =", optimzed_upper_bound)
    return optimzed_upper_bound

class Node:
    # @profile
    def __init__(self, orig_cost_matrix, required_cells, excluded_cells, orig_cost_matrix_index, gumbel_truncation):
        '''
        Following the terminology used by [1], a node is defined to be a nonempty subset of possible
        assignments to a cost matrix.  Every assignment in node N is required to contain
        required_cells and exclude excluded_cells.

        Inputs:
        - orig_cost_matrix: (2d numpy array) the original cost matrix
        - required_cells: (list of pairs) where each pair represents a (zero indexed) location
            in the assignment matrix that must be a 1
        - excluded_cells: (list of pairs) where each pair represents a (zero indexed) location
            in the assignment matrix that must be a 0
        - orig_cost_matrix_index: index of the cost matrix this Node is descended from, used when
            when finding the k lowest cost assignments among a group of assignment matrices
            (k_best_assign_mult_cost_matrices)
        - gumbel_truncation: (float) truncate gumbels for this Node to this value (https://cmaddis.github.io/)
        '''
        self.orig_cost_matrix = np.array(orig_cost_matrix, copy=True)
        self.required_cells = required_cells[:]
        self.excluded_cells = excluded_cells[:]
        self.gumbel_truncation = gumbel_truncation
        rows_containing_excluded_cells = set()
        cols_containing_excluded_cells = set()
        for row, col in self.excluded_cells:
            rows_containing_excluded_cells.add(row)
            cols_containing_excluded_cells.add(col)
        for row, col in self.required_cells:
            if row in rows_containing_excluded_cells:
                rows_containing_excluded_cells.remove(row)
            if col in cols_containing_excluded_cells:
                cols_containing_excluded_cells.remove(col)
        # if len(rows_containing_excluded_cells) > 1 and len(cols_containing_excluded_cells) > 1:
        if len(rows_containing_excluded_cells) > 1:
            # print( "more than 1 col and more than 1 row contain excluded cells!!!")
            print( "more than 1 row contain excluded cells!!!")
            print( "self.excluded_cells:", self.excluded_cells)
            print( "self.required_cells:", self.required_cells)
            print( "rows_containing_excluded_cells:", rows_containing_excluded_cells)
            print( "cols_containing_excluded_cells:", cols_containing_excluded_cells)
            assert(False)

        self.orig_cost_matrix_index = orig_cost_matrix_index
        #the number of assignments this node contains
        self.assignment_count = self.count_assignments()
        if self.assignment_count == 0: #this node is empty
            return

        # print( "self.assignment_count:", self.assignment_count)

        if USE_1_GUMBEL:
            compare_gumbel_vals = compare_truncated_gumbel(n_vals=[1, self.assignment_count], truncation=gumbel_truncation)
            self.max_gumbel_1 = compare_gumbel_vals[0]
            self.max_gumbel = compare_gumbel_vals[1]
        else:
            self.max_gumbel = truncated_gumbel(n=self.assignment_count, truncation=gumbel_truncation)


        if DEBUG:
            print( "New Node:")
            print( "self.required_cells:", self.required_cells )
            print( "self.excluded_cells:", self.excluded_cells )

        #we will transform the cost matrix into the "remaining cost matrix" as described in [1]
        self.remaining_cost_matrix = self.construct_remaining_cost_matrix()
        assert((self.remaining_cost_matrix > -.000000001).all()), self.remaining_cost_matrix
        #solve the assignment problem for the remaining cost matrix

        if ASSIGNMENT_SOLVER == 'munkres':
            hm = Munkres()
            # we get a list of (row, col) associations, or 1's in the minimum assignment matrix
            association_list = hm.compute(self.remaining_cost_matrix.tolist())
        elif ASSIGNMENT_SOLVER == 'scipy':
            row_ind, col_ind = linear_sum_assignment(self.remaining_cost_matrix)
            assert(len(row_ind) == len(col_ind))
            association_list = zip(row_ind, col_ind)
        else:
            assert(ASSIGNMENT_SOLVER == 'pymatgen')
            lin_assign = linear_assignment.LinearAssignment(self.remaining_cost_matrix)
            solution = lin_assign.solution
            association_list = zip([i for i in range(len(solution))], solution)
#                association_list = [(i, i) for i in range(orig_cost_matrix.shape[0])]

        if DEBUG:
            print( "remaining cost matrix:")
            print( self.remaining_cost_matrix)
            print( "association_list")
            print( association_list)


        #compute the minimum cost assignment for the node
        self.minimum_cost = 0
        # print "self.remaining_cost_matrix.shape:", self.remaining_cost_matrix.shape
        # print "association_list:", association_list
        for (row,col) in association_list:
#            print( 'a', self.minimum_cost, type(self.minimum_cost))
#            print( 'b', self.remaining_cost_matrix[row][col], type(self.remaining_cost_matrix[row][col]))
#            print( 'c', self.minimum_cost +self.remaining_cost_matrix[row][col], type(self.minimum_cost +self.remaining_cost_matrix[row][col]))
            #np.asscalar important for avoiding overflow problems
            self.minimum_cost += np.asscalar(self.remaining_cost_matrix[row][col])
        for (row, col) in self.required_cells:
            #np.asscalar important for avoiding overflow problems
            self.minimum_cost += np.asscalar(orig_cost_matrix[row][col])

        #store the minimum cost associations with indices consistent with the original cost matrix
        self.min_cost_associations = self.get_orig_indices(association_list)

        #the largest gumbel perturbed state is the max gumbel in this partition + the largest log weight 
        # (which is the same as the negative min_cost)
        self.upper_bound_gumbel_perturbed_state = -self.minimum_cost + self.max_gumbel

        TEST_SUBMATRIX_BOUND1 = False
        TEST_SUBMATRIX_BOUND2 = False
        TEST_SUBMATRIX_BOUND3 = False

        # if TEST_SUBMATRIX_BOUND1:
        #     #test out using an upper bound on the permanent for the submatrix
        #     #gumbel perturbation upper bound
        #     sub_matrix = np.exp(-self.remaining_cost_matrix)
        #     gumbel_expectation_upper_bound = compute_gumbel_upper_bound(sub_matrix, num_perturbations=100)
        #     for row, col in self.required_cells:
        #         gumbel_expectation_upper_bound -= self.orig_cost_matrix[row][col]
        #     self.test_submatrix_upper_bound = gumbel_expectation_upper_bound + truncated_gumbel(n=1, truncation=gumbel_truncation)

        if TEST_SUBMATRIX_BOUND2:
            #from https://arxiv.org/pdf/1412.1933.pdf, Corollary 1.
            sub_matrix = np.exp(-self.remaining_cost_matrix)            
            minc_upper_bound = 1.0
            assert(sub_matrix.shape[0] == sub_matrix.shape[1])
            N = sub_matrix.shape[0]            
            for row in range(N):
                row_sum = 0
                for col in range(N):
                    row_sum += sub_matrix[row][col]
                minc_upper_bound *= math.factorial(np.ceil(row_sum))**(1/np.ceil(row_sum))
            self.log_minc_upper_bound = np.log(minc_upper_bound)
            for row, col in self.required_cells:
                self.log_minc_upper_bound -= self.orig_cost_matrix[row][col]
    
            #another bound, worse than minc_upper_bound above 
            #from https://arxiv.org/pdf/1412.1933.pdf, Corollary 2.
            # matrix_sum = 0
            # for row in range(N):
            #     for col in range(N):
            #         matrix_sum += sub_matrix[row][col]
            # gamma = matrix_sum/N
            # minc_upper_bound_2 = (gamma + 1)**N * np.exp(-N) * (np.e * np.sqrt(gamma + 1))**(N/(gamma + 1))
            # assert(minc_upper_bound_2 > minc_upper_bound)


            #another bound
            #https://www.sciencedirect.com/science/article/pii/S002437959810040X
            # matrix must be fully indecomposable, check this holds !!
            smallest_pos_element_product = 1.0
            row_sum_product = 1.0
            for row in range(N):
                row_sum = 0
                s_i = np.inf
                for col in range(N):
                    row_sum += sub_matrix[row][col]
                    if (sub_matrix[row][col] > 0) and (sub_matrix[row][col] < s_i):
                        s_i = sub_matrix[row][col]
                smallest_pos_element_product *= s_i
                row_sum_product *= row_sum - s_i

            fully_indecomposable_upper_bound = smallest_pos_element_product + row_sum_product

            #another bound
            #https://www.tandfonline.com/doi/abs/10.1080/03081080008818633
            minc_extended_upper_bound = 1.0
            for row in range(N):
                row_sum = 0
                largest_row_element = 0.0
                num_pos_elements_in_row = 0
                for col in range(N):
                    row_sum += sub_matrix[row][col]
                    if sub_matrix[row][col] > largest_row_element:
                        largest_row_element = sub_matrix[row][col]
                    if sub_matrix[row][col] > 0:
                        num_pos_elements_in_row += 1

                minc_extended_upper_bound *= largest_row_element * mu_func(num_pos_elements_in_row, row_sum/largest_row_element)
                # print( "minc_extended_upper_bound =", minc_extended_upper_bound, num_pos_elements_in_row, row_sum, largest_row_element)
            self.log_minc_extended_upper_bound = np.log(minc_extended_upper_bound)
            for row, col in self.required_cells:
                self.log_minc_extended_upper_bound -= self.orig_cost_matrix[row][col]


            #another bound
            #https://dukespace.lib.duke.edu/dspace/bitstream/handle/10161/1054/D_Law_Wai_a_200904.pdf?sequence=1&isAllowed=y
            def h_func(r):
                if r >= 1:
                    return r + .5*math.log(r) + np.e - 1
                else:
                    return 1 + (np.e - 1)*r

            assert((sub_matrix <= 1).all())
            assert((sub_matrix >= 0).all())
            bregman_extended_upper_bound = 1
            for row in range(N):
                row_sum = 0
                for col in range(N):
                    row_sum += sub_matrix[row][col]

                bregman_extended_upper_bound *= h_func(row_sum)/np.e
            self.log_bregman_extended_upper_bound = np.log(bregman_extended_upper_bound)

            assert(self.log_bregman_extended_upper_bound > np.log(minc_extended_upper_bound))

            #another bound
            #https://arxiv.org/pdf/math/0508096.pdf
            hadamard_upper_bound = math.factorial(N)/(N**(N/2))
            for row in range(N):
                row_length_squared = 0
                for col in range(N):
                    row_length_squared += (sub_matrix[row][col])**2
                row_length = np.sqrt(row_length_squared)
                hadamard_upper_bound *= row_length
            self.log_hadamard_upper_bound = np.log(hadamard_upper_bound)

            # assert(self.log_hadamard_upper_bound > np.log(minc_extended_upper_bound))

            #another bound
            #http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.331.2073&rep=rep1&type=pdf
            arithmetic_mean_UB = 0.0
            for row in range(N):
                for col in range(N):
                    arithmetic_mean_UB += (sub_matrix[row][col])**N
            arithmetic_mean_UB /= N**2 
            arithmetic_mean_UB *= math.factorial(N)

            #another bound
            #https://ac-els-cdn-com.stanford.idm.oclc.org/S002437950400299X/1-s2.0-S002437950400299X-main.pdf?_tid=fa4d00ee-39a5-4030-b7c1-28bb5fbc76c0&acdnat=1534454814_a7411b3006e0e092622de35cbf015275
            # equation (6), U^M(A)
            minc_extended_upper_bound2 = 1.0
            for row in range(N):
                sorted_row = sorted(sub_matrix[row], reverse=True)
                row_sum = 0
                for col in range(N):
                    row_sum += sorted_row[col] * delta(col+1)
                minc_extended_upper_bound2 *= row_sum
            assert(minc_extended_upper_bound2 < minc_extended_upper_bound)

            #another bound
            #https://ac-els-cdn-com.stanford.idm.oclc.org/S002437950400299X/1-s2.0-S002437950400299X-main.pdf?_tid=fa4d00ee-39a5-4030-b7c1-28bb5fbc76c0&acdnat=1534454814_a7411b3006e0e092622de35cbf015275
            # optimize equation (6), U^M(A) according to equation (9) (that is, rescaling columns)

            # optimized_minc_extended_upper_bound2 = optimized_minc_extened_UB2(sub_matrix)
            # assert(optimized_minc_extended_upper_bound2 < minc_extended_upper_bound2)


            TEST_BOUNDS = True
            if TEST_BOUNDS:
                # exact_permanent = calc_permanent_rysers(sub_matrix)
                exact_permanent = 0
                log_exact_permanent = np.log(exact_permanent)
                if(log_exact_permanent <= np.log(fully_indecomposable_upper_bound) + .00000001 and np.log(fully_indecomposable_upper_bound) < np.log(minc_upper_bound) and np.log(fully_indecomposable_upper_bound) < np.log(minc_extended_upper_bound)):
                    MINC_UB_BEST.append(0)
                    EXTENDEND_MINC_UB_BEST.append(0)
                    FULLY_INDECOMPOSABLE_UB_BEST.append(1)
                    print( 'FULLY_INDECOMPOSABLE_UB_BEST!, N=', N)
                elif (np.log(minc_upper_bound) < np.log(fully_indecomposable_upper_bound) and np.log(minc_upper_bound) < np.log(minc_extended_upper_bound)):
                    MINC_UB_BEST.append(1)
                    EXTENDEND_MINC_UB_BEST.append(0)
                    FULLY_INDECOMPOSABLE_UB_BEST.append(0)
                else:
                    assert((np.log(minc_extended_upper_bound)) < np.log(minc_upper_bound) and (np.log(minc_extended_upper_bound) < np.log(fully_indecomposable_upper_bound) or log_exact_permanent > np.log(fully_indecomposable_upper_bound) + .00000001))
                    MINC_UB_BEST.append(0)
                    EXTENDEND_MINC_UB_BEST.append(1)
                    FULLY_INDECOMPOSABLE_UB_BEST.append(0)
                if log_exact_permanent <= np.log(fully_indecomposable_upper_bound) + .00000001:
                    FULLY_INDECOMPOSABLE_UB_INVALID.append(0)
                else:
                    FULLY_INDECOMPOSABLE_UB_INVALID.append(1)

                print( "N =", N)
                print( "log exact permanent =", log_exact_permanent)
                print( "log of minc_extended_upper_bound", np.log(minc_extended_upper_bound))
                print( "log of minc_extended_upper_bound2", np.log(minc_extended_upper_bound2))
                print( "log of arithmetic_mean_UB", np.log(arithmetic_mean_UB))
                # print( "log of optimized_minc_extended_upper_bound2", np.log(optimized_minc_extended_upper_bound2))
                print( "log minc_upper_bound =", np.log(minc_upper_bound))
                print( "log of fully_indecomposable_upper_bound =", np.log(fully_indecomposable_upper_bound))
                print( "log of bregman_extended_upper_bound", self.log_bregman_extended_upper_bound)
                print( "log of hadamard_upper_bound", self.log_hadamard_upper_bound)
                # print( "log minc_upper_bound_2 =", np.log(minc_upper_bound_2))
                print( "np.mean(MINC_UB_BEST) =", np.mean(MINC_UB_BEST))
                print( "np.mean(EXTENDEND_MINC_UB_BEST) =", np.mean(EXTENDEND_MINC_UB_BEST))
                print( "np.mean(FULLY_INDECOMPOSABLE_UB_BEST) =", np.mean(FULLY_INDECOMPOSABLE_UB_BEST))
                print( "np.mean(FULLY_INDECOMPOSABLE_UB_INVALID) =", np.mean(FULLY_INDECOMPOSABLE_UB_INVALID))
                print( )
                assert(log_exact_permanent < minc_upper_bound), "minc_upper_bound invalid!!"
                # assert(log_exact_permanent <= np.log(fully_indecomposable_upper_bound + .00000001)), ("fully_indecomposable_upper_bound invalid!!", sub_matrix)
                assert(log_exact_permanent < np.log(minc_extended_upper_bound)), ("minc_extended_upper_bound invalid!!", np.log(minc_extended_upper_bound), np.log(minc_extended_upper_bound) is np.nan, minc_extended_upper_bound)


                compare_gumbel_vals = compare_truncated_gumbel(n_vals=[1, self.assignment_count], truncation=gumbel_truncation)
                max_1_gumbel_val = compare_gumbel_vals[0]
                max_n_gumbel_vals = compare_gumbel_vals[1]
                compare_max_state_bound = -self.minimum_cost + max_n_gumbel_vals
                compare_minc_bound = self.log_minc_extended_upper_bound + max_1_gumbel_val
                # assert(compare_minc_bound < compare_max_state_bound)
                print("compare_max_state_bound:", compare_max_state_bound)
                print("compare_minc_bound:", compare_minc_bound)
                print()
                print()

        # self.random_associations, self.random_association_cost = self.sample_association_non_uniform()
        self.random_associations, self.random_association_cost = self.sample_association_uniform()
        self.rand_assoc_gumbel_perturbed_state = -self.random_association_cost + self.max_gumbel

        # self.test_single_gumbel = truncated_gumbel(n=1, truncation=gumbel_truncation)
        self.test_single_gumbel = truncated_gumbel(n=1, truncation=self.max_gumbel)
        self.test_perturbing_max_state = -self.minimum_cost + self.test_single_gumbel

        # if self.test_perturbing_max_state > self.rand_assoc_gumbel_perturbed_state:
        #     print( "test_perturbing_max_state larger :):):):):):):):):):) diff=", self.test_perturbing_max_state - self.rand_assoc_gumbel_perturbed_state)

        # else:
        
        #     print( "test_perturbing_max_state smaller :<:<:<:<:<:<:<:<:<:< diff=", self.test_perturbing_max_state - self.rand_assoc_gumbel_perturbed_state)
        # print( "-self.minimum_cost =", -self.minimum_cost)
        # print( "-self.random_association_cost =", -self.random_association_cost)
        # print( "test_single_gumbel =", test_single_gumbel)
        # print( "self.max_gumbel =", self.max_gumbel)
        # print( "test_perturbing_max_state =", test_perturbing_max_state)
        # print( "self.rand_assoc_gumbel_perturbed_state =", self.rand_assoc_gumbel_perturbed_state)
        # print( )


        # if self.rand_assoc_gumbel_perturbed_state > self.upper_bound_gumbel_perturbed_state:
        #     print( "self.minimum_cost:", self.minimum_cost)
        #     print( "self.random_association_cost:", self.random_association_cost)
        #     print( "self.max_gumbel:", self.max_gumbel)
        #     print( "self.rand_assoc_gumbel_perturbed_state:", self.rand_assoc_gumbel_perturbed_state)
        #     print( "self.upper_bound_gumbel_perturbed_state:", self.upper_bound_gumbel_perturbed_state)


        if TEST_SUBMATRIX_BOUND1:
            # print( 'trivial upper bound: ', self.upper_bound_gumbel_perturbed_state)
            # print( 'submatrix upper bound: ', self.test_submatrix_upper_bound)
            # print( 'N =', self.remaining_cost_matrix.shape[0])
            self.upper_bound_gumbel_perturbed_state = self.test_submatrix_upper_bound        
        elif TEST_SUBMATRIX_BOUND2:
            # print( 'trivial upper bound: ', self.upper_bound_gumbel_perturbed_state)
            # print( 'gumbel submatrix upper bound: ', self.test_submatrix_upper_bound)
            # print( 'minc submatrix upper bound: ', self.log_minc_upper_bound)
            # print( 'N =', self.remaining_cost_matrix.shape[0])

            # self.upper_bound_gumbel_perturbed_state = self.log_minc_upper_bound + truncated_gumbel(n=1, truncation=gumbel_truncation)
            self.upper_bound_gumbel_perturbed_state = self.log_minc_extended_upper_bound + truncated_gumbel(n=1, truncation=gumbel_truncation)

        elif TEST_SUBMATRIX_BOUND3:
            #another bound
            #https://ac-els-cdn-com.stanford.idm.oclc.org/S002437950400299X/1-s2.0-S002437950400299X-main.pdf?_tid=fa4d00ee-39a5-4030-b7c1-28bb5fbc76c0&acdnat=1534454814_a7411b3006e0e092622de35cbf015275
            # equation (6), U^M(A)
            sub_matrix = np.exp(-self.remaining_cost_matrix)            
            assert(sub_matrix.shape[0] == sub_matrix.shape[1])
            N = sub_matrix.shape[0]            

            minc_extended_upper_bound2 = 1.0
            for row in range(N):
                sorted_row = sorted(sub_matrix[row], reverse=True)
                row_sum = 0
                for col in range(N):
                    row_sum += sorted_row[col] * delta(col+1)
                minc_extended_upper_bound2 *= row_sum
            self.log_minc_extended_upper_bound2 = np.log(minc_extended_upper_bound2)                
            for row, col in self.required_cells:
                self.log_minc_extended_upper_bound2 -= self.orig_cost_matrix[row][col]
            if USE_1_GUMBEL:
                pass #use original A star bound

                #use the log_minc_extended_upper_bound2 permanent upper bound for this submatrix
                # self.upper_bound_gumbel_perturbed_state = self.log_minc_extended_upper_bound2 + self.max_gumbel_1
                
                #test using exact permanent as upper bound
                # exact_permanent = calc_permanent_rysers(sub_matrix)
                # self.upper_bound_gumbel_perturbed_state = np.log(exact_permanent) + self.max_gumbel_1

            else:
                self.upper_bound_gumbel_perturbed_state = self.log_minc_extended_upper_bound2 + truncated_gumbel(n=1, truncation=gumbel_truncation)
   
        else:
            assert(self.rand_assoc_gumbel_perturbed_state <= self.upper_bound_gumbel_perturbed_state + .0001), (self.rand_assoc_gumbel_perturbed_state, self.upper_bound_gumbel_perturbed_state, -self.minimum_cost, -self.random_association_cost)

        if DEBUG:
            print( "New Node:")
            print( "self.required_cells:", self.required_cells )
            print( "self.excluded_cells:", self.excluded_cells )
            print()
            print()

    def sample_association_uniform(self):
        '''
        sample an association from among this node's asscociations uniformly, assumes
        only one row contains excluded cells
        '''
        assert(self.orig_cost_matrix.shape[0] == self.orig_cost_matrix.shape[1])
        N = self.orig_cost_matrix.shape[0]

        rows_containing_required_cells = set()
        cols_containing_required_cells = set()
        rows_containing_excluded_cells = set()
        cols_containing_excluded_cells = set()
        for row, col in self.excluded_cells:
            rows_containing_excluded_cells.add(row)
            cols_containing_excluded_cells.add(col)
        for row, col in self.required_cells:
            assert(row not in rows_containing_required_cells)
            rows_containing_required_cells.add(row)
            assert(col not in cols_containing_required_cells)
            cols_containing_required_cells.add(col)

            if row in rows_containing_excluded_cells:
                rows_containing_excluded_cells.remove(row)
            if col in cols_containing_excluded_cells:
                cols_containing_excluded_cells.remove(col)
        if len(rows_containing_excluded_cells) > 1:
            print( "more than 1 row contains excluded cells!!!")
            print( "self.excluded_cells:", self.excluded_cells)
            print( "self.required_cells:", self.required_cells)
            print( "rows_containing_excluded_cells:", rows_containing_excluded_cells)
            print( "cols_containing_excluded_cells:", cols_containing_excluded_cells)
            assert(False)


        sampled_associations = []
        sampled_associations.extend(self.required_cells)
        assert(len(rows_containing_excluded_cells) == 1 or len(rows_containing_excluded_cells) == 0), len(rows_containing_excluded_cells)
        if len(rows_containing_excluded_cells) == 1:
            row_with_excluded_cells = rows_containing_excluded_cells.pop()
            included_columns = []
            for col in range(N):
                if (col not in cols_containing_excluded_cells) and (col not in cols_containing_required_cells):
                    included_columns.append(col)
            #sample the column for the row with excluded cells
            row_w_excl_cells_sampled_col = np.random.choice(included_columns)
            sampled_associations.append((row_with_excluded_cells, row_w_excl_cells_sampled_col))
            # print( "for row with excluded cells, sampled:", (row_with_excluded_cells, row_w_excl_cells_sampled_col))
            # print( "included_columns:", included_columns)
            # print( "cols_containing_excluded_cells:", cols_containing_excluded_cells)
        else:
            row_w_excl_cells_sampled_col = -99 #junk value a col will never take
            row_with_excluded_cells = -99 #junk value a row will never take

        remaining_cols = set()
        for col in range(N):
            if (col not in cols_containing_required_cells) and (col != row_w_excl_cells_sampled_col):
                remaining_cols.add(col)

        # print( "before sampling remaining rows:")
        # print( "remaining_cols:", remaining_cols)
        # print( "sampled_associations:", sampled_associations)
        for row in range(N):
            if (row != row_with_excluded_cells) and (row not in rows_containing_required_cells):
                sampled_col = np.random.choice(list(remaining_cols))
                sampled_associations.append((row, sampled_col))
                remaining_cols.remove(sampled_col)

        assert(len(remaining_cols) == 0), (remaining_cols, sampled_associations, len(sampled_associations), self.required_cells, cols_containing_required_cells)
        assert(len(sampled_associations) == N)

        random_association_cost = 0
        for (row, col) in sampled_associations:
            #np.asscalar important for avoiding overflow problems
            random_association_cost += np.asscalar(self.orig_cost_matrix[row][col])

        return sampled_associations, random_association_cost

    def sample_association_non_uniform(self):
        '''
        sample an association from among this node's asscociations
        NOTE: this is not uniform, even though it should be for gumbels, FIX if promising
        '''
        assert(self.orig_cost_matrix.shape[0] == self.orig_cost_matrix.shape[1])
        N = self.orig_cost_matrix.shape[0]

        required_rows = {}
        required_columns = {}
        for (row, col) in self.required_cells:
            assert(row not in required_rows)
            assert(col not in required_columns)
            required_rows[row] = col
            required_columns[col] = row

        sampled_associations = []
        sampled_columns = []
        sampled_rows = []
        possible_columns_for_each_row = defaultdict(set)

        for row in range(N):
            if row in required_rows:
                sampled_associations.append((row, required_rows[row]))
                sampled_columns.append(required_rows[row])
                sampled_rows.append(row)
                possible_columns_for_each_row[row] = set()
        for row in range(N):
            if row in required_rows:
                continue
            for col in range(N):
                if (col in required_columns) or ((row, col) in self.excluded_cells):
                    continue
                possible_columns_for_each_row[row].add(col)

        # print()
        # print( 'required_cells:')
        # print( self.required_cells)
        # print( 'excluded_cells:')
        # print( self.excluded_cells)
        while len(sampled_associations) < N:
            # print( )
            # print( "len(sampled_associations):", len(sampled_associations))
            # print( "possible_columns_for_each_row:", possible_columns_for_each_row)
            row_with_fewest_possible_columns = None
            fewest_possible_columns = None
            for row, possible_columns in possible_columns_for_each_row.items():
                if (row not in sampled_rows) and \
                   (row_with_fewest_possible_columns == None or len(possible_columns) < fewest_possible_columns):
                    row_with_fewest_possible_columns = row
                    fewest_possible_columns = len(possible_columns)
            # print( "sampling row:", row_with_fewest_possible_columns)
            #sample a col for the row with fewest possible column associations (or tied for fewest)
            # if len(possible_columns_for_each_row[row_with_fewest_possible_columns]) == 0:
            #     print( 'required_cells:')
            #     print( self.required_cells)
            #     print( 'excluded_cells:')
            #     print( self.excluded_cells)
            assert(len(possible_columns_for_each_row[row_with_fewest_possible_columns]) > 0), "error in sample_association_non_uniform, this row has no possible column associations"
            sampled_column = np.random.choice(list(possible_columns_for_each_row[row_with_fewest_possible_columns]))
            sampled_associations.append((row_with_fewest_possible_columns, sampled_column))
            sampled_columns.append(sampled_column)
            sampled_rows.append(row_with_fewest_possible_columns)
            #remove the sampled column from associations with other rows
            for row, possible_columns in possible_columns_for_each_row.items():
                if sampled_column in possible_columns_for_each_row[row]:
                    possible_columns_for_each_row[row].remove(sampled_column)
        for row, possible_columns in possible_columns_for_each_row.items():
            assert(len(possible_columns) == 0)

        random_association_cost = 0
        for (row, col) in sampled_associations:
            #np.asscalar important for avoiding overflow problems
            random_association_cost += np.asscalar(self.orig_cost_matrix[row][col])
 
        return sampled_associations, random_association_cost

    def get_min_cost_assignment(self):
        min_cost_assignment = self.required_cells[:]
        min_cost_assignment.extend(self.min_cost_associations)
        if DEBUG:
            return (self.minimum_cost, min_cost_assignment, self.excluded_cells, self.required_cells[:], self.min_cost_associations)
        else:
            return (self.minimum_cost, min_cost_assignment, self.orig_cost_matrix_index)

    def partition(self):
        '''
        Partition this node by associations_to_partition_by, as described in [1]

        Output:
        - partition: a list of mutually disjoint Nodes, whose union with the minimum assignment
            of this node forms the set of possible assignments represented by this node
        '''
        # print( '#'*80)
        # print( "partition called on node with assignment_count =", self.assignment_count)

        #test
        test_max_gumbel = truncated_gumbel(n=self.assignment_count, truncation=self.gumbel_truncation)
        if self.max_gumbel > test_max_gumbel:
            # print "first gumbel larger"
            FIRST_GUMBEL_LARGER.append(1)
        else:
            # print "second gumbel larger"
            FIRST_GUMBEL_LARGER.append(0)
        # print "first gumbel larger fraction of time:", np.mean(FIRST_GUMBEL_LARGER)

        #done testing
        associations_to_partition_by = []
        for association in self.random_associations:
            if association not in self.required_cells:
                associations_to_partition_by.append(association)
        associations_to_partition_by = sorted(associations_to_partition_by, key=lambda x: x[0])
        # print( "associations_to_partition_by:", associations_to_partition_by)
        partition = []
        cur_required_cells = self.required_cells[:]

        #the number of assignments in each partitioned Node
        partition_assignment_counts = []
        if DEBUG:
            print( '!'*40, 'Debug partition()', '!'*40)
            print( len(associations_to_partition_by) - 1)

        for idx in range(len(associations_to_partition_by) - 1):
            cur_assoc = associations_to_partition_by[idx]
            cur_excluded_cells = self.excluded_cells[:]
            cur_excluded_cells.append(cur_assoc)
            if DEBUG:
                print( "idx:", idx)
                print( "cur_required_cells:", cur_required_cells)
                print( "cur_excluded_cells:", cur_excluded_cells)
                print( "self.excluded_cells: ", self.excluded_cells)
                print( "self.required_cells: ", self.required_cells)
                #check we haven't made a mistake
                for assoc in cur_excluded_cells:
                    assert(not(assoc in cur_required_cells))
                for i in range(len(cur_required_cells)):
                    for j in range(i+1, len(cur_required_cells)):
                        assert(cur_required_cells[i][0] != cur_required_cells[j][0] and
                               cur_required_cells[i][1] != cur_required_cells[j][1])
                          
            new_node = Node(self.orig_cost_matrix, cur_required_cells, cur_excluded_cells,
                                  self.orig_cost_matrix_index, gumbel_truncation=self.max_gumbel)
            cur_required_cells.append(cur_assoc)
           
            if new_node.assignment_count > 0:
                partition.append(new_node)
                partition_assignment_counts.append(new_node.assignment_count)
            else: #this node contains 0 assignments, don't add to partition
                pass

        #the sum of assignments over each partitioned node + 1 (the minimum assignment in this node)
        #should be equal to the number of assignments in this node
        assert(self.assignment_count == sum(partition_assignment_counts) + 1), (self.assignment_count, partition_assignment_counts, sum(partition_assignment_counts))
        return partition

    def partition_pick_order(self):
        '''
        Partition this node by associations_to_partition_by, as described in [1]
        pick the order to partition, greedily choosing the smallest upper bound
        Output:
        - partition: a list of mutually disjoint Nodes, whose union with the minimum assignment
            of this node forms the set of possible assignments represented by this node
        '''
        # print( '#'*80)
        # print( "partition called on node with assignment_count =", self.assignment_count)

        associations_to_partition_by = set()
        for association in self.random_associations:
            if association not in self.required_cells:
                associations_to_partition_by.add(association)
        associations_to_partition_by = sorted(associations_to_partition_by, key=lambda x: x[0])
        # print( "associations_to_partition_by:", associations_to_partition_by)
        partition = []
        cur_required_cells = self.required_cells[:]

        #the number of assignments in each partitioned Node
        partition_assignment_counts = []

        #if we have excluded cells, first partition using the association in the same row as the excluded cells
        if len(self.excluded_cells) > 0:
            for excluded_cell in self.excluded_cells:
                assert(excluded_cell[0] == self.excluded_cells[0][0]) #make sure all excluded cells are in the same row
            first_assoc = None
            for cur_assoc in associations_to_partition_by:
                if cur_assoc[0] == self.excluded_cells[0][0]: #found the assignment in the same row as excluded cells
                    first_assoc = cur_assoc
                    break
            assert(first_assoc is not None)
            associations_to_partition_by.remove(first_assoc)
            cur_excluded_cells = self.excluded_cells[:]
            cur_excluded_cells.append(cur_assoc)
                          
            new_node = Node(self.orig_cost_matrix, cur_required_cells, cur_excluded_cells,
                                  self.orig_cost_matrix_index, gumbel_truncation=self.max_gumbel)
            cur_required_cells.append(cur_assoc)
           
            if new_node.assignment_count > 0:
                partition.append(new_node)
                partition_assignment_counts.append(new_node.assignment_count)
            else: #this node contains 0 assignments, don't add to partition
                pass

        while len(associations_to_partition_by) > 0:
            best_assoc = None
            best_bound = np.inf
            for cur_assoc in associations_to_partition_by:
                init_len = len(self.excluded_cells)
                self.excluded_cells.append(cur_assoc)
                temp_remaining_cost_matrix = self.construct_remaining_cost_matrix()
                del self.excluded_cells[-1]
                assert(len(self.excluded_cells) == init_len)                
                sub_matrix = np.exp(-temp_remaining_cost_matrix)
                cur_sub_matrix_permanent_UB = minc_extended_UB2(sub_matrix)
                if cur_sub_matrix_permanent_UB < best_bound:
                    best_bound = cur_sub_matrix_permanent_UB
                    best_assoc = cur_assoc
            assert(best_assoc is not None)
            associations_to_partition_by.remove(best_assoc)

            cur_excluded_cells = self.excluded_cells[:]
            cur_excluded_cells.append(best_assoc)
                          
            new_node = Node(self.orig_cost_matrix, cur_required_cells, cur_excluded_cells,
                                  self.orig_cost_matrix_index, gumbel_truncation=self.max_gumbel)
            cur_required_cells.append(best_assoc)
           
            if new_node.assignment_count > 0:
                partition.append(new_node)
                partition_assignment_counts.append(new_node.assignment_count)
            else: #this node contains 0 assignments, don't add to partition
                pass

        #the sum of assignments over each partitioned node + 1 (the minimum assignment in this node)
        #should be equal to the number of assignments in this node
        assert(self.assignment_count == sum(partition_assignment_counts) + 1), (self.assignment_count, partition_assignment_counts, sum(partition_assignment_counts))
        return partition


    #transform the cost matrix into the "remaining cost matrix" as described in [1]
    def construct_remaining_cost_matrix(self):
        remaining_cost_matrix = np.array(self.orig_cost_matrix, copy=True)
      
        #replace excluded_cell locations with infinity in the remaining cost matrix
        for (row, col) in self.excluded_cells:
            remaining_cost_matrix[row][col] = sys.maxint

        rows_to_delete = []
        cols_to_delete = []
        for (row, col) in self.required_cells: #remove required rows and columns
            rows_to_delete.append(row)
            cols_to_delete.append(col)

        #create sorted lists of rows and columns to delete, where indices are sorted in increasing
        #order, e.g. [1, 4, 5, 9]
        sorted_rows_to_delete = sorted(rows_to_delete)
        sorted_cols_to_delete = sorted(cols_to_delete)

        #delete rows and cols, starting with LARGEST indices to preserve validity of smaller indices
        for row in reversed(sorted_rows_to_delete):
            remaining_cost_matrix = np.delete(remaining_cost_matrix, row, 0)
        for col in reversed(sorted_cols_to_delete):
            remaining_cost_matrix = np.delete(remaining_cost_matrix, col, 1)


        return remaining_cost_matrix

    def get_orig_indices(self, rcm_indices):
        '''
        Take a list of indices in the remaining cost matrix and transform them into indices
        in the original cost matrix

        Inputs:
        - rcm_indices: (list of pairs) indices in the remaining cost matrix

        Outputs:
        - orig_indices: (list of pairs) converted indices in the original cost matrix
        '''

        orig_indices = rcm_indices

        deleted_rows = []
        deleted_cols = []
        for (row, col) in self.required_cells: #remove required rows and columns
            deleted_rows.append(row)
            deleted_cols.append(col)

        #create sorted lists of rows and columns that were deleted in the remaining cost matrix,
        #where indices are sorted in increasing order, e.g. [1, 4, 5, 9]
        sorted_deleted_rows = sorted(deleted_rows)
        sorted_deleted_cols = sorted(deleted_cols)

        for deleted_row in sorted_deleted_rows:
            for idx, (row, col) in enumerate(orig_indices):
                if deleted_row <= row:
                    orig_indices[idx] = (orig_indices[idx][0] + 1, orig_indices[idx][1])

        for deleted_col in sorted_deleted_cols:
            for idx, (row, col) in enumerate(orig_indices):
                if deleted_col <= col:
                    orig_indices[idx] = (orig_indices[idx][0], orig_indices[idx][1] + 1)

        return orig_indices

    def count_assignments(self, use_brute_force = False):
        '''
        Count the number of assignments in this Node.  

        If the node contains only required cells this is easy.  For a Node containing A
        required cells and an orig_cost_matrix of size (N, N), the number of assignments
        is (N - A)!

        If the node contains excluded cells we can use the Inclusion-Exclusion formula
        (https://www.whitman.edu/mathematics/cgt_online/book/section02.01.html) to transform
        the problem.  If the node contains B excluded cells, we brute force enumerate
        the number of assignments in 2^B sub-problems over only required cells.

        - orig_cost_matrix: (2d numpy array) the original cost matrix
        - required_cells: (list of pairs) where each pair represents a (zero indexed) location
            in the assignment matrix that must be a 1
        - excluded_cells: (list of pairs) where each pair represents a (zero indexed) location
            in the assignment matrix that must be a 0

        '''
        # make sure the required cells are valid: we can't require a column or row twice
        required_rows = []
        required_columns = []
        for (row, col) in self.required_cells:
            assert(row not in required_rows)
            assert(col not in required_columns)
            required_rows.append(row)
            required_columns.append(col)
        # make sure the excluded cells are valid: we can't exclude a required cell and
        # we can't exclude a cell more than once
        for (row, col) in self.excluded_cells:
            assert((row, col) not in self.required_cells)
            assert(self.excluded_cells.count((row, col)) == 1), self.excluded_cells.count((row, col))
        # prune unnecessary excluded cells: we don't need to exclude a cell that is in the
        # same row or column as a required cell
        pruned_excluded_cells = []
        for (row, col) in self.excluded_cells:
            if (row not in required_rows) and (col not in required_columns):
                pruned_excluded_cells.append((row, col))
        self.excluded_cells = pruned_excluded_cells



        assert(self.orig_cost_matrix.shape[0] == self.orig_cost_matrix.shape[1])
        N = self.orig_cost_matrix.shape[0]
        num_required_cells = len(self.required_cells)
        num_excluded_cells = len(self.excluded_cells)

        analytic_assignment_count = math.factorial(N - num_required_cells - 1) * (N - num_required_cells - num_excluded_cells)

        
        if use_brute_force:
            #iterate over 2^num_excluded_cells possible intersections of excluded cells
            intersections = list(itertools.product([0, 1], repeat=num_excluded_cells))
            assert(len(intersections) == 2**num_excluded_cells)
            assignment_count = 0
            for cur_intersection in intersections:
                cur_intersection_rows = []
                cur_intersection_cols = []
                cur_intersection_empty = False
                for cell_idx, cur_cell_included in enumerate(cur_intersection):
                    if cur_cell_included == 1:
                        (cur_cell_row, cur_cell_col) = self.excluded_cells[cell_idx]
                        # we intersect the complement of these excluded cells so the intersection when two
                        # excluded cells contain the same row or col is empty (can't require the same row or col)
                        if (cur_cell_row in cur_intersection_rows) or (cur_cell_col in cur_intersection_cols):
                            cur_intersection_empty = True
                            break
                        cur_intersection_rows.append(cur_cell_row)
                        cur_intersection_cols.append(cur_cell_col)
                if not cur_intersection_empty:
                    num_required_cells_in_cur_intersection = cur_intersection.count(1)
                    assignment_count += (-1)**num_required_cells_in_cur_intersection * math.factorial(N - num_required_cells - num_required_cells_in_cur_intersection)
            
            assert(analytic_assignment_count == assignment_count)

        # return assignment_count
        return analytic_assignment_count


def brute_force_k_best_assignments(k, cost_matrix):
    assert(cost_matrix.shape[0] == cost_matrix.shape[1])
    n = cost_matrix.shape[0]
    all_perm_mats = gen_permutation_matrices(n)
    costs = []
    for pm in all_perm_mats:
        costs.append(np.trace(np.dot(pm, np.transpose(cost_matrix))))

    min_costs = [] #list of triples (smallest k costs, corresponding permutation matrix, 0)
    for i in range(k):
        (min_key, min_cost) = min(enumerate(costs), key=itemgetter(1)) #find the next smallest cost
        min_costs.append((min_cost, all_perm_mats[min_key], 0))
        del all_perm_mats[min_key]
        del costs[min_key]

    return min_costs


def convert_perm_list_to_array(list_):
    '''
    Input:
    - list_: a list of length n, where each element is a pair representing an element in an nxn
    permutation matrix that is a 1. 

    Output:
    - matrix: numpy array of the permutation matrix
    '''
    array = np.zeros((len(list_), len(list_)))
    for indices in list_:
        array[indices] = 1
    return array

def convert_perm_array_to_list(arr):
    '''
    Input:
    - matrix: numpy array of the permutation matrix

    Output:
    - list_: a list of length n, where each element is a pair representing an element in an nxn
    permutation matrix that is a 1. 

    '''
    list_ = []
    assert(arr.shape[0] == arr.shape[1])
    n = arr.shape[0]
    for row in range(n):
        for col in range(n):
            if(arr[(row, col)] == 1):
                list_.append((row,col))
    assert(len(list_) == n)
    return list_

def check_assignments_match(best_assignments1, best_assignments2):
    for (index, (cost, assignment_list, cost_matrix_index)) in enumerate(best_assignments1):
        #assert(cost == best_assignments2[index][0]), (cost, best_assignments2[index][0])
        np.testing.assert_allclose(cost, best_assignments2[index][0], rtol=1e-5, atol=0), (cost, best_assignments2[index][0])
#        assert((convert_perm_list_to_array(assignment_list) == best_assignments2[index][1]).all), (assignment_list, best_assignments1[index][1])
        assert(cost_matrix_index == best_assignments2[index][2]), (cost_matrix_index, best_assignments2[index][2], cost, best_assignments2[index][0], convert_perm_list_to_array(assignment_list), best_assignments2[index][1])

def gen_permutation_matrices(n):
    '''
    return a list of all nxn permutation matrices (numpy arrays)
    '''
    all_permutation_matrices = []
    for cur_permutation in itertools.permutations([i for i in range(n)], n):
        cur_perm_mat = np.zeros((n,n))
        for row, col in enumerate(cur_permutation):
            cur_perm_mat[row][col] = 1
        all_permutation_matrices.append(cur_perm_mat)
    return all_permutation_matrices


def test_against_brute_force(N,k,iters):
    '''
    Test our implementation of Murty's algorithm to find the k best assignments for a given cost
    matrix against a brute force approach.
    Inputs:
    - N: use a random cost matrix of size (NxN)
    - k: find k best solutions
    - iters: number of random problems to solve and check
    '''
    for test_iter in range(iters):
        cost_matrix = np.random.rand(N,N)*1000

        best_assignments = k_best_assignments(k, cost_matrix)
        if DEBUG:
            for (idx, assignment) in enumerate(best_assignments):
                print( idx, ":   ", assignment)
            print()
        print( "calculated with Hungarian")
        best_assignments_brute_force = brute_force_k_best_assignments(k, cost_matrix)
        print( "calculated with brute force")

        if DEBUG:
            for (idx, (cost, perm)) in enumerate(best_assignments_brute_force):
                print( idx, ":   ", (cost, convert_perm_array_to_list(perm)))
        check_assignments_match(best_assignments, best_assignments_brute_force)
        print( "match!")

def test_mult_cost_matrices(num_cost_matrices, N,k,iters):
    '''
    Inputs:
    - num_cost_matrices: number of cost matrices to use
    - N: use a random cost matrices of size (NxN)
    - k: find k best solutions
    - iters: number of random problems to solve and check
    '''
    for test_iter in range(iters):
        cost_matrices = []
        for i in range(num_cost_matrices):
            cost_matrices.append(np.random.rand(N,N))

        best_assignments_mult = k_best_assign_mult_cost_matrices(k, cost_matrices)
#        print( best_assignments_mult)
        print( 'calculated')
        #now try using k_best_assignments k times
        best_assignments_naive = []
        for (idx, cur_cost_matrix) in enumerate(cost_matrices):
            cur_best_assignments = k_best_assignments(k, cur_cost_matrix)
            for (idx1, cur_assignment) in enumerate(cur_best_assignments):
                cur_best_assignments[idx1] = (cur_assignment[0], cur_assignment[1], idx)
            best_assignments_naive.extend(cur_best_assignments)
        best_assignments_naive.sort(key=itemgetter(0))
        best_assignments_naive = best_assignments_naive[0:k]
        print( "calculated naive")

#        print()
#        print( best_assignments_naive)
        check_assignments_match(best_assignments_mult, best_assignments_naive)
        print( 'match!')
#        if DEBUG:
#            for (idx, assignment) in enumerate(best_assignments):
#                print( idx, ":   ", assignment)
#            print()
#        print( "calculated with Hungarian")
#        best_assignments_brute_force = brute_force_k_best_assignments(k, cost_matrix)
#        print( "calculated with brute force")
#
#        if DEBUG:
#            for (idx, (cost, perm)) in enumerate(best_assignments_brute_force):
#                print( idx, ":   ", (cost, convert_perm_array_to_list(perm)))
#        check_assignments_match(best_assignments, best_assignments_brute_force)
#        print( "match!")


def calc_permanent_rysers(matrix):
    '''
    Exactly calculate the permanent of the given matrix user Ryser's method (faster than calc_permanent)
    '''
    N = matrix.shape[0]
    assert(N == matrix.shape[1])
    #this looks complicated because the method takes and returns a complex matrix,
    #we are only dealing with real matrices so set complex component to 0
    return np.real(rysers_permanent(1j*np.zeros((N,N)) + matrix))

def check_permanent_fraction_in_top_k(N,k,iters):
    '''
    Find the sum of the top k assignments and compare with the trivial bound
    on the remaining assignments of (N!-k)*(the kth best assignment)
    Inputs:
    - N: use a random cost matrix of size (NxN)
    - k: find k best solutions
    - iters: number of random problems to solve and check
    '''
    for test_iter in range(iters):
        matrix = np.random.rand(N,N)
        for row in range(N):
            for col in range(N):
                if matrix[row][col] < .5:
                    matrix[row][col] = matrix[row][col] ** 2
                else:
                    matrix[row][col] = 1 - (1 - matrix[row][col])**2
        # print(("matrix:", matrix))
        cost_matrix = -np.log(matrix)

        best_assignments, partition = k_best_assignments(k, cost_matrix)
        if DEBUG:
            for (idx, assignment) in enumerate(best_assignments):
                print( idx, ":   ", assignment)
            print()

        assert(len(best_assignments) == k)
        top_k_assignment_sum = 0
        for idx in range(k):
            top_k_assignment_sum += np.exp(-best_assignments[idx][0])
        trivial_bound_on_remaining_assignments_sum = (math.factorial(N)-k)*np.exp(-best_assignments[k-1][0])

        #note this assumes we have been given the negative of the assignments, and have thus found the max not min assignments
        bound_on_remaining_assignments_sum = 0
        node_bounds = []
        for node in partition:
            node_bounds.append(node.assignment_count * np.exp(-node.minimum_cost))
        bound_on_remaining_assignments_sum = sum(node_bounds)
        node_bounds = np.sort(node_bounds)

# @@@@@@@@@@@@
#         self.orig_cost_matrix = np.array(orig_cost_matrix, copy=True)
#         self.required_cells = required_cells[:]
#         self.excluded_cells = excluded_cells[:]
#         self.orig_cost_matrix_index = orig_cost_matrix_index
#         #the number of assignments this node contains
#         self.assignment_count = self.count_assignments()
# @@@@@@@@@@@@


        bound_on_remaining_assignments_sum2 = 0
        node_bounds2 = []
        for node in partition:
            matrix = np.exp(-node.orig_cost_matrix)
            assert(matrix.shape[0] == matrix.shape[1])
            N = matrix.shape[0]
            required_rows = []
            required_columns = []
            for (row, col) in node.required_cells:
                assert(row not in required_rows)
                assert(col not in required_columns)
                required_rows.append(row)
                required_columns.append(col)
            node_upper_bound = 1.0
            for row in range(N):
                if row in required_rows:
                    continue
                row_sum = 0
                for col in range(N):
                    if (col in required_columns) or ((row, col) in node.excluded_cells):
                        continue
                    row_sum += matrix[row][col]
                node_upper_bound *= math.factorial(np.ceil(row_sum))**(1/np.ceil(row_sum))
            for (row, col) in node.required_cells:
                node_upper_bound *= matrix[row][col]

            node_bounds2.append(node_upper_bound)

        bound_on_remaining_assignments_sum2 = sum(node_bounds2)
        node_bounds2 = np.sort(node_bounds2)


        combined_bounds = []
        assert(len(node_bounds2) == len(node_bounds))
        for idx in range(len(node_bounds)):
            combined_bounds.append(np.min((node_bounds[idx], node_bounds2[idx])))
        combined_bound_on_remaining_assignments_sum = sum(combined_bounds)

        exact_permanent = calc_permanent_rysers(matrix)

        print( "len(partition) =", len(partition))
        print( "node_bounds[-10] =", node_bounds[-10])
        print( "node_bounds2[-10] =", node_bounds2[-10])
        print( "np.min(node_bounds) =", np.min(node_bounds))
        print( "np.min(node_bounds2) =", np.min(node_bounds2))
        print( "np.max(node_bounds) =", np.max(node_bounds))
        print( "np.max(node_bounds2) =", np.max(node_bounds2))
        print( "np.median(node_bounds) =", np.median(node_bounds))
        print( "np.median(node_bounds2) =", np.median(node_bounds2))
        print( "np.mean(node_bounds) =", np.mean(node_bounds))
        print( "np.mean(node_bounds2) =", np.mean(node_bounds2))
        print( "bound on permanent fraction found =", top_k_assignment_sum/(top_k_assignment_sum + trivial_bound_on_remaining_assignments_sum), "N=", N, "k=", k, "top_k_assignment_sum=", top_k_assignment_sum, "trivial_bound_on_remaining_assignments_sum=", trivial_bound_on_remaining_assignments_sum)
        print( "bound on permanent fraction found =", top_k_assignment_sum/(top_k_assignment_sum + bound_on_remaining_assignments_sum), "N=", N, "k=", k, "top_k_assignment_sum=", top_k_assignment_sum, "bound_on_remaining_assignments_sum=", bound_on_remaining_assignments_sum)
        print( "bound on permanent fraction found =", top_k_assignment_sum/(top_k_assignment_sum + bound_on_remaining_assignments_sum2), "N=", N, "k=", k, "top_k_assignment_sum=", top_k_assignment_sum, "bound_on_remaining_assignments_sum2=", bound_on_remaining_assignments_sum2)
        print( "bound on permanent fraction found =", top_k_assignment_sum/(top_k_assignment_sum + combined_bound_on_remaining_assignments_sum), "N=", N, "k=", k, "top_k_assignment_sum=", top_k_assignment_sum, "combined_bound_on_remaining_assignments_sum=", combined_bound_on_remaining_assignments_sum)
        print( "top_k_assignment_sum/exact_permanent =", top_k_assignment_sum/exact_permanent)
        print( "(top assignment)/(kth assignment) =", np.exp(-best_assignments[0][0])/np.exp(-best_assignments[k-1][0]), "top assignment =", np.exp(-best_assignments[0][0]), "kth assignment =", np.exp(-best_assignments[k-1][0]))
        print( )


def test_sub_permanant_differences(N):
    '''
    1. generate a random (N x N) matrix
    2. calculate the permanent when the ith row and column are removed, for i=1 to N
    3. compare permanents from 2.
    4. estimate/bound permanents from 2 and see how we compare with true values
    '''
    matrix = np.random.rand(N,N)
    for row in range(N):
        for col in range(N):
            if matrix[row][col] < .5:
                matrix[row][col] = matrix[row][col] ** 1
            else:
                matrix[row][col] = 1 - (1 - matrix[row][col])**1
    sub_matrix_permanents = []
    print( "matrix:")
    print( matrix)
    print()
    for idx in range(N):
        sub_matrix = np.delete(matrix, idx, 0)
        sub_matrix = np.delete(sub_matrix, idx, 1)
        # print( "sub_matrix:")
        # print( sub_matrix)
        # print()
        # gumbel_expectation_upper_bound = compute_gumbel_upper_bound(sub_matrix, num_perturbations=100)
        # print( "gumbel bound:", gumbel_expectation_upper_bound)
        # exact_permanent = calc_permanent_rysers(sub_matrix)
        # print( "exact_permanent:", np.log(exact_permanent))

        # hm = Munkres()
        # cost_matrix = -np.log(sub_matrix)
        # min_cost_association_list = hm.compute(cost_matrix.tolist())
        # minimum_cost = 0
        # for (row,col) in min_cost_association_list:
        #     minimum_cost += np.asscalar(cost_matrix[row][col])
        # max_assignment = np.exp(-minimum_cost)
        # trivial_bound = math.factorial(N)*max_assignment

        # (barv_log_perm_estimate, lower_bound, upper_bound) = \
        #     approx_permanent3(sub_matrix, k_2=100, log_base=np.e, debugged=True, w_min=None)

        # print( "trivial_bound:", np.log(trivial_bound))
        # print( "barv_log_perm_estimate:", barv_log_perm_estimate)

        USE_EXTENSION_1 = False
        if USE_EXTENSION_1:
            minc_extended_upper_bound = 1.0
            for row in range(sub_matrix.shape[0]):
                row_sum = 0
                largest_row_element = 0.0
                num_pos_elements_in_row = 0
                for col in range(sub_matrix.shape[1]):
                    row_sum += sub_matrix[row][col]
                    if sub_matrix[row][col] > largest_row_element:
                        largest_row_element = sub_matrix[row][col]
                    if sub_matrix[row][col] > 0:
                        num_pos_elements_in_row += 1

                minc_extended_upper_bound *= largest_row_element * mu_func(num_pos_elements_in_row, row_sum/largest_row_element)
                # print( "minc_extended_upper_bound =", minc_extended_upper_bound, num_pos_elements_in_row, row_sum, largest_row_element)
            log_minc_extended_upper_bound = np.log(minc_extended_upper_bound)       

        else:
            minc_extended_upper_bound2 = 1.0
            for row in range(sub_matrix.shape[0]):
                sorted_row = sorted(sub_matrix[row], reverse=True)
                row_sum = 0
                for col in range(sub_matrix.shape[1]):
                    row_sum += sorted_row[col] * delta(col+1)
                minc_extended_upper_bound2 *= row_sum      
            log_minc_extended_upper_bound = np.log(minc_extended_upper_bound2)   
        # sub_matrix_permanents.append((np.log(exact_permanent), log_minc_extended_upper_bound, barv_log_perm_estimate, gumbel_expectation_upper_bound, np.log(trivial_bound)))
        sub_matrix_permanents.append([log_minc_extended_upper_bound])

    sub_matrix_permanents = sorted(sub_matrix_permanents, key=lambda x: x[0])
    print( "sub_matrix_permanents:", sub_matrix_permanents)


def plot_top_assignments(N):
    matrix = np.random.rand(N,N)
    for row in range(N):
        for col in range(N):
            if matrix[row][col] < .5:
                matrix[row][col] = matrix[row][col] ** 1
            else:
                matrix[row][col] = 1 - (1 - matrix[row][col])**1

    cost_matrix = -np.log(matrix)

    k = math.factorial(N)
    exact_permanent = calc_permanent_rysers(matrix)
    permanent_fraction = 20
    lowest_cost_assignments, partition = k_best_assignments(k=k, cost_matrix=cost_matrix, min_sum=exact_permanent/permanent_fraction)

    # assert(len(lowest_cost_assignments) == k)
    top_k_assignment_sum = 0
    largest_assignments = []
    top_k_assignment_sums = []
    for idx in range(len(lowest_cost_assignments)):
        top_k_assignment_sum += np.exp(-lowest_cost_assignments[idx][0])
        largest_assignments.append(np.exp(-lowest_cost_assignments[idx][0]))
        top_k_assignment_sums.append(top_k_assignment_sum)
    # assert(np.abs(top_k_assignment_sum - exact_permanent) < 0.0001)


    normalized_largest_assignments = [assignmnet_val/largest_assignments[0] for assignmnet_val in largest_assignments]
    normalized_top_k_assignment_sums = [assignment_sum/top_k_assignment_sum for assignment_sum in top_k_assignment_sums]
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot([i for i in range(len(largest_assignments))], normalized_largest_assignments, 'r+', label='top k assignment' , markersize=10)
    ax.plot([i for i in range(len(largest_assignments))], normalized_top_k_assignment_sums, 'g*', label='top k sum/(permanent*%d)'%permanent_fraction , markersize=10)

    plt.title('sorted assignments, N=%d' % N)
    plt.xlabel('assignment index')
    plt.ylabel('assignment value)')
    # Put a legend below current axis
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
              fancybox=False, shadow=False, ncol=2, numpoints = 1)

    fig.savefig('sorted_assignments_N=%d' % N, bbox_extra_artists=(lgd,), bbox_inches='tight')    

    plt.close()

def truncated_gumbel(n, truncation):
    '''
    https://cmaddis.github.io/
    sample the max of n gumbels with location 0 and scale 1, truncated at truncation
    '''
    assert(n>0), n
    gumbel = np.random.gumbel() + math.log(n)
    return -np.log(np.exp(-gumbel) + np.exp(-truncation))

def compare_truncated_gumbel(n_vals, truncation):
    '''
    https://cmaddis.github.io/
    sample the max of n gumbels with location 0 and scale 1, truncated at truncation
    sample 1 gumbel and return the truncated value scaled for each value of n in n_vals
    '''
    return_vals = []
    gumbel = np.random.gumbel() 

    for n in n_vals:
        assert(n>0), n    
        cur_gumbel = gumbel + math.log(n)
        return_vals.append(-np.log(np.exp(-cur_gumbel) + np.exp(-truncation)))
    return return_vals

def sample_max_of_gumbels(n):
    '''
    sample the max of n gumbels with location 0 and scale 1

    see, e.g. https://cs.stanford.edu/~ermon/papers/kim-sabharwal-ermon.pdf,
    equation 1 where all n weights are set to 1
    '''
    return np.random.gumbel(loc=np.log(n))

def test_permanent_matrix_with_swapped_rows_cols(N=12):
    matrix = np.random.rand(N,N)
    print( 'original matrix exact permanent:', calc_permanent_rysers(matrix))
    row_swapped_matrix = np.array(matrix, copy=True)
    row_swapped_matrix[3][:] = matrix[8][:]
    row_swapped_matrix[8][:] = matrix[3][:]
    print( 'row_swapped_matrix exact permanent:', calc_permanent_rysers(row_swapped_matrix))


    row_col_swapped_matrix = np.array(row_swapped_matrix, copy=True)
    row_col_swapped_matrix[:][3] = row_swapped_matrix[:][8]
    row_col_swapped_matrix[:][8] = row_swapped_matrix[:][3]
    print( 'row_col_swapped_matrix exact permanent:', calc_permanent_rysers(row_col_swapped_matrix))

# @profile
def test_gumbel_permanent_estimation(N,iters,num_samples=1,matrix=None, exact_log_Z=None):
    '''
    Find the sum of the top k assignments and compare with the trivial bound
    on the remaining assignments of (N!-k)*(the kth best assignment)
    Inputs:
    - N: use a random cost matrix of size (NxN)
    - iters: number of random problems to solve and check
    '''
    if matrix is None:
        matrix = np.random.rand(N,N)
        for row in range(N):
            for col in range(N):
                if matrix[row][col] < .5:
                    # matrix[row][col] = matrix[row][col] ** 1
                    matrix[row][col] = 0
                else:
                    # matrix[row][col] = 1 - (1 - matrix[row][col])**1
                    matrix[row][col] = 1

    # matrix, exact_permanent = create_diagonal2(N=N, k=10, zero_one=False)
    # exact_log_Z = np.log(exact_permanent) 


    # print(("matrix:", matrix))
    cost_matrix = -np.log(matrix)
    all_samples_of_log_Z = []
    node_count_plus_heap_sizes_list = []
    number_of_times_partition_called_list = []
    runtimes_list = []
    all_sampled_associations = []
    wall_time = 0
    for test_iter in range(iters):
        # print test_iter
        if test_iter % 1000 == 0:
            print "completed", test_iter, "iters"
        t1 = time.time()
        samples_of_log_Z, node_count_plus_heap_size, number_of_times_partition_called_for_each_sample, cur_sampled_associations = \
            sample_log_permanent_with_gumbels(cost_matrix, num_samples=num_samples)
        t2 = time.time()
        runtimes_list.append(t2-t1)
        all_sampled_associations.extend(cur_sampled_associations)
        cur_wall_time = t2-t1
        wall_time += cur_wall_time
        # print( "np.mean(samples_of_log_Z):", np.mean(samples_of_log_Z), "node_count_plus_heap_size:", node_count_plus_heap_size, "wall time =", t2-t1)
        all_samples_of_log_Z.append(np.mean(samples_of_log_Z))
        node_count_plus_heap_sizes_list.append(node_count_plus_heap_size)
        number_of_times_partition_called_list.append(number_of_times_partition_called_for_each_sample)
    print()
    # print( "exact log(permanent):", np.log(calc_permanent_rysers(matrix)))
    print( "np.mean(all_samples_of_log_Z) =", np.mean(all_samples_of_log_Z))
    print( "np.mean(node_count_plus_heap_sizes_list) =", np.mean(node_count_plus_heap_sizes_list))
    print( "np.mean(number_of_times_partition_called_list) =", np.mean(number_of_times_partition_called_list))
    print( "number_of_times_partition_called_list :", number_of_times_partition_called_list)
    print( "len(node_count_plus_heap_sizes_list) =", len(node_count_plus_heap_sizes_list))

    log_Z_estimate = np.mean(all_samples_of_log_Z)
    return number_of_times_partition_called_list, node_count_plus_heap_sizes_list, runtimes_list, all_sampled_associations, wall_time, log_Z_estimate, all_samples_of_log_Z, exact_log_Z


def plot_runtime_vs_N(pickle_file_paths=['./number_of_times_partition_called_for_each_n.pickle'], pickle_file_paths2=None):
    n_vals_mean = []
    run_time_vals_mean = []
    number_of_times_partition_called_vals_mean = []
    all_n_vals = []
    all_run_time_vals = []
    all_number_of_times_partition_called_vals = []
    for pickle_file_path in pickle_file_paths:
        f = open(pickle_file_path, 'rb')
        number_of_times_partition_called_for_each_n = pickle.load(f)
        f.close()

        for n, (number_of_times_partition_called_list, node_count_plus_heap_sizes_list, runtimes_list) in number_of_times_partition_called_for_each_n.items():
            if n < 5:
                continue
            all_n_vals.extend([n for i in range(len(number_of_times_partition_called_list))])
            all_run_time_vals.extend(runtimes_list)
            all_number_of_times_partition_called_vals.extend(number_of_times_partition_called_list)
            # n_vals_mean.append(math.log(n))
            n_vals_mean.append(n)
            run_time_vals_mean.append(math.log(np.mean(runtimes_list)))
            # run_time_vals_mean.append(np.mean(runtimes_list))
            number_of_times_partition_called_vals_mean.append(math.log(np.mean(number_of_times_partition_called_list)))
            # number_of_times_partition_called_vals_mean.append(np.mean(number_of_times_partition_called_list))
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(n_vals_mean, run_time_vals_mean, 'r+', label='run_time_vals_mean' , markersize=10)
    # ax.plot(all_n_vals, all_run_time_vals, 'r+', label='run_time_vals_mean' , markersize=10)
    ax.plot(n_vals_mean, number_of_times_partition_called_vals_mean, 'gx', label='number_of_times_partition_called_vals_mean' , markersize=10)
    plt.title('permanent sampling scaling')
    plt.xlabel('N (matrix dimension)')
    plt.ylabel('')
    # Put a legend below current axis
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
              fancybox=False, shadow=False, ncol=2, numpoints = 1)
    # print( 'hi1')
    # plt.show()
    # print( 'hi2')

    POLY_FIT = False #fit with some polynomials
    if POLY_FIT:
        p3 = np.poly1d(np.polyfit(x=n_vals_mean, y=number_of_times_partition_called_vals_mean, deg=3))
        p5 = np.poly1d(np.polyfit(x=n_vals_mean, y=number_of_times_partition_called_vals_mean, deg=5))
        # p2 = np.poly1d(np.polyfit(x=n_vals_mean, y=number_of_times_partition_called_vals_mean, deg=2))
        p4 = np.poly1d(np.polyfit(x=n_vals_mean, y=number_of_times_partition_called_vals_mean, deg=4))
        xp = np.linspace(0, 60, 200)
        _ = plt.plot(xp, p3(xp), '-', xp, p5(xp), '--', xp, p4(xp), '--')

    # fig.savefig('permanent sampling scaling old perm UB', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent sampling_amortized scaling new perm UB', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent sampling scaling pick partition order perm UB', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent sampling scaling new2 perm UB', bbox_extra_artists=(lgd,), bbox_inches='tight')    

    if pickle_file_paths2 is not None:
        n_vals_mean = []
        run_time_vals_mean = []
        number_of_times_partition_called_vals_mean = []
        all_n_vals = []
        all_run_time_vals = []
        all_number_of_times_partition_called_vals = []

        for pickle_file_path in pickle_file_paths2:
            f = open(pickle_file_path, 'rb')
            number_of_times_partition_called_for_each_n = pickle.load(f)
            f.close()

            for n, (number_of_times_partition_called_list, node_count_plus_heap_sizes_list, runtimes_list) in number_of_times_partition_called_for_each_n.items():
                if n < 5:
                    continue
                all_n_vals.extend([n for i in range(len(number_of_times_partition_called_list))])
                all_run_time_vals.extend(runtimes_list)
                all_number_of_times_partition_called_vals.extend(number_of_times_partition_called_list)
                # n_vals_mean.append(math.log(n))
                n_vals_mean.append(n)
                run_time_vals_mean.append(math.log(np.mean(runtimes_list)))
                # run_time_vals_mean.append(np.mean(runtimes_list))
                number_of_times_partition_called_vals_mean.append(math.log(np.mean(number_of_times_partition_called_list)))
                # number_of_times_partition_called_vals_mean.append(np.mean(number_of_times_partition_called_list))

        ax.plot(n_vals_mean, run_time_vals_mean, 'y+', label='run_time_vals_mean 2' , markersize=10)
        # ax.plot(all_n_vals, all_run_time_vals, 'r+', label='run_time_vals_mean' , markersize=10)
        ax.plot(n_vals_mean, number_of_times_partition_called_vals_mean, 'mx', label='number_of_times_partition_called_vals_mean 2' , markersize=10)

    fig.savefig('permanent sampling scaling 2 plots perm UB', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent sampling scaling single gumbel perm UB', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent sampling scaling single gumbel, pick partition order perm UB', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent sampling scaling single gumbel, close to 01 matrix perm UB', bbox_extra_artists=(lgd,), bbox_inches='tight')    

    plt.close()


def test_optimized_minc(z, N):
    A = np.ones((N, N))
    for row in range(N):
        for col in range(N):
            if col > row:
                A[row][col] = z
    exact_permanent = calc_permanent_rysers(A)

    #another bound
    #https://ac-els-cdn-com.stanford.idm.oclc.org/S002437950400299X/1-s2.0-S002437950400299X-main.pdf?_tid=fa4d00ee-39a5-4030-b7c1-28bb5fbc76c0&acdnat=1534454814_a7411b3006e0e092622de35cbf015275
    # equation (6), U^M(A)

    minc_extended_upper_bound2 = 1.0
    for row in range(N):
        sorted_row = sorted(A[row], reverse=True)
        row_sum = 0
        for col in range(N):
            row_sum += sorted_row[col] * delta(col+1)
        minc_extended_upper_bound2 *= row_sum

    #another bound
    #https://ac-els-cdn-com.stanford.idm.oclc.org/S002437950400299X/1-s2.0-S002437950400299X-main.pdf?_tid=fa4d00ee-39a5-4030-b7c1-28bb5fbc76c0&acdnat=1534454814_a7411b3006e0e092622de35cbf015275
    # optimize equation (6), U^M(A) according to equation (9) (that is, rescaling columns)

    optimized_minc_extended_upper_bound2 = optimized_minc_extened_UB2(A)
    assert(optimized_minc_extended_upper_bound2 < minc_extended_upper_bound2)    

    print("exact_permanent =", exact_permanent)
    print("minc_extended_upper_bound2 =", minc_extended_upper_bound2)
    print("optimized_minc_extended_upper_bound2 =", optimized_minc_extended_upper_bound2)

def test_sampling_correctness(N=3, ITERS=10000000, matrix_to_use='rand'):
    # check for smaller n
    # check total variatianal distance and compare with sampling normally
    '''
    Test that we're sampling from the correct distributions over assocations
    '''
    #key: length n tuple of associations, each is a tuple of length 2
    #value: dict, with keys:
    #   - 'true probability', value: (float)
    #   - 'empirical probability', value: (float)

    assert(N in [3, 4,5])

    if matrix_to_use == 'rand':
        matrix = np.random.rand(N,N)
        for row in range(N):
            for col in range(N):
                if matrix[row][col] < .5:
                    matrix[row][col] = matrix[row][col] ** 1
                else:
                    matrix[row][col] = 1 - (1 - matrix[row][col])**1
    elif matrix_to_use == 'matrix1':
        #matrix for seed=11 where we don't seem to have a problem
        matrix = np.array([[0.18026969, 0.01947524, 0.46321853],
                           [0.72493393, 0.4202036 , 0.4854271 ],
                           [0.01278081, 0.48737161, 0.94180665]])
    elif matrix_to_use == 'matrix2':
        #matrix for seed=10 where we seem to have a problem
        matrix = np.array([[0.77132064, 0.02075195, 0.63364823],
                           [0.74880388, 0.49850701, 0.22479665],
                           [0.19806286, 0.76053071, 0.16911084]])
    else:
        assert(False), "wrong parameter for matrix_to_use!!"
    exact_permanent = calc_permanent_rysers(matrix)


    all_associations = {}
    list_of_all_associations = []
    list_of_all_true_probabilities = []
    for row1 in range(N):
        for row2 in range(N):
            if row2 == row1:
                continue
            for row3 in range(N):
                if row3 in [row1, row2]:
                    continue
                if N == 3:
                    cur_association = ((0, row1), (1, row2), (2, row3))
                    true_probability = matrix[0, row1]*matrix[1, row2]*matrix[2, row3]/exact_permanent
                    all_associations[cur_association] = {'true probability': true_probability,
                                                         'empirical probability': 0.0}
                    list_of_all_associations.append(cur_association)                                                         
                    list_of_all_true_probabilities.append(true_probability)
                    continue

                for row4 in range(N):
                    if row4 in [row1, row2, row3]:
                        continue
                    if N == 4:
                        cur_association = ((0, row1), (1, row2), (2, row3), (3, row4))
                        true_probability = matrix[0, row1]*matrix[1, row2]*matrix[2, row3]*matrix[3, row4]/exact_permanent
                        all_associations[cur_association] = {'true probability': true_probability,
                                                             'empirical probability': 0.0}
                        list_of_all_associations.append(cur_association)                                                             
                        list_of_all_true_probabilities.append(true_probability)
                        continue
                    for row5 in range(N):
                        if row5 in [row1, row2, row3, row4]:
                            continue
                        cur_association = ((0, row1), (1, row2), (2, row3), (3, row4), (4, row5))
                        true_probability = matrix[0, row1]*matrix[1, row2]*matrix[2, row3]*matrix[3, row4]*matrix[4, row5]/exact_permanent
                        all_associations[cur_association] = {'true probability': true_probability,
                                                             'empirical probability': 0.0}
                        list_of_all_associations.append(cur_association)                                                             
                        list_of_all_true_probabilities.append(true_probability)
    # print(all_associations)
    # sleep(temp)
    number_of_times_partition_called_list, node_count_plus_heap_sizes_list, runtimes_list, all_sampled_associations, wall_time, log_Z_estimate, all_samples_of_log_Z = \
        test_gumbel_permanent_estimation(N, iters=ITERS, num_samples=1, matrix=matrix)
        # test_gumbel_permanent_estimation(N, iters=1, num_samples=1, matrix=matrix)
    print("wall_time =", wall_time)
    print("log_Z_estimate =", log_Z_estimate)
    print("np.log(exact_permanent) =", np.log(exact_permanent))

    for sampled_association in all_sampled_associations:
        all_associations[tuple(sorted(sampled_association, key=lambda x: x[0]))]['empirical probability'] += 1/len(all_sampled_associations)

    #key: association
    #value: empirical probability based on ITERS standard samples from the true distribution
    empirical_probs_sampled_standard = defaultdict(int)
    for i in range(ITERS):
        association_idx = np.random.choice(len(list_of_all_associations), p=list_of_all_true_probabilities)
        empirical_probs_sampled_standard[list_of_all_associations[association_idx]] += 1/ITERS

    empirical_probs = []
    empirical_probs_sampled_standard_list = []
    true_probs = []
    standard_tv_distance = 0
    gumbel_tv_distance = 0
    max_standard_error = 0
    max_gumbel_error = 0
    for assoc, probs in all_associations.items():
        true_probs.append(probs['true probability'])
        empirical_probs.append(probs['empirical probability'])
        empirical_probs_sampled_standard_list.append(empirical_probs_sampled_standard[assoc])
        gumbel_tv_distance += np.abs(true_probs[-1] - empirical_probs[-1])
        standard_tv_distance += np.abs(true_probs[-1] - empirical_probs_sampled_standard_list[-1])
        # print "cur gumbel error =", np.abs(true_probs[-1] - empirical_probs[-1])
        if np.abs(true_probs[-1] - empirical_probs[-1]) > max_gumbel_error:
            max_gumbel_error = np.abs(true_probs[-1] - empirical_probs[-1])
        if np.abs(true_probs[-1] - empirical_probs_sampled_standard_list[-1]) > max_standard_error:
            max_standard_error = np.abs(true_probs[-1] - empirical_probs_sampled_standard_list[-1])
    arbitrary_prob_indices = [i for i in range(len(all_associations))]


    print "gumbel_tv_distance =", gumbel_tv_distance
    print "standard_tv_distance =", standard_tv_distance

    print "max_gumbel_error =", max_gumbel_error
    print "max_standard_error =", max_standard_error

    print
    
    simulated_gumbel_mean = 0
    for i in range(ITERS):
        simulated_gumbel_mean += np.random.gumbel()
    simulated_gumbel_mean /= ITERS
    simulated_gumbel_error = np.abs(simulated_gumbel_mean - np.euler_gamma)
    print "hypothesized gumbel log_Z_estimate error =", np.abs(log_Z_estimate - np.log(exact_permanent))
    print "simulated_gumbel_error =", simulated_gumbel_error

    print
    statistic, critical_values, significance_level = scipy.stats.anderson(all_samples_of_log_Z, dist='gumbel_r')
    print "Anderson-Darling statistic:", statistic
    print "Anderson-Darling critical_values:", critical_values
    print "Anderson-Darling significance_level:", significance_level

    statistic, critical_values, significance_level = scipy.stats.anderson(np.random.gumbel(loc=0, size=ITERS), dist='gumbel_r')
    print "true gumbel Anderson-Darling statistic:", statistic
    print "true gumbel Anderson-Darling critical_values:", critical_values
    print "true gumbel Anderson-Darling significance_level:", significance_level

    statistic, critical_values, significance_level = scipy.stats.anderson(np.random.gumbel(loc=1, scale=10, size=ITERS), dist='gumbel_r')
    print "true gumbel location 1 Anderson-Darling statistic:", statistic
    print "true gumbel location 1 Anderson-Darling critical_values:", critical_values
    print "true gumbel location 1 Anderson-Darling significance_level:", significance_level

    statistic, critical_values, significance_level = scipy.stats.anderson(np.random.normal(loc=0, size=ITERS), dist='gumbel_r')
    print "normal Anderson-Darling statistic:", statistic
    print "normal Anderson-Darling critical_values:", critical_values
    print "normal Anderson-Darling significance_level:", significance_level

    print
    print "ITERS =", ITERS
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(arbitrary_prob_indices, empirical_probs, 'r+', label='empirical_probs' , markersize=10)
    ax.plot(arbitrary_prob_indices, empirical_probs_sampled_standard_list, 'b+', label='empirical_probs sampled standard' , markersize=10)
    ax.plot(arbitrary_prob_indices, true_probs, 'gx', label='true_probs' , markersize=10)
    plt.title('permutation probabilities')
    plt.xlabel('arbitrary index')
    plt.ylabel('probability')
    # Put a legend below current axis
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
              fancybox=False, shadow=False, ncol=2, numpoints = 1)
    # plt.show()
    fig_file_name = "4test_sampling_empirical_vs_true_probabilities_N=%d_seed=%d_1gumbel_iters=%d_matrix=%s"%(N, SEED, ITERS, matrix_to_use)
    pickle_file_name = "./pickle_experiment_results/" + fig_file_name + ".pickle"
    fig.savefig(fig_file_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

    f = open(pickle_file_name, 'wb')
    pickle.dump((empirical_probs, empirical_probs_sampled_standard_list, true_probs, all_sampled_associations, log_Z_estimate, all_samples_of_log_Z), f)
    f.close() 


    return gumbel_tv_distance, standard_tv_distance

    # sleep(-3)


    # for assoc, probs in all_associations.items():
    #     assert(np.abs(probs['true probability'] - probs['empirical probability']) < .01), (probs['true probability'], probs['empirical probability'])

    # print('Looks good!!')

def report_test_sampling_correctness_from_pickle(N=3, ITERS=10000, matrix_to_use='rand'):
    # check for smaller n
    # check total variatianal distance and compare with sampling normally
    '''
    Test that we're sampling from the correct distributions over assocations
    '''
    #key: length n tuple of associations, each is a tuple of length 2
    #value: dict, with keys:
    #   - 'true probability', value: (float)
    #   - 'empirical probability', value: (float)

    assert(N in [3, 4,5])

    if matrix_to_use == 'rand':
        matrix = np.random.rand(N,N)
        for row in range(N):
            for col in range(N):
                if matrix[row][col] < .5:
                    matrix[row][col] = matrix[row][col] ** 1
                else:
                    matrix[row][col] = 1 - (1 - matrix[row][col])**1
    elif matrix_to_use == 'matrix1':
        #matrix for seed=11 where we don't seem to have a problem
        matrix = np.array([[0.18026969, 0.01947524, 0.46321853],
                           [0.72493393, 0.4202036 , 0.4854271 ],
                           [0.01278081, 0.48737161, 0.94180665]])
    elif matrix_to_use == 'matrix2':
        #matrix for seed=10 where we seem to have a problem
        matrix = np.array([[0.77132064, 0.02075195, 0.63364823],
                           [0.74880388, 0.49850701, 0.22479665],
                           [0.19806286, 0.76053071, 0.16911084]])
    else:
        assert(False), "wrong parameter for matrix_to_use!!"
    exact_permanent = calc_permanent_rysers(matrix)

    all_associations = {}
    list_of_all_associations = []
    list_of_all_true_probabilities = []
    for row1 in range(N):
        for row2 in range(N):
            if row2 == row1:
                continue
            for row3 in range(N):
                if row3 in [row1, row2]:
                    continue
                if N == 3:
                    cur_association = ((0, row1), (1, row2), (2, row3))
                    true_probability = matrix[0, row1]*matrix[1, row2]*matrix[2, row3]/exact_permanent
                    all_associations[cur_association] = {'true probability': true_probability,
                                                         'empirical probability': 0.0}
                    list_of_all_associations.append(cur_association)                                                         
                    list_of_all_true_probabilities.append(true_probability)
                    continue

                for row4 in range(N):
                    if row4 in [row1, row2, row3]:
                        continue
                    if N == 4:
                        cur_association = ((0, row1), (1, row2), (2, row3), (3, row4))
                        true_probability = matrix[0, row1]*matrix[1, row2]*matrix[2, row3]*matrix[3, row4]/exact_permanent
                        all_associations[cur_association] = {'true probability': true_probability,
                                                             'empirical probability': 0.0}
                        list_of_all_associations.append(cur_association)                                                             
                        list_of_all_true_probabilities.append(true_probability)
                        continue
                    for row5 in range(N):
                        if row5 in [row1, row2, row3, row4]:
                            continue
                        cur_association = ((0, row1), (1, row2), (2, row3), (3, row4), (4, row5))
                        true_probability = matrix[0, row1]*matrix[1, row2]*matrix[2, row3]*matrix[3, row4]*matrix[4, row5]/exact_permanent
                        all_associations[cur_association] = {'true probability': true_probability,
                                                             'empirical probability': 0.0}
                        list_of_all_associations.append(cur_association)                                                             
                        list_of_all_true_probabilities.append(true_probability)
    # print(all_associations)
    # sleep(temp)
    fig_file_name = "4test_sampling_empirical_vs_true_probabilities_N=%d_seed=%d_1gumbel_iters=%d_matrix=%s"%(N, SEED, ITERS, matrix_to_use)
    pickle_file_name = "./pickle_experiment_results/" + fig_file_name + ".pickle"
    print "pickle_file_name:", pickle_file_name
    f = open(pickle_file_name, 'rb')
    (empirical_probs, empirical_probs_sampled_standard_list, true_probs, all_sampled_associations, log_Z_estimate, all_samples_of_log_Z) = pickle.load(f)
    f.close() 

    # print("wall_time =", wall_time)
    # print("log_Z_estimate =", log_Z_estimate)
    print("np.log(exact_permanent) =", np.log(exact_permanent))

    for sampled_association in all_sampled_associations:
        all_associations[tuple(sorted(sampled_association, key=lambda x: x[0]))]['empirical probability'] += 1/len(all_sampled_associations)

    #key: association
    #value: empirical probability based on ITERS standard samples from the true distribution
    empirical_probs_sampled_standard = defaultdict(int)
    for i in range(ITERS):
        association_idx = np.random.choice(len(list_of_all_associations), p=list_of_all_true_probabilities)
        empirical_probs_sampled_standard[list_of_all_associations[association_idx]] += 1/ITERS

    empirical_probs = []
    empirical_probs_sampled_standard_list = []
    true_probs = []
    standard_tv_distance = 0
    gumbel_tv_distance = 0
    max_standard_error = 0
    max_gumbel_error = 0
    for assoc, probs in all_associations.items():
        true_probs.append(probs['true probability'])
        empirical_probs.append(probs['empirical probability'])
        empirical_probs_sampled_standard_list.append(empirical_probs_sampled_standard[assoc])
        gumbel_tv_distance += np.abs(true_probs[-1] - empirical_probs[-1])
        standard_tv_distance += np.abs(true_probs[-1] - empirical_probs_sampled_standard_list[-1])
        # print "cur gumbel error =", np.abs(true_probs[-1] - empirical_probs[-1])
        if np.abs(true_probs[-1] - empirical_probs[-1]) > max_gumbel_error:
            max_gumbel_error = np.abs(true_probs[-1] - empirical_probs[-1])
        if np.abs(true_probs[-1] - empirical_probs_sampled_standard_list[-1]) > max_standard_error:
            max_standard_error = np.abs(true_probs[-1] - empirical_probs_sampled_standard_list[-1])
    arbitrary_prob_indices = [i for i in range(len(all_associations))]


    print "gumbel_tv_distance =", gumbel_tv_distance
    print "standard_tv_distance =", standard_tv_distance

    print "max_gumbel_error =", max_gumbel_error
    print "max_standard_error =", max_standard_error

    print
    
    simulated_gumbel_mean = 0
    for i in range(ITERS):
        simulated_gumbel_mean += np.random.gumbel()
    simulated_gumbel_mean /= ITERS
    simulated_gumbel_error = np.abs(simulated_gumbel_mean - np.euler_gamma)
    print "hypothesized gumbel log_Z_estimate error =", np.abs(log_Z_estimate - np.log(exact_permanent))
    print "simulated_gumbel_error =", simulated_gumbel_error

    print
    statistic, critical_values, significance_level = scipy.stats.anderson(all_samples_of_log_Z, dist='gumbel_r')
    print "Anderson-Darling statistic:", statistic
    print "Anderson-Darling critical_values:", critical_values
    print "Anderson-Darling significance_level:", significance_level

    statistic, critical_values, significance_level = scipy.stats.anderson(np.random.gumbel(loc=0, size=ITERS), dist='gumbel_r')
    print "true gumbel Anderson-Darling statistic:", statistic
    print "true gumbel Anderson-Darling critical_values:", critical_values
    print "true gumbel Anderson-Darling significance_level:", significance_level

    statistic, critical_values, significance_level = scipy.stats.anderson(np.random.gumbel(loc=1, scale=10, size=ITERS), dist='gumbel_r')
    print "true gumbel location 1 Anderson-Darling statistic:", statistic
    print "true gumbel location 1 Anderson-Darling critical_values:", critical_values
    print "true gumbel location 1 Anderson-Darling significance_level:", significance_level

    statistic, critical_values, significance_level = scipy.stats.anderson(np.random.normal(loc=0, size=ITERS), dist='gumbel_r')
    print "normal Anderson-Darling statistic:", statistic
    print "normal Anderson-Darling critical_values:", critical_values
    print "normal Anderson-Darling significance_level:", significance_level

    print
    print "ITERS =", ITERS
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(arbitrary_prob_indices, empirical_probs, 'r+', label='empirical_probs' , markersize=10)
    ax.plot(arbitrary_prob_indices, empirical_probs_sampled_standard_list, 'b+', label='empirical_probs sampled standard' , markersize=10)
    ax.plot(arbitrary_prob_indices, true_probs, 'gx', label='true_probs' , markersize=10)
    plt.title('permutation probabilities')
    plt.xlabel('arbitrary index')
    plt.ylabel('probability')
    # Put a legend below current axis
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
              fancybox=False, shadow=False, ncol=2, numpoints = 1)
    # plt.show()
    fig.savefig(fig_file_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

    return gumbel_tv_distance, standard_tv_distance





def test_logZ_error(N=3, ITERS=10000000, matrix_to_use='rand'):
    # check for smaller n
    # check total variatianal distance and compare with sampling normally
    '''
    Test that we're sampling from the correct distributions over assocations
    '''
    #key: length n tuple of associations, each is a tuple of length 2
    #value: dict, with keys:
    #   - 'true probability', value: (float)
    #   - 'empirical probability', value: (float)

    if matrix_to_use == 'rand':
        matrix = np.random.rand(N,N)
        for row in range(N):
            for col in range(N):
                if matrix[row][col] < .5:
                    matrix[row][col] = (matrix[row][col] ** 5)
                else:
                    matrix[row][col] = (1 - (1 - matrix[row][col])**5)
    elif matrix_to_use == 'matrix1':
        #matrix for seed=11 where we don't seem to have a problem
        matrix = np.array([[0.18026969, 0.01947524, 0.46321853],
                           [0.72493393, 0.4202036 , 0.4854271 ],
                           [0.01278081, 0.48737161, 0.94180665]])
    elif matrix_to_use == 'matrix2':
        #matrix for seed=10 where we seem to have a problem
        matrix = np.array([[0.77132064, 0.02075195, 0.63364823],
                           [0.74880388, 0.49850701, 0.22479665],
                           [0.19806286, 0.76053071, 0.16911084]])
    else:
        assert(False), "wrong parameter for matrix_to_use!!"
    exact_permanent = calc_permanent_rysers(matrix)

    number_of_times_partition_called_list, node_count_plus_heap_sizes_list, runtimes_list, all_sampled_associations, wall_time, log_Z_estimate, all_samples_of_log_Z = \
        test_gumbel_permanent_estimation(N, iters=ITERS, num_samples=1, matrix=matrix)
        # test_gumbel_permanent_estimation(N, iters=1, num_samples=1, matrix=matrix)
    print("wall_time =", wall_time)
    print("log_Z_estimate =", log_Z_estimate)
    print("np.log(exact_permanent) =", np.log(exact_permanent))

    
    simulated_gumbel_mean = 0
    for i in range(ITERS):
        simulated_gumbel_mean += np.random.gumbel()
    simulated_gumbel_mean /= ITERS
    simulated_gumbel_error = np.abs(simulated_gumbel_mean - np.euler_gamma)
    print "hypothesized gumbel log_Z_estimate error =", np.abs(log_Z_estimate - np.log(exact_permanent))
    print "simulated_gumbel_error =", simulated_gumbel_error





def test_total_variation_distance(N=3, tv_ITERS=10):
    gumbel_tv_distance_list = []
    standard_tv_distance_list = []
    for i in range(tv_ITERS):
        gumbel_tv_distance, standard_tv_distance = test_sampling_correctness(N=3, ITERS=1000)
        gumbel_tv_distance_list.append(gumbel_tv_distance)
        standard_tv_distance_list.append(standard_tv_distance)

    print "np.mean(gumbel_tv_distance_list) =", np.mean(gumbel_tv_distance_list)
    print "np.mean(standard_tv_distance_list) =", np.mean(standard_tv_distance_list)

def test_permanent_bound_tightness(N):
    matrix = np.random.rand(N,N)
    for row in range(N):
        for col in range(N):
            if matrix[row][col] < .5:
                matrix[row][col] = matrix[row][col] ** 1
                # matrix[row][col] = 0
            else:
                matrix[row][col] = 1 - (1 - matrix[row][col])**1
                # matrix[row][col] = 1

    exact_permanent = calc_permanent_rysers(matrix)
    # exact_permanent = 0

    minc2_sub_matrix_min_bound = 0
    row = 0
    sub_matrix = np.delete(matrix, row, 0)
    for col in range(matrix.shape[1]):
        cur_sub_matrix = np.delete(sub_matrix, col, 1)
        minc2_sub_matrix_min_bound += matrix[row, col] * minc_extended_UB2(cur_sub_matrix)

    minc2_sub_matrix2_min_bound = 0
    row = 0
    sub_matrix = np.delete(matrix, row, 0)
    for col in range(matrix.shape[1]):
        cur_sub_matrix = np.delete(sub_matrix, col, 1)
        for col1 in range(cur_sub_matrix.shape[1]):
            cur_sub_matrix2 = np.delete(cur_sub_matrix, 0, 0)
            cur_sub_matrix2 = np.delete(cur_sub_matrix2, col1, 1)
            assert(cur_sub_matrix2.shape[0] == cur_sub_matrix2.shape[1] and cur_sub_matrix2.shape[1] == matrix.shape[1]-2)
            minc2_sub_matrix2_min_bound += matrix[row, col]* matrix[1, col1] * minc_extended_UB2(cur_sub_matrix2)

    minc_UB2 = minc_extended_UB2(matrix)
    minc_UB2_of_transpose = minc_extended_UB2(np.transpose(matrix))
    # optimized_minc_extended_upper_bound2 = optimized_minc_extened_UB2(matrix)
    optimized_minc_extended_upper_bound2 = 0

    print 'log(exact_permanent) =', np.log(exact_permanent)
    print 'log extended minc2 UB =', np.log(minc_UB2)
    print 'log extended minc2 UB of transpose =', np.log(minc_UB2_of_transpose)
    print 'log optimized extended minc2 UB =', np.log(optimized_minc_extended_upper_bound2)
    print 'difference =', np.log(minc_UB2) - np.log(exact_permanent)
    print     
    print 'log extended minc2 submatrix UB =', np.log(minc2_sub_matrix_min_bound)    
    print 'log(minc2) - log(submatrix) =', np.log(minc_UB2) - np.log(minc2_sub_matrix_min_bound)
    print 'log extended minc2 submatrix2 UB =', np.log(minc2_sub_matrix2_min_bound)    
    print 'log(minc2) - log(submatrix2) =', np.log(minc_UB2) - np.log(minc2_sub_matrix2_min_bound)

def test_gumbel_mean_concentration(samples):
    mean_off_by_more_than_point1 = []
    for idx in range(1000):
        mean = 0
        for i in range(samples):
            mean += np.random.gumbel()
        mean /= samples
        mean -= np.euler_gamma
        if abs(mean) > .1:
            mean_off_by_more_than_point1.append(1)
        else:
            mean_off_by_more_than_point1.append(0)
    # print "gumbel mean =", mean
    print "fraction mean_off_by_more_than_point1 =", np.mean(mean_off_by_more_than_point1)


def calculate_minc2_proposal_probability(matrix, associations):
    '''
    - matrix: square np.array of size NxN
    - associations: list with length N, containing some permutation of the numbers
        0 through N-1.
    '''
    
    proposal_probability = 1.0
    # for row, col in enumerate(associations):
    while (True):
        assert(len(associations) == matrix.shape[0])
        row = 0
        col = associations[0]
        assert(matrix.shape[0] == matrix.shape[1] and matrix.shape[0] >= 2)
        if matrix.shape[0] == 2:
            proposal_probability *= matrix[0,col]*matrix[1,1-col]/(matrix[0,col]*matrix[1,1-col] + matrix[0,1-col]*matrix[1,col])
            return proposal_probability
        den = 0.0
        for col1 in range(matrix.shape[1]):
            sub_matrix = np.delete(matrix, row, 0)
            sub_matrix = np.delete(sub_matrix, col1, 1)
            den += matrix[row, col1] * minc_extended_UB2(sub_matrix)
        sub_matrix = np.delete(matrix, row, 0)
        sub_matrix = np.delete(sub_matrix, col, 1)
        num = matrix[row, col] * minc_extended_UB2(sub_matrix)
        proposal_probability *= num/den
        associations = associations[1:]
        for idx in range(len(associations)):
            if associations[idx] > col:
                associations[idx] -= 1
        matrix = sub_matrix

def test_max_proposal_probability_error(N):
    matrix = np.random.rand(N,N)
    min_proposal_prob = np.inf
    max_true_prob = -np.inf
    max_proposal_error_ratio = -np.inf
    exact_permanent = calc_permanent_rysers(matrix)
    for cur_associations in itertools.permutations(range(N)):
        cur_associations = list(cur_associations)
        proposal_probability = calculate_minc2_proposal_probability(matrix, cur_associations)
        true_probability = 1.0
        for row, col in enumerate(cur_associations):
            true_probability *= matrix[row, col]
        true_probability /= exact_permanent
        proposal_error_ratio = true_probability/proposal_probability

        if proposal_probability < min_proposal_prob:
            min_proposal_prob = proposal_probability
        if true_probability > max_true_prob:
            max_true_prob = true_probability
        if proposal_error_ratio > max_proposal_error_ratio:
            max_proposal_error_ratio = proposal_error_ratio

    print "min_proposal_prob =", min_proposal_prob
    print "max_true_prob =", max_true_prob
    print "max_true_prob/min_proposal_prob =", max_true_prob/min_proposal_prob
    print "max_proposal_error_ratio =", max_proposal_error_ratio

def create_diagonal2(N, k, zero_one=False):
    '''
    create NxN matrix with blocks on the diagonal of size at most kxk
    '''    
    diag_matrix = np.zeros((N, N))
    diag_matrix_permanent = 1.0
    print diag_matrix.shape
    for i in range(N): #only has to go up to the number of blocks
        if N > k:
            cur_block = np.random.rand(k,k)
            if zero_one:
                for row in range(k):
                    for col in range(k):
                        if cur_block[row][col] < .1:
                            cur_block[row][col] = 0
                        else:
                            cur_block[row][col] = 1

            cur_block_exact_permanent = calc_permanent_rysers(cur_block)
            diag_matrix_permanent *= cur_block_exact_permanent
            diag_matrix[i*k:(i+1)*k, \
                        i*k:(i+1)*k] = cur_block
            N -= k
        else:
            cur_block = np.random.rand(N,N)
            if zero_one:            
                for row in range(N):
                    for col in range(N):
                        if cur_block[row][col] < .1:
                            cur_block[row][col] = 0
                        else:
                            cur_block[row][col] = 1

            cur_block_exact_permanent = calc_permanent_rysers(cur_block)
            diag_matrix_permanent *= cur_block_exact_permanent
            diag_matrix[i*k:, \
                        i*k:] = cur_block
            return diag_matrix, diag_matrix_permanent


if __name__ == "__main__":
    # test_max_proposal_probability_error(6)
    # sleep(-2)

    # test_gumbel_mean_concentration(samples=1000)
    # sleep(-4)
    # test_permanent_bound_tightness(N=20)   
    # exit(0)
    # test_total_variation_distance()
    # test_sampling_correctness(ITERS=100000, matrix_to_use='rand')
    # test_sampling_correctness(ITERS=100000, matrix_to_use='rand')
    # report_test_sampling_correctness_from_pickle(ITERS=100000, matrix_to_use='rand')
    # sleep(-2)
    # test_optimized_minc(z=1/6, N=16)
    # test_sub_permanant_differences(N=100)

    # test_logZ_error(N=12, ITERS=100, matrix_to_use='rand')
    # sleep(-1)


    ITERS = 5
    NUM_SAMPLES = 1
    # pickle_file_path = './number_of_times_partition_called_for_each_n_%diters.pickle' % (ITERS)
    # pickle_file_path = './number_of_times_partition_called_for_each_n_%diters_newPermUB2_numSamples=%d_close01matrix.pickle' % (ITERS, NUM_SAMPLES)
    # pickle_file_path = './number_of_times_partition_called_for_each_n_%diters_newPermUB2_numSamples=%d.pickle' % (ITERS, NUM_SAMPLES)
    # pickle_file_path = './number_of_times_partition_called_for_each_n_%diters_newPermUB2_numSamples=%d_pickPartitionOrder.pickle' % (ITERS, NUM_SAMPLES)
    
    # pickle_file_path = './number_of_times_partition_called_for_each_n_%diters_origAStar_diagMatrix.pickle' % (ITERS)
    # pickle_file_path = './number_of_times_partition_called_for_each_n_%diters_origAStar.pickle' % (ITERS)
    pickle_file_path = './number_of_times_partition_called_for_each_n_%diters_origAStar_01matrix.pickle' % (ITERS)

    # pickle_file_path = './number_of_times_partition_called_for_each_n_%diters_singleGumbel_n81plus.pickle' % (ITERS)
    # pickle_file_path = './number_of_times_partition_called_for_each_n_%diters_singleGumbel_pickPartitionOrder.pickle' % (ITERS)
    # pickle_file_path = './number_of_times_partition_called_for_each_n_%diters_singleGumbel_close01matrix.pickle' % (ITERS)
    
    # pickle_file_path = './number_of_times_partition_called_for_each_n_%diters_newPermUB_numSamples=%d.pickle' % (ITERS, NUM_SAMPLES)
    # pickle_file_path = './number_of_times_partition_called_for_each_n_%diters_newPermUB_n71plus.pickle' % ITERS

    # plot_runtime_vs_N()
    # plot_runtime_vs_N(pickle_file_paths = ['./number_of_times_partition_called_for_each_n_%diters.pickle' % ITERS, \
    #                                        './number_of_times_partition_called_for_each_n_%diters_n61plus.pickle' % ITERS])
    
    # plot_runtime_vs_N(pickle_file_paths = [pickle_file_path])
   
    # plot_runtime_vs_N(pickle_file_paths = ['./number_of_times_partition_called_for_each_n_%diters_newPermUB.pickle' % ITERS, \
    #                                        './number_of_times_partition_called_for_each_n_%diters_newPermUB_n61plus.pickle' % ITERS, \
    #                                        './number_of_times_partition_called_for_each_n_%diters_newPermUB_n71plus.pickle' % ITERS])
    
    # plot_runtime_vs_N(pickle_file_paths = ['./number_of_times_partition_called_for_each_n_%diters_singleGumbel.pickle' % (ITERS)],
    #                   pickle_file_paths2 =['./number_of_times_partition_called_for_each_n_%diters_newPermUB.pickle' % ITERS, \
    #                                        './number_of_times_partition_called_for_each_n_%diters_newPermUB_n61plus.pickle' % ITERS, \
    #                                        './number_of_times_partition_called_for_each_n_%diters_newPermUB_n71plus.pickle' % ITERS])

    # sleep(3)

    number_of_times_partition_called_for_each_n = {}
    for N in range(5, 140):
    # for N in range(30, 60):
        print( "N =", N)
        number_of_times_partition_called_list, node_count_plus_heap_sizes_list, runtimes_list, all_sampled_associations, wall_time, log_Z_estimate, all_samples_of_log_Z, exact_log_Z = test_gumbel_permanent_estimation(N, iters=ITERS, num_samples=NUM_SAMPLES)
        number_of_times_partition_called_for_each_n[N] = (runtimes_list, all_samples_of_log_Z, exact_log_Z)
        
        # number_of_times_partition_called_for_each_n[N] = (number_of_times_partition_called_list, node_count_plus_heap_sizes_list, runtimes_list)

    
        f = open(pickle_file_path, 'wb')
        pickle.dump(number_of_times_partition_called_for_each_n, f)
        f.close() 

    sleep(-1)
    N = 40 # cost matrices of size (NxN) 

    test_permanent_matrix_with_swapped_rows_cols(N)
    sleep(3)




