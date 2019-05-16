from __future__ import division
import numpy as np
from munkres import Munkres, print_matrix
import sys
import itertools
import math
from operator import itemgetter
# from permanent import permanent as rysers_permanent
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
import scipy.io
import copy
import os
import networkx as nx
# from libc.stdlib cimport malloc, free
# from libc.math cimport pow

# sys.path.insert(0, '/Users/jkuck/research/atlas_AAAI_2017/refactored_multi_model/')
# from permanent_model import compute_gumbel_upper_bound, approx_permanent3
# from boundZ import calculate_gumbel_slack
DEBUG_3 = False

class permanent_Upper_Bounds:
    def __init__(self):
        self.upper_bounds_dictionary = {}

    def print_bounds(self):
        print "MATRIX_PERMANENT_UBS:", self.upper_bounds_dictionary

    def get_upper_bound(self, matrix_idx, required_cells, sub_matrix=None, compare_naive_max_weight_bound=True, verbose=False):
        '''
        Return an upper bound on the specified sub matrix.  
        If a cached (and possibly tightened) bound
        exists return this, otherwise compute from scratch

        Inputs:
        - matrix_idx: (int) specifies an index for the original matrix
        - required_cells: (tuple of tuples) specifies a submatrix
        - sub_matrix: (numpy array) this is the actual submatrix whose permanent
            we upper bound if the value isn't cached yet.  If None, return a cached
            value or fail if the value is not cached.

        Outputs:
        - upper_bound: (float) upper bound on the specified sub matrix
        '''
        assert(type(required_cells) == type((1,2)))
        if sub_matrix is None:
            assert((matrix_idx, required_cells) in self.upper_bounds_dictionary), "permanent_Upper_Bounds.get_upper_bound called without specifying the sub_matrix without a cached upper bound"
        if (matrix_idx, required_cells) in self.upper_bounds_dictionary:
            upper_bound = self.upper_bounds_dictionary[(matrix_idx, required_cells)]
            if verbose:
                print "cached UB for", (matrix_idx, required_cells), "=", upper_bound

        #original matrix has shape (0, 0), we have no measurements and no targets.  probability of remaining in this state is 1
        elif(required_cells == () and sub_matrix.shape[0] == 0):
            upper_bound = 1.0
            self.upper_bounds_dictionary[(matrix_idx, required_cells)] = upper_bound

        else:
            upper_bound = minc_extended_UB2(sub_matrix)
            if compare_naive_max_weight_bound:
                (association_list, max_assignment) = find_max_assignment(sub_matrix)
                N = sub_matrix.shape[0]
                assert(N == sub_matrix.shape[1])
                naive_max_weight_bound = max_assignment*math.factorial(N)
                upper_bound = min(upper_bound, naive_max_weight_bound)

            if verbose:
                print "newly computed UB for", (matrix_idx, required_cells), "=", upper_bound
            assert(upper_bound > -.000000001)
            if upper_bound < 0:
                upper_bound = 0
            self.upper_bounds_dictionary[(matrix_idx, required_cells)] = upper_bound

        return upper_bound


    def tighten_upper_bound(self, matrix_idx, required_cells, slack, assert_zero=False):
        '''
        Tighten the specified upper bound by subtracting slack from it.

        Inputs:
        - matrix_idx: (int) specifies an index for the original matrix
        - required_cells: (tuple of tuples) specifies a submatrix
        - slack: (float) value to be subtracted from the specified upper bound
        - assert_zero: (bool) if True, require the upper bound is zero after tightening,
            don't correct errors or tighten slack.  Used if we sample an original matrix
            with shape (0, 0) representing 0 measurements and 0 targets: only one thing
            can happen here, remain in the same state

        Outputs:
        - corrected_slack: (float) if subtracting slack made the bound negative, presumably due to a numerical
            error, we subtracted too much slack and reset the bound to zero.  This is the actual slack
            we subtracted after correcting if necessary.
        '''
        # begin debugging
        # print "type(required_cells)", type(required_cells)
        # print "required_cells", required_cells
        assert(type(required_cells) == type((1,2)))
        # if matrix_idx == 0 and required_cells == tuple([(0, 9)]):
        #     print 'tighten_upper_bound called on 0, [(0, 9)], old upper bound =', self.upper_bounds_dictionary[(matrix_idx, required_cells)], 'new upper bound =', self.upper_bounds_dictionary[(matrix_idx, required_cells)] - slack
        # print 'tighten_upper_bound called on', (matrix_idx, required_cells), 'old upper bound =', self.upper_bounds_dictionary[(matrix_idx, required_cells)], 'new upper bound =', self.upper_bounds_dictionary[(matrix_idx, required_cells)] - slack
        
        # end debugging

        assert((matrix_idx, required_cells) in self.upper_bounds_dictionary), "permanent_Upper_Bounds.tighten_upper_bound called on a bound we haven't computed!!"
        original_bound = self.upper_bounds_dictionary[(matrix_idx, required_cells)]
        self.upper_bounds_dictionary[(matrix_idx, required_cells)] -= slack
        if assert_zero: #
            assert(self.upper_bounds_dictionary[(matrix_idx, required_cells)] == 0.0)
            return original_bound

        elif self.upper_bounds_dictionary[(matrix_idx, required_cells)] < 0.0:
            print "< 0.0, tried to subtract too much slack might be ok due to numerical errors but maybe check into more"
            # assert(False), "subtracting too much slack might be ok due to numerical errors, disabled for now" 
            self.upper_bounds_dictionary[(matrix_idx, required_cells)] = 0.0
            return original_bound

        else:
            return slack

    def set_upper_bound(self, matrix_idx, required_cells, new_upper_bound, sub_matrix=None):
        '''
        Update the specified submatrix upper bound with a new bound, possibly computed in a 
        different way, if the new bound is tighter.  NOT RIGHT NOW: Then check that nesting still holds.

        don't require the bound always exists in the dictionary
        check nesting not implemented yet!!

        Inputs:
        - matrix_idx: (int) specifies an index for the original matrix
        - required_cells: (tuple of tuples) specifies a submatrix
        - new_upper_bound: (float) the new upper bound
        - sub_matrix: (numpy array) this is the actual submatrix whose permanent
            upper bound we are updating      

        '''
        assert(type(required_cells) == type((1,2)))
        if matrix_idx == 0 and required_cells == tuple([(0, 9)]):
            print 'set_upper_bound called on 0, [(0, 9)], old upper bound =', self.upper_bounds_dictionary[(matrix_idx, required_cells)], 'new upper bound =', new_upper_bound
        # print 'set_upper_bound called on', (matrix_idx, required_cells), 'old upper bound =', self.upper_bounds_dictionary[(matrix_idx, required_cells)], 'new upper bound =', new_upper_bound
        self.upper_bounds_dictionary[(matrix_idx, required_cells)] = new_upper_bound
        #self.check_nesting

    def correct_errors_during_tightening(self, local_matrix, matrix_idx, prv_required_cells, global_row_indices, global_col_indices):
        '''
        correct for numerical errors during bound tightening

        Outputs:
        - upper_bound_increase: (float) we increased the specified upper bound by this much

        '''
        if self.upper_bounds_dictionary[(matrix_idx, tuple(prv_required_cells))] < 0.0:
            self.upper_bounds_dictionary[(matrix_idx, tuple(prv_required_cells))] = 0.0
            assert("false"), "negative bound during tightenting" #can take this out if just due to numerical errors, in for debugging now
        assert(local_matrix.shape[0] == local_matrix.shape[1]) 
        N = local_matrix.shape[0]


        sum_of_nested_UBs = 0
        for col in range(N):
            cur_required_cells = tuple(prv_required_cells + [(global_row_indices[0], global_col_indices[col])])
            cur_submatrix_UB = self.get_upper_bound(matrix_idx, cur_required_cells)
            sum_of_nested_UBs += cur_submatrix_UB * local_matrix[0, col]

        if sum_of_nested_UBs > self.get_upper_bound(matrix_idx, tuple(prv_required_cells)): #numerical error
            upper_bound_increase = sum_of_nested_UBs - self.get_upper_bound(matrix_idx, tuple(prv_required_cells))
            self.upper_bounds_dictionary[(matrix_idx, tuple(prv_required_cells))] = sum_of_nested_UBs
         
        else:
            upper_bound_increase = 0

        return upper_bound_increase

    def check_nesting(self, local_matrix, matrix_idx, prv_required_cells, global_row_indices, global_col_indices, recurse=False, verbose=False):
        '''
        correct for numerical errors during bound tightening
        '''
        # print "permanent upper bounds:"
        # print self.upper_bounds_dictionary
        pass
        assert(local_matrix.shape[0] == local_matrix.shape[1]) 
        N = local_matrix.shape[0]
        # assert(len(prv_required_cells) < 8)
        sum_of_nested_UBs = 0
        for col in range(N):
            cur_submatrix = np.delete(local_matrix, col, 1) #delete columnumns
            cur_submatrix = np.delete(cur_submatrix, 0, 0) #delete rows
            # print "N:", N, "M:", M, "T:", T, "global_row_indices:", global_row_indices, "global_col_indices:", global_col_indices, "col:", col, "local_matrix:"
            # print local_matrix
            cur_required_cells = tuple(prv_required_cells + [(global_row_indices[0], global_col_indices[col])])
            cur_submatrix_UB = self.get_upper_bound(matrix_idx, cur_required_cells, cur_submatrix)
            sum_of_nested_UBs += cur_submatrix_UB * local_matrix[0, col]

        assert(sum_of_nested_UBs <= self.get_upper_bound(matrix_idx, tuple(prv_required_cells))), (recurse, len(prv_required_cells), prv_required_cells, matrix_idx, sum_of_nested_UBs, self.get_upper_bound(matrix_idx, tuple(prv_required_cells)))
        if verbose:
            print "check_nesting called and checked out for prv_required_cells =", prv_required_cells, ".  sum_of_nested_UBs =", sum_of_nested_UBs, "self.get_upper_bound(matrix_idx, tuple(prv_required_cells)) =", self.get_upper_bound(matrix_idx, tuple(prv_required_cells))
        if recurse: #THIS IS BUGGY, need to find the best row to partition on
            for col in range(N):
                cur_required_cells = prv_required_cells + [(global_row_indices[0], global_col_indices[col])]            
                submatrix_global_row_indices = np.delete(global_row_indices, 0)
                submatrix_global_col_indices = np.delete(global_col_indices, col)
                cur_submatrix = np.delete(local_matrix, col, 1) #delete columnumns
                cur_submatrix = np.delete(cur_submatrix, 0, 0) #delete rows
                self.check_nesting(cur_submatrix, matrix_idx, cur_required_cells, submatrix_global_row_indices, submatrix_global_col_indices, recurse=False)



sys.path.insert(0, '/atlas/u/jkuck/rbpf_fireworks/mht_helpers')
from k_best_assignment import k_best_assignments


#'pymatgen' should be fastest, significantly
#pick from ['munkres', 'scipy', 'pymatgen'], 
ASSIGNMENT_SOLVER = 'pymatgen'

# random.seed(0)
SEED=2
# np.random.seed(SEED)
PICK_PARTITION_ORDER = False
USE_1_GUMBEL = True

DEBUG = False
DEBUG1 = False

FIRST_GUMBEL_LARGER = []
BEST_ROW_CACHE={}
MATRIX_PERMANENT_UBS = permanent_Upper_Bounds()

# every permanent UB implicitly contains U = #unassociated measurements + #unassocatiated targets
# this dictionary counts the number of times we have tightened each upper bound so we can set the
# upper bound to exactly 0 in case of numerical rounding issues on the 2^U time we tighten
# key: (matrix_idx, required_cells)
# val: (2^U, remaining times to tighten = 2^U - times we have previously tightened)
UB_TIGHTENING_COUNTS = {}

COMPARE_WAI = False

#default dictionary with 
# key: ((i_1, ..., i_m), (j_1, ..., j_t)) where i_1 to i_m are sorted measurement indices in (0, M-1]
#                                           and j_1 to j_t are sorted target indices in (0, T-1]
# value: list of tuples ((k_1, ..., k_m), (l_1, ..., l_t), prob) specifying each sampled instantiation, where:
#   k_1 = 1 represents that the first measurement is a birth
#   k_1 = 0 represents that the first measurement is clutter
#   l_1 = 1 represents that the first target dies
#   l_1 = 0 represents that the first target lives
#   prob (float) is the joint probability of the specified measurements
# being birth or clutter as specified (conditioned on the fact that they are unassociated) and the specified targets
# living or dying as specified (conditioned on the fact that they are unassociated)
SAMPLED_BIRTHS_DEATHS = defaultdict(list)
#
#
#References:
# [1] K. G. Murty, "Letter to the Editor--An Algorithm for Ranking all the Assignments in Order of
#     Increasing Cost," Oper. Res., vol. 16, no. May 2016, pp. 682-687, 1968.
#
# [2] I. J. Cox and M. L. Miller, "On finding ranked assignments with application to multitarget
#     tracking and motion correspondence," IEEE Trans. Aerosp. Electron. Syst., vol. 31, no. 1, pp.
#     486-489, Jan. 1995.

class associationMatrix:
    def __init__(self, matrix, M, T, conditional_birth_probs, conditional_death_probs, prior_prob):
        '''
        Inputs:
        - matrix: numpy matrix with dimensions (M+T)x(M+T) containing probabilities
            [a_11    ...     a_1T   um_1 0   ...  0]
            [.               .      0   .          ]
            [.               .      .      .       ]
            [.               .      .         .    ]
            [a_M1    ...     a_MT   0    ...   um_M]
            [ut_1    ...     ut_T   1    ...      1]
            [.               .      .             .]
            [.               .      .             .]
            [ut_1    ...     ut_T   1    ...      1]    
            - upper left quadrant is a MxT submatrix and composed of a_ij = the association probability of
              measurement i to target j
            - upper right quadrant is an MxM submatrix.  Row i is composed of M repetitions of 
              um_i = the probability that measurement i is unassociated with a target (marginalized over whether the
              measurement is clutter or a birth)
            - lower left quadrant is a TxT submatrix.  It is a diagonal matrix with elements ut_i = the
              probability that target i doesn't emit a measuremnt (marginalized over
              whether it lives or dies)
            - lower right quadrant is an TxM submatrix of all 1's
        - M: (int) number of measurements, indicates form of the association matrix
        - T: (int) number of targets, indicates form of the association matrix
        - conditional_birth_probs: (list of length M) probability of birth for each measurement, given that it is 
            not associated with a target (it is either birth or clutter)
        - conditional_death_probs: (list of length T) probability of death for each target, given that it is not
            associated with a measurement
        - prior_prob: (float) the probability of this state of targets before seeing these measurements

        '''        
        self.matrix = matrix
        self.M = M
        self.T = T
        self.conditional_birth_probs = conditional_birth_probs
        self.conditional_death_probs = conditional_death_probs
        self.prior_prob = prior_prob


def multi_matrix_sample_associations_without_replacement(num_samples, all_association_matrices, try_rescaling_all_matrices=True, verbose=False,\
    allow_exhausting_samples=True):
    '''
    Inputs:
    - all_association_matrices: list of associationMatrix
    - allow_exhausting_samples: (bool) if True we allow for the possibility that we sample every state and return fewer than num_samples
        samples.  In tracking this is possible, e.g. going from 0 targets to 1 target there are very few states or even going from 0 to 10
        targets there are few states if we require birth.
    Outputs
    - samples: (list of associationInfo)
    '''
    # print "multi_matrix_sample_associations_without_replacement called"
    random_number = np.random.random()
    matrix_file_name = './%f' % random_number
    if not os.path.exists(os.path.dirname(matrix_file_name)):
        try:
            os.makedirs(os.path.dirname(matrix_file_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # print "saving matrices in %s" % matrix_file_name    
    # f = open(matrix_file_name, 'w')
    # pickle.dump(all_association_matrices, f)
    # f.close()


    global MATRIX_PERMANENT_UBS
    global BEST_ROW_CACHE
    global SAMPLED_BIRTHS_DEATHS
    global UB_TIGHTENING_COUNTS

    MATRIX_PERMANENT_UBS = permanent_Upper_Bounds()
    BEST_ROW_CACHE = {} 
    SAMPLED_BIRTHS_DEATHS = defaultdict(list)
    UB_TIGHTENING_COUNTS = {}

    # print "reset caches:"

    # MATRIX_PERMANENT_UBS.print_bounds()
    # print "BEST_ROW_CACHE:", BEST_ROW_CACHE
    # print "SAMPLED_BIRTHS_DEATHS:", SAMPLED_BIRTHS_DEATHS

    #begin debugging
    (association_list, max_assignment) = find_max_assignment(all_association_matrices[0].matrix)
    # print "maximum assignment =", max_assignment, "maximum assignment*math.factorial(N) =", max_assignment*math.factorial(all_association_matrices[0].matrix.shape[0])
    #end debugging


    if try_rescaling_all_matrices:
        for matrix_idx, a_matrix in enumerate(all_association_matrices):
            N = a_matrix.matrix.shape[0]
            if N < 2: #don't need to rescale small matrices and code below crashes
                continue
            assert(N == a_matrix.matrix.shape[1])
            M_remaining = 0
            matrix_contains_no_targets = False
            while a_matrix.matrix[M_remaining, N-1] != 1:
                M_remaining += 1
                if M_remaining == N:
                    matrix_contains_no_targets = True
                    break
            if matrix_contains_no_targets:
                continue

            T_remaining=0
            while a_matrix.matrix[M_remaining, T_remaining] != 1:
                T_remaining += 1

            column_rescalings = a_matrix.matrix[M_remaining, :T_remaining]
            permanent_rescaling = np.prod(column_rescalings)
            rescaled_matrix = copy.copy(a_matrix.matrix)
            rescaled_matrix[:,:T_remaining] /= column_rescalings

            original_permanent_UB = minc_extended_UB2(a_matrix.matrix)
            rescaled_permanent_UB = minc_extended_UB2(rescaled_matrix)
            # print
            # print '-'*80
            # print "a_matrix.matrix:"
            # print a_matrix.matrix
            (association_list, max_assignment) = find_max_assignment(a_matrix.matrix)
            # print "maximum assignment =", max_assignment, "maximum assignment*math.factorial(N) =", max_assignment*math.factorial(N)

            # print "original_permanent_UB =", original_permanent_UB, "rescaled_permanent_UB =", rescaled_permanent_UB*permanent_rescaling
            if rescaled_permanent_UB*permanent_rescaling < original_permanent_UB:
                a_matrix.matrix = rescaled_matrix
                a_matrix.prior_prob *= permanent_rescaling

            # print '-'*80
            # print

    # sleep(323)
    assert(num_samples >= 1)
    samples = []
    while len(samples) < num_samples:
        if verbose:
            print '@'*80
            MATRIX_PERMANENT_UBS.print_bounds()
            print

        distribution_over_matrices = []
        for matrix_idx, a_matrix in enumerate(all_association_matrices):
            permanent_UB = MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, required_cells=(), sub_matrix=a_matrix.matrix)

            if verbose:
                print "matrix_idx, permanent_UB, a_matrix.prior_prob:", matrix_idx, permanent_UB, a_matrix.prior_prob
            # print "matrix_idx, permanent_UB, a_matrix.prior_prob:", matrix_idx, permanent_UB, a_matrix.prior_prob
            distribution_over_matrices.append(a_matrix.prior_prob*permanent_UB)
        # sleep(232)
        if verbose:
            print "distribution_over_matrices before normalization: ", distribution_over_matrices
        if allow_exhausting_samples and np.sum(distribution_over_matrices) == 0:
            break
        assert(np.sum(distribution_over_matrices) > 0), (np.sum(distribution_over_matrices), distribution_over_matrices)
        distribution_over_matrices /= np.sum(distribution_over_matrices)
        if verbose:
            print "distribution_over_matrices after normalization: ", distribution_over_matrices
        sampled_idx = np.random.choice(len(distribution_over_matrices), p=distribution_over_matrices)
        sampled_matrix = all_association_matrices[sampled_idx]
        if verbose:
            print '#'*80
            print 'sampling from matrix with index:', sampled_idx, 'with UB =', MATRIX_PERMANENT_UBS.get_upper_bound(sampled_idx, required_cells=()), 'and prior probability =', all_association_matrices[sampled_idx].prior_prob
        sampled_a_info = sample_association_single_matrix(sampled_matrix, sampled_idx)
        if sampled_a_info is None:
            continue
        sampled_a_info.matrix_index = sampled_idx
        samples.append(sampled_a_info)
        # print "got a sample! now we have", len(samples), "samples"
        # sampled_a_info.print_info()

    print "multi_matrix_sample_associations_without_replacement returning", len(samples), "samples"
    assert(len(samples) > 0)
    return samples

# @profile
def sample_association_single_matrix(a_matrix, matrix_idx, verbose=False):
    '''

    Inputs:
    - a_matrix: (associationMatrix)
    - matrix_idx: (int) the index of the original association matrix, when dealing with multiple states in the
        sequential setting

    Output: either
    -None: we sampled slack
    OR
    -sampled_a_info: (associationInfo)
    '''
    global MATRIX_PERMANENT_UBS
    global BEST_ROW_CACHE

    if a_matrix.matrix.shape[0] == 0: #0 measurements and 0 targets
        sampled_a_info = associationInfo(meas_grp_associations=[], dead_target_indices=[], complete_assoc_probability=a_matrix.prior_prob,\
            bottom_prob=1.0, conditional_unassociated_probability=1.0, a_matrix=a_matrix)
        MATRIX_PERMANENT_UBS.tighten_upper_bound(matrix_idx, required_cells=(), slack=1.0, assert_zero=True)
        return sampled_a_info

    N = a_matrix.matrix.shape[0]
    assert(N == a_matrix.matrix.shape[1])
    #convert 2d array to tuple of tuples
    hashable_matrix = tuple([tuple(row) for row in a_matrix.matrix])
    no_required_cells = ()

    global_row_indices = range(N)
    global_col_indices = range(N)
    if verbose:
        print "permanent UB from sample_association_single_matrix:", MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, no_required_cells)
    assert(MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, no_required_cells) > 0)
    if DEBUG_3:
        print '2 calling sample_association_01matrix_plusSlack with prv_required_cells:', []

    sampled_association, sub_tree_slack, prv_required_cells_at_sample, sampled_a_info = sample_association_01matrix_plusSlack(a_matrix.matrix, matrix_idx, MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, no_required_cells), \
        prv_required_cells=[], depth=1, \
        global_row_indices=global_row_indices, global_col_indices=global_col_indices, M=a_matrix.M, T=a_matrix.T, orig_a_matrix=a_matrix)

    if sampled_association is None: #we sampled a weight 0 association from proposal
        return None

    assert(prv_required_cells_at_sample is not None)


    return sampled_a_info


def find_implicit_associations(sampled_association, M, T):
    '''
    When sampling from the following matrix:
        probs: numpy matrix with dimensions (M+T)x(M+T) with probabilities 
        [a_11    ...     a_1T   um_1 0   ...  0]
        [.               .      0   .          ]
        [.               .      .      .       ]
        [.               .      .         .    ]
        [a_M1    ...     a_MT   0    ...   um_M]
        [ut_1    ...     ut_T   1    ...      1]
        [.               .      .             .]
        [.               .      .             .]
        [ut_1    ...     ut_T   1    ...      1]    
        - upper left quadrant is a MxT submatrix and composed of a_ij = the association probability of
          measurement i to target j
        - upper right quadrant is an MxM submatrix.  Row i is composed of M repetitions of 
          um_i = the probability that measurement i is unassociated with a target (marginalized over whether the
          measurement is clutter or a birth)
        - lower left quadrant is a TxT submatrix.  It is a diagonal matrix with elements ut_i = the
          probability that target i doesn't emit a measuremnt (marginalized over
          whether it lives or dies)
        - lower right quadrant is an TxM submatrix of all 1's
    We stop after sampling the first M rows.  This function appends the tuples (some value >= M, t) to sampled_association for every
    t in (0, T-1] that is not already present as the second value in a tuple in sampled_association.  (We're explicitly
    adding the implicit associations that these targets are not associated with any measurement.)
    '''
    implicit_associations = []
    unassociated_targets = set(range(T))
    for (m, t) in sampled_association:
        if t in unassociated_targets:
            unassociated_targets.remove(t)
    M_val = M 
    for t in unassociated_targets:
        implicit_associations.append((M_val, t))
        M_val += 1

    return implicit_associations


class associationInfo:
    def __init__(self, meas_grp_associations, dead_target_indices, complete_assoc_probability, bottom_prob, conditional_unassociated_probability, a_matrix,\
        unassociated_target_count=0, unassociated_measurement_count=0, min_possible_conditional_unassociated_probability=None,
        unassociated_measurements=None, unassociated_targets=None):
        '''
        Inputs:
        -meas_grp_associations: (list of ints) each element specifices the assocation of a measurement group
            [0,T-1]: target association
            T: birth
            -1: clutter
        -dead_target_indices: (list of ints) each element specifies the index of a dead target
        -complete_assoc_probability: (float) prior probability of target states multipled by
            probability of all associations (obtained by multiplying all
            matrix elements and multiplying sampled birth/clutter and life/death probabilities).  This is proportional
            (by a normalization constant that is the same for all particles) to the final probability of the target states
            after measurement associations.
        -bottom_prob: (float) obtained by multiplying matrix elements in the bottom T rows, without multiplying
            sampled birth/clutter probabilities.  This is useful because we need to subtract T! times this quantify
            from the corresponding permanent upper bound (and propogate to parents as slack)
        -conditional_unassociated_probability: (float) product of conditional birth/clutter and life/death probabilities
        -matrix_index:(int) specifies which matrix (particle representing the pre-association state) this sample was drawn from
        -unassociated_target_count: (int) the number of targets that are not associated with a measurement
        -unassociated_measurement_count: (int) the number of measurements that are not associated with a target
        -unassociated_measurements: (list of ints) each element is the index of a measurement that is not associated with a target
        -unassociated_targets: (list of ints) each element is the index of a target that is not associated with a measurement
        '''


        self.meas_grp_associations = meas_grp_associations
        self.dead_target_indices = dead_target_indices
        self.complete_assoc_probability = complete_assoc_probability
        self.bottom_prob = bottom_prob
        self.conditional_unassociated_probability = conditional_unassociated_probability
        self.matrix_index = None

        self.unassociated_target_count = unassociated_target_count
        self.unassociated_measurement_count = unassociated_measurement_count
        self.min_possible_conditional_unassociated_probability = min_possible_conditional_unassociated_probability
        # if a_matrix.T == 0: #sketchy is ther something better? also we seemed to get upper bound > 0 on a matrix of size (0,0) without this?!?!
        #     self.bottom_prob = 0

        self.unassociated_measurements = unassociated_measurements
        self.unassociated_targets = unassociated_targets

    def print_info(self):
        print '-'*40, 'begin associationInfo', '-'*40
        print "self.meas_grp_associations:", self.meas_grp_associations
        print "self.dead_target_indices:", self.dead_target_indices
        print "self.complete_assoc_probability:", self.complete_assoc_probability
        print "self.bottom_prob:", self.bottom_prob
        print "self.conditional_unassociated_probability:", self.conditional_unassociated_probability
        print "self.matrix_index:", self.matrix_index
        print "self.unassociated_measurements:", self.unassociated_measurements
        print "self.unassociated_targets:", self.unassociated_targets
        print "self.min_possible_conditional_unassociated_probability:", self.min_possible_conditional_unassociated_probability
        print '-'*40, 'end associationInfo', '-'*40
        print 

def sample_unassoc_measurementsAndTargets(required_cells, a_matrix, matrix_idx, verbose=False):
    '''
    Inputs:
    - required_cells: (list of tuples of ints)
    - a_matrix: (associationMatrix) the original matrix with extra info
    

    '''
    implicit_associations = find_implicit_associations(required_cells, a_matrix.M, a_matrix.T)
    # print "implicit_associations:", implicit_associations
    complete_associations = list(copy.copy(required_cells))
    complete_associations.extend(implicit_associations)

    meas_grp_associations = []
    dead_target_indices = []
    complete_associations.sort() #sort by m_indices 
    complete_assoc_probability = 1.0 #probability of all associations
    bottom_prob = 1.0 #product of elements with row > M
    unassociated_targets = []
    unassociated_target_vals = []

    unassociated_measurements = []
    unassociated_measurement_vals = []
    M = a_matrix.M
    T = a_matrix.T
    # print "M =", M
    # print "T =", T
    # print "complete_associations:", complete_associations
    for assoc_idx, (m_idx, t_idx) in enumerate(complete_associations):
        assert(assoc_idx == m_idx), (complete_associations, assoc_idx, m_idx, t_idx)
        if m_idx < M and t_idx < T: #measurement-target association
            meas_grp_associations.append(t_idx)
            complete_assoc_probability *= a_matrix.matrix[m_idx, t_idx]
        elif m_idx < M and t_idx >= T: #measurement is clutter or birth
            unassociated_measurements.append(m_idx)
            assert(t_idx < M+T)
            complete_assoc_probability *= a_matrix.matrix[m_idx, t_idx]
            meas_grp_associations.append(-2) #filler value
        elif m_idx >= M and t_idx < T: #target is unnassociated
            unassociated_targets.append(t_idx)
            assert(m_idx < M+T)
            complete_assoc_probability *= a_matrix.matrix[m_idx, t_idx] 
            bottom_prob *= a_matrix.matrix[m_idx, t_idx] 
        else:
            assert(t_idx >= T and t_idx < M+T and m_idx >= M and m_idx < M+T)
            assert(a_matrix.matrix[m_idx, t_idx] == 1)
            #dummy variable to dummy variable assignment, don't need to do anything
        # print "bottom_prob =", bottom_prob

    unassociated_target_count = len(unassociated_targets)
    unassociated_measurement_count = len(unassociated_measurements)

    if verbose:
        print
        print
        print
        print
        print "a_matrix.matrix"
        print a_matrix.matrix
    (meas_associations, dead_target_indices, conditional_unassociated_probability, min_possible_conditional_unassociated_probability) = sample_unassoc_measurementsAndTargets_helper\
        (unassociated_targets, unassociated_measurements, a_matrix.conditional_birth_probs, a_matrix.conditional_death_probs, matrix_idx, T,\
         required_cells)
             
    complete_assoc_probability *= conditional_unassociated_probability * a_matrix.prior_prob
    for m_idx, assoc in meas_associations:
        assert(meas_grp_associations[m_idx] == -2)
        meas_grp_associations[m_idx] = assoc

    dead_target_indices.sort()


    assert(meas_grp_associations.count(-2) == 0), meas_grp_associations.count(-2)
    sampled_a_info = associationInfo(meas_grp_associations, dead_target_indices, complete_assoc_probability, bottom_prob, conditional_unassociated_probability, a_matrix=a_matrix,
                                     unassociated_target_count=unassociated_target_count, unassociated_measurement_count=unassociated_measurement_count, \
                                     min_possible_conditional_unassociated_probability=min_possible_conditional_unassociated_probability, \
                                     unassociated_measurements=unassociated_measurements,\
                                     unassociated_targets=unassociated_targets)
    return sampled_a_info

def sample_unassoc_measurementsAndTargets_helper(unassociated_targets, unassociated_measurements, conditional_birth_probs,\
                                                 conditional_death_probs, matrix_idx, T, required_cells=None, verbose=False):
    '''

    Outputs:
        -min_possible_conditional_unassociated_probability: the smallest conditional_unassociated_probability that could be
        sampled, so we can tell when permanent upper bounds should be set to zero
    '''
    # print "sample_unassoc_measurementsAndTargets_helper called"
    # print "unassociated_targets:", unassociated_targets
    # print "unassociated_measurements:", unassociated_measurements
    # print "matrix_idx:", matrix_idx
    if verbose:
        print '-'*10
        print "sample_unassoc_measurementsAndTargets_helper called"
        print "matrix_idx:", matrix_idx 
        print "required_cells:", required_cells
        MATRIX_PERMANENT_UBS.print_bounds
    unassociated_targets.sort()
    unassociated_measurements.sort()

    list_of_sampled_values = SAMPLED_BIRTHS_DEATHS[(matrix_idx, tuple(unassociated_measurements), tuple(unassociated_targets), required_cells)]


    if verbose:
        print "matrix_idx:", matrix_idx, "list_of_sampled_values:", list_of_sampled_values
        print

    previous_measurement_vals = []
    previous_target_vals = []

    meas_associations = [] #list of tuples of (m_idx, assoc) where assoc = T (birth) or -1 (clutter)
    dead_target_indices = [] #indices of targets with sampled death
    conditional_unassociated_probability = 1.0
    min_possible_conditional_unassociated_probability = 1.0
    list_of_conditional_unassociated_probabilities = []
    for m_idx in unassociated_measurements:
        birth_prob = conditional_birth_probs[m_idx]
        orig_birth_prob = conditional_birth_probs[m_idx]
        clutter_prob = 1 - birth_prob
        min_birth_clutter_prob = min(birth_prob, clutter_prob)
        if min_birth_clutter_prob == 0.0: #cannot sample zero probability event
            min_birth_clutter_prob = 1.0
        min_possible_conditional_unassociated_probability *= min_birth_clutter_prob
        if (matrix_idx, tuple(unassociated_measurements), tuple(unassociated_targets), required_cells) in SAMPLED_BIRTHS_DEATHS:
            if verbose:
                print "(matrix_idx, tuple(unassociated_measurements), tuple(unassociated_targets), required_cells):", (matrix_idx, tuple(unassociated_measurements), tuple(unassociated_targets), required_cells)
            list_of_sampled_values = SAMPLED_BIRTHS_DEATHS[(matrix_idx, tuple(unassociated_measurements), tuple(unassociated_targets), required_cells)]
            print "list_of_sampled_values:", list_of_sampled_values
            for sampled_unassoc_meas_vals, sampled_unassoc_targ_vals, list_of_sampled_probs in list_of_sampled_values:
                if verbose:
                    print "1sampled_unassoc_meas_vals, sampled_unassoc_targ_vals, list_of_sampled_probs:", sampled_unassoc_meas_vals, sampled_unassoc_targ_vals, list_of_sampled_probs, matrix_idx
                    print "tuple(previous_measurement_vals):", tuple(previous_measurement_vals)
                    print "sampled_unassoc_meas_vals[:len(previous_measurement_vals)]:", sampled_unassoc_meas_vals[:len(previous_measurement_vals)]
                    print "tuple(previous_measurement_vals) == tuple(sampled_unassoc_meas_vals[:len(previous_measurement_vals)]):", tuple(previous_measurement_vals) == tuple(sampled_unassoc_meas_vals[:len(previous_measurement_vals)])
                if tuple(previous_measurement_vals) == tuple(sampled_unassoc_meas_vals[:len(previous_measurement_vals)]):
                    if sampled_unassoc_meas_vals[len(previous_measurement_vals)] == 1:
                        birth_prob -= np.prod(list_of_sampled_probs[len(previous_measurement_vals):])
                        if verbose:
                            print "1"
                    else:
                        assert(sampled_unassoc_meas_vals[len(previous_measurement_vals)] == 0)
                        clutter_prob -= np.prod(list_of_sampled_probs[len(previous_measurement_vals):])
                        if verbose:
                            print "2"
                else:
                    if verbose:
                        print "3"
                    else:
                        pass

        MATRIX_PERMANENT_UBS.print_bounds()
        print "(matrix_idx, required_cells):", (matrix_idx, required_cells)
        print "birth_prob:", birth_prob, "orig_birth_prob:", orig_birth_prob, "clutter_prob:", clutter_prob
        birth_prob = birth_prob/(birth_prob + clutter_prob)
        assert((birth_prob >= 0 or np.allclose(0, birth_prob, atol=0)) and (birth_prob <= 1 or np.allclose(birth_prob, 1, atol=0))), birth_prob
        assert((clutter_prob >= 0 or np.allclose(0, clutter_prob, atol=0)) and (clutter_prob <= 1 or np.allclose(clutter_prob, 1, atol=0))), clutter_prob
        if verbose:
            print "birth_prob:", birth_prob
            print "orig_birth_prob:", orig_birth_prob
            print "clutter_prob:", clutter_prob
            print "clutter_prob original:", 1 - orig_birth_prob
        if np.random.rand() < birth_prob:
            meas_associations.append((m_idx,T)) #sampled birth
            conditional_unassociated_probability *= orig_birth_prob
            list_of_conditional_unassociated_probabilities.append(orig_birth_prob)
            previous_measurement_vals.append(1)
        else:
            meas_associations.append((m_idx,-1)) #sampled clutter
            conditional_unassociated_probability *= (1 - orig_birth_prob)
            list_of_conditional_unassociated_probabilities.append((1 - orig_birth_prob))
            previous_measurement_vals.append(0)

    for t_idx in unassociated_targets:
        death_prob = conditional_death_probs[t_idx]
        orig_death_prob = conditional_death_probs[t_idx]
        # if death_prob > .999:
        #     death_prob = 1.0
        #     orig_death_prob = 1.0
        life_prob = 1 - death_prob
        min_life_death_prob = min(death_prob, life_prob)
        if min_life_death_prob == 0.0: #cannot sample zero probability event
            min_life_death_prob = 1.0 
        min_possible_conditional_unassociated_probability *= min_life_death_prob

        if (matrix_idx, tuple(unassociated_measurements), tuple(unassociated_targets), required_cells) in SAMPLED_BIRTHS_DEATHS:
            if verbose:
                print"(matrix_idx, tuple(unassociated_measurements), tuple(unassociated_targets)):", (matrix_idx, tuple(unassociated_measurements), tuple(unassociated_targets))
            list_of_sampled_values = SAMPLED_BIRTHS_DEATHS[(matrix_idx, tuple(unassociated_measurements), tuple(unassociated_targets), required_cells)]
            for sampled_unassoc_meas_vals, sampled_unassoc_targ_vals, list_of_sampled_probs in list_of_sampled_values:
                if verbose:
                    print "2sampled_unassoc_meas_vals, sampled_unassoc_targ_vals, list_of_sampled_probs:", sampled_unassoc_meas_vals, sampled_unassoc_targ_vals, list_of_sampled_probs, matrix_idx
                    print"tuple(previous_measurement_vals):", tuple(previous_measurement_vals)
                    print "tuple(sampled_unassoc_meas_vals):", tuple(sampled_unassoc_meas_vals)
                    print "tuple(previous_target_vals):", tuple(previous_target_vals)
                    print "tuple(sampled_unassoc_targ_vals[:len(previous_target_vals)]):", tuple(sampled_unassoc_targ_vals[:len(previous_target_vals)])
                if tuple(previous_measurement_vals) == tuple(sampled_unassoc_meas_vals) and\
                   tuple(previous_target_vals) == tuple(sampled_unassoc_targ_vals[:len(previous_target_vals)]):
                    if sampled_unassoc_targ_vals[len(previous_target_vals)] == 1:
                        death_prob -= np.prod(list_of_sampled_probs[len(previous_measurement_vals)+len(previous_target_vals):])
                        if verbose:
                            print"4"
                    else:
                        assert(sampled_unassoc_targ_vals[len(previous_target_vals)] == 0)
                        life_prob -= np.prod(list_of_sampled_probs[len(previous_measurement_vals)+len(previous_target_vals):])
                        if verbose:
                            print"5"
                else:
                    if verbose:
                        print"6"


        MATRIX_PERMANENT_UBS.print_bounds()
        print "(matrix_idx, required_cells):", (matrix_idx, required_cells)
        print "death_prob:", death_prob, "orig_death_prob:", orig_death_prob, "life_prob:", life_prob

        death_prob = death_prob/(death_prob + life_prob)
        if verbose:
            print "death_prob:", death_prob
            print "orig_death_prob:", orig_death_prob
            print "life_prob:", life_prob
            print "life_prob original:", 1 - orig_death_prob

        assert((death_prob >= -.000000001) and (death_prob <= 1 or np.allclose(death_prob, 1, atol=0))), death_prob
        assert((life_prob >= -.000000001) and (life_prob <= 1 or np.allclose(life_prob, 1, atol=0))), life_prob

        if np.random.rand() < death_prob:
            dead_target_indices.append(t_idx) #sampled target death
            conditional_unassociated_probability *= orig_death_prob
            list_of_conditional_unassociated_probabilities.append(orig_death_prob)
            previous_target_vals.append(1)
        else:
            conditional_unassociated_probability *= (1 - orig_death_prob)
            list_of_conditional_unassociated_probabilities.append((1 - orig_death_prob))
            previous_target_vals.append(0)


    for (pmvs, ptvs, cups) in SAMPLED_BIRTHS_DEATHS[(matrix_idx, tuple(unassociated_measurements), tuple(unassociated_targets), required_cells)]:
        if pmvs == previous_measurement_vals and ptvs == previous_target_vals:
            if verbose:
                print "already sampled this!! ?!"
                print "pmvs:", pmvs
                print "previous_measurement_vals:", previous_measurement_vals
                print "ptvs:", ptvs
                print "previous_target_vals:", previous_target_vals
                print "unassociated_targets:", unassociated_targets
                print "unassociated_measurements:", unassociated_measurements
                print "required_cells:", required_cells
                print "SAMPLED_BIRTHS_DEATHS:"
                print SAMPLED_BIRTHS_DEATHS
                MATRIX_PERMANENT_UBS.print_bounds()
            assert(False), "already sampled this!! ?!"

    if verbose:
        print "sampled for matrix_idx =", matrix_idx, "required_cells:", required_cells, previous_measurement_vals, previous_target_vals
        print "conditional_unassociated_probability =", conditional_unassociated_probability

    SAMPLED_BIRTHS_DEATHS[(matrix_idx, tuple(unassociated_measurements), tuple(unassociated_targets), required_cells)].append(\
        (previous_measurement_vals, previous_target_vals, list_of_conditional_unassociated_probabilities))

    if verbose:
        print "just set key:", (matrix_idx, tuple(unassociated_measurements), tuple(unassociated_targets)), "in SAMPLED_BIRTHS_DEATHS"

    if DEBUG_3:
        print "min_possible_conditional_unassociated_probability:", min_possible_conditional_unassociated_probability
    return (meas_associations, dead_target_indices, conditional_unassociated_probability, min_possible_conditional_unassociated_probability)

def correct_sampled_association_indices(incorrect_sampled_association):
    used_cols = defaultdict(int)
    corrected_row = 0
    correct_sampled_association = []
    print "incorrect_sampled_association:", incorrect_sampled_association
    for cur_assoc in incorrect_sampled_association:
        cur_incorrect_col = cur_assoc[1]
        cur_corrected_col = cur_incorrect_col
        for col, col_count in used_cols.iteritems():
            if cur_incorrect_col >= col:
                cur_corrected_col += col_count
        used_cols[cur_incorrect_col] += 1
        correct_sampled_association.append((corrected_row, cur_corrected_col))
        corrected_row += 1
    print "correct_sampled_association:", correct_sampled_association
    print 
    print        
    return correct_sampled_association



def sinkhorn_scale(matrix, debug=False):
    assert(matrix.shape[0] == matrix.shape[1])
    N = matrix.shape[0]
    cols_scaling = np.ones(N)
    rows_scaling = np.ones(N)
    original_matrix = copy.copy(matrix)
    iters = 0
    while(True):            
        col_sums = np.sum(matrix, axis=0)
        col_normalized_matrix = matrix / col_sums

        row_sums = np.sum(col_normalized_matrix, axis=1)
        double_stochastic_matrix = col_normalized_matrix/row_sums[:,None]

        cols_scaling /= col_sums
        rows_scaling /= row_sums

        if np.allclose(matrix, double_stochastic_matrix):
            break
        else:
            matrix = double_stochastic_matrix

    if debug:
        iters += 1
        print "iters:", iters
        assert(np.allclose(original_matrix*cols_scaling*rows_scaling[:,None], double_stochastic_matrix))
        assert(np.allclose(np.sum(double_stochastic_matrix, axis=0), np.ones(N)))
        assert(np.allclose(np.sum(double_stochastic_matrix, axis=1), np.ones(N)))

    return double_stochastic_matrix, cols_scaling, rows_scaling

def conjectured_optimal_bound(matrix):
    assert(matrix.shape[0] == matrix.shape[1])
    N = matrix.shape[0]

    double_stochastic_matrix, cols_scaling, rows_scaling = sinkhorn_scale(matrix)

    one_minus_matrix = 1 - double_stochastic_matrix
    permanent_UB = 2**(N/2) * np.prod(np.power(one_minus_matrix, one_minus_matrix)) / (np.prod(cols_scaling)*np.prod(rows_scaling))
    # permanent_UB = 2**(N) * np.prod(np.power(one_minus_matrix, one_minus_matrix)) / (np.prod(cols_scaling)*np.prod(rows_scaling))

    return permanent_UB

# @profile
def minc_extended_UB2_excludeRowCol(matrix, excluded_row, excluded_col):
    #another bound
    #https://ac-els-cdn-com.stanford.idm.oclc.org/S002437950400299X/1-s2.0-S002437950400299X-main.pdf?_tid=fa4d00ee-39a5-4030-b7c1-28bb5fbc76c0&acdnat=1534454814_a7411b3006e0e092622de35cbf015275
    # equation (6), U^M(A)

    assert(matrix.shape[0] == matrix.shape[1])
    N = matrix.shape[0]

    minc_extended_upper_bound2 = 1.0
    for row in range(N):
        if row == excluded_row:
            continue
        sorted_row = sorted(zip(matrix[row], range(N)), reverse=True)
        row_sum = 0
        delta_idx = 0
        for col in range(N):
            if sorted_row[col][1] == excluded_col:
                continue
            row_sum += sorted_row[col][0] * delta(delta_idx+1)
            delta_idx += 1
            # row_sum += sorted_row[col][0] * numba_delta(col+1)
        minc_extended_upper_bound2 *= row_sum
    return minc_extended_upper_bound2


# @profile
def find_best_row_to_partition_matrix(matrix, matrix_idx, prv_required_cells, rows_to_select, global_row_indices,\
                                      global_col_indices, verbose=False, verbose1=False):
    '''
    Inputs:
    - matrix_idx: (int) the index of the original association matrix, when dealing with multiple states in the
        sequential setting
    - rows_to_select: (int) select the best row to partition from among the first rows_to_select rows
    '''
    if COMPARE_WAI:
        return 0
    global BEST_ROW_CACHE
    
    if rows_to_select == 1 or rows_to_select == 2:
        return 0

    if (matrix_idx, tuple(prv_required_cells)) in BEST_ROW_CACHE:
        if verbose:
            print "returning cached result"
        if verbose1:
            print "find_best_row_to_partition_matrix returning cached result"
            print 'best row to partition for (matrix_idx, prv_required_cells):', (matrix_idx, prv_required_cells), "is:", BEST_ROW_CACHE[(matrix_idx, tuple(prv_required_cells))]
            print
        return BEST_ROW_CACHE[(matrix_idx, tuple(prv_required_cells))]

    N = matrix.shape[0]
    assert(N == matrix.shape[1])

    fixed_column_options = list(itertools.permutations(range(N), 1))
    matrix_UB = (minc_extended_UB2(matrix))
    if verbose:
        print "find_best_row_to_partition_matrix", '*'*80        
        print "matrix_UB:", matrix_UB

    deltas = np.array([delta(i + 1) for i in range(N - 1)])
    row_sum = np.empty_like(matrix, dtype=float)
    for col in range(N):
        matrix_sorted = np.sort(np.delete(matrix, col, 1), axis=1)[:, ::-1]
        row_sum[:, col] = (matrix_sorted * deltas).sum(axis=1)
    # Can't use this trick to multiply all the rows and then divide, as we might get 0 / 0
    # upper_bounds_excluding_row_col = row_sum.prod(axis=0) / row_sum
    upper_bounds_excluding_row_col = np.empty_like(matrix[:rows_to_select,:], dtype=float)
    # for row in range(N):
    for row in range(rows_to_select):
        upper_bounds_excluding_row_col[row] = np.delete(row_sum, row, 0).prod(axis=0)
    # The (i, j)-element is the upper bound of the submatrix after deleting the i-th row and j-th column

    partitioned_UB = (upper_bounds_excluding_row_col * matrix[:rows_to_select,:]).sum(axis=1)
    row_with_smallest_partitioned_UB = np.argmin(partitioned_UB)
    smallest_partitioned_upper_bound = partitioned_UB[row_with_smallest_partitioned_UB]

    if verbose:
        print "returning new result"
        print "smallest_partitioned_upper_bound =", smallest_partitioned_upper_bound, "matrix_UB =", matrix_UB

    if verbose:
        print "smallest_partitioned_upper_bound =", smallest_partitioned_upper_bound, "matrix_UB =", matrix_UB

    if not (smallest_partitioned_upper_bound < matrix_UB or np.allclose(smallest_partitioned_upper_bound, matrix_UB, atol=0)):
        #FOR DEBUGGING
        # assert(False), "debugging, about to call find_best_row_to_partition_matrix_with_rescaling, remove this assert later if desired"
        row_with_smallest_partitioned_UB = find_best_row_to_partition_matrix_with_rescaling(matrix, matrix_idx, prv_required_cells, rows_to_select,\
                                                                                            global_row_indices, global_col_indices)
    # assert(smallest_partitioned_upper_bound < matrix_UB or np.allclose(smallest_partitioned_upper_bound, matrix_UB)), (smallest_partitioned_upper_bound, matrix_UB, matrix.shape, matrix, prv_required_cells)

    if verbose1:
        print "find_best_row_to_partition_matrix returning new result"
        print 'best row to partition for (matrix_idx, prv_required_cells):', (matrix_idx, prv_required_cells), "is:", row_with_smallest_partitioned_UB
        print
    BEST_ROW_CACHE[(matrix_idx, tuple(prv_required_cells))] = row_with_smallest_partitioned_UB

    return row_with_smallest_partitioned_UB

def find_best_row_to_partition_matrix_constantNumTargets(matrix, matrix_idx, prv_required_cells, verbose=False):
    '''
    Inputs:
    - matrix_idx: (int) the index of the original association matrix, when dealing with multiple states in the
        sequential setting
    '''

    global BEST_ROW_CACHE
    # print "BEST_ROW_CACHE:"
    # print BEST_ROW_CACHE

    if tuple(prv_required_cells) in BEST_ROW_CACHE:
        if verbose:
            print "returning cached result"
        # print "smallest_partitioned_upper_bound =", BEST_ROW_CACHE[(matrix_idx, tuple(prv_required_cells))]

        return BEST_ROW_CACHE[(matrix_idx, tuple(prv_required_cells))]

    N = matrix.shape[0]
    assert(N == matrix.shape[1])

    fixed_column_options = list(itertools.permutations(range(N), 1))
    matrix_UB = (minc_extended_UB2(matrix))
    if verbose:
        print "find_best_row_to_partition_matrix", '*'*80        
        print "matrix_UB:", matrix_UB

    deltas = np.array([delta(i + 1) for i in range(N - 1)])
    row_sum = np.empty_like(matrix, dtype=float)
    for col in range(N):
        matrix_sorted = np.sort(np.delete(matrix, col, 1), axis=1)[:, ::-1]
        row_sum[:, col] = (matrix_sorted * deltas).sum(axis=1)
    # Can't use this trick to multiply all the rows and then divide, as we might get 0 / 0
    # upper_bounds_excluding_row_col = row_sum.prod(axis=0) / row_sum
    upper_bounds_excluding_row_col = np.empty_like(matrix, dtype=float)
    for row in range(N):
        upper_bounds_excluding_row_col[row] = np.delete(row_sum, row, 0).prod(axis=0)
    # The (i, j)-element is the upper bound of the submatrix after deleting the i-th row and j-th column
    partitioned_UB = (upper_bounds_excluding_row_col * matrix).sum(axis=1)
    row_with_smallest_partitioned_UB = np.argmin(partitioned_UB)
    smallest_partitioned_upper_bound = partitioned_UB[row_with_smallest_partitioned_UB]

    if verbose:
        print "returning new result"
        print "smallest_partitioned_upper_bound =", smallest_partitioned_upper_bound, "matrix_UB =", matrix_UB

    if verbose:
        print "smallest_partitioned_upper_bound =", smallest_partitioned_upper_bound, "matrix_UB =", matrix_UB
    assert(smallest_partitioned_upper_bound < matrix_UB or np.allclose(smallest_partitioned_upper_bound, matrix_UB)), (smallest_partitioned_upper_bound, matrix_UB, matrix.shape, matrix)

    BEST_ROW_CACHE[(matrix_idx, tuple(prv_required_cells))] = row_with_smallest_partitioned_UB
    # print "BEST_ROW_CACHE:"
    # print BEST_ROW_CACHE

    return row_with_smallest_partitioned_UB

def find_best_row_to_partition_matrix_with_rescaling(matrix, matrix_idx, prv_required_cells, rows_to_select, \
                                                     global_row_indices, global_col_indices, verbose=False):

    '''
    Inputs:
    - matrix_idx: (int) the index of the original association matrix, when dealing with multiple states in the
        sequential setting
    - rows_to_select: (int) select the best row to partition from among the first rows_to_select rows
    '''
    global BEST_ROW_CACHE
    global MATRIX_PERMANENT_UBS

    assert((matrix_idx, tuple(prv_required_cells)) not in BEST_ROW_CACHE)

    N = matrix.shape[0]
    assert(N == matrix.shape[1])

    #find M_remaining and T_remaining by iterating until we hit the 1's in the bottom right corner
    M_remaining = 0
    while matrix[M_remaining, N-1] != 1:
        M_remaining += 1
    T_remaining=0
    while matrix[M_remaining, T_remaining] != 1:
        T_remaining += 1

    matrix_copy = copy.copy(matrix)
    column_rescalings = matrix_copy[M_remaining, :T_remaining]
    matrix_copy[:,:T_remaining] /= column_rescalings

    #!! BE CAREFUL HERE, may need to use this bound, or the rescaled bound !!
    matrix_UB = min(minc_extended_UB2(matrix_copy), rescaled_tracking_UB(matrix_copy)) 

    deltas = np.array([delta(i + 1) for i in range(N - 1)])
    row_sum = np.empty_like(matrix_copy, dtype=float)
    for col in range(N):
        matrix_sorted = np.sort(np.delete(matrix_copy, col, 1), axis=1)[:, ::-1]
        row_sum[:, col] = (matrix_sorted * deltas).sum(axis=1)
    # Can't use this trick to multiply all the rows and then divide, as we might get 0 / 0
    # upper_bounds_excluding_row_col = row_sum.prod(axis=0) / row_sum
    upper_bounds_excluding_row_col = np.empty_like(matrix_copy[:rows_to_select,:], dtype=float)
    # for row in range(N):
    for row in range(rows_to_select):
        upper_bounds_excluding_row_col[row] = np.delete(row_sum, row, 0).prod(axis=0)
    # The (i, j)-element is the upper bound of the submatrix after deleting the i-th row and j-th column

    partitioned_UB = (upper_bounds_excluding_row_col * matrix_copy[:rows_to_select,:]).sum(axis=1)
    row_with_smallest_partitioned_UB = np.argmin(partitioned_UB)
    #could use upper_bounds_excluding_row_col to make this faster rather than recomputing
    smallest_partitioned_upper_bound = 0
    for col in range(N):
        cur_submatrix = np.delete(matrix, col, 1) #delete columns
        cur_submatrix = np.delete(cur_submatrix, row_with_smallest_partitioned_UB, 0) #delete rows
        curUB = rescaled_tracking_UB(cur_submatrix)
        cur_required_cells = copy.copy(prv_required_cells)
        cur_required_cells.append((global_row_indices[row_with_smallest_partitioned_UB], global_col_indices[col]))
        cur_required_cells = tuple(cur_required_cells)
        MATRIX_PERMANENT_UBS.set_upper_bound(matrix_idx, cur_required_cells, new_upper_bound=curUB, sub_matrix=cur_submatrix)
        smallest_partitioned_upper_bound += curUB * matrix[row_with_smallest_partitioned_UB, col]

    # print

    assert(smallest_partitioned_upper_bound < matrix_UB or np.allclose(smallest_partitioned_upper_bound, matrix_UB, atol=0)), (smallest_partitioned_upper_bound, matrix_UB, matrix.shape, matrix, prv_required_cells)

    BEST_ROW_CACHE[(matrix_idx, tuple(prv_required_cells))] = row_with_smallest_partitioned_UB

    return row_with_smallest_partitioned_UB


def rescaled_tracking_UB(matrix, M_remaining=None, T_remaining=None):
    '''
    divide first T_remaining columns by the constant value matrix in each matrix[i, M_remaining:] entry (i < T_remaining)
    so that the bottom T_remaining rows are all ones

    note: M_remaining + T_remaining may not be the matrix size because we may have deleted rows/columns

    Inputs:
    - matrix: (np.array) the matrix whose permanent we are upper bounding
    - M_remaining: (int) number of measurements
    - T_remaining: (int) number of targets

    Outputs:
    - permanent_upper_bound: (float) upper bound on the permanent
    '''
    # print '*'*80
    #find M_remaining and T_remaining by iterating until we hit the 1's in the bottom right corner
    N = matrix.shape[0]
    assert(N == matrix.shape[1])
    
    M_remaining = 0
    while matrix[M_remaining, N-1] != 1:
        M_remaining += 1
    T_remaining=0
    while matrix[M_remaining, T_remaining] != 1:
        T_remaining += 1

    matrix_copy = copy.copy(matrix)
    column_rescalings = matrix_copy[M_remaining, :T_remaining]
    # print "column_rescalings:", column_rescalings
    permanent_rescaling = np.prod(column_rescalings)
    matrix_copy[:,:T_remaining] /= column_rescalings
    # print "rescaled matrix_copy:", matrix_copy
    assert((matrix_copy[M_remaining:, :] == 1).all)
    permanent_upper_bound = minc_extended_UB2(matrix_copy, try_rescaling=False)

    permanent_upper_bound *= permanent_rescaling

    return permanent_upper_bound



def sample_association_01matrix_plusSlack(matrix, matrix_idx, permanentUB, prv_required_cells, depth, \
    global_row_indices, global_col_indices, M, T, orig_a_matrix, verbose=False, tracking_constant_num_targets=False):
    '''
    Inputs: 
        - matrix: (np.array of shape NxN)
        - prv_required_cells: (list of tuples), [(row, col), ...]
        - M: (int) number of measurements, indicates form of the association matrix
        - T: (int) number of targets, indicates form of the association matrix
        - matrix_idx: (int) the index of the original association matrix, when dealing with multiple states in the
            sequential setting
        - orig_a_matrix: (associationMatrix) the original matrix with extra info

    Outputs: list of length N of tuples representing (row, col) associations
    '''
    # print "depth =", depth
    # print
    # print '-'*80
    # print "sample_association_01matrix_plusSlack called"
    # print "matrix_idx:", matrix_idx
    # print "prv_required_cells:", prv_required_cells
    # print "orig_a_matrix.matrix:"
    # print orig_a_matrix.matrix
    # MATRIX_PERMANENT_UBS.print_bounds
    # print 
    # print "sample_association_01matrix_plusSlack called, (matrix_idx, prv_required_cells):", (matrix_idx, prv_required_cells)

    # print "sample_association_01matrix_plusSlack called, matrix_idx, tuple(prv_required_cells):", matrix_idx, tuple(prv_required_cells)

    # if prv_required_cells == []:
    #     "matrix_idx:", matrix_idx
    #     print '#'*80
    if verbose:
        print '-'*80
        print "permanentUB =", permanentUB

        print "T:", T

        print "M:", M

    assert(permanentUB > 0)
    global MATRIX_PERMANENT_UBS
    if not tracking_constant_num_targets:
        assert(orig_a_matrix.matrix.shape[0] == M + T), (orig_a_matrix.matrix.shape[0], M, T)
    else:
        assert(orig_a_matrix.matrix.shape[0] == M), (orig_a_matrix.matrix.shape[0], M, T)


    if DEBUG1:
        MATRIX_PERMANENT_UBS.print_bounds
    local_matrix = np.copy(matrix)
    N = local_matrix.shape[0]
    assert(N == local_matrix.shape[1])
    prv_required_cells_copy = copy.copy(prv_required_cells)

    if N == 1:
        assert(permanentUB <= matrix[0, 0]), (permanentUB, matrix[0, 0], matrix[0, 0])
        sampled_association_global_indices = [(global_row_indices[0], global_col_indices[0])]

        required_cells = tuple(prv_required_cells_copy + sampled_association_global_indices)
        if verbose:
            print "N = 1, about to call sample_unassoc_measurementsAndTargets"

        # print "(matrix_idx, tuple(required_cells)):", (matrix_idx, tuple(required_cells))
        # assert((matrix_idx, tuple(required_cells)) in MATRIX_PERMANENT_UBS.upper_bounds_dictionary)

        sampled_a_info = sample_unassoc_measurementsAndTargets(required_cells, orig_a_matrix, matrix_idx)
        if verbose:
            print "1 sampled:", (matrix_idx, sampled_a_info.unassociated_measurements, sampled_a_info.unassociated_targets)

            print "sampled matrix_idx =", matrix_idx, "required_cells:", required_cells
            print "sampled_a_info.conditional_unassociated_probability =", sampled_a_info.conditional_unassociated_probability
            print "MATRIX_PERMANENT_UBS[(matrix_idx, tuple(prv_required_cells))]:", MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells))
            print "matrix[0, 0] * sampled_a_info.conditional_unassociated_probability", matrix[0, 0] * sampled_a_info.conditional_unassociated_probability
        # assert(sampled_a_info.bottom_prob == 1.0), sampled_a_info.bottom_prob 
        if verbose:
            print "1matrix_permanent_UBs[(matrix_idx, tuple(prv_required_cells))] =", MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells))
        
        slack = matrix[0, 0] * sampled_a_info.conditional_unassociated_probability
        slack = MATRIX_PERMANENT_UBS.tighten_upper_bound(matrix_idx, tuple(prv_required_cells), slack)

        if verbose:
            print "2matrix_permanent_UBs[(matrix_idx, tuple(prv_required_cells))] =", MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells))


        #numerical issue, set to 0
        if MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells)) < matrix[0, 0]*sampled_a_info.min_possible_conditional_unassociated_probability/1.01 and\
           MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells)) > 0:
            print
            print "sampled_a_info.min_possible_conditional_unassociated_probability:", sampled_a_info.min_possible_conditional_unassociated_probability
            print "1 numerical issue, set to 0"
            print "prv_required_cells:", prv_required_cells
            print "MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells)):", MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells))
            slack += MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells))
            print "slack = ", slack
            print
            MATRIX_PERMANENT_UBS.set_upper_bound(matrix_idx, tuple(prv_required_cells), 0.0)
        return sampled_association_global_indices, slack, prv_required_cells, sampled_a_info



    num_possible_rows_to_partition = M-len(prv_required_cells)
    # print "M:", M
    # print "len(prv_required_cells):", len(prv_required_cells)
    if num_possible_rows_to_partition == 0: #we've sampled a complete association
        # print "(matrix_idx, tuple(prv_required_cells)):", (matrix_idx, tuple(prv_required_cells))
        assert((matrix_idx, tuple(prv_required_cells)) in MATRIX_PERMANENT_UBS.upper_bounds_dictionary)
        sampled_a_info = sample_unassoc_measurementsAndTargets(tuple(prv_required_cells), orig_a_matrix, matrix_idx)

        assert(MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells)) > math.factorial(T)*sampled_a_info.bottom_prob*sampled_a_info.conditional_unassociated_probability or\
            np.allclose(math.factorial(T)*sampled_a_info.bottom_prob*sampled_a_info.conditional_unassociated_probability, MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells)), atol=0)), (MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells)), math.factorial(T)*sampled_a_info.bottom_prob*sampled_a_info.conditional_unassociated_probability, T, local_matrix, sampled_a_info.bottom_prob, sampled_a_info.conditional_unassociated_probability, sampled_submatrix, orig_a_matrix.matrix, orig_a_matrix.M, orig_a_matrix.T, orig_a_matrix.conditional_birth_probs, orig_a_matrix.conditional_death_probs)
        
        #first time this matrix has been sampled, the bound is loose
        if(MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells)) >= math.factorial(T)*sampled_a_info.bottom_prob):
            sampled_slack = MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells)) - math.factorial(T)*sampled_a_info.bottom_prob * (1 - sampled_a_info.conditional_unassociated_probability)
            #set upper bound to the exact permanent (calculate because rows are constant) - (the sampled value)*T!
            MATRIX_PERMANENT_UBS.set_upper_bound(matrix_idx, tuple(prv_required_cells), math.factorial(T)*sampled_a_info.bottom_prob * (1 - sampled_a_info.conditional_unassociated_probability))
            assert(MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells)) >= 0), MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells))
        else:
            #the value we sampled
            sampled_slack = math.factorial(T)*sampled_a_info.bottom_prob * sampled_a_info.conditional_unassociated_probability
            assert((MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells)) > sampled_slack) or np.allclose(sampled_slack, MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells)), atol=0))
            sampled_slack = MATRIX_PERMANENT_UBS.tighten_upper_bound(matrix_idx, tuple(prv_required_cells), sampled_slack)

        exact_permanent1, check_exact = minc_extended_UB2(matrix, check_exact=True)
        assert(check_exact)
        #numerical issue, set to 0
        if MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells)) < exact_permanent1*sampled_a_info.min_possible_conditional_unassociated_probability/1.01:
            sampled_slack += MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells))
            MATRIX_PERMANENT_UBS.set_upper_bound(matrix_idx, tuple(prv_required_cells), 0.0)

        sampled_association_global_indices = []

        return sampled_association_global_indices, sampled_slack, prv_required_cells, sampled_a_info









    # Get all permutations of length depth of numbers 0 through N-1
    fixed_column_options = list(itertools.permutations(range(N), depth))
    
    # print
    # print "about to call find_best_row_to_partition_matrix"
    # print "M=", M
    # print "len(prv_required_cells):", len(prv_required_cells)
    # print "local_matrix:"
    # print local_matrix
    if tracking_constant_num_targets:
        best_row_to_partition = find_best_row_to_partition_matrix_constantNumTargets(local_matrix, matrix_idx, prv_required_cells_copy)
    else:
        best_row_to_partition = find_best_row_to_partition_matrix(local_matrix, matrix_idx, prv_required_cells_copy, rows_to_select=M-len(prv_required_cells), global_row_indices=global_row_indices,\
                                                                  global_col_indices=global_col_indices)

    #swap rows
    # temp_row = np.copy(local_matrix[0])
    # local_matrix[0] = local_matrix[best_row_to_partition]
    # local_matrix[best_row_to_partition] = temp_row
    local_matrix[[0,best_row_to_partition]] = local_matrix[[best_row_to_partition,0]]

    #swap global indices
    if DEBUG1:
        print "about to swap global_row_indices"
        print "prv_required_cells:", prv_required_cells, "best_row_to_partition:", best_row_to_partition, "global_row_indices:", global_row_indices
    temp_idx = global_row_indices[0]
    global_row_indices[0] = global_row_indices[best_row_to_partition]
    global_row_indices[best_row_to_partition] = temp_idx
    if DEBUG1:
        print "after swap global_row_indices"
        print "prv_required_cells:", prv_required_cells, "best_row_to_partition:", best_row_to_partition, "global_row_indices:", global_row_indices

    proposal_distribution = []
    if DEBUG1:
        print "submatrix upper bounds:"
    assert(local_matrix.shape[0] > 1)
    if verbose:
        print "permanentUB:", permanentUB
        print "proposal distribution submatrix_permanent_UBs:",

    # print "proposal distribution for required cells:"     
    for fixed_columns in (fixed_column_options):
        cur_submatrix = np.delete(local_matrix, fixed_columns, 1) #delete columns
        cur_submatrix = np.delete(cur_submatrix, range(depth), 0) #delete rows

        hashable_matrix = tuple([tuple(row) for row in cur_submatrix])
        required_cells = tuple(prv_required_cells_copy + [(global_row_indices[row], global_col_indices[fixed_columns[row]]) for row in range(depth)])

        submatrix_permanent_UB = MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, required_cells, cur_submatrix)            
        if verbose:
            print submatrix_permanent_UB,

        # print submatrix_permanent_UB
        upper_bound_submatrix_count = submatrix_permanent_UB

        # print   
        # print "fixed_columns:", fixed_columns
        # print "upper_bound_submatrix_count:", upper_bound_submatrix_count  
        # print "cur_submatrix:", cur_submatrix
        for row in range(depth):
            upper_bound_submatrix_count *= local_matrix[row, fixed_columns[row]]
            # print "local_matrix[row, fixed_columns[row]]:", local_matrix[row, fixed_columns[row]]   
        assert(submatrix_permanent_UB >= 0), submatrix_permanent_UB
        if verbose:
            print upper_bound_submatrix_count, ','
        proposal_distribution.append(upper_bound_submatrix_count)
        if DEBUG1:
            print upper_bound_submatrix_count,

    # print
    if DEBUG1:
        print

    sum_of_submatrix_UBs = np.sum(proposal_distribution)
    # print "proposal_distribution:", proposal_distribution
    assert(sum_of_submatrix_UBs > 0), (sum_of_submatrix_UBs, proposal_distribution, matrix)

    if sum_of_submatrix_UBs <= permanentUB or np.allclose(sum_of_submatrix_UBs, permanentUB, atol=0):
        # print "1 prv_required_cells:", prv_required_cells, "sum_of_submatrix_UBs:", sum_of_submatrix_UBs, "permanentUB:", permanentUB
        cur_level_slack = permanentUB - sum_of_submatrix_UBs
        if cur_level_slack < 0.0:
            cur_level_slack = 0.0
        proposal_distribution.append(cur_level_slack)
        unnormalized_proposal_dist = np.copy(proposal_distribution)
        proposal_distribution /= np.sum(proposal_distribution)

        sampled_association_idx = np.random.choice(len(proposal_distribution), p=proposal_distribution)

        if sampled_association_idx == len(proposal_distribution) - 1:
            sampled_association = None #we sampled a weight 0 association

            # print 'about to call tighten_upper_bound 1'
            cur_level_slack = MATRIX_PERMANENT_UBS.tighten_upper_bound(matrix_idx, tuple(prv_required_cells), cur_level_slack)
                                
            slack_correction_term = MATRIX_PERMANENT_UBS.correct_errors_during_tightening(local_matrix, matrix_idx, prv_required_cells, global_row_indices, global_col_indices)
            MATRIX_PERMANENT_UBS.check_nesting(local_matrix, matrix_idx, prv_required_cells, global_row_indices, global_col_indices)
            cur_level_slack -= slack_correction_term
            assert(MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, tuple(prv_required_cells)) >= 0)
            
            
            return sampled_association, cur_level_slack, None, None
        else:
            sampled_fixed_columns = fixed_column_options[sampled_association_idx]
            sampled_submatrix = np.delete(local_matrix, sampled_fixed_columns, 1) #delete columns
            sampled_submatrix = np.delete(sampled_submatrix, range(depth), 0) #delete rows
            sampled_association = [(row, sampled_fixed_columns[row]) for row in range(depth)]
            sampled_association_global_indices = [(global_row_indices[local_row], global_col_indices[local_col]) for (local_row, local_col) in sampled_association]
           
            hashable_matrix = tuple([tuple(row) for row in sampled_submatrix])
            required_cells = tuple(prv_required_cells_copy + sampled_association_global_indices)
            sampled_submatrix_permanent_UB = MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, required_cells)

            prv_required_cells_copy.extend(sampled_association_global_indices)
            submatrix_global_row_indices = np.delete(global_row_indices, range(depth))
            submatrix_global_col_indices = np.delete(global_col_indices, sampled_fixed_columns)

            if DEBUG_3:
                print '1 calling sample_association_01matrix_plusSlack with prv_required_cells:', prv_required_cells_copy
            remaining_sampled_associations, sub_tree_slack, prv_required_cells_at_sample, sampled_a_info = sample_association_01matrix_plusSlack(sampled_submatrix, matrix_idx, sampled_submatrix_permanent_UB, prv_required_cells_copy, depth=1, global_row_indices=submatrix_global_row_indices, global_col_indices=submatrix_global_col_indices, M=M, T=T, orig_a_matrix=orig_a_matrix)

                
            MATRIX_PERMANENT_UBS.check_nesting(local_matrix, matrix_idx, prv_required_cells, global_row_indices, global_col_indices)            
            # print "cur_level_slack + sub_tree_slack*local_matrix[0, sampled_fixed_columns[0]]:", cur_level_slack + sub_tree_slack*local_matrix[0, sampled_fixed_columns[0]]
            # print "cur_level_slack :", cur_level_slack 
            # print "sub_tree_slack*local_matrix[0, sampled_fixed_columns[0]]:", sub_tree_slack*local_matrix[0, sampled_fixed_columns[0]]
            # print "unnormalized_proposal_dist:", unnormalized_proposal_dist
            
            # print 'about to call tighten_upper_bound 2'
            corrected_total_slack = MATRIX_PERMANENT_UBS.tighten_upper_bound(matrix_idx, tuple(prv_required_cells), slack=cur_level_slack + sub_tree_slack*local_matrix[0, sampled_fixed_columns[0]])

            slack_correction_term = MATRIX_PERMANENT_UBS.correct_errors_during_tightening(local_matrix, matrix_idx, prv_required_cells, global_row_indices, global_col_indices)
            MATRIX_PERMANENT_UBS.check_nesting(local_matrix, matrix_idx, prv_required_cells, global_row_indices, global_col_indices)
            corrected_total_slack -= slack_correction_term



            if remaining_sampled_associations is None: #we sampled some slack
                sampled_association_global_indices = None
                return sampled_association_global_indices, corrected_total_slack, prv_required_cells_at_sample, sampled_a_info
            else:
                sampled_association_global_indices.extend(remaining_sampled_associations)
                return sampled_association_global_indices, corrected_total_slack, prv_required_cells_at_sample, sampled_a_info
    else:
        print "sum_of_submatrix_UBs > permanentUB :(:(:("
        print "matrix_idx, tuple(prv_required_cells):", matrix_idx, tuple(prv_required_cells)
        print "sum_of_submatrix_UBs: ", sum_of_submatrix_UBs
        print "permanentUB: ", permanentUB

        print "sum_of_submatrix_UBs-permanentUB: ", sum_of_submatrix_UBs-permanentUB
        print "(sum_of_submatrix_UBs-permanentUB)/permanentUB: ", (sum_of_submatrix_UBs-permanentUB)/permanentUB
        print "np.log(sum_of_submatrix_UBs)-np.log(permanentUB): ", np.log(sum_of_submatrix_UBs)-np.log(permanentUB)
        print "try other partitionings"
        print "matrix:",
        print matrix
        assert(False), "not expecting this! also fix find_best_row_to_partition_matrix and caching there, etc."
        find_best_row_to_partition_matrix(local_matrix, matrix_idx, prv_required_cells_copy)
        print

        sampled_association_global_indices = sample_association_01matrix_plusSlack(local_matrix, matrix_idx, permanentUB, prv_required_cells_copy, depth=depth+1, global_row_indices=global_row_indices, global_col_indices=global_col_indices, M=M, T=T, orig_a_matrix=orig_a_matrix)
        return sampled_association_global_indices
    # return sampled_association_global_indices


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

def minc_extended_UB2(matrix, verbose=False, check_exact=False, try_rescaling=False):
    # print "minc_extended_UB2 called on matrix:"
    # print matrix

    if COMPARE_WAI:
        return immediate_nesting_extended_bregman(matrix)
    assert(matrix.shape[0] == matrix.shape[1])
    N = matrix.shape[0]
    assert(N>0)
    # print "matrix.shape:", matrix.shape
    # print matrix
    # if matrix.shape == (0,0): #does this make sense!?!?
    #     return 1.0
    if(matrix == matrix[0]).all(): #if all columns are constants, calculate exact permanent
        if verbose:
            print "all columns are constants:"
            print matrix
        if check_exact:
            return math.factorial(N) * np.prod(matrix[0]), True
        else:
            return math.factorial(N) * np.prod(matrix[0])

    elif(matrix[1:, :] == matrix[1]).all(): #if all columns are constants, excluding first element, calculate exact permanent
        if verbose:
            print "all columns, excluding first element, are constants:"
            print matrix

        exact_permanent = 0
        for row in range(N):
            exact_permanent += matrix[0, row] * np.prod(matrix[1, :row]) * np.prod(matrix[1, row+1:])
        exact_permanent *= math.factorial(N-1)
        if check_exact:
            return exact_permanent, True
        else:
            return exact_permanent

    if verbose:
        print "all columns are not constants:"
        print matrix

    deltas = np.array([delta(i + 1) for i in range(N)])
    matrix_sorted = np.sort(matrix, axis=1)[:, ::-1]
    computed_minc_extended_upper_bound = (matrix_sorted * deltas).sum(axis=1).prod()
    if try_rescaling:
        computed_minc_extended_upper_bound = min(computed_minc_extended_upper_bound, rescaled_tracking_UB(matrix))
    # if computed_minc_extended_upper_bound > 0: #just testing potential speed here
    #     test_conjectured_optimal_bound = conjectured_optimal_bound(matrix)
    #     upper_bound_return_val = min(test_conjectured_optimal_bound, computed_minc_extended_upper_bound)
    # else:
    #     upper_bound_return_val = computed_minc_extended_upper_bound
    if check_exact:
        return computed_minc_extended_upper_bound, False
    else:
        return computed_minc_extended_upper_bound

# @profile
# @nb.jit
# @nb.guvectorize(["float64(float64[:,:])"], "(n,n)->()")
def minc_extended_UB2_not_vectorized(matrix):
    #another bound
    #https://ac-els-cdn-com.stanford.idm.oclc.org/S002437950400299X/1-s2.0-S002437950400299X-main.pdf?_tid=fa4d00ee-39a5-4030-b7c1-28bb5fbc76c0&acdnat=1534454814_a7411b3006e0e092622de35cbf015275
    # equation (6), U^M(A)

    if COMPARE_WAI:
        return immediate_nesting_extended_bregman(matrix)

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

# @profile
def immediate_nesting_extended_bregman(matrix):
    #https://dukespace.lib.duke.edu/dspace/bitstream/handle/10161/1054/D_Law_Wai_a_200904.pdf?sequence=1&isAllowed=y
    assert((matrix <= 1).all())
    assert((matrix >= 0).all())
    N = matrix.shape[0]
    assert(N == matrix.shape[1])
    col_sum = matrix.sum(axis=0)
    h_func = np.where(col_sum >= 1, col_sum + 0.5 * np.log(col_sum) + np.e - 1, 1 + (np.e - 1) * col_sum) / np.e
    return h_func.prod()

def h_func(r):
    if r >= 1:
        return r + .5*math.log(r) + np.e - 1
    else:
        return 1 + (np.e - 1)*r


# @profile
def immediate_nesting_extended_bregman_not_vectorized(matrix):
    #https://dukespace.lib.duke.edu/dspace/bitstream/handle/10161/1054/D_Law_Wai_a_200904.pdf?sequence=1&isAllowed=y


    assert((matrix <= 1).all())
    assert((matrix >= 0).all())
    N = matrix.shape[0]
    assert(N == matrix.shape[1])
    bregman_extended_upper_bound = 1
    for col in range(N):
        col_sum = 0
        for row in range(N):
            col_sum += matrix[row][col]

        bregman_extended_upper_bound *= h_func(col_sum)/np.e

    return bregman_extended_upper_bound

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
        assert((self.remaining_cost_matrix > 0).all()), self.remaining_cost_matrix
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
        TEST_SUBMATRIX_BOUND3 = True

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
                # pass #use origina A star bound
                #use the log_minc_extended_upper_bound2 permanent upper bound for this submatrix
                # self.upper_bound_gumbel_perturbed_state = self.log_minc_extended_upper_bound2 + self.max_gumbel_1
                exact_permanent = calc_permanent_rysers(sub_matrix)
                #test using exact permanent as upper bound
                self.upper_bound_gumbel_perturbed_state = np.log(exact_permanent) + self.max_gumbel_1

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




def plot_runtime_vs_N(pickle_file_paths=['./number_of_times_partition_called_for_each_n.pickle'], pickle_file_paths2=None,\
                      plot_filename=None):
    n_vals_mean1 = []
    log_n_vals_mean1 = []
    run_time_vals_mean1 = []
    number_of_times_partition_called_vals_mean = []
    all_n_vals1 = []
    all_run_time_vals1 = []
    all_number_of_times_partition_called_vals = []
    for pickle_file_path in pickle_file_paths:
        f = open(pickle_file_path, 'rb')
        number_of_times_partition_called_for_each_n = pickle.load(f)
        f.close()
        # for n, (number_of_times_partition_called_list, node_count_plus_heap_sizes_list, runtimes_list) in number_of_times_partition_called_for_each_n.items():
        for n, (runtimes_list, all_samples_of_log_Z, exact_log_Z, permanent_UBs) in number_of_times_partition_called_for_each_n.items():
            if n < 5:
                continue
            all_n_vals1.extend([n for i in range(len(runtimes_list))])
            all_run_time_vals1.extend(runtimes_list)
            # all_number_of_times_partition_called_vals.extend(number_of_times_partition_called_list)
            log_n_vals_mean1.append(math.log(n))
            n_vals_mean1.append(n)
            run_time_vals_mean1.append(math.log(np.mean(runtimes_list)))
            # run_time_vals_mean1.append(np.mean(runtimes_list))
            # number_of_times_partition_called_vals_mean.append(math.log(np.mean(number_of_times_partition_called_list)))
            # number_of_times_partition_called_vals_mean.append(np.mean(number_of_times_partition_called_list))
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(n_vals_mean1, run_time_vals_mean1, 'r+', label='ours' , markersize=10)
    # ax.plot(all_n_vals1, all_run_time_vals1, 'r+', label='run_time_vals_mean' , markersize=10)

    if pickle_file_paths2 is not None:
        n_vals_mean2 = []
        run_time_vals_mean2 = []
        log_n_vals_mean2 = []

        for pickle_file_path in pickle_file_paths2:
            f = open(pickle_file_path, 'rb')
            number_of_times_partition_called_for_each_n = pickle.load(f)
            f.close()

            for n, (runtimes_list, all_samples_of_log_Z, exact_log_Z, permanent_UBs) in number_of_times_partition_called_for_each_n.items():
                if n < 5:
                    continue
                n_vals_mean2.append(n)
                log_n_vals_mean2.append(math.log(n))
                run_time_vals_mean2.append(math.log(np.mean(runtimes_list)))
                # run_time_vals_mean2.append(np.mean(runtimes_list))

        ax.plot(n_vals_mean2, run_time_vals_mean2, 'y+', label='baseline' , markersize=10)

    matplotlib.rcParams.update({'font.size': 20})

    # ax.plot(n_vals_mean, number_of_times_partition_called_vals_mean, 'gx', label='number_of_times_partition_called_vals_mean' , markersize=10)
    plt.title('Runtime Scaling')
    plt.xlabel('N (matrix dimension)')
    plt.ylabel('log(runtime) (log(seconds))')
    # Put a legend below current axis
    lgd = ax.legend(loc='lower right',# bbox_to_anchor=(0.5, -.11),
              fancybox=False, shadow=False, ncol=1, numpoints = 1)
    fig.savefig(plot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent_estimation_scaling_plots_diagMatrix', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent_estimation_scaling_plots_uniformMatrix', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent_estimation_scaling_plots_01Matrix', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    plt.close()

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(log_n_vals_mean1, run_time_vals_mean1, 'r+', label='runtime' , markersize=10)
    if pickle_file_paths2 is not None:
        ax.plot(log_n_vals_mean2, run_time_vals_mean2, 'y+', label='baseline' , markersize=10)

    plt.title('Runtime Scaling')
    plt.xlabel('log(N) (log(matrix dimension))')
    plt.ylabel('log(runtime) (log(seconds))')
    # Put a legend below current axis
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
              fancybox=False, shadow=False, ncol=2, numpoints = 1)
    fig.savefig(plot_filename + 'logN', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent_estimation_scaling_plots_logN_diagMatrix', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent_estimation_scaling_plots_logN_uniformMatrix', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent_estimation_scaling_plots_logN_01Matrix', bbox_extra_artists=(lgd,), bbox_inches='tight')    

    plt.close()

    return 

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

    # fig.savefig('permanent_estimation_scaling_plots', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    fig.savefig('permanent_estimation_scaling_plots_diagMatrix', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent sampling scaling single gumbel perm UB', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent sampling scaling single gumbel, pick partition order perm UB', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent sampling scaling single gumbel, close to 01 matrix perm UB', bbox_extra_artists=(lgd,), bbox_inches='tight')    

    plt.close()


def plot_estimateAndExactPermanent_vs_N(pickle_file_paths=['./number_of_times_partition_called_for_each_n.pickle'], pickle_file_paths2=None):
    n_vals_mean = []
    mle_permanent_estimates = []
    exact_permanents = []
    upper_bounds  = []
    lower_bounds = []
    for pickle_file_path in pickle_file_paths:
        f = open(pickle_file_path, 'rb')
        number_of_times_partition_called_for_each_n = pickle.load(f)
        f.close()
        # for n, (number_of_times_partition_called_list, node_count_plus_heap_sizes_list, runtimes_list) in number_of_times_partition_called_for_each_n.items():
        for n, (runtimes_list, all_samples_of_log_Z, exact_log_Z) in number_of_times_partition_called_for_each_n.items():
            if n < 5:
                continue
            n_vals_mean.append(n)
            cur_permanent_estimate = -np.log(np.mean(np.exp(-np.array(all_samples_of_log_Z))))
            mle_permanent_estimates.append(cur_permanent_estimate)
            lower_bounds.append(cur_permanent_estimate-1)
            upper_bounds.append(cur_permanent_estimate+1)
            exact_permanents.append(exact_log_Z)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(n_vals_mean, mle_permanent_estimates, 'rx', label='permanent estimates' , markersize=5)
    ax.plot(n_vals_mean, lower_bounds, 'bo', label='lower bounds' , markersize=2)
    ax.plot(n_vals_mean, upper_bounds, 'yo', label='upper bounds' , markersize=2)
    ax.plot(n_vals_mean, exact_permanents, 'g+', label='exact permanents' , markersize=5)
    plt.title('High Probability Permanent Bounds')
    plt.xlabel('N (matrix dimension)')
    plt.ylabel('log(permanent)')
    # Put a legend below current axis
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
              fancybox=False, shadow=False, ncol=2, numpoints = 1)


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

    # fig.savefig('permanent_estimate_VS_exact_plots', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    fig.savefig('permanent_estimate_VS_exact_plots_diagMatrix', bbox_extra_artists=(lgd,), bbox_inches='tight')    
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

# @profile
def test_sampling_correctness(N=5, ITERS=10000000, matrix_to_use='rand'):
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
                if matrix[row][col] < .1:
                    # matrix[row][col] = 0
                    matrix[row][col] = matrix[row][col] ** 1
                else:
                    # matrix[row][col] = 1
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
    number_of_times_partition_called_list, node_count_plus_heap_sizes_list, runtimes_list, all_sampled_associations, wall_time, log_Z_estimate, all_samples_of_log_Z, permanent_UBs = \
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
    fig_file_name = "01matrix_improvedslack_test_sampling_empirical_vs_true_probabilities_N=%d_seed=%d_1gumbel_iters=%d_matrix=%s"%(N, SEED, ITERS, matrix_to_use)
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

    number_of_times_partition_called_list, node_count_plus_heap_sizes_list, runtimes_list, all_sampled_associations, wall_time, log_Z_estimate, all_samples_of_log_Z, permanent_UBs = \
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

def test_permanent_bound_tightness(N, use_matrix=False, matrix=None):
    if use_matrix == False:
        use_diag_matrix = True
        if use_diag_matrix:
            matrix, exact_permanent = create_diagonal2(N, k=5, zero_one=False)

        else:
            matrix = np.random.rand(N,N)
            for row in range(N):
                for col in range(N):
                    if matrix[row][col] < .5:
                        matrix[row][col] = matrix[row][col] ** 1
                        # matrix[row][col] = 0
                    else:
                        matrix[row][col] = 1 - (1 - matrix[row][col])**1
                        # matrix[row][col] = 1


    # matrix = np.array([[0.02119613, 0.63227319, 0.76980874, 0.48558316, 0.09312535, 0.21782214, 0.08640624, 0.45764297, 0.94977823, 0.53365085],
    #               [0.8760028,  0.71496163, 0.02277475, 0.18681426, 0.30267468, 0.16571689, 0.98345952, 0.85441607, 0.15731342, 0.19554486],
    #               [0.40401111, 0.88049258, 0.91961023, 0.59388085, 0.08951876, 0.89382644, 0.76628156, 0.43223284, 0.43246381, 0.46189491],
    #               [0.83991298, 0.20435737, 0.5447851,  0.63195192, 0.24507631, 0.69310934, 0.8518319,  0.39510064, 0.63028636, 0.93082819],
    #               [0.33150593, 0.58859955, 0.48313886, 0.46273538, 0.76226034, 0.40406069, 0.41718348, 0.26634174, 0.84638477, 0.27546787],
    #               [0.81861929, 0.32600449, 0.0542015,  0.37139681, 0.17534188, 0.65221004, 0.95801876, 0.47385952, 0.4470294,  0.30665919],
    #               [0.1329523,  0.55384001, 0.75787698, 0.68159521, 0.45342187, 0.88853756, 0.44781194, 0.6250724,  0.68744354, 0.11769418],
    #               [0.30805376, 0.1971726,  0.66141022, 0.61088737, 0.72586821, 0.24768436, 0.47490583, 0.35234266, 0.83246263, 0.81291092],
    #               [0.95881507, 0.88264871, 0.98512152, 0.6529703,  0.52572222, 0.87133197, 0.38387505, 0.2867429,  0.80746319, 0.89256982],
    #               [0.37829088, 0.57979053, 0.29453573, 0.21166937, 0.48988732, 0.52620089, 0.67418419, 0.30791013, 0.05857503, 0.16759526]])

    # exact_permanent = calc_permanent_rysers(matrix)
    exact_permanent = 0


    # minc2_sub_matrix_min_bound = 0
    # row = 0
    # sub_matrix = np.delete(matrix, row, 0)
    # for col in range(matrix.shape[1]):
    #     cur_sub_matrix = np.delete(sub_matrix, col, 1)
    #     minc2_sub_matrix_min_bound += matrix[row, col] * minc_extended_UB2(cur_sub_matrix)

    # minc2_sub_matrix2_min_bound = 0
    # row = 0
    # sub_matrix = np.delete(matrix, row, 0)
    # for col in range(matrix.shape[1]):
    #     cur_sub_matrix = np.delete(sub_matrix, col, 1)
    #     for col1 in range(cur_sub_matrix.shape[1]):
    #         cur_sub_matrix2 = np.delete(cur_sub_matrix, 0, 0)
    #         cur_sub_matrix2 = np.delete(cur_sub_matrix2, col1, 1)
    #         assert(cur_sub_matrix2.shape[0] == cur_sub_matrix2.shape[1] and cur_sub_matrix2.shape[1] == matrix.shape[1]-2)
    #         minc2_sub_matrix2_min_bound += matrix[row, col]* matrix[1, col1] * minc_extended_UB2(cur_sub_matrix2)

    minc_UB2 = minc_extended_UB2(matrix)
    minc_UB2_of_transpose = minc_extended_UB2(np.transpose(matrix))
    optimized_minc_extended_upper_bound2 = optimized_minc_extened_UB2(matrix)
    # optimized_minc_extended_upper_bound2 = 0
    bregman_extended_upper_bound = immediate_nesting_extended_bregman(matrix)

    conjectured_optimal_bound_val = conjectured_optimal_bound(matrix)

    print 'log(exact_permanent) =', np.log(exact_permanent)
    print 'log(bregman_extended_upper_bound) =', np.log(bregman_extended_upper_bound)
    print 'log extended minc2 UB =', np.log(minc_UB2)
    print 'log extended minc2 UB of transpose =', np.log(minc_UB2_of_transpose)
    print 'log optimized extended minc2 UB =', np.log(optimized_minc_extended_upper_bound2)
    print 'log conjectured_optimal_bound_val =', np.log(conjectured_optimal_bound_val)
    # print 'difference =', np.log(minc_UB2) - np.log(exact_permanent)
    # print     
    # print 'log extended minc2 submatrix UB =', np.log(minc2_sub_matrix_min_bound)    
    # print 'log(minc2) - log(submatrix) =', np.log(minc_UB2) - np.log(minc2_sub_matrix_min_bound)
    # print 'log extended minc2 submatrix2 UB =', np.log(minc2_sub_matrix2_min_bound)    
    # print 'log(minc2) - log(submatrix2) =', np.log(minc_UB2) - np.log(minc2_sub_matrix2_min_bound)

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

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def create_diagonal(in_matrix, n):
    '''
    create diag_matrix, a diagonal matrix with n copies of in_matrix on it's diagonal 
    Inputs:
    - in_matrix: numpy array,
    - n: int, 

    Ouputs:
    - diag_matrix: numpy array, the array we created with shape of 
        in_matrix.shape[0]*n X in_matrix.shape[1]*n
    '''    
    in_matrix_exact_permanent = calc_permanent_rysers(in_matrix)
    diag_matrix_permanent = in_matrix_exact_permanent**n 
    diag_matrix = np.zeros((in_matrix.shape[0]*n, in_matrix.shape[1]*n))
    print diag_matrix.shape
    for i in range(n):
        diag_matrix[i*in_matrix.shape[0]:(i+1)*in_matrix.shape[0], \
                    i*in_matrix.shape[1]:(i+1)*in_matrix.shape[1]] = in_matrix
    return diag_matrix, diag_matrix_permanent

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

    

def per(mtx, column, selected, prod, output=False):
    """
    Row expansion for the permanent of matrix mtx.
    The counter column is the current column, 
    selected is a list of indices of selected rows,
    and prod accumulates the current product.
    http://homepages.math.uic.edu/~jan/mcs507f13/permanent.py
    """
    if column == mtx.shape[1]:
        if output:
            print selected, prod
        return prod
    else:
        result = 0
        for row in range(mtx.shape[0]):
            if not row in selected:
                result = result \
                + per(mtx, column+1, selected+[row], prod*mtx[row,column])
        return result

def permanent_check(mat):
    """
    Returns the permanent of the matrix mat.
    http://homepages.math.uic.edu/~jan/mcs507f13/permanent.py
    """
    return per(mat, 0, [], 1)

# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)
# @cython.cdivision(True)
# def permfunc_cython(np.ndarray [double, ndim = 2, mode = 'c'] M):
#     cdef:
#         int n = M.shape[0], s=1, i, j
#         int *f = <int*>malloc(n*sizeof(int))
#         double *d = <double*>malloc(n*sizeof(double))
#         double *v = <double*>malloc(n*sizeof(double))
#         double p = 1, prod

#     for i in range(n):
#         v[i] = 0.
#         for j in range(n):
#             v[i] += M[j,i]
#         p *= v[i]
#         f[i] = i
#         d[i] = 2
#     j = 0
#     while (j < n-1):
#         prod = 1.
#         for i in range(n):
#             v[i] -= d[j]*M[j, i]
#             prod *= v[i]
#         d[j] = -d[j]
#         s = -s            
#         p += s*prod
#         f[0] = 0
#         f[j] = f[j+1]
#         f[j+1] = j+1
#         j = f[0]

#     free(d)
#     free(f)
#     free(v)
#     return p/pow(2.,(n-1))

def npperm(M):
    #https://github.com/scipy/scipy/issues/7151    
    n = M.shape[0]
    d = np.ones(n)
    j =  0
    s = 1
    f = np.arange(n)
    v = M.sum(axis=0)
    p = np.prod(v)
    while (j < n-1):
        v -= 2*d[j]*M[j]
        d[j] = -d[j]
        s = -s
        prod = np.prod(v)
        p += s*prod
        f[0] = 0
        f[j] = f[j+1]
        f[j+1] = j+1
        j = f[0]
    return p/2**(n-1) 

def test_create_diagonal():
    N=11
    matrix = np.random.rand(N,N)
    for row in range(N):
        for col in range(N):
            if matrix[row][col] < .1:
                matrix[row][col] = 0
            else:
                matrix[row][col] = 1

    diag_matrix, permanent = create_diagonal(matrix, n=2)
    check_permanent = calc_permanent_rysers(diag_matrix)
    # check_permanent1 = permanent_check(diag_matrix)
    # check_permanent1 = permfunc_cython(diag_matrix)
    check_permanent1 = npperm(diag_matrix)
    print "permanent - check_permanent =", permanent - check_permanent
    print "permanent - check_permanent1 =", permanent - check_permanent1
    print "check_permanent1 - check_permanent =", check_permanent1 - check_permanent 
    print "check_permanent =", check_permanent
    print "check_permanent1 =", check_permanent1
    print "permanent =", permanent
    assert(permanent == check_permanent), (permanent, check_permanent)

def test_create_diagonal2():
    N=18
    diag_matrix, permanent = create_diagonal2(N=N, k=5, zero_one=True)
    check_permanent = calc_permanent_rysers(diag_matrix)
    # check_permanent1 = permanent_check(diag_matrix)
    # check_permanent1 = permfunc_cython(diag_matrix)
    check_permanent1 = npperm(diag_matrix)
    print "permanent - check_permanent =", permanent - check_permanent
    print "permanent - check_permanent1 =", permanent - check_permanent1
    print "check_permanent1 - check_permanent =", check_permanent1 - check_permanent 
    print "check_permanent =", check_permanent
    print "check_permanent1 =", check_permanent1
    print "permanent =", permanent
    print diag_matrix
    assert(permanent == check_permanent), (permanent, check_permanent)    
 

def plot_pruning_effect(pickle_file_paths=['./number_of_times_partition_called_for_each_n.pickle'], pickle_file_paths2=None):
    n_vals_mean = []
    log_n_vals_mean = []
    run_time_vals_mean = []
    number_of_times_partition_called_vals_mean = []
    all_n_vals = []
    all_run_time_vals = []
    all_number_of_times_partition_called_vals = []
    fig = plt.figure()
    ax = plt.subplot(111)


    for pickle_file_path in pickle_file_paths:
        f = open(pickle_file_path, 'rb')
        number_of_times_partition_called_for_each_n = pickle.load(f)
        f.close()
        # for n, (number_of_times_partition_called_list, node_count_plus_heap_sizes_list, runtimes_list) in number_of_times_partition_called_for_each_n.items():
        for n, (runtimes_list, all_samples_of_log_Z, exact_log_Z, permanent_UBs) in number_of_times_partition_called_for_each_n.items():
            if n < 5:
                continue
            smooth_int = 1
            sample_number = range(int(np.floor(len(runtimes_list)/smooth_int)))
            smoothed_runtime_list = [np.mean(runtimes_list[i*smooth_int:(i+1)*smooth_int]) for i in range(int(np.floor(len(runtimes_list)/smooth_int)))]
            print (sample_number)
            print (smoothed_runtime_list)
            # ax.plot(sample_number, smoothed_runtime_list, 'r+', label='matrix size = %d'%n , markersize=10)
            ax.plot(sample_number, permanent_UBs, 'r+', label='matrix size = %d'%n , markersize=10)
    # ax.plot(all_n_vals, all_run_time_vals, 'r+', label='run_time_vals_mean' , markersize=10)

    # ax.plot(n_vals_mean, number_of_times_partition_called_vals_mean, 'gx', label='number_of_times_partition_called_vals_mean' , markersize=10)
    plt.title('Runtime with pruning')
    plt.xlabel('sample number')
    plt.ylabel('runtime')
    # Put a legend below current axis
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
              fancybox=False, shadow=False, ncol=2, numpoints = 1)
    fig.savefig('pruning_effect', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent_estimation_scaling_plots_uniformMatrix', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent_estimation_scaling_plots_01Matrix', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    plt.close()

 
def find_max_assignment(matrix):
    assert((matrix >= 0).all()), matrix
    cost_matrix = -np.log(matrix)

    lin_assign = linear_assignment.LinearAssignment(cost_matrix)
    solution = lin_assign.solution
    association_list = zip([i for i in range(len(solution))], solution)

    minimum_cost = 0
    for (row,col) in association_list:
        minimum_cost += np.asscalar(cost_matrix[row][col])
    max_assignment = np.exp(-minimum_cost)
    # print "association_list:", association_list
    # print "max_assignment:", max_assignment
    return (association_list, max_assignment)

def test_nesting(N, verbose=True):
    use_diag_matrix = False
    if use_diag_matrix:
        matrix, exact_permanent = create_diagonal2(N, k=5, zero_one=False)

    else:
        matrix = np.random.rand(N,N)
        for row in range(N):
            for col in range(N):
                if matrix[row][col] < .5:
                    matrix[row][col] = matrix[row][col] ** 1
                    # matrix[row][col] = 0
                else:
                    matrix[row][col] = 1 - (1 - matrix[row][col])**1
                    # matrix[row][col] = 1

    fixed_column_options = list(itertools.permutations(range(N), 1))
    matrix_UB = (conjectured_optimal_bound(matrix))
    if verbose:
        print "find_best_row_to_partition_matrix", '*'*80
        print "matrix_UB:", matrix_UB
    row_with_smallest_partitioned_UB = None
    smallest_partitioned_upper_bound = None
    for row in range(N):
        proposal_distribution = []
        for fixed_columns in (fixed_column_options):
            cur_submatrix = np.delete(matrix, fixed_columns, 1) #delete columns
            cur_submatrix = np.delete(cur_submatrix, [row], 0) #delete rows

            submatrix_permanent_UB = (conjectured_optimal_bound(cur_submatrix)) #add a little for potential computational error, would be nice to make this cleaner
            # print row, fixed_columns, submatrix_permanent_UB

            upper_bound_submatrix_count = submatrix_permanent_UB
            upper_bound_submatrix_count *= matrix[row, fixed_columns[0]]

            proposal_distribution.append(upper_bound_submatrix_count)
        cur_partitioned_UB = np.sum(proposal_distribution)
        if smallest_partitioned_upper_bound is None or cur_partitioned_UB < smallest_partitioned_upper_bound:
            smallest_partitioned_upper_bound = cur_partitioned_UB
            row_with_smallest_partitioned_UB = row
        if verbose:
            print "partitioned UB:", np.sum(proposal_distribution)
            print "(partitioned UB)/matrix_UB:", np.sum(proposal_distribution)/matrix_UB
        # assert(np.sum(proposal_distribution)/matrix_UB < 1), (np.sum(proposal_distribution), matrix_UB, matrix)

    if verbose:
        print "smallest_partitioned_upper_bound =", smallest_partitioned_upper_bound, "matrix_UB =", matrix_UB
    assert(smallest_partitioned_upper_bound < matrix_UB or np.allclose(smallest_partitioned_upper_bound, matrix_UB, atol=0)), (smallest_partitioned_upper_bound, matrix_UB)

    return row_with_smallest_partitioned_UB


EXAMPLE_KITTI_LOG_PROBS = np.array([[-5.15127189e+01, -1.10841131e+02, -9.99206432e+02, -2.91639890e+02, -9.99206432e+02, -1.95518885e+02, -6.65377970e+00, -2.21951832e+01, -1.34019613e+01, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16,],
                                    [-5.22585082e+02, -1.01998073e+02, -5.02113298e+02, -6.65872496e+00, -1.75266741e+02, -2.34396841e+01, -2.95342619e+02, -2.11116076e+02, -1.00000000e+16, -1.34019613e+01, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16,],
                                    [-9.99206432e+02, -5.83120074e+02, -9.72371421e+01, -2.15083819e+02, -1.70181875e+01, -3.43052740e+02, -9.99206432e+02, -9.99206432e+02, -1.00000000e+16, -1.00000000e+16, -1.34019613e+01, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16,],
                                    [-6.85421419e+00, -2.92220579e+02, -9.99206432e+02, -5.06691717e+02, -9.99206432e+02, -3.96701224e+02, -5.51136754e+01, -1.21561611e+02, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.34019613e+01, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16,],
                                    [-3.00533374e+02, -6.74163542e+00, -9.99206432e+02, -1.00220189e+02, -5.22537275e+02, -3.98876843e+01, -1.12800305e+02, -5.00661214e+01, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.34019613e+01, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16,],
                                    [-1.25983114e+02, -4.43814171e+01, -9.99206432e+02, -2.04722564e+02, -6.82956098e+02, -1.17013745e+02, -2.30256843e+01, -6.72843104e+00, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.34019613e+01, -1.00000000e+16, -1.00000000e+16,],
                                    [-4.33138135e+02, -5.07776044e+01, -6.44637523e+02, -1.52624337e+01, -2.64649219e+02, -7.62289369e+00, -2.17590109e+02, -1.39784871e+02, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.34019613e+01, -1.00000000e+16,],
                                    [-3.57204943e+02, -1.38451762e+01, -9.99206432e+02, -5.61238128e+01, -4.12724683e+02, -1.67157409e+01, -1.52332890e+02, -8.02932540e+01, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.00000000e+16, -1.34019613e+01,],
                                    [-1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,],
                                    [-1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,],
                                    [-1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,],
                                    [-1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,],
                                    [-1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,],
                                    [-1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,],
                                    [-1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,],
                                    [-1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00, -1.67922622e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,],])

EXAMPLE_MOT_LOG_PROBS2 = np.array([[-24.355210671802567, -1000.1663992658117, -1000.1663992658117, -36.870827956111334, -1000.1663992658117, -475.19072722227423, -1000.1663992658117, -272.90900740761344, -1000.1663992658117, -672.444503981268, -1000.1663992658117, -34.694573956457894, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-1000.1663992658117, -20.338725810683293, -43.778886616310395, -1000.1663992658117, -71.08578932873469, -441.55378259005806, -168.25763766032964, -709.3213894219876, -29.107564425445258, -281.4779990098786, -38.268747920024836, -1e+16, -30.962702620127537, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-40.572599081912465, -1000.1663992658117, -1000.1663992658117, -21.953526857822897, -1000.1663992658117, -617.5860700359992, -1000.1663992658117, -367.77283961852976, -1000.1663992658117, -1000.1663992658117, -1000.1663992658117, -1e+16, -1e+16, -33.36671239520714, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-241.01975946009154, -710.6875607896874, -1000.1663992658117, -347.67740746967036, -466.7902798344983, -62.53831538826367, -1000.1663992658117, -20.312715270431962, -603.3514032372916, -149.24918949740962, -546.310663585588, -1e+16, -1e+16, -1e+16, -30.051319987599175, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-1002.8004660670537, -63.463800553418096, -130.81843972613137, -1002.8004660670537, -18.658311312382903, -235.45852526617642, -194.92006909410492, -450.38542991128077, -32.58422564738656, -119.74409453489001, -24.7100893117449, -1e+16, -1e+16, -1e+16, -1e+16, -28.82269200714537, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-232.21337347358906, -1002.8004660670537, -1002.8004660670537, -318.263515930725, -536.4300169393584, -84.70720429953157, -1002.8004660670537, -23.872312311855328, -691.3232678283437, -185.48486893676426, -629.2034804774426, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -26.007729094746836, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-1003.4844952458901, -13.228279252246505, -34.523674112630694, -1003.4844952458901, -33.816865697572375, -324.7791346776469, -86.17181166863355, -545.3208705571587, -13.88162363973695, -193.60464093639558, -18.701920236586353, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.315950937462159, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-1000.1663992658117, -47.130651681257575, -22.82272062625238, -1000.1663992658117, -142.32723260934125, -658.2451033355129, -108.64500939852357, -1000.1663992658117, -72.53420363346427, -449.79443908029975, -93.54078661884165, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -33.668849976174855, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-436.91317509262547, -440.06785911335584, -654.6844773497525, -596.9730989441566, -240.27797783695814, -20.292075216270753, -1000.1663992658117, -60.79941970231152, -348.7000105016711, -44.33881569941978, -304.0133991320408, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -30.2946882214933, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-108.45572333206171, -1003.4844952458901, -1003.4844952458901, -118.50472151752031, -1003.4844952458901, -239.77652231704602, -1003.4844952458901, -107.83281698882007, -1003.4844952458901, -387.0319782612454, -1003.4844952458901, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.315950937462159, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-387.6997707888343, -338.35051928567174, -535.8808541651676, -556.2434467976257, -199.5517672655454, -35.56364258639937, -1000.7021867475559, -84.96606133402246, -273.86006002805766, -48.012052622410884, -238.42263212414832, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -20.05715257947975, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-334.36809182506744, -387.12353406263827, -597.6814678080613, -491.3293902859333, -240.59146236918662, -37.39040014655388, -1000.7021867475559, -68.6213576674913, -319.99343654736657, -63.45572562747119, -282.1120127525583, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -19.93034381112003, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-551.7673167558246, -316.4349672527511, -497.6054481550612, -738.8712501306561, -145.6191779077575, -26.275421790540996, -619.5564422713114, -110.38404114393883, -236.35819213209334, -16.596418128938755, -199.19271542580316, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -21.221338629464995, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-139.5618659103101, -1000.7021867475559, -1000.7021867475559, -197.44088377491755, -733.530207713676, -162.27904344277937, -1000.7021867475559, -48.004417673867806, -1000.7021867475559, -304.07402261704465, -1000.7021867475559, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -20.466868334102255, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-515.2753719231084, -189.33348826770262, -340.63489363478146, -712.8618225075435, -85.99108722742696, -56.167134475485135, -486.465275450971, -167.13876921306908, -141.3661898559051, -21.09675681784439, -115.68065185615464, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.533573322770245, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-1000.7021867475559, -129.00856856663867, -242.11478208368374, -1000.7021867475559, -33.648177041947164, -123.01554749893069, -329.7177234907404, -291.9751652755724, -78.4060946415705, -46.222956913845394, -58.75359777591825, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -19.984306325574735, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-1000.7021867475559, -194.90670289845698, -123.95140765195156, -1000.7021867475559, -244.15912064522456, -1000.7021867475559, -16.887876858913845, -1000.7021867475559, -196.20873004012952, -624.8452979842921, -220.0401235168564, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -19.97152341744201, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-1002.4525108295774, -85.58424518518032, -182.8206934830407, -1002.4525108295774, -18.769182270694795, -152.14385029631734, -261.3972094286581, -337.16753985795054, -50.39861810152122, -60.89087957429499, -36.01468603635624, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.533573322770245, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-1002.4525108295774, -107.1717001263035, -54.23050484611797, -1002.4525108295774, -179.77814812994015, -1002.4525108295774, -10.553301688838882, -1002.4525108295774, -124.15811039605168, -527.2435536285909, -147.14130847014317, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.533573322770245, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-1000.7021867475559, -28.059143189132982, -70.40416179080567, -1000.7021867475559, -30.345950601859975, -338.79110807609123, -155.00463939159417, -590.4268306166928, -16.668497388042525, -194.33874315912294, -17.720068411162522, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -26.61943444754994, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-649.1708868563987, -211.54976858476257, -354.6517288441339, -1000.807601544663, -73.32385063988744, -52.30925822953563, -437.25554131079565, -172.254121071811, -144.29221265054838, -11.99853796624713, -115.56749518148487, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.733686138162138, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-1000.807601544663, -26.52619247932692, -77.94030085343647, -1000.807601544663, -15.65204927798669, -278.6064840979491, -158.0829505085938, -505.48676805515987, -10.508121527633397, -150.82925656770018, -8.557959910723719, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.733686138162138, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16],
[-1000.807601544663, -65.27426226057757, -62.89755533928566, -1000.807601544663, -68.4223630723506, -491.26071801806563, -37.95231381517581, -1000.807601544663, -51.62065030909821, -304.69024850064505, -59.24440257581686, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.733686138162138, -1e+16, -1e+16, -1e+16, -1e+16],
[-591.093734302068, -255.72035669390448, -413.33117637750206, -1000.807601544663, -100.59819393747841, -35.00365946284423, -501.8719179932106, -137.08879806227552, -181.81005449987794, -8.89468395897759, -149.26406601909778, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.733686138162138, -1e+16, -1e+16, -1e+16],
[-1000.807601544663, -166.14996726356256, -177.51291398578388, -1000.807601544663, -101.60535447285496, -476.52349437210563, -69.94831996815238, -1000.807601544663, -122.66256977458762, -297.1390219376432, -121.45329433494425, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.733686138162138, -1e+16, -1e+16],
[-561.7767978033512, -262.5651393556461, -424.3057021241917, -1000.807601544663, -107.73281284125517, -30.0112722894805, -523.2379361541159, -125.56132264082774, -189.07396213982884, -8.561434992631986, -156.00948176536917, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.733686138162138, -1e+1],
[-1000.807601544663, -37.951288754210076, -44.899055759874955, -1000.807601544663, -50.2812038119471, -446.6876354428979, -57.999470083503425, -736.3458873601479, -29.185244963851588, -271.72738263339113, -36.29793222476295, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.733686138162138  ],
[-1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[-1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0                                                     ],
[-1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[-1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[-1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[-1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[-1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[-1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[-1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[-1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[-1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, -1.3211838600817223, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],])

EXAMPLE_MOT_LOG_PROBS3 = np.array([[-999.3854000809093, -999.3854000809093, -7.118789605007705, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -298.9858815578605, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -207.07662365192135, -233.62071389109096, -345.84932602218163, -999.3854000809093, -93.79366345503512, -999.3854000809093, -999.3854000809093, -153.63506968938523, -552.2898146071772, -256.61370460736964, -999.3854000809093, -243.18096590874038, -161.84859362452545, -355.266066097446, -999.3854000809093, -999.3854000809093, -556.7022520756074, -361.38315683281104, -271.0657913989043, -376.1645101140728, -999.3854000809093, -146.2155039658525, -8.80533608271862, -999.3854000809093, -999.3854000809093, -48.44645556495659, -999.3854000809093, -265.1444586847153, -266.2387173457947, -162.7728895839737, -999.3854000809093, -684.3313309580265, -999.3854000809093, -999.3854000809093, -217.00350209548233, -266.01980359236717, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -301.1779102884419, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -7.132011805379882, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-7.623618013205445, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -396.87177778530037, -999.3854000809093, -999.3854000809093, -999.3854000809093, -463.71782009221494, -999.3854000809093, -999.3854000809093, -651.8103102381282, -999.3854000809093, -999.3854000809093, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -7.270836623000588, -272.4944875666481, -430.504536781814, -143.4732961492682, -999.3854000809093, -261.18032115570765, -446.03090709623916, -257.666855693391, -544.8089223753102, -723.1022547955752, -590.2551828203489, -355.9225001366313, -25.06135033046543, -999.3854000809093, -999.3854000809093, -77.52338172010309, -23.597796856105994, -222.9213131169498, -467.6587069416193, -582.819657604555, -405.3691193030719, -233.0364935544095, -196.1826081582684, -999.3854000809093, -187.98553939761044, -999.3854000809093, -241.11650885670326, -999.3854000809093, -320.9691512664751, -999.3854000809093, -744.8254720022906, -999.3854000809093, -999.3854000809093, -412.3160463391294, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -7.849354837600856, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -185.85438164769113, -999.3854000809093, -999.3854000809093, -177.0273550013524, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -599.1353820920733, -999.3854000809093, -999.3854000809093, -389.05334700238035, -632.7095450044569, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -242.0957604865952, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -76.92948094137515, -999.3854000809093, -999.3854000809093, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -166.14118008558847, -999.3854000809093, -999.3854000809093, -267.7201984854395, -468.538262981551, -193.17809891349305, -999.3854000809093, -999.3854000809093, -7.126774413709398, -999.3854000809093, -674.8311454377932, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -152.08632658725637, -999.3854000809093, -999.3854000809093, -68.29607900647484, -210.61470590289701, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -13.033547901903997, -999.3854000809093, -720.4781428394037, -999.3854000809093, -508.00890826919516, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -392.51927288230576, -999.3854000809093, -999.3854000809093, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -220.97121766998364, -999.3854000809093, -999.3854000809093, -202.20825392107193, -406.3886718820772, -199.86102083646264, -653.4117179629139, -999.3854000809093, -11.453621994876206, -999.3854000809093, -593.3153128087496, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -104.3678436691742, -999.3854000809093, -999.3854000809093, -39.85253451415274, -156.49260799544783, -730.2805199437122, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -7.490612900241544, -999.3854000809093, -614.7330689529631, -999.3854000809093, -436.56071610589237, -999.3854000809093, -682.2117738512658, -999.3854000809093, -999.3854000809093, -456.72938231820683, -999.3854000809093, -999.3854000809093, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -515.4348568867601, -143.5121266783533, -545.1274219021185, -999.3854000809093, -7.353714495980436, -999.3854000809093, -999.3854000809093, -94.98405721526599, -194.67059547461798, -140.27018706768004, -390.37126015076444, -162.88728424793848, -56.29388056196626, -260.25687018433496, -999.3854000809093, -999.3854000809093, -395.0450649031291, -188.14496208182118, -50.28397474494987, -134.47582339929653, -329.86558510291644, -428.4388097897684, -81.85852827765858, -647.6713118014542, -451.13262544755037, -189.56570110034937, -999.3854000809093, -458.66514453281076, -624.0357247999874, -410.1198935502267, -999.3854000809093, -328.81503390469396, -999.3854000809093, -999.3854000809093, -83.04176788876393, -595.2653681059107, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -583.8521430875761, -999.3854000809093, -999.3854000809093, -26.659506688542425, -234.45011544765123, -354.5733057022839, -265.7266365795905, -999.3854000809093, -145.97265850514322, -648.9513937477286, -357.79651885982383, -999.3854000809093, -999.3854000809093, -999.3854000809093, -540.3629883159373, -7.14973502792283, -999.3854000809093, -999.3854000809093, -29.851487347018413, -33.63823684498238, -361.7000541336751, -667.5488574720717, -999.3854000809093, -459.96122501915266, -357.7384845657264, -97.67435492760228, -999.3854000809093, -253.55009563074287, -999.3854000809093, -224.0083970418977, -999.3854000809093, -351.2948882177111, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -607.1888740110552, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -8.612032076305722, -999.3854000809093, -999.3854000809093, -999.3854000809093, -548.6855685239666, -439.5281762599825, -999.3854000809093, -409.12323142569534, -999.3854000809093, -542.181810068024, -999.3854000809093, -466.73534416027303, -514.7048555633663, -999.3854000809093, -999.3854000809093, -388.35802029342335, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -268.5836493656315, -255.36053716861235, -999.3854000809093, -999.3854000809093, -328.262989182339, -999.3854000809093, -646.356376325176, -8.582670271004154, -378.9357189854065, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -576.233515401346, -10.56901623734305, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -629.198829077861, -226.9021516077141, -12.707336871609181, -999.3854000809093, -442.13008687799675, -999.3854000809093, -500.5740289780708, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -734.217452324328, -221.36954247171352, -999.3854000809093, -999.3854000809093, -348.9303557025111, -359.60552598963574, -735.8083523679774, -999.3854000809093, -999.3854000809093, -97.79504949098278, -295.481359449488, -410.891723129187, -999.3854000809093, -100.20497831387324, -999.3854000809093, -7.1282814441885956, -668.1955716039087, -45.552096111046886, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -737.9481759307931, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -265.1732222209858, -7.140195667386519, -999.3854000809093, -537.1962188422057, -999.3854000809093, -467.9613515480643, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -239.19443591142533, -999.3854000809093, -999.3854000809093, -353.0528489380661, -396.08321670150167, -999.3854000809093, -999.3854000809093, -999.3854000809093, -140.06152230070185, -385.266118698848, -386.45420492266516, -999.3854000809093, -154.41899737550912, -999.3854000809093, -13.391412193031176, -999.3854000809093, -75.83558028765373, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -275.48138212050833, -999.3854000809093, -420.7252787797356, -216.65139182653758, -999.3854000809093, -673.0964502871338, -399.9453455064881, -7.212292159389522, -379.132894273087, -158.3805810027073, -457.9696580327109, -282.3504656500042, -359.1664043089043, -999.3854000809093, -999.3854000809093, -379.3782852604136, -196.0416958985932, -82.29893137361913, -179.9899100016724, -87.46902012084576, -999.3854000809093, -517.0955211623115, -610.3943115353469, -232.98657493784802, -678.9622825177415, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -237.96082180491223, -999.3854000809093, -999.3854000809093, -274.0110040956254, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -467.80242484778125, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -31.797575247547556, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -8.248789702063608, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -441.353647111948, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -142.63594605990204, -999.3854000809093, -387.1101016320459, -553.8527834619449, -999.3854000809093, -999.3854000809093, -999.3854000809093, -181.3201989591633, -999.3854000809093, -478.36612026400695, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -361.692566720229, -437.91173362292915, -999.3854000809093, -999.3854000809093, -90.59734104065686, -999.3854000809093, -999.3854000809093, -7.147189804956261, -382.922698763714, -21.407814169906477, -385.0673448651296, -19.934302282097814, -16.5842344947546, -632.011724400158, -999.3854000809093, -999.3854000809093, -999.3854000809093, -518.4082244977265, -129.30733793224306, -93.17716167744621, -384.83767515686225, -568.7891828535387, -124.94264783565309, -999.3854000809093, -395.46745487940524, -338.4897880956825, -999.3854000809093, -999.3854000809093, -462.43748064959925, -607.7791678986108, -999.3854000809093, -243.6605036594441, -999.3854000809093, -999.3854000809093, -22.20068506911096, -405.7702766289441, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -371.48999613921427, -579.1080866058376, -999.3854000809093, -999.3854000809093, -157.89169672177866, -999.3854000809093, -999.3854000809093, -16.21709326502383, -469.04235574860803, -15.743687810208554, -397.6703445522365, -8.21012796013228, -39.43214084686253, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -666.4602622975004, -186.1741766748404, -106.40696956194867, -419.12030922271146, -680.6718778688444, -188.7499814389925, -999.3854000809093, -390.2129359011798, -443.9809771172252, -999.3854000809093, -999.3854000809093, -473.3074920037877, -737.4647320992323, -999.3854000809093, -233.91501668421208, -999.3854000809093, -999.3854000809093, -34.92944450076637, -404.8964311630204, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -485.9665809964926, -546.0859892961254, -999.3854000809093, -999.3854000809093, -142.19140307017125, -999.3854000809093, -999.3854000809093, -21.606758871652662, -368.1467691419336, -7.122484693228624, -287.98453076786416, -11.695595597240137, -27.776172319953066, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -607.0242554171451, -132.60379060707473, -56.16883235028137, -306.9837296519285, -999.3854000809093, -221.4917150535351, -999.3854000809093, -283.67785086689446, -487.9187866387562, -999.3854000809093, -999.3854000809093, -602.7857417715509, -999.3854000809093, -999.3854000809093, -154.6952032653655, -999.3854000809093, -999.3854000809093, -16.49615646306154, -528.3736325412539, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-441.38970625339476, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -211.6967016363745, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -150.31561965869733, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -700.8280207598775, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -7.3149449570434975, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -157.727786120233, -999.3854000809093, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -94.37910872203719, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -182.90530776005286, -999.3854000809093, -999.3854000809093, -400.4409297921385, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -616.9182438515114, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -485.49908239728535, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -7.1445276699319225, -999.3854000809093, -999.3854000809093, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -380.6315521491783, -999.3854000809093, -999.3854000809093, -81.38299198155855, -364.9346339769549, -201.94671825357995, -396.02327714726067, -999.3854000809093, -68.43483696916516, -999.3854000809093, -362.4007029091192, -999.3854000809093, -999.3854000809093, -999.3854000809093, -707.71337063821, -33.024195677761185, -999.3854000809093, -999.3854000809093, -7.315802235794962, -47.664898440411754, -449.51797446663625, -999.3854000809093, -999.3854000809093, -690.7386673932532, -554.3524763249717, -42.57083380534934, -999.3854000809093, -435.0931164104616, -999.3854000809093, -367.2419907890331, -999.3854000809093, -550.4785393022789, -999.3854000809093, -999.3854000809093, -599.946246297397, -999.3854000809093, -999.3854000809093, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -634.8618613471772, -999.3854000809093, -999.3854000809093, -27.642744999199003, -396.26810592422135, -284.47729352771285, -193.03565860814746, -999.3854000809093, -209.4884138517023, -528.7165938681279, -189.89490271788532, -609.5555345660896, -648.9681180913075, -671.933241795669, -414.3701766369403, -33.381943271040264, -999.3854000809093, -999.3854000809093, -50.46065387756999, -7.131239685219141, -221.74944895900532, -474.350610328534, -503.7011628623658, -588.141418628151, -348.7657522842739, -159.54533102589093, -999.3854000809093, -315.73988125409124, -999.3854000809093, -368.1599231930725, -999.3854000809093, -482.88613159923574, -999.3854000809093, -714.0926739939467, -999.3854000809093, -999.3854000809093, -461.5462728865796, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -363.99553298041843, -310.1126526887304, -76.32409756535431, -999.3854000809093, -393.7444328397969, -744.8254720022906, -999.3854000809093, -609.9110092812806, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -613.0874287691062, -353.43896244627587, -999.3854000809093, -507.9168425739879, -540.017318090464, -480.9193829904406, -709.3645941133842, -999.3854000809093, -999.3854000809093, -17.467571316020635, -193.05750767067926, -663.2421572840753, -999.3854000809093, -51.004098157074125, -999.3854000809093, -46.03805982211841, -386.8886262599885, -7.130137757381371, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -713.3251486698074, -444.7084149573837, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -253.5126776146427, -368.5395863827223, -141.1557548678739, -999.3854000809093, -379.68481989414965, -629.5006620065004, -999.3854000809093, -536.0949038252384, -999.3854000809093, -721.6789925546094, -999.3854000809093, -688.4984396002469, -557.6995220091743, -438.56796300422934, -999.3854000809093, -425.00211806602135, -653.7844607110549, -555.3370033976106, -699.4262786680495, -999.3854000809093, -999.3854000809093, -7.710643425032301, -158.63873660758892, -999.3854000809093, -999.3854000809093, -50.03447343410268, -999.3854000809093, -96.87667185764393, -270.5842170950953, -18.448881785237656, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -653.1967900071677, -320.1013585259783, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -214.049722086682, -999.3854000809093, -734.0482881155127, -48.99217374696924, -999.3854000809093, -999.3854000809093, -128.69264515042298, -82.6148661056058, -133.39860268701426, -197.68792020251362, -174.95100680958993, -67.84507900552289, -339.16223921559043, -999.3854000809093, -999.3854000809093, -443.52325019633, -211.87396506665814, -7.470248922397891, -60.85058971824221, -150.90478674967463, -708.5420409288579, -217.08218347262985, -717.417379260366, -249.4031891224966, -382.88137949384475, -999.3854000809093, -707.965360831478, -999.3854000809093, -676.9264680811281, -999.3854000809093, -177.68866887720426, -999.3854000809093, -999.3854000809093, -72.66653323818856, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -474.5082888746513, -999.3854000809093, -999.3854000809093, -142.3569190668565, -999.3854000809093, -999.3854000809093, -94.3293291084005, -176.01289208541453, -56.526614204250684, -111.16330522914409, -90.00330821734003, -57.36003373158115, -659.7374602158571, -999.3854000809093, -999.3854000809093, -999.3854000809093, -475.1225490118734, -56.04142738753984, -7.118950705563775, -114.78326483640541, -999.3854000809093, -327.636472920024, -999.3854000809093, -121.9607329122526, -598.7477839904353, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -53.91577869713202, -999.3854000809093, -999.3854000809093, -35.45657225243508, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -460.48621820219097, -357.5303326378598, -999.3854000809093, -999.3854000809093, -57.73183068264247, -999.3854000809093, -999.3854000809093, -17.760554612377298, -272.36372743857993, -27.314800142851237, -300.64884057321467, -38.41408468819126, -7.121188907110584, -533.5093986350755, -999.3854000809093, -999.3854000809093, -713.7297181621068, -413.41961012280655, -70.41286832016769, -56.741958832735854, -288.6160022801857, -597.2686335149681, -132.98681601947192, -999.3854000809093, -321.6898400567835, -336.75153385472146, -999.3854000809093, -739.047819679068, -572.98195820391, -616.5621813685588, -999.3854000809093, -194.0941817860476, -999.3854000809093, -999.3854000809093, -11.669668530658052, -516.752453505097, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -303.1781777689908, -172.59984235167622, -161.3993493284981, -999.3854000809093, -164.60364773740554, -999.3854000809093, -716.184157020734, -323.5687674770887, -640.9716261913319, -463.95454377378934, -999.3854000809093, -454.52561545692737, -316.0043118836284, -247.05170506205508, -999.3854000809093, -739.4364002724741, -421.5598295795587, -303.1040638164593, -386.13541969330004, -573.4878556941314, -999.3854000809093, -66.87264014415008, -55.501771195527176, -597.0159696246843, -999.3854000809093, -7.394891802672017, -999.3854000809093, -108.63821863191662, -353.5988788745152, -58.144799782030994, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -391.1741862770825, -379.65636571939046, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -194.46290818016644, -999.3854000809093, -999.3854000809093, -434.1688105279037, -999.3854000809093, -7.2571562271993075, -999.3854000809093, -999.3854000809093, -182.72632883966747, -999.3854000809093, -427.14299631354606, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -345.705243897677, -999.3854000809093, -999.3854000809093, -203.43914485339803, -278.3392554054358, -733.3662216132675, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -206.5147686532112, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -182.46244866075335, -999.3854000809093, -999.3854000809093, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -593.580012095913, -999.3854000809093, -999.3854000809093, -346.302674783528, -999.3854000809093, -999.3854000809093, -390.115521121179, -87.30142759040767, -310.63915644254877, -19.577096663103394, -386.9294203866204, -290.9674106857369, -744.1323248217307, -999.3854000809093, -999.3854000809093, -999.3854000809093, -502.1140714667173, -136.6813961476704, -114.76048491756245, -7.125039652511705, -999.3854000809093, -695.374049567604, -999.3854000809093, -47.40055327928312, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -72.63850817604009, -999.3854000809093, -999.3854000809093, -246.6792536049554, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-704.222460105795, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -121.14800475331077, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -188.67455769645903, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -154.85004101715202, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -7.837696464051018, -999.3854000809093, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -346.22545563211514, -999.3854000809093, -999.3854000809093, -248.25947235559005, -237.79181471587574, -158.75786808513382, -38.37911191452917, -207.04769695620368, -198.76932248670994, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -718.6280240658384, -166.87540310268318, -55.350928212728284, -71.4996262957824, -999.3854000809093, -623.764854304771, -999.3854000809093, -27.157131633353696, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -7.128795720156712, -999.3854000809093, -999.3854000809093, -145.20418230060025, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -60.79774058783561, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -147.73096354059027, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -247.21056224053305, -999.3854000809093, -713.075538922537, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -44.53074850750043, -999.3854000809093, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -523.5602422043625, -423.6082902204012, -999.3854000809093, -999.3854000809093, -87.80222512786142, -999.3854000809093, -999.3854000809093, -23.349239027213272, -269.41521780911984, -16.6032865544823, -248.2899112919877, -30.331126399946466, -12.207053314477024, -612.10599438008, -999.3854000809093, -999.3854000809093, -999.3854000809093, -469.90396122305293, -74.34787370211569, -35.813468111824214, -248.19408725930447, -706.9661043286695, -186.23162514536432, -999.3854000809093, -260.08986901405996, -421.2575734122816, -999.3854000809093, -999.3854000809093, -644.478634607675, -729.6266665771942, -999.3854000809093, -144.02225572197582, -999.3854000809093, -999.3854000809093, -7.1352183825196605, -577.9586362529321, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -19.352594476526953, -999.3854000809093, -999.3854000809093, -999.3854000809093, -654.2190012541519, -355.74206192890375, -999.3854000809093, -506.09430224935784, -999.3854000809093, -652.1095690666226, -999.3854000809093, -568.0243960512231, -623.1250173596185, -999.3854000809093, -999.3854000809093, -320.595250473068, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -289.56054312105704, -323.5031721234496, -999.3854000809093, -999.3854000809093, -381.96583667972413, -999.3854000809093, -689.923247975954, -8.291398964515293, -408.69443335483186, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -689.9860158437585, -16.4530037319984, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -436.2940269385784, -999.3854000809093, -999.3854000809093, -368.71509330309766, -219.42348342164024, -262.5599821714549, -13.814707243577097, -326.48316344465843, -296.8668606336945, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -209.97801974159606, -105.54125306517152, -42.4174383489769, -999.3854000809093, -999.3854000809093, -999.3854000809093, -7.734069481020888, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -22.37478216356602, -999.3854000809093, -999.3854000809093, -234.05932849504626, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16, -1e+16,],
[-999.3854000809093, -999.3854000809093, -999.3854000809093, -11.16395672651408, -999.3854000809093, -999.3854000809093, -999.3854000809093, -575.3492732485022, -473.87752620827825, -999.3854000809093, -403.0324523928629, -999.3854000809093, -526.4160481279233, -999.3854000809093, -448.27063799279017, -515.5012249318232, -999.3854000809093, -999.3854000809093, -436.9101769437612, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -328.68259736714964, -283.442980665566, -999.3854000809093, -999.3854000809093, -381.675659068778, -999.3854000809093, -734.595382190072, -12.442560425727278, -448.48459173383, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -570.9693845886441, -7.1195000935384805, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965, -1e+16,],
[-441.4799062281059, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -7.144244543126312, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -999.3854000809093, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -1e+16, -14.32678299763965,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
[-1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, -1.139992111702299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],])

test_array = np.array([[4.34375494e-04,  1.28275053e-03, 3.22266818e-04, 1.46793625e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
 [6.38616167e-04,  3.96672526e-05, 6.86774019e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.46793625e-06],
 [1.34941249e-01,  1.34941249e-01, 1.34941249e-01, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00],
 [1.34941249e-01,  1.34941249e-01, 1.34941249e-01, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00],
 [1.34941249e-01,  1.34941249e-01, 1.34941249e-01, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00],
 [1.34941249e-01,  1.34941249e-01, 1.34941249e-01, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00],
 [1.34941249e-01,  1.34941249e-01, 1.34941249e-01, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00],])

test_array1 = np.array([[7.10141662e-04, 1.46793625e-06, 0.00000000e+00],
                        [9.15447086e-81, 0.00000000e+00, 1.46793625e-06],
                        [1.34941249e-01, 1.00000000e+00, 1.00000000e+00]])


test_array2 = np.array([[3.10097235e-060, 7.70299370e-044, 1.08677827e-003,
        3.81715238e-024, 0.00000000e+000, 0.00000000e+000,
        5.84856525e-004, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 1.46793625e-006, 0.00000000e+000,
        0.00000000e+000],
       [0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        3.10254263e-311, 1.06873819e-003, 1.21128481e-003,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 1.46793625e-006,
        0.00000000e+000],
       [1.20110187e-022, 3.15350171e-004, 1.09923697e-044,
        6.87951966e-004, 0.00000000e+000, 0.00000000e+000,
        6.48534466e-052, 0.00000000e+000, 1.46793625e-006,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000],
       [1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000],
       [1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000],
       [1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000],
       [1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000],
       [1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000],
       [1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000],
       [1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000],
       [1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000],
       [1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000],
       [1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000],
       [1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000],
       [1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000],
       [1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.34941249e-001, 1.34941249e-001,
        1.34941249e-001, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
        1.00000000e+000]])

error_array = np.array([[7.26840310e-016, 1.19102641e-003, 7.68354243e-103, 4.32954251e-103, 1.38240822e-003, 1.00901386e-019, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.46793625e-006, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                        [1.12117284e-003, 1.25976951e-015, 1.34030479e-151, 5.18948230e-152, 5.19901707e-014, 2.37626455e-004, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.46793625e-006, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                        [1.62804365e-301, 2.32410022e-187, 4.08543343e-004, 4.44555427e-004, 5.84875399e-194, 6.07700744e-322, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.46793625e-006, 0.00000000e+000],
                        [1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000],
                        [1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000],
                        [1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000],
                        [1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000],
                        [1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000],
                        [1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000],
                        [1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000],
                        [1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000],
                        [1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000],
                        [1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000],
                        [1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000],
                        [1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000],
                        [1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.34941249e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000]])

error_array1 = np.array([[2.36734057e-089, 3.07686161e-011, 0.00000000e+000, 0.00000000e+000, 7.24334078e-216, 4.94914073e-254, 2.23537993e-028, 2.68138672e-003, 2.26902149e-002, 2.93823303e-012, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                         [6.90278736e-007, 2.67565317e-055, 2.64128155e-155, 9.07741078e-097, 6.67092046e-040, 4.46041929e-073, 8.42267663e-139, 1.01405912e-001, 3.90398386e-001, 0.00000000e+000, 1.95129079e-011, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                         [3.82492022e-151, 0.00000000e+000, 4.87993429e-007, 2.79493335e-027, 2.04027618e-083, 3.40040380e-025, 0.00000000e+000, 8.32895456e-001, 3.98859718e-001, 0.00000000e+000, 0.00000000e+000, 1.54253505e-011, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                         [4.76098959e-090, 1.32911984e-264, 1.00321854e-023, 6.55373085e-007, 9.75199927e-029, 6.77407540e-006, 0.00000000e+000, 7.04584930e-001, 7.88708968e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.62210081e-011, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                         [1.96450603e-035, 4.87568965e-167, 7.59681691e-079, 1.52953128e-028, 6.16679742e-007, 4.22993516e-020, 0.00000000e+000, 3.24464576e-001, 8.33463068e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.62342074e-011, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                         [3.69452493e-082, 4.87573911e-253, 4.10819952e-025, 9.49358177e-005, 8.80606896e-023, 7.39011684e-004, 0.00000000e+000, 6.96335166e-001, 8.51373508e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 2.73478534e-008, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                         [1.47180609e-202, 0.00000000e+000, 1.76617352e-016, 2.08737753e-032, 1.25915364e-100, 1.00211687e-029, 0.00000000e+000, 9.17238139e-001, 3.66431574e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.68311492e-011, 0.00000000e+000, 0.00000000e+000,],
                         [7.92404422e-224, 3.96227321e-069, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 2.82083383e-006, 2.63886876e-003, 4.81288068e-004, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 2.73478534e-008, 0.00000000e+000,],
                         [1.92052699e-214, 0.00000000e+000, 4.90938158e-019, 1.26642688e-041, 3.28096399e-115, 1.19744825e-037, 0.00000000e+000, 9.12754448e-001, 2.99790513e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 3.21739452e-009,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 2.31639238e-001, 9.94313750e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 2.31639238e-001, 9.94313750e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 2.31639238e-001, 9.94313750e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 2.31639238e-001, 9.94313750e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 2.31639238e-001, 9.94313750e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 2.31639238e-001, 9.94313750e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 2.31639238e-001, 9.94313750e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 2.31639238e-001, 9.94313750e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 2.31639238e-001, 9.94313750e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],])

error_array2 = np.array([[2.36734057e-089, 3.07686161e-011, 8.30253172e-010, 0.00000000e+000, 7.24334078e-216, 2.23537993e-028, 2.68138672e-003, 2.93823303e-012, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                         [1.96450603e-035, 4.87568965e-167, 6.59843275e-034, 7.59681691e-079, 6.16679742e-007, 0.00000000e+000, 3.24464576e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.62342074e-011, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                         [6.90278736e-007, 2.67565317e-055, 7.75274434e-019, 2.64128155e-155, 6.67092046e-040, 8.42267663e-139, 1.01405912e-001, 0.00000000e+000, 1.95129079e-011, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                         [1.47180609e-202, 0.00000000e+000, 1.43817824e-067, 1.76617352e-016, 1.25915364e-100, 0.00000000e+000, 9.17238139e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.68311492e-011, 0.00000000e+000, 0.00000000e+000,],
                         [7.92404422e-224, 3.96227321e-069, 1.27380126e-007, 0.00000000e+000, 0.00000000e+000, 2.82083383e-006, 2.63886876e-003, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 2.73478534e-008, 0.00000000e+000,],
                         [1.92052699e-214, 0.00000000e+000, 3.17942999e-071, 4.90938158e-019, 3.28096399e-115, 0.00000000e+000, 9.12754448e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 3.21739452e-009,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 9.94313750e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 9.94313750e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 9.94313750e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 9.94313750e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 9.94313750e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 9.94313750e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 9.94313750e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 9.94313750e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 9.94313750e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                         [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 9.94313750e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],])

slow_array = np.array([[1.03255159e-056, 1.22128011e-144, 2.71459761e-239, 8.76592419e-115, 4.24704637e-051, 3.98006564e-045, 1.28276307e-008, 1.25542105e-095, 0.00000000e+000, 2.05437656e-088, 1.12293348e-004, 4.90818356e-001, 5.36329732e-001, 2.29802893e-012, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
 [2.52353869e-043, 1.87140548e-009, 1.36535621e-039, 8.21347291e-007, 1.30430431e-060, 3.11310211e-079, 4.04637753e-101, 1.53864913e-007, 0.00000000e+000, 2.43439506e-011, 1.98672370e-008, 2.35338384e-001, 4.32976560e-001, 0.00000000e+000, 1.90603563e-011, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
 [0.00000000e+000, 0.00000000e+000, 5.56673335e-283, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.15477409e-011, 4.47901508e-181, 2.84772199e-042, 4.91288066e-004, 4.91288066e-004, 0.00000000e+000, 0.00000000e+000, 6.14426788e-013, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
 [2.99699972e-145, 1.07073459e-027, 8.00122188e-007, 1.33934670e-030, 7.53951472e-178, 8.35558371e-204, 6.86612175e-229, 8.16053563e-051, 1.00939265e-286, 1.48387389e-004, 6.26340260e-020, 2.79148452e-002, 7.83106498e-002, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.80811013e-011, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
 [4.64800195e-050, 1.17897872e-007, 6.64223115e-033, 4.23037644e-006, 6.28823194e-069, 1.29840487e-088, 5.21081212e-113, 1.07641330e-008, 0.00000000e+000, 4.07213253e-009, 2.58568924e-009, 1.99498666e-001, 3.81774252e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 2.39486101e-010, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
 [6.29410896e-010, 1.13058041e-082, 3.69371978e-179, 4.39631798e-066, 4.88592388e-008, 1.31915160e-009, 1.14846867e-045, 1.72195838e-050, 0.00000000e+000, 4.72830155e-060, 9.33679447e-002, 8.21667981e-001, 8.71472001e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 3.26898827e-012, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
 [1.05730397e-021, 3.87214136e-129, 9.70119615e-247, 6.81419048e-106, 1.42711573e-013, 1.02341576e-007, 4.19374252e-047, 1.32659585e-086, 0.00000000e+000, 2.61046494e-085, 2.04295224e-001, 8.82761862e-001, 7.86163187e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 5.17269005e-012, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
 [4.71654373e-008, 2.54630446e-059, 2.51573431e-142, 3.70411071e-046, 5.38482807e-010, 1.09391981e-015, 2.63865378e-047, 4.26038908e-033, 0.00000000e+000, 1.14399828e-046, 2.11971362e-002, 7.26359891e-001, 8.61764095e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 4.29071286e-012, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
 [2.14984806e-010, 1.33793562e-097, 2.18363824e-202, 1.38454715e-078, 8.23139893e-006, 1.60027208e-004, 3.96797463e-041, 4.04680951e-061, 0.00000000e+000, 1.24386517e-068, 3.50883937e-001, 9.16803894e-001, 9.01251316e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 2.73478534e-008, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
 [9.60025627e-058, 6.42615647e-007, 4.22335881e-028, 8.93765850e-007, 2.93799308e-078, 5.50048303e-099, 1.14611029e-130, 1.30174595e-011, 0.00000000e+000, 2.26512496e-007, 2.55635859e-010, 1.65294072e-001, 3.26884725e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.68008007e-011, 0.00000000e+000, 0.00000000e+000,],
 [9.00373196e-113, 4.49655282e-017, 3.16267753e-010, 3.90353770e-020, 2.43960563e-141, 9.84501983e-166, 1.30731196e-200, 9.92963076e-036, 0.00000000e+000, 1.13223922e-003, 1.45696876e-016, 5.19464943e-002, 1.27761875e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 4.47531824e-011, 0.00000000e+000,],
 [3.30751627e-011, 3.67960436e-096, 1.71945945e-197, 6.72432845e-077, 9.99517331e-007, 2.01682864e-005, 1.05958881e-033, 1.83043966e-059, 0.00000000e+000, 2.01921751e-067, 2.77736108e-001, 9.00875149e-001, 8.96055124e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 6.70290525e-009,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 9.08137191e-001, 9.92684090e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 9.08137191e-001, 9.92684090e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 9.08137191e-001, 9.92684090e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 9.08137191e-001, 9.92684090e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 9.08137191e-001, 9.92684090e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 9.08137191e-001, 9.92684090e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 9.08137191e-001, 9.92684090e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 9.08137191e-001, 9.92684090e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 9.08137191e-001, 9.92684090e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 9.08137191e-001, 9.92684090e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 9.08137191e-001, 9.92684090e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 9.08137191e-001, 9.92684090e-001, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],])

slow_array1 = np.array([[2.07586701e-072, 1.40161629e-135, 5.56215914e-281, 6.73882352e-176, 1.43951665e-051, 2.40751623e-009, 2.42654990e-009, 1.26191170e-125, 0.00000000e+000, 5.97195948e-106, 2.50358259e-065, 9.27962780e-010, 1.32997688e-065, 8.94373741e-013, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
 [1.01179886e-044, 8.34170459e-007, 7.46166670e-042, 7.72511523e-010, 2.87793257e-068, 4.35468786e-015, 4.84906464e-056, 2.19029201e-006, 0.00000000e+000, 2.34178707e-012, 1.87521651e-055, 1.32590443e-115, 6.32138953e-108, 0.00000000e+000, 1.94756833e-011, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
 [3.60379480e-009, 3.94845470e-063, 3.65629647e-187, 1.64118482e-085, 2.87093654e-007, 3.25640561e-002, 1.11884131e-026, 2.68924609e-057, 0.00000000e+000, 1.12251104e-063, 1.78633706e-007, 1.72947800e-060, 1.18364273e-013, 0.00000000e+000, 0.00000000e+000, 1.05185352e-011, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
 [5.53390803e-052, 3.30369834e-007, 4.88803522e-035, 4.54017644e-008, 1.08376463e-076, 5.82102907e-017, 3.04838591e-062, 9.91598919e-007, 0.00000000e+000, 5.10936881e-010, 8.34928660e-064, 2.24542309e-129, 8.52096072e-120, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.82064234e-011, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
 [0.00000000e+000, 0.00000000e+000, 6.29424776e-305, 0.00000000e+000, 0.00000000e+000, 7.99258994e-074, 0.00000000e+000, 0.00000000e+000, 6.20905977e-012, 4.47901508e-181, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 6.97357870e-013, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
 [4.33891354e-007, 7.16588416e-043, 2.32642124e-149, 1.72927441e-061, 5.61712396e-012, 3.32724538e-003, 2.36266277e-028, 7.30656040e-039, 0.00000000e+000, 9.07860651e-050, 7.21592306e-008, 7.04275257e-062, 3.40082802e-023, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.48249046e-011, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
 [3.65546861e-152, 1.23062340e-042, 7.27128917e-007, 7.67789217e-029, 1.89273307e-180, 5.95055705e-039, 5.99082330e-118, 1.04585054e-039, 0.00000000e+000, 1.63975525e-004, 9.28885057e-173, 4.22018089e-250, 1.65158400e-261, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.74066316e-011, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
 [1.18682757e-024, 8.13433095e-109, 1.24921503e-262, 2.87872805e-138, 5.87583219e-009, 1.85980187e-002, 7.92778066e-028, 4.59125840e-099, 0.00000000e+000, 2.84919991e-092, 1.24734574e-018, 1.77588957e-064, 1.50671165e-009, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 5.79260307e-013, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
 [1.80137192e-060, 4.02415896e-009, 2.24808755e-029, 4.61557160e-007, 3.54921269e-086, 3.61190029e-019, 2.78963743e-071, 2.29700056e-008, 0.00000000e+000, 6.36688413e-008, 1.55665476e-073, 5.86434831e-150, 5.74721477e-133, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.29148701e-011, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
 [8.80020994e-014, 4.63961481e-083, 3.73445907e-222, 4.93000342e-109, 1.68871499e-005, 8.77098029e-002, 7.24049607e-025, 1.66248824e-075, 0.00000000e+000, 8.03303181e-077, 6.86250518e-010, 3.57853075e-059, 2.66437368e-008, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 2.20232440e-010, 0.00000000e+000, 0.00000000e+000,],
 [2.19114593e-121, 3.16460250e-030, 1.12786515e-009, 1.74601252e-018, 3.49357780e-149, 1.30838283e-032, 2.54493293e-107, 5.70031748e-028, 0.00000000e+000, 1.26853236e-003, 6.67999226e-140, 2.51614883e-225, 7.04052170e-219, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 5.31887239e-011, 0.00000000e+000,],
 [1.47337410e-011, 4.24749570e-074, 6.11421222e-206, 5.36398187e-100, 3.25636538e-005, 1.55267460e-001, 1.98669541e-019, 1.52605284e-067, 0.00000000e+000, 1.79291081e-071, 4.25387473e-008, 2.41864244e-044, 8.54826221e-009, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 6.70290525e-009,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 8.38371808e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 8.38371808e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 8.38371808e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 8.38371808e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 8.38371808e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 8.38371808e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 8.38371808e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 8.38371808e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 8.38371808e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 8.38371808e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 8.38371808e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 8.38371808e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
 [4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 2.31639238e-001, 8.38371808e-001, 4.51487234e-002, 1.22757249e-001, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],])

problem_array = np.array([[4.30408436e-015, 7.25073878e-048, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 4.47901508e-181, 0.00000000e+000, 0.00000000e+000, 4.47459242e-216, 5.16046422e-012, 0.00000000e+000, 2.28119079e-002, 6.44759685e-013, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                          [1.18249816e-045, 0.00000000e+000, 1.08033774e-009, 3.43604628e-097, 2.54594701e-078, 6.64679151e-048, 4.16802524e-041, 6.22576746e-041, 7.98650708e-200, 0.00000000e+000, 9.83958249e-042, 7.27159690e-002, 0.00000000e+000, 9.64825877e-013, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                          [1.29046143e-023, 0.00000000e+000, 1.00021138e-071, 2.28092844e-007, 2.43500253e-007, 2.94743297e-004, 4.06381611e-058, 1.41233945e-081, 3.16524668e-039, 2.15764198e-300, 3.55291329e-053, 6.92401868e-001, 0.00000000e+000, 0.00000000e+000, 1.64429006e-011, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                          [2.43528606e-014, 0.00000000e+000, 1.09237112e-178, 1.31007246e-031, 4.09072370e-045, 1.43743523e-011, 6.31892425e-168, 3.50912200e-208, 6.06368918e-007, 9.24262353e-182, 1.49961678e-156, 8.49868280e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.57995820e-011, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                          [2.58428515e-022, 0.00000000e+000, 1.67453498e-081, 1.95480851e-006, 9.31276035e-008, 1.94934469e-003, 7.39026889e-065, 5.42072053e-090, 3.75663236e-033, 1.88529223e-292, 2.49101005e-059, 7.48547113e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 2.78520200e-010, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                          [4.79301671e-047, 0.00000000e+000, 8.06636755e-035, 1.43843112e-067, 2.08828352e-050, 1.68021726e-032, 1.47696880e-007, 8.47247259e-010, 2.37894460e-172, 0.00000000e+000, 1.86335622e-007, 1.39491258e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 8.18916640e-012, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                          [1.39235683e-050, 0.00000000e+000, 8.57715656e-036, 3.92714576e-085, 2.89766830e-065, 2.11868502e-040, 5.29524196e-008, 7.22874443e-007, 3.93002911e-201, 0.00000000e+000, 2.01360529e-008, 9.58424372e-002, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.95200205e-010, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                          [2.22540624e-042, 0.00000000e+000, 6.34527594e-034, 8.62810501e-049, 6.86010711e-035, 4.18908604e-024, 4.70045872e-010, 2.48549148e-016, 1.38144530e-139, 0.00000000e+000, 2.63787248e-009, 2.10053893e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 3.68155709e-012, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                          [1.36047796e-055, 0.00000000e+000, 7.99524294e-042, 5.67955490e-107, 2.26165520e-084, 5.11047869e-050, 5.78754479e-012, 2.43091404e-007, 2.09286591e-234, 0.00000000e+000, 9.45811801e-013, 6.18385561e-002, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.14401676e-010, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                          [3.17219297e-022, 0.00000000e+000, 5.15224530e-096, 1.67277382e-007, 2.27817031e-010, 2.54093598e-003, 2.09827335e-074, 1.69032725e-101, 1.25135221e-028, 1.41949691e-289, 4.21386964e-068, 7.92007123e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.49576424e-011, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
                          [5.82525392e-018, 0.00000000e+000, 1.75908165e-152, 2.20358123e-020, 1.92584204e-030, 3.85712570e-007, 1.68388833e-131, 3.90631440e-167, 4.46050157e-011, 3.67994775e-218, 7.75608943e-122, 8.83088870e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 5.22418474e-011, 0.00000000e+000, 0.00000000e+000,],
                          [1.00704658e-054, 0.00000000e+000, 1.91629213e-038, 2.97124425e-106, 9.14005979e-084, 9.05829851e-050, 3.02574044e-011, 2.92281819e-006, 3.51200869e-232, 0.00000000e+000, 4.01976857e-012, 6.24363169e-002, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 6.70290525e-009, 0.00000000e+000,],
                          [9.08485762e-044, 0.00000000e+000, 2.45483882e-024, 4.18910612e-063, 2.78034977e-046, 6.58796921e-031, 1.01493836e-005, 9.84753757e-008, 2.47166331e-164, 0.00000000e+000, 6.95822318e-006, 1.48732042e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 6.70290525e-009,],
                          [4.51487234e-002, 2.31639238e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 7.42919967e-002, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                          [4.51487234e-002, 2.31639238e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 7.42919967e-002, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                          [4.51487234e-002, 2.31639238e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 7.42919967e-002, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                          [4.51487234e-002, 2.31639238e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 7.42919967e-002, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                          [4.51487234e-002, 2.31639238e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 7.42919967e-002, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                          [4.51487234e-002, 2.31639238e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 7.42919967e-002, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                          [4.51487234e-002, 2.31639238e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 7.42919967e-002, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                          [4.51487234e-002, 2.31639238e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 7.42919967e-002, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                          [4.51487234e-002, 2.31639238e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 7.42919967e-002, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                          [4.51487234e-002, 2.31639238e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 7.42919967e-002, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                          [4.51487234e-002, 2.31639238e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 7.42919967e-002, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],
                          [4.51487234e-002, 2.31639238e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 6.01224008e-001, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 4.51487234e-002, 7.42919967e-002, 9.92684090e-001, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000,],])

if __name__ == "__main__":
    # test_permanent_bound_tightness(N=-1, use_matrix=True, matrix=test_array2)
    # sleep(-1)

    # a = np.array([[1.51906459e-51, 1.00016664e-10, 1.19640860e-03, 0.00000000e+00,
    #     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    #     1.51217537e-06, 0.00000000e+00, 0.00000000e+00],
    #    [4.89124407e-04, 3.17553091e-95, 1.95975541e-61, 0.00000000e+00,
    #     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    #     0.00000000e+00, 1.51217537e-06, 0.00000000e+00],
    #    [1.22243883e-85, 1.28914032e-03, 2.29485057e-10, 1.51217537e-06,
    #     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    #     0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    #    [1.86518244e-01, 1.86518244e-01, 1.86518244e-01, 1.00000000e+00,
    #     1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
    #     1.00000000e+00, 1.00000000e+00, 1.00000000e+00],
    #    [1.86518244e-01, 1.86518244e-01, 1.86518244e-01, 1.00000000e+00,
    #     1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
    #     1.00000000e+00, 1.00000000e+00, 1.00000000e+00],
    #    [1.86518244e-01, 1.86518244e-01, 1.86518244e-01, 1.00000000e+00,
    #     1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
    #     1.00000000e+00, 1.00000000e+00, 1.00000000e+00],
    #    [1.86518244e-01, 1.86518244e-01, 1.86518244e-01, 1.00000000e+00,
    #     1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
    #     1.00000000e+00, 1.00000000e+00, 1.00000000e+00],
    #    [1.86518244e-01, 1.86518244e-01, 1.86518244e-01, 1.00000000e+00,
    #     1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
    #     1.00000000e+00, 1.00000000e+00, 1.00000000e+00],
    #    [1.86518244e-01, 1.86518244e-01, 1.86518244e-01, 1.00000000e+00,
    #     1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
    #     1.00000000e+00, 1.00000000e+00, 1.00000000e+00],
    #    [1.86518244e-01, 1.86518244e-01, 1.86518244e-01, 1.00000000e+00,
    #     1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
    #     1.00000000e+00, 1.00000000e+00, 1.00000000e+00],
    #    [1.86518244e-01, 1.86518244e-01, 1.86518244e-01, 1.00000000e+00,
    #     1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
    #     1.00000000e+00, 1.00000000e+00, 1.00000000e+00]])
    if False:
        #Test nesting of upper bound when we can only choose between 1 or 2 rows
        print "complete UB= ", minc_extended_UB2(error_array)
        nesting_UB_r0 = 0
        for c in range(error_array.shape[0]):
            cur_submatrix = np.delete(error_array, c, 1) #delete columns
            cur_submatrix = np.delete(cur_submatrix, 0, 0) #delete rows

            nesting_UB_r0 += error_array[0, c]*minc_extended_UB2(cur_submatrix)
            print "intermediate nesting_UB_r0= ", nesting_UB_r0

        print "nesting_UB_r0= ", nesting_UB_r0

        nesting_UB_r1 = 0
        for c in range(error_array.shape[0]):
            cur_submatrix = np.delete(error_array, c, 1) #delete columns
            cur_submatrix = np.delete(cur_submatrix, 1, 0) #delete rows

            nesting_UB_r1 += error_array[1, c]*minc_extended_UB2(cur_submatrix)
            print "intermediate nesting_UB_r1= ", nesting_UB_r1

        print "nesting_UB_r1= ", nesting_UB_r1    

        nesting_UB_r2 = 0
        for c in range(error_array.shape[0]):
            cur_submatrix = np.delete(error_array, c, 1) #delete columns
            cur_submatrix = np.delete(cur_submatrix, 2, 0) #delete rows

            nesting_UB_r2 += error_array[2, c]*minc_extended_UB2(cur_submatrix)
            print "intermediate nesting_UB_r2= ", nesting_UB_r2

        print "nesting_UB_r2= ", nesting_UB_r2      

        sleep(-10)

    # matrix=np.exp(EXAMPLE_KITTI_LOG_PROBS)
    # M=8
    # T=8

    # matrix=np.exp(EXAMPLE_MOT_LOG_PROBS2)
    # M=27
    # T=11
    
    # matrix=np.exp(EXAMPLE_MOT_LOG_PROBS3)
    # M=38
    # T=39

    # print minc_extended_UB2(slow_array1)
    # print rescaled_tracking_UB(slow_array1)
    # print conjectured_optimal_bound(slow_array1)
    # sleep(-2)
    matrix=problem_array

    # N = matrix.shape[0]
    # assert(N == matrix.shape[1])
    # M_remaining = 0
    # while matrix[M_remaining, N-1] != 1:
    #     M_remaining += 1
    # T_remaining=0
    # while matrix[M_remaining, T_remaining] != 1:
    #     T_remaining += 1

    # column_rescalings = matrix[M_remaining, :T_remaining]
    # permanent_rescaling = np.prod(column_rescalings)
    # matrix[:,:T_remaining] /= column_rescalings    # print matrix
    if False:
        M=13
        T=12
        matrix[:M,:] = matrix[:M,:]
        # matrix = 1000*matrix
        # print matrix
        # sleep(02)

        conditional_birth_probs=np.ones(M)
        conditional_death_probs=np.ones(T)
        prior_prob=1.0

        samples = multi_matrix_sample_associations_without_replacement(30, [associationMatrix(matrix, M, T, conditional_birth_probs, conditional_death_probs, prior_prob)])
    
    if True:
        # matrix_file_name = '/atlas/u/jkuck/rbpf_fireworks/inspect_matrices0.136455'
        # matrix_file_name = '/atlas/u/jkuck/rbpf_fireworks/inspect_matrices0.501049'
        # matrix_file_name = '/atlas/u/jkuck/rbpf_fireworks/inspect_matrices0.417022'
        # matrix_file_name = '/atlas/u/jkuck/rbpf_fireworks/matrices_for_debugging/inspect_matrices0.061037'
        # matrix_file_name = '/atlas/u/jkuck/rbpf_fireworks/matrices_for_debugging/inspect_matrices0.440232'
        # matrix_file_name = '/atlas/u/jkuck/rbpf_fireworks/matrices_for_debugging/inspect_matrices0.747122'
        # matrix_file_name = '/atlas/u/jkuck/rbpf_fireworks/matrices_for_debugging/inspect_matrices0.067089'
        matrix_file_name = '/atlas/u/jkuck/rbpf_fireworks/matrices_for_debugging/inspect_matrices0.455643'
 

        f = open(matrix_file_name, 'r')
        all_association_matrices = pickle.load(f)
        f.close()
        samples = multi_matrix_sample_associations_without_replacement(10, all_association_matrices)


    for sample in samples:
        print "sample_probability =", sample.complete_assoc_probability, "meas_grp_associations:", sample.meas_grp_associations, "dead_target_indices:", sample.dead_target_indices
