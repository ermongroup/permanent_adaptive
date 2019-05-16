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
import copy
import os
from itertools import combinations
import operator as op
from functools import reduce
from gumbel_sample_permanent import optimized_minc_extened_UB2
# sys.path.insert(0, '/Users/jkuck/tracking_research/rbpf_fireworks/mht_helpers')
sys.path.insert(0, '../rbpf_fireworks/mht_helpers/')

from constant_num_targets_sample_permenant import conjectured_optimal_bound, sink_horn_scale_then_soules

# sys.path.insert(0, '/Users/jkuck/research/bp_permanent')
# import matlab.engine
# eng = matlab.engine.start_matlab()

def calc_permanent_rysers(matrix):
    '''
    Exactly calculate the permanent of the given matrix user Ryser's method (faster than calc_permanent)
    '''
    N = matrix.shape[0]
    assert(N == matrix.shape[1])
    #this looks complicated because the method takes and returns a complex matrix,
    #we are only dealing with real matrices so set complex component to 0
    return np.real(rysers_permanent(1j*np.zeros((N,N)) + matrix))

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

def minc_extended_UB2(matrix):
    #another bound
    #https://ac-els-cdn-com.stanford.idm.oclc.org/S002437950400299X/1-s2.0-S002437950400299X-main.pdf?_tid=fa4d00ee-39a5-4030-b7c1-28bb5fbc76c0&acdnat=1534454814_a7411b3006e0e092622de35cbf015275
    # equation (6), U^M(A)
    # return immediate_nesting_extended_bregman(matrix)

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



def nCr(n, r):
    #https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def test_decompose_minc_extended_UB2(matrix):
    #another bound
    #https://ac-els-cdn-com.stanford.idm.oclc.org/S002437950400299X/1-s2.0-S002437950400299X-main.pdf?_tid=fa4d00ee-39a5-4030-b7c1-28bb5fbc76c0&acdnat=1534454814_a7411b3006e0e092622de35cbf015275
    # equation (6), U^M(A)
    # return immediate_nesting_extended_bregman(matrix)

    assert(matrix.shape[0] == matrix.shape[1])
    N = matrix.shape[0]
    assert(N%2 == 0)

    half_N = int(N/2)

    UB_upper_matrix = 1.0
    for row in range(half_N):
        sorted_row = sorted(matrix[row], reverse=True)
        row_sum = 0
        for col in range(half_N):
            row_sum += sorted_row[col] * delta(col+1)
            # row_sum += sorted_row[col] * numba_delta(col+1)
        UB_upper_matrix *= row_sum

    UB_lower_matrix = 1.0
    for row in range(half_N, N):
        sorted_row = sorted(matrix[row], reverse=True)
        row_sum = 0
        for col in range(half_N):
            row_sum += sorted_row[col] * delta(col+1)
            # row_sum += sorted_row[col] * numba_delta(col+1)
        UB_lower_matrix *= row_sum   

    total_UB = UB_upper_matrix * UB_lower_matrix * nCr(N, half_N)

    return total_UB


def h_func(r):
    if r >= 1:
        return r + .5*math.log(r) + np.e - 1
    else:
        return 1 + (np.e - 1)*r

def immediate_nesting_extended_bregman(matrix):
    #https://dukespace.lib.duke.edu/dspace/bitstream/handle/10161/1054/D_Law_Wai_a_200904.pdf?sequence=1&isAllowed=y
    #bound the transpose due to our partitioning

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

def create_diagonal2(N, k, zero_one=False):
    '''
    create NxN matrix with blocks on the diagonal of size at most kxk
    '''    

    diag_matrix = np.zeros((N, N))
    # diag_matrix = np.random.rand(N, N)/100
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


def singular_value_bound(matrix):
    #https://arxiv.org/abs/1212.0025
    u, s, vh = np.linalg.svd(matrix, full_matrices=True)
    assert(max(s) == s[0])
    largest_singular_value = s[0]
    n = matrix.shape[0]
    assert(n == matrix.shape[1])
    permanent_upper_bound = largest_singular_value ** n
    return permanent_upper_bound

def test_permanent_bound_tightness(N):
    use_diag_matrix = False
    if use_diag_matrix:
        # matrix, exact_permanent = create_diagonal2(N, k=10, zero_one=False)
        matrix, exact_permanent = create_diagonal2(N, k=10, zero_one=False)

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

    exact_permanent = calc_permanent_rysers(matrix)

    minc_UB2 = minc_extended_UB2(matrix)
    bregman_extended_upper_bound = immediate_nesting_extended_bregman(matrix)
    # optimized_soules = 0
    optimized_soules = optimized_minc_extened_UB2(matrix)
    # print np.sum(matrix, axis=0)
    # print np.reciprocal(np.sum(matrix, axis=0))
    guess_at_hurt_col_scalings = np.sum(matrix, axis=0)
    guess_at_col_scalings = np.reciprocal(np.sum(matrix, axis=0))
    # print matrix
    guess_optimize_soules = minc_extended_UB2(matrix * guess_at_col_scalings)/np.prod(guess_at_col_scalings)
    guess_hurt_soules = minc_extended_UB2(matrix * guess_at_hurt_col_scalings)/np.prod(guess_at_hurt_col_scalings)

    lower_bound, conjectured_optimal_UB = conjectured_optimal_bound(matrix, return_lower_bound=True)

    sinkhorn_soules_bound = sink_horn_scale_then_soules(matrix)

    get_BP_lower_bound = False
    if get_BP_lower_bound:
        matlab_matrix = eng.magic(N)
        for row in range(N):
            for col in range(N):
                matlab_matrix[row][col] = matrix[row][col]
        bp_lower_bound = eng.estperslow(matlab_matrix)
    else:
        bp_lower_bound = 1

    print 'log(exact_permanent) =', np.log(exact_permanent)
    print 'log(bregman_extended_upper_bound) =', np.log(bregman_extended_upper_bound)
    print 'log extended minc2 UB =', np.log(minc_UB2)
    print 'log optimized_soules =', np.log(optimized_soules)
    print 'log guess_optimize_soules =', np.log(guess_optimize_soules)
    print 'log guess_hurt_soules =', np.log(guess_hurt_soules)
    print 'log bp_lower_bound =', np.log(bp_lower_bound)
    print 'log sinkhorn_soules_bound =', np.log(sinkhorn_soules_bound)
    return bp_lower_bound, lower_bound, conjectured_optimal_UB, guess_optimize_soules, optimized_soules, minc_UB2, bregman_extended_upper_bound, exact_permanent, sinkhorn_soules_bound

def get_bp_lower_bound(matrix):
    assert(matrix.shape[0] == matrix.shape[1])
    N = matrix.shape[0]
    matlab_matrix = eng.magic(N)
    for row in range(N):
        for col in range(N):
            matlab_matrix[row][col] = matrix[row][col]
    bp_lower_bound = eng.estperslow(matlab_matrix)   
    return bp_lower_bound 

def plot_permanent_bound_tightness_VS_n(max_n):
    law_ratios = []
    soules_ratios = []
    optimized_soules_ratios = []
    guess_optimized_soules_ratios = []
    conjectured_optimal_bound_ratios = []
    lower_bound_ratios = []
    bp_lower_bound_ratios = []
    sinkhorn_soules_ratios = []
    n_vals = range(3, max_n)

    n_vals.extend(n_vals)
    n_vals.extend(n_vals)
    print n_vals

    law_over_soules = []
    for n in n_vals:
        print "n=", n
        bp_lower_bound, lower_bound, conjectured_optimal_UB, guess_optimize_soules, optimized_soules, soules_UB, law_UB, exact_permanent, sinkhorn_soules = test_permanent_bound_tightness(n)

        cur_law_ratio = law_UB/exact_permanent
        law_ratios.append(cur_law_ratio)

        cur_soules_ratio = soules_UB/exact_permanent
        soules_ratios.append(cur_soules_ratio)

        law_over_soules.append(law_UB/soules_UB)

        optimized_soules_ratios.append(optimized_soules/exact_permanent)
        guess_optimized_soules_ratios.append(guess_optimize_soules/exact_permanent)


        conjectured_optimal_bound_ratios.append(conjectured_optimal_UB/exact_permanent)
        lower_bound_ratios.append(lower_bound/exact_permanent)

        bp_lower_bound_ratios.append(bp_lower_bound/exact_permanent)
        sinkhorn_soules_ratios.append(sinkhorn_soules/exact_permanent)

        fig = plt.figure()
        ax = plt.subplot(111)
        matplotlib.rcParams.update({'font.size': 15})

        # ax.semilogx(n_vals[:len(law_over_soules)], law_over_soules, 'x', label='law over soules')
        # ax.loglog(n_vals[:len(law_over_soules)], law_ratios, 'x', label='Law ratios')
        ax.semilogy(n_vals[:len(law_over_soules)], soules_ratios, 'x', label='Soules ratios')
        ax.semilogy(n_vals[:len(law_over_soules)], guess_optimized_soules_ratios, 'x', label='guess optimize Soules ratios')
        ax.semilogy(n_vals[:len(law_over_soules)], optimized_soules_ratios, 'x', label='optimized soules ratios')
        ax.semilogy(n_vals[:len(law_over_soules)], conjectured_optimal_bound_ratios, 'x', label='conjectured optimal ratios')
        ax.semilogy(n_vals[:len(law_over_soules)], sinkhorn_soules_ratios, 'x', label='sinkhorn soules ratios')
        # ax.semilogy(n_vals[:len(law_over_soules)], bp_lower_bound_ratios, 'x', label='bp lower bound ratios')
        # ax.semilogy(n_vals[:len(law_over_soules)], lower_bound_ratios, 'x', label='lower bound ratios')
        # ax.plot(xp, np.log(p6), '-', label=r'$e^{-9.5} n^2 + e^{-20} n^5$')

        plt.title('Bound tightness comparison')
        plt.xlabel('n (matrix dimension)')
        plt.ylabel('upper_bound/permanent')
        lgd = ax.legend(loc='upper left', #prop={'size': 9},# bbox_to_anchor=(0.5, -.11),
                  fancybox=False, shadow=False, ncol=1, numpoints = 1)
        plt.setp(lgd.get_title(),fontsize='xx-small')

        # plt.show()

        if not os.path.exists('./scaling_plots'):
            os.makedirs('./scaling_plots')

        # fig.savefig('loglog_bound_tightness_comparison_sinkhornSoules_uniformMatrix', bbox_extra_artists=(lgd,), bbox_inches='tight')    
        fig.savefig('./scaling_plots/loglog_bound_tightness_comparison_sinkhornSoules_blockDiagk=10', bbox_extra_artists=(lgd,), bbox_inches='tight')    
        plt.close()


def test_permanent_bound_tightness1(N):
    use_diag_matrix = True
    if use_diag_matrix:
        matrix, exact_permanent = create_diagonal2(N, k=10, zero_one=False)

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

    # exact_permanent = calc_permanent_rysers(matrix)

    minc_UB2 = minc_extended_UB2(matrix)
    bregman_extended_upper_bound = immediate_nesting_extended_bregman(matrix)



    singular_value_upper_bound = singular_value_bound(matrix)
    decomposed_minc_UB2 = test_decompose_minc_extended_UB2(matrix)

    print 'log(exact_permanent) =', np.log(exact_permanent)
    print 'log(bregman_extended_upper_bound) =', np.log(bregman_extended_upper_bound)
    print 'log extended minc2 UB =', np.log(minc_UB2)
    print 'log(singular_value_upper_bound)', np.log(singular_value_upper_bound)
    print 'log decomposed_minc_UB2 =', np.log(decomposed_minc_UB2)


    levels = []
    upper_bounds = []
    for level in range(2, N-1):
        print 'level:', level
        levels.append(level)
        if level < N-3:
            upper_bounds.append(minc_UB2)
            continue

        cur_UB = 0
        for columns in combinations(range(N), level):  # 2 for pairs, 3 for triplets, etc
            upper_matrix = np.delete(matrix, columns, 1) #delete columnumns
            upper_matrix = np.delete(upper_matrix, range(N-level,N), 0) #delete rows

            cols_to_detete = [i for i in range(N) if i not in columns]
            lower_matrix = np.delete(matrix, cols_to_detete, 1) #delete columnumns
            lower_matrix = np.delete(lower_matrix, range(0,N-level), 0) #delete rows
            cur_UB += calc_permanent_rysers(upper_matrix) * minc_extended_UB2(lower_matrix)
            # cur_UB += calc_permanent_rysers(upper_matrix) * immediate_nesting_extended_bregman(lower_matrix)
        upper_bounds.append(cur_UB)

    fig = plt.figure()
    ax = plt.subplot(111)
    matplotlib.rcParams.update({'font.size': 15})

    # ax.semilogx(n_vals[:len(law_over_soules)], law_over_soules, 'x', label='law over soules')
    ax.plot(levels, upper_bounds, 'x', label='Law ratios')
    # ax.semilogy(n_vals[:len(law_over_soules)], soules_ratios, 'x', label='Soules ratios')
    # ax.plot(xp, np.log(p6), '-', label=r'$e^{-9.5} n^2 + e^{-20} n^5$')

    # ax.axhline(y=bregman_extended_upper_bound, label="law", c='r')
    ax.axhline(y=minc_UB2, label="soules", c='g')
    # ax.axhline(y=decomposed_minc_UB2, label="soules decomposed", c='g')
    ax.axhline(y=exact_permanent, label="exact_permanent", c='b')

    plt.title('Bound tightness comparison')
    plt.xlabel('level')
    plt.ylabel('permenant/UB')
    lgd = ax.legend(loc='upper left', #prop={'size': 9},# bbox_to_anchor=(0.5, -.11),
              fancybox=False, shadow=False, ncol=1, numpoints = 1)
    plt.setp(lgd.get_title(),fontsize='xx-small')

    # plt.show()

    fig.savefig('with_possible_lower_bound_UB_tightness_comparison', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    plt.close()

    return minc_UB2, bregman_extended_upper_bound, exact_permanent

if __name__ == "__main__":
    # N = 17
    # matrix = np.random.rand(N,N)
    # m = eng.magic(N)
    # for row in range(N):
    #     for col in range(N):
    #         m[row][col] = matrix[row][col]
    # t1 = time.time()
    # lb = eng.estperslow(m)
    # t2 = time.time()

    # print "bethe permanent (lb) = ", lb, "runtime =", t2-t1
    # t3 = time.time()
    # lower_bound, conjectured_optimal_UB = conjectured_optimal_bound(matrix, return_lower_bound=True)
    # t4 = time.time()

    # print "sinkhorn scaling, possibly bethe permanent (lb) = ", lower_bound, "runtime =", t4-t3

    # sleep(2341)

    # test_permanent_bound_tightness1(30)

    plot_permanent_bound_tightness_VS_n(max_n = 60)
    # test_permanent_bound_tightness(N=50)

