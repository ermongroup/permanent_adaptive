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


def test_permanent_bound_tightness(N):
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

    exact_permanent = 0

    minc_UB2 = minc_extended_UB2(matrix)
    bregman_extended_upper_bound = immediate_nesting_extended_bregman(matrix)

    print 'log(exact_permanent) =', np.log(exact_permanent)
    print 'log(bregman_extended_upper_bound) =', np.log(bregman_extended_upper_bound)
    print 'log extended minc2 UB =', np.log(minc_UB2)


if __name__ == "__main__":
    test_permanent_bound_tightness(N=80)
