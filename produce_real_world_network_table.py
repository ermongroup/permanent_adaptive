from __future__ import division
import numpy as np
import networkx as nx
import scipy.io
import time

from constant_num_targets_sample_permenant import conjectured_optimal_bound, sample_association_01matrix_plusSlack, permanent_Upper_Bounds

def read_network_file(filename):
    '''
    Inputs:
    - filename: (string) filename for a .mtx or .edges matrix

    Outputs:
    - edge_matrix: (np.array i think) array representation for permanent input (double check...)
    '''
    f = open(filename, 'rb')
    if filename.split('.')[-1] == 'mtx': # for .mtx
        edge_matrix = scipy.io.mmread(f).toarray()#[0:2, 0:5]
    else:#for .edges
        assert(filename.split('.')[-1] == 'edges'), (filename.split('.')[-1] )
        graph = nx.read_edgelist(f)
        sparse_matrix = nx.adjacency_matrix(graph)
        edge_matrix = np.asarray(sparse_matrix.todense())
    f.close()
    return edge_matrix

def create_sinkhorn_values_in_table(matrix_filenames):
    '''
    Code used to create Sinkhorn bounds in table 1 of NeurIPS 2019 submission
    '''

    for cur_matrix_filename in matrix_filenames:
        edge_matrix = read_network_file(cur_matrix_filename)
        permanent_LB, conjectured_permanent_UB, valid_sinkhorn_UB, runtime = conjectured_optimal_bound(edge_matrix, return_lower_bound=True)
        print('-'*80)
        print(cur_matrix_filename)
        # print(edge_matrix)
        print("sinkhorn approximation from gurvits:")
        print("ln(permanent_LB)", np.log(permanent_LB))
        print("ln(valid_sinkhorn_UB)", np.log(valid_sinkhorn_UB))
        print("ln(conjectured_permanent_UB)", np.log(conjectured_permanent_UB))
        print("runtime", runtime)


def estimate_acceptance_probability_with_nesting_upper_bounds(matrix, required_accepted_sample_count, compare_wai):
    '''
    Code used to get sampling bounds in table 1 of NeurIPS 2019 submission
    Inputs:
    - matrix: (np.array)
    - required_accepted_sample_count: (int) the number of samples to accept
    - compare_wai: (bool) if True, use the bound from Law that provable nests with a fixed partitioning
                   If False, use the bound from Soules


    Outputs:
    - p_hat: (float) estimate of the acceptance probability
    '''
    # be careful with setting COMPARE_WAI correctly 
    # (messy, created a global variable in constant_num_targets_sample_permenant to adjust
    # whether we use Soules upper bound on the permanent or Law's upper bound)

    MATRIX_PERMANENT_UBS = permanent_Upper_Bounds()
    BEST_ROW_CACHE = {}

    N = matrix.shape[0]
    #only using 1 matrix, this is for target tracking when we want to sample from the distribution
    #defined by the permanents of multiple matrices
    matrix_idx = 0 
    assert(N == matrix.shape[1])
    no_required_cells = ()

    global_row_indices = range(N)
    global_col_indices = range(N)
    total_matrix_upper_bound = MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, no_required_cells, sub_matrix=matrix, compare_wai=compare_wai)
    assert(total_matrix_upper_bound > 0)

    total_sample_count = 0
    accepted_sample_count = 0
    while(True):
        #sample_association_01matrix_plusSlack was named and then changed, handles general non-negative matrices 
        sampled_association, sub_tree_slack = sample_association_01matrix_plusSlack(matrix, matrix_idx, permanentUB=total_matrix_upper_bound, \
            prv_required_cells=[], depth=1, \
            global_row_indices=global_row_indices, global_col_indices=global_col_indices, with_replacement=True,\
            tighten_slack=False, matrix_permanent_ubs=MATRIX_PERMANENT_UBS, best_row_cache=BEST_ROW_CACHE,\
            compare_wai=compare_wai)

        total_sample_count += 1
        if sampled_association is not None: #we accepted a sample
            print"got a sample!"
            accepted_sample_count += 1
            if accepted_sample_count == required_accepted_sample_count:
                p_hat = accepted_sample_count/total_sample_count
                assert(total_matrix_upper_bound == MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, no_required_cells, compare_wai=compare_wai))
                permanent_estimate = p_hat*total_matrix_upper_bound
                return p_hat, permanent_estimate


    return sampled_a_info


def create_sampling_values_in_table(matrix_filenames, required_accepted_sample_count=10):
    '''
    Code used to get sampling bounds in table 1 of NeurIPS 2019 submission.  
    Using 10 samples, we get an upper bound on ln(permanent) of estimate + .6 and a lower
    bound of estimate - .6 that holds with probability > .95, as seen by running test_binomial_concentration
    with num_gumbel_samples = 10, additive_log_error = .6, and varying values of p in the range .5 to .000000005
    because (binomial fraction of log estiamtes off by more than 0.600000) is less than .95

    Note that for the fast performance on ENZYMES-g479
    SAMPLE_WITH_REPLACEMENT_ONLY_TIGHTEN_CUR_LEVEL_SLACK must be set to False in constant_num_targets_sample_permenant
    and sample_association_01matrix_plusSlack must be called with tighten_slack=True in 
    estimate_acceptance_probability_with_nesting_upper_bounds (in this file)

    Additionally, gumbel methods or a other methods must be used to compute the high probability bounds
    '''
    for compare_wai in [False, True]:
        for cur_matrix_filename in matrix_filenames:
            edge_matrix = read_network_file(cur_matrix_filename)
            t0 = time.time()
            p_hat, permanent_estimate = estimate_acceptance_probability_with_nesting_upper_bounds(edge_matrix, required_accepted_sample_count, compare_wai)
            t1 = time.time()
            print('-'*80)
            print(cur_matrix_filename)
            print("sampling bounds with compare_wai:", compare_wai)
            print("ln(permanent_estimate)", np.log(permanent_estimate))
            print("lower bound (estimate - .6 for 10 accepted samples) = ", np.log(permanent_estimate) - .6)
            print("upper bound (estimate + .6 for 10 accepted samples) = ", np.log(permanent_estimate) + .6)
            print("p_hat", p_hat, 'required_accepted_sample_count:', required_accepted_sample_count)
            print("runtime", t1-t0)




if __name__ == "__main__":



#WORKING EXAMPLES
    # matrix_filename = "./networkrepository_data/cage5.mtx"
    # matrix_filename = "./networkrepository_data/bcspwr01.mtx"

    #matrix_filename = "./networkrepository_data/edge_defined/ENZYMES_g192.edges" #change file eading for .edges!!
    # matrix_filename = "./networkrepository_data/edge_defined/ENZYMES_g230.edges" #change file eading for .edges!!
    # matrix_filename = "./networkrepository_data/edge_defined/ENZYMES_g479.edges" #change file eading for .edges!!
    # matrix_filename = "./networkrepository_data/edge_defined/ENZYMES_g490.edges" #change file eading for .edges!!


# WAI faster:
    # matrix_filename = "./networkrepository_data/smaller_networks/can_24.mtx"


    matrix_filenames = ["./networkrepository_data/edge_defined/ENZYMES_g192.edges",\
                    "./networkrepository_data/edge_defined/ENZYMES_g230.edges",\
                    "./networkrepository_data/edge_defined/ENZYMES_g479.edges",\
                    "./networkrepository_data/cage5.mtx",\
                    "./networkrepository_data/bcspwr01.mtx"]
    matrix_filenames = ["./networkrepository_data/edge_defined/ENZYMES_g479.edges",\
                    "./networkrepository_data/cage5.mtx",\
                    "./networkrepository_data/bcspwr01.mtx"]

    # create_sinkhorn_values_in_table(matrix_filenames)
    create_sampling_values_in_table(matrix_filenames, required_accepted_sample_count=10)
