from __future__ import division
import numpy as np
import networkx as nx
import scipy.io
import time
import math

from constant_num_targets_sample_permenant import conjectured_optimal_bound, sample_association_01matrix_plusSlack_oldHopefullyFast, permanent_Upper_Bounds
from compare_bounds import create_diagonal2

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
    edge_matrix = edge_matrix.astype(float)
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


def truncated_gumbel(n, truncation):
    '''
    https://cmaddis.github.io/
    sample the max of n gumbels with location 0 and scale 1, truncated at truncation
    '''
    assert(n>0), n
    gumbel = np.random.gumbel() + math.log(n)
    return -np.log(np.exp(-gumbel) + np.exp(-truncation))

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
    - permanent_estimate: (float) estimate of the permanent when we don't tighten upper bounds
    - permanent_estimate_with_tightening: (float) estimate of the permanent when we tighten upper bounds
    '''
    # be careful with setting COMPARE_WAI correctly 
    # (messy, created a global variable in constant_num_targets_sample_permenant to adjust
    # whether we use Soules upper bound on the permanent or Law's upper bound)
    global MATRIX_PERMANENT_UBS
    global BEST_ROW_CACHE
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
    samples_since_last_accepted_sample = 0
    permanent_estimate_with_tightening = 0
    upper_bounds_since_last_accepted_sample = []
    all_upper_bounds = []
    gumbel_samples_of_lnZ = []
    cur_truncation = np.inf
    inverse_upper_bounds = []
    if compare_wai:
        tighten_slack=False #we don't tighten slack for comparison with Law's algorithm (fixed partitioning, no slack tightening, weaker UB)
    else:
        tighten_slack=True #using our method we tighten slack
    while(True):
        cur_total_matrix_upper_bound = MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, no_required_cells, compare_wai=compare_wai)
        cur_gumbel_permanent_estimate = truncated_gumbel(n=cur_total_matrix_upper_bound, truncation=cur_truncation)
        # print("cur_total_matrix_upper_bound", cur_total_matrix_upper_bound)
        upper_bounds_since_last_accepted_sample.append(cur_total_matrix_upper_bound)
        all_upper_bounds.append(cur_total_matrix_upper_bound)
        inverse_upper_bounds.append(1/cur_total_matrix_upper_bound)
        #sample_association_01matrix_plusSlack_oldHopefullyFast was named and then changed, handles general non-negative matrices 
        # print'before calling sample_association_01matrix_plusSlack_oldHopefullyFast'
        # MATRIX_PERMANENT_UBS.check_nesting2(matrix, matrix_idx, [], global_row_indices, global_col_indices)    

        #VERY IMPORTANT to reset global_row_indices and global_col_indices to range(N) very time this is called on the complete matrix
        sampled_association, sub_tree_slack = sample_association_01matrix_plusSlack_oldHopefullyFast(matrix, matrix_idx, permanentUB=cur_total_matrix_upper_bound, \
            prv_required_cells=[], depth=1, \
            global_row_indices=range(N), global_col_indices=range(N), with_replacement=True,\
            tighten_slack=tighten_slack, matrix_permanent_ubs=MATRIX_PERMANENT_UBS, best_row_cache=BEST_ROW_CACHE,\
            compare_wai=compare_wai)

        # print "MATRIX_PERMANENT_UBS.upper_bounds_dictionary"
        # print MATRIX_PERMANENT_UBS.upper_bounds_dictionary

        # print 'after returning to estimate_acceptance_probability_with_nesting_upper_bounds'
        # MATRIX_PERMANENT_UBS.check_nesting2(matrix, matrix_idx, [], global_row_indices, global_col_indices)    


        total_sample_count += 1
        samples_since_last_accepted_sample += 1
        if sampled_association is not None: #we accepted a sample
            print"got a sample!"
            accepted_sample_count += 1
            # permanent_estimate_with_tightening += cur_total_matrix_upper_bound/samples_since_last_accepted_sample
            # permanent_estimate_with_tightening += cur_total_matrix_upper_bound**2/np.sum(upper_bounds_since_last_accepted_sample)
            permanent_estimate_with_tightening += cur_total_matrix_upper_bound**2
            samples_since_last_accepted_sample = 0
            upper_bounds_since_last_accepted_sample = []
            gumbel_samples_of_lnZ.append(cur_gumbel_permanent_estimate)
            cur_truncation = np.inf
            if accepted_sample_count == required_accepted_sample_count:
                print("first upper bound", all_upper_bounds[0])
                print("last upper bound", all_upper_bounds[-1])
                print("(first upper bound)/(last upper bound)", all_upper_bounds[0]/all_upper_bounds[-1])
                print("permanent_estimate_with_tightening", permanent_estimate_with_tightening)
               
                # p_hat = accepted_sample_count/total_sample_count
                # assert(total_matrix_upper_bound == MATRIX_PERMANENT_UBS.get_upper_bound(matrix_idx, no_required_cells, compare_wai=compare_wai))
                # permanent_estimate = p_hat*total_matrix_upper_bound
                # permanent_estimate_with_tightening /= np.sum(all_upper_bounds)
                gumbel_mean_permanent_estimate = np.mean(gumbel_samples_of_lnZ) - np.euler_gamma
                gumbel_mle_estimate = -np.log(np.mean(np.exp(-np.array(gumbel_samples_of_lnZ))))
                permanent_estimate = accepted_sample_count/np.sum(inverse_upper_bounds)

                t0 = time.time()
                bootstrap_LB, bootstrap_UB = bootstrap_adaptive_rejecting_sampling_confidence_interval(Z_hat=permanent_estimate, Z_UBs=np.array(all_upper_bounds), N=100000, alpha=.05)
                t1 = time.time()

                #checking that bootstrap is repeatable.  may have an off by one error because binomial distribution has to return an integer, but this 
                #should be a small difference in log domain,
                for i in range(10):
                    t0 = time.time()
                    bootstrap_LB, bootstrap_UB = bootstrap_adaptive_rejecting_sampling_confidence_interval(Z_hat=permanent_estimate, Z_UBs=np.array(all_upper_bounds), N=100000, alpha=.05)
                    t1 = time.time()
                    print 'lb', bootstrap_LB, 'ub', bootstrap_UB, 'ln(lb)', np.log(bootstrap_LB), 'lb(UB)', np.log(bootstrap_UB), "time", t1-t0

                # print "bootstrap, 100", bootstrap_adaptive_rejecting_sampling_confidence_interval(Z_hat=permanent_estimate, Z_UBs=np.array(all_upper_bounds), N=100, alpha=.05)
                # print "bootstrap, 1000", bootstrap_adaptive_rejecting_sampling_confidence_interval(Z_hat=permanent_estimate, Z_UBs=np.array(all_upper_bounds), N=1000, alpha=.05)
                # print "bootstrap, 10000", bootstrap_adaptive_rejecting_sampling_confidence_interval(Z_hat=permanent_estimate, Z_UBs=np.array(all_upper_bounds), N=10000, alpha=.05)
                # print "bootstrap, 100000", bootstrap_adaptive_rejecting_sampling_confidence_interval(Z_hat=permanent_estimate, Z_UBs=np.array(all_upper_bounds), N=100000, alpha=.05)
                bootstrap_runtime = t1-t0
                # print"all_upper_bounds"
                # print all_upper_bounds
                # sleep(-1)
                # return p_hat, permanent_estimate, permanent_estimate_with_tightening, gumbel_mean_permanent_estimate, gumbel_mle_estimate, stefano_corrected_permanent_est_with_tightening
                return permanent_estimate, bootstrap_LB, bootstrap_UB, bootstrap_runtime, gumbel_mean_permanent_estimate, gumbel_mle_estimate
        else:
            cur_truncation = cur_gumbel_permanent_estimate

def bootstrap_adaptive_rejecting_sampling_confidence_interval(Z_hat, Z_UBs, N, alpha=.05):
    '''
    Inputs:
    - Z_hat: (float) estimate of the partition function, used for bootstrapping
    - Z_UBs: (np.array of length T) The upper bounds on the partition function
        used to get T samples (including accepted and rejected samples)
    - N:(int) # of simulations to perform for bootstrap estimate
    - alpha: (float) confidence interval holds with probability (1 - alpha)

    Outputs:
    - lower_bound: (float) lower bound on Z, holding with probability (1 - alpha)
    - upper_bound: (float) upper bound on Z, holding with probability (1 - alpha)

    *note if we were to repeat the experiment, the true value of Z would fall within
    our bounds with probability (1-alpha)
    '''
    inverse_UBs = 1/Z_UBs
    bootstrap_probs = Z_hat*inverse_UBs
    assert(len(Z_UBs.shape) == 1)
    T = Z_UBs.shape[0]
    bootstrapped_accepted_sample_counts = np.sum(np.random.binomial(n=1, p=bootstrap_probs, size=(N,T)), axis = 1)
    bootstrapped_Z_estimates = bootstrapped_accepted_sample_counts/np.sum(inverse_UBs)

    percentiles = np.percentile(bootstrapped_Z_estimates, [100*alpha/2, 100 - 100*alpha/2])
    lower_bound = percentiles[0]
    upper_bound = percentiles[1]
    return lower_bound, upper_bound

def create_sampling_values_in_table(matrix_filenames, required_accepted_sample_count=10):
    '''
    Code used to get sampling bounds in table 1 of NeurIPS 2019 submission.  
    Using 10 samples, we get an upper bound on ln(permanent) of estimate + .6 and a lower
    bound of estimate - .6 that holds with probability > .95, as seen by running test_binomial_concentration
    with num_gumbel_samples = 10, additive_log_error = .6, and varying values of p in the range .5 to .000000005
    because (binomial fraction of log estiamtes off by more than 0.600000) is less than .95

    Note that for the fast performance on ENZYMES-g479
    SAMPLE_WITH_REPLACEMENT_ONLY_TIGHTEN_CUR_LEVEL_SLACK must be set to False in constant_num_targets_sample_permenant
    and sample_association_01matrix_plusSlack_oldHopefullyFast must be called with tighten_slack=True in 
    estimate_acceptance_probability_with_nesting_upper_bounds (in this file)

    Additionally, gumbel methods or another method must be used to compute the high probability bounds
    '''
    for compare_wai in [False, True]:
        for cur_matrix_filename in matrix_filenames:
            edge_matrix = read_network_file(cur_matrix_filename)
            t0 = time.time()
            # p_hat, permanent_estimate, permanent_estimate_with_tightening, gumbel_mean_permanent_estimate, gumbel_mle_estimate, stefano_corrected_permanent_est_with_tightening = estimate_acceptance_probability_with_nesting_upper_bounds(edge_matrix, required_accepted_sample_count, compare_wai)
            permanent_estimate, bootstrap_LB, bootstrap_UB, bootstrap_runtime, gumbel_mean_permanent_estimate, gumbel_mle_estimate = estimate_acceptance_probability_with_nesting_upper_bounds(edge_matrix, required_accepted_sample_count, compare_wai)
            t1 = time.time()
            print('-'*80)
            print(cur_matrix_filename)
            print("sampling bounds with compare_wai:", compare_wai)
            # print("ln(stefano_corrected_permanent_est_with_tightening)", np.log(stefano_corrected_permanent_est_with_tightening))
            # print("ln(permanent_estimate, no tightening)", np.log(permanent_estimate))
            # print("lower bound no tightening(estimate - .6 for 10 accepted samples) = ", np.log(permanent_estimate) - .6)
            # print("upper bound no tightening(estimate + .6 for 10 accepted samples) = ", np.log(permanent_estimate) + .6)
            # print("ln(permanent_estimate, with tightening)", np.log(permanent_estimate_with_tightening))
            # print("lower bound with tightening(estimate - .6 for 10 accepted samples) = ", np.log(permanent_estimate_with_tightening) - .6)
            # print("upper bound with tightening(estimate + .6 for 10 accepted samples) = ", np.log(permanent_estimate_with_tightening) + .6)
            print("gumbel ln(permanent_estimate)", gumbel_mean_permanent_estimate)
            
            # print("p_hat", p_hat, 'required_accepted_sample_count:', required_accepted_sample_count)
            print("independent bernoulli ln permanent estimate =", np.log(permanent_estimate))
            print("independent bernoulli ln permanent LB =", np.log(bootstrap_LB))
            print("independent bernoulli ln permanent UB =", np.log(bootstrap_UB))
            print("bootstrap_runtime:", bootstrap_runtime)

            print("total runtime", t1-t0)


def debug(required_accepted_sample_count=500):
    a = np.random.rand()
    a = np.random.rand()
    compare_wai = False
    diag_matrix, diag_matrix_exact_permanent = create_diagonal2(N=15, k=15)
    t0 = time.time()
    p_hat, permanent_estimate, permanent_estimate_with_tightening, gumbel_mean_permanent_estimate, gumbel_mle_estimate, stefano_corrected_permanent_est_with_tightening = estimate_acceptance_probability_with_nesting_upper_bounds(diag_matrix, required_accepted_sample_count, compare_wai)
    t1 = time.time()
    print('-'*80)
    print("sampling bounds with compare_wai:", compare_wai)
    print("exact permanent", diag_matrix_exact_permanent)
    print("ln(exact permanent)", np.log(diag_matrix_exact_permanent))
    print("ln(stefano_corrected_permanent_est_with_tightening)", np.log(stefano_corrected_permanent_est_with_tightening))
    print("ln(permanent_estimate, no tightening)", np.log(permanent_estimate))
    print("lower bound no tightening(estimate - .6 for 10 accepted samples) = ", np.log(permanent_estimate) - .6)
    print("upper bound no tightening(estimate + .6 for 10 accepted samples) = ", np.log(permanent_estimate) + .6)
    print("ln(permanent_estimate, with tightening)", np.log(permanent_estimate_with_tightening))
    print("lower bound with tightening(estimate - .6 for 10 accepted samples) = ", np.log(permanent_estimate_with_tightening) - .6)
    print("upper bound with tightening(estimate + .6 for 10 accepted samples) = ", np.log(permanent_estimate_with_tightening) + .6)
    print("gumbel mean ln(permanent_estimate)", gumbel_mean_permanent_estimate)
    print("gumbel MLE ln(permanent_estimate)", gumbel_mle_estimate)
    
    print("p_hat", p_hat, 'required_accepted_sample_count:', required_accepted_sample_count)
    print("runtime", t1-t0)



if __name__ == "__main__":
    # debug()
    # exit(0)



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
    # matrix_filenames = ["./networkrepository_data/bcspwr01.mtx"]

    # create_sinkhorn_values_in_table(matrix_filenames)
    create_sampling_values_in_table(matrix_filenames, required_accepted_sample_count=10)
