from __future__ import division
import numpy as np
import sys
import itertools
import math
from operator import itemgetter
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

# SEED=32 #incorrect method seems to do well with this seed??
SEED=1
np.random.seed(SEED)

UPPER_BOUND_MULTIPLIER = 0.0
def accelerated_a_star_sample(probs):
    '''

    Inputs:
    - probs: (numpy array)  

    Output:
    - best_assignments: (list of pairs) best_assignments[i][0] is the cost of the ith best
        assignment.  best_assignments[i][1] is the ith best assignment, which is a list of pairs
        where each pair represents an association in the assignment (1's in assignment matrix)
    '''
    init_node = Node(probs, gumbel_truncation=np.inf)

    #heap, storing the smallest negative gumbel perturbed states, or largest gumbel perturbed states
    #each element is a tuple of (-gumbel_perturbed_state, Node)
    negative_gumbel_perturbed_heap = []
    heapq.heappush(negative_gumbel_perturbed_heap, (-init_node.rand_assoc_gumbel_perturbed_state, init_node)) 
       
    cur_partition = init_node.partition()

    while(True):
        np.set_printoptions(linewidth=300)
        max_gumbel_upper_bound = -negative_gumbel_perturbed_heap[0][0]

        node_idx_with_max_gumbel_ub = None
        for cur_idx, cur_node in enumerate(cur_partition):
            if cur_node.upper_bound_gumbel_perturbed_state > max_gumbel_upper_bound:
                max_gumbel_upper_bound = cur_node.upper_bound_gumbel_perturbed_state
                node_idx_with_max_gumbel_ub = cur_idx
        if node_idx_with_max_gumbel_ub is None:
            break #we found the maximum gumbel perturbed state

        heapq.heappush(negative_gumbel_perturbed_heap, (-cur_partition[node_idx_with_max_gumbel_ub].rand_assoc_gumbel_perturbed_state, cur_partition[node_idx_with_max_gumbel_ub])) 

        if cur_partition[node_idx_with_max_gumbel_ub].assignment_count > 1:
            cur_partition.extend(cur_partition[node_idx_with_max_gumbel_ub].partition())

        del cur_partition[node_idx_with_max_gumbel_ub]

    smallest_gumbel_perturbed_cost = heapq.heappop(negative_gumbel_perturbed_heap)
    sample_of_log_Z = -smallest_gumbel_perturbed_cost[0] - np.euler_gamma
    # sampled_state =smallest_gumbel_perturbed_cost[1].random_state_idx
    sampled_state = np.where(probs == smallest_gumbel_perturbed_cost[1].random_state_prob)
    assert(len(sampled_state) == 1)
    sampled_state = int(sampled_state[0])
    #the total number of assignments is N! or the number of assignments in each of the partitioned
    #nodes plus the number of explicitlitly found assignments in negative_gumbel_perturbed_heap
    total_state_count = sum([node.assignment_count for node in cur_partition]) + len(negative_gumbel_perturbed_heap) + 1
    assert(total_state_count == len(probs))

    return sample_of_log_Z, sampled_state


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


class Node:
    # @profile
    def __init__(self, probs, gumbel_truncation):
        '''
        Following the terminology used by [1], a node is defined to be a nonempty subset of possible
        assignments to a cost matrix.  Every assignment in node N is required to contain
        required_cells and exclude excluded_cells.

        Inputs:
        - probs: (numpy array) the (unnormalized) probabilities contained in this node
        '''
        self.probs = probs
        self.number_of_states = len(probs)
        self.gumbel_truncation = gumbel_truncation

        self.assignment_count = self.count_assignments()
        if self.assignment_count == 0: #this node is empty
            return

        # compare_gumbel_vals = compare_truncated_gumbel(n_vals=[1, self.assignment_count], truncation=gumbel_truncation)
        compare_gumbel_vals = compare_truncated_gumbel(n_vals=[1, np.sum(self.probs)], truncation=gumbel_truncation)
        self.max_gumbel_1 = compare_gumbel_vals[0]
        self.max_gumbel = compare_gumbel_vals[1]
        # assert(self.max_gumbel == self.max_gumbel_1 + np.log(len(self.probs))), (self.max_gumbel, self.max_gumbel_1 + np.log(len(self.probs)), self.gumbel_truncation)

        self.random_state_idx, self.random_state_prob = self.sample_state_uniform()
        self.rand_assoc_gumbel_perturbed_state = np.log(self.random_state_prob) + self.max_gumbel
        # self.rand_assoc_gumbel_perturbed_state = np.log(self.random_state_prob) + self.max_gumbel

        self.upper_bound_on_sum_of_probs = np.sum(self.probs) + UPPER_BOUND_MULTIPLIER*np.random.rand()

        # #improved upper bound, WRONG
        # self.upper_bound_gumbel_perturbed_state = np.log(self.upper_bound_on_sum_of_probs) + self.max_gumbel_1
        
        # hypothesized bounds
        #improved upper bound, hypothesized
        # self.upper_bound_gumbel_perturbed_state = np.log(self.upper_bound_on_sum_of_probs) + self.max_gumbel_1 + np.log(np.max(self.probs)) - np.log(self.upper_bound_on_sum_of_probs/len(self.probs))
        self.upper_bound_gumbel_perturbed_state = self.max_gumbel + np.log(np.max(self.probs)) - np.log(self.upper_bound_on_sum_of_probs/len(self.probs))
        # self.upper_bound_gumbel_perturbed_state = np.inf
        #matching lower bound, hypothesized
        # self.random_state_idx = np.random.choice(len(self.probs), p=self.probs/np.sum(self.probs))
        self.random_state_idx = np.random.choice(len(self.probs))

        self.random_state_prob = self.probs[self.random_state_idx]
        #need to change this wehn UPPER_BOUND_MULTIPLIER != 0
        # self.rand_assoc_gumbel_perturbed_state = self.max_gumbel_1 + np.log(self.upper_bound_on_sum_of_probs) + np.log(self.random_state_prob) - np.log(self.upper_bound_on_sum_of_probs/len(self.probs))
        self.rand_assoc_gumbel_perturbed_state = self.max_gumbel + np.log(self.random_state_prob) - np.log(self.upper_bound_on_sum_of_probs/len(self.probs))
        self.rand_assoc_gumbel_perturbed_state = self.max_gumbel + np.log(self.random_state_prob) - np.log(self.random_state_prob)
        # self.rand_assoc_gumbel_perturbed_state = self.max_gumbel + np.log(self.random_state_prob)
        # self.gumbel_max = self.max_gumbel_1 + np.log(self.upper_bound_on_sum_of_probs) #fix truncation during partitioning

        # original A* sampling upper bound
        # self.upper_bound_gumbel_perturbed_state = np.log(np.max(self.probs)) + self.max_gumbel

        
        # assert(np.abs(self.upper_bound_gumbel_perturbed_state - self.hyp_upper_bound_gumbel_perturbed_state)<.0001), (self.upper_bound_gumbel_perturbed_state, self.max_gumbel_1 + np.log(len(self.probs)) + np.log(np.max(self.probs)), self.hyp_upper_bound_gumbel_perturbed_state)
        # assert(np.abs(self.rand_assoc_gumbel_perturbed_state - self.hyp_rand_assoc_gumbel_perturbed_state)<.0001), (self.rand_assoc_gumbel_perturbed_state, self.hyp_rand_assoc_gumbel_perturbed_state)


    def sample_state_uniform(self):
        '''
        sample a state from among this node's states uniformly, assumes
        '''
        random_state_idx = np.random.choice(len(self.probs))
        random_state_prob = self.probs[random_state_idx]
        return random_state_idx, random_state_prob

    def partition(self):
        '''
        Partition this node

        Output:
        - new_partition: a list of mutually disjoint Nodes, whose union with the minimum assignment
            of this node forms the set of possible assignments represented by this node
        '''
        # print( '#'*80)
        # print( "new_partition called on node with assignment_count =", self.assignment_count)
        partition1_count = (self.assignment_count-1)//2
        partition2_count = self.assignment_count - 1 - partition1_count

        probs_to_partition = self.probs.copy()
        probs_to_partition = np.delete(probs_to_partition, self.random_state_idx)
        partition1_indices = np.random.choice(len(probs_to_partition), size=partition1_count, replace=False)
        partition2_indices = np.array([i for i in range(len(probs_to_partition)) if (i not in partition1_indices)])
        assert(len(partition1_indices) == partition1_count and len(partition2_indices) == partition2_count)
        partition1_probs = np.array([probs_to_partition[i] for i in partition1_indices])
        partition2_probs = np.array([probs_to_partition[i] for i in partition2_indices])
        # print partition1_probs
        # print partition2_probs
        # print self.probs
        # print self.random_state_prob
        # print self.random_state_idx
        # print probs_to_partition
        assert(np.abs((np.sum(self.probs) - self.random_state_prob) - (np.sum(partition1_probs) + np.sum(partition2_probs))) < .00000001), (np.sum(self.probs)-self.random_state_prob,  np.sum(partition1_probs) + np.sum(partition2_probs))

        new_partition = []
        partition_assignment_counts = []
        if len(partition1_probs) > 0:
            partition1_node = Node(probs=partition1_probs, gumbel_truncation=self.max_gumbel)
            new_partition.append(partition1_node)
            partition_assignment_counts.append(partition1_node.assignment_count)
        if len(partition2_probs) > 0:
            partition2_node = Node(probs=partition2_probs, gumbel_truncation=self.max_gumbel)
            new_partition.append(partition2_node)
            partition_assignment_counts.append(partition2_node.assignment_count)

        #the sum of assignments over each partitioned node + 1 (the minimum assignment in this node)
        #should be equal to the number of assignments in this node
        assert(self.assignment_count == np.sum(partition_assignment_counts) + 1), (self.assignment_count, partition_assignment_counts, sum(partition_assignment_counts))
        return new_partition



    def count_assignments(self, use_brute_force = False):
        '''
        Count the number of states in this Node.  
        '''
        return len(self.probs)



# @profile
def test_accelerated_a_star_sample(N,iters,probs=None):
    '''
    Find the sum of the top k assignments and compare with the trivial bound
    on the remaining assignments of (N!-k)*(the kth best assignment)
    Inputs:
    - N: use a random cost matrix of size (NxN)
    - iters: number of random problems to solve and check
    '''
    if probs is None:
        probs = np.random.rand(N)
    all_samples_of_log_Z = []
    runtimes_list = []
    all_sampled_states = []
    wall_time = 0
    for test_iter in range(iters):
        if test_iter % 1000 == 0:
            print "completed", test_iter, "iters"
        t1 = time.time()
        sample_of_log_Z, sampled_state = accelerated_a_star_sample(probs)
        t2 = time.time()
        runtimes_list.append(t2-t1)
        all_sampled_states.append(sampled_state)
        cur_wall_time = t2-t1
        wall_time += cur_wall_time
        all_samples_of_log_Z.append(sample_of_log_Z)
    print()
    # print( "exact log(permanent):", np.log(calc_permanent_rysers(matrix)))
    print( "np.mean(all_samples_of_log_Z) =", np.mean(all_samples_of_log_Z))

    log_Z_estimate = np.mean(all_samples_of_log_Z)
    return runtimes_list, all_sampled_states, wall_time, log_Z_estimate, all_samples_of_log_Z




def test_sampling_correctness(N=5, ITERS=10000000, probs_to_use='0_1'):
    # check for smaller n
    # check total variatianal distance and compare with sampling normally
    '''
    Test that we're sampling from the correct distributions over assocations
    '''
    #key: length n tuple of associations, each is a tuple of length 2
    #value: dict, with keys:
    #   - 'true probability', value: (float)
    #   - 'empirical probability', value: (float)

    if probs_to_use == 'rand':
        probs = np.random.rand(N)
    elif probs_to_use == '0_1':
        probs = np.random.rand(N)
        for idx in range(len(probs)):
            if probs[idx] < .5:
                probs[idx] = 0 + probs[idx]/10000000000
            else:
                probs[idx] = 1.0 - probs[idx]/10000000000
    elif probs_to_use == 'step':
        probs = np.array([.1, .101, .3, .301, .302, .6, .601, .8, .801, .802])
    else:
        assert(False), "wrong parameter for probs_to_use!!: %s" % probs_to_use
    exact_partition_function = np.sum(probs)

    all_state_probs = {}
    list_of_all_true_probabilities = []
    for idx in range(len(probs)):
        true_probability = probs[idx]/exact_partition_function
        all_state_probs[idx] = {'true probability': true_probability,
                                             'empirical probability': 0.0}
        list_of_all_true_probabilities.append(true_probability)

    runtimes_list, all_sampled_states, wall_time, log_Z_estimate, all_samples_of_log_Z =\
        test_accelerated_a_star_sample(N, iters=ITERS, probs=probs)
    print("wall_time =", wall_time)
    print("log_Z_estimate =", log_Z_estimate)
    print("np.log(exact_partition_function) =", np.log(exact_partition_function))

    for sampled_state in all_sampled_states:
        all_state_probs[sampled_state]['empirical probability'] += 1/ITERS

    #key: association
    #value: empirical probability based on ITERS standard samples from the true distribution
    empirical_probs_sampled_standard = defaultdict(int)
    assert(ITERS == len(all_sampled_states))
    assert(N == len(list_of_all_true_probabilities))
    for i in range(ITERS):
        sampled_state_idx = np.random.choice(N, p=list_of_all_true_probabilities)
        empirical_probs_sampled_standard[sampled_state_idx] += 1/ITERS

    empirical_probs_exponential_gumbel = defaultdict(int)
    for i in range(ITERS):
        gumbel_perturbed_states = probs.copy()
        gumbel_perturbed_states = np.log(gumbel_perturbed_states)
        for idx in range(len(gumbel_perturbed_states)):
            compare_gumbel_vals = compare_truncated_gumbel(n_vals=[1], truncation=np.inf)
            gumbel_perturbed_states[idx] += compare_gumbel_vals[0]
        sampled_state_idx = np.argmax(gumbel_perturbed_states)
        empirical_probs_exponential_gumbel[sampled_state_idx] += 1/ITERS

        
 

    empirical_probs = []
    empirical_probs_sampled_standard_list = []
    empirical_probs_exponential_gumbel_list = []
    true_probs = []
    standard_tv_distance = 0
    gumbel_tv_distance = 0
    max_standard_error = 0
    max_gumbel_error = 0
    for state_idx, probabilities in all_state_probs.items():
        true_probs.append(probabilities['true probability'])
        empirical_probs.append(probabilities['empirical probability'])
        empirical_probs_sampled_standard_list.append(empirical_probs_sampled_standard[state_idx])
        empirical_probs_exponential_gumbel_list.append(empirical_probs_exponential_gumbel[state_idx])        
        gumbel_tv_distance += np.abs(true_probs[-1] - empirical_probs[-1])
        standard_tv_distance += np.abs(true_probs[-1] - empirical_probs_sampled_standard_list[-1])
        # print "cur gumbel error =", np.abs(true_probs[-1] - empirical_probs[-1])
        if np.abs(true_probs[-1] - empirical_probs[-1]) > max_gumbel_error:
            max_gumbel_error = np.abs(true_probs[-1] - empirical_probs[-1])
        if np.abs(true_probs[-1] - empirical_probs_sampled_standard_list[-1]) > max_standard_error:
            max_standard_error = np.abs(true_probs[-1] - empirical_probs_sampled_standard_list[-1])

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
    print "hypothesized gumbel log_Z_estimate error =", np.abs(log_Z_estimate - np.log(exact_partition_function))
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
    ax.plot(range(N), empirical_probs, 'r+', label='empirical_probs A*' , markersize=10)
    ax.plot(range(N), empirical_probs_sampled_standard_list, 'b+', label='empirical_probs sampled standard' , markersize=10)
    ax.plot(range(N), empirical_probs_exponential_gumbel_list, 'm+', label='empirical_probs sampled exponential gumbel' , markersize=10)
    ax.plot(range(N), true_probs, 'gx', label='true_probs' , markersize=10)
    plt.title('permutation probabilities')
    plt.xlabel('arbitrary index')
    plt.ylabel('probability')
    # Put a legend below current axis
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.1),
              fancybox=False, shadow=False, ncol=2, numpoints = 1)
    # plt.show()
    fig_file_name = "2hypothesis_test_accel_astar_correctness_UPPER_BOUND_MULTIPLIER=%d_N=%d_seed=%d_1gumbel_iters=%d_matrix=%s"%(UPPER_BOUND_MULTIPLIER, N, SEED, ITERS, probs_to_use)
    pickle_file_name = "./pickle_experiment_results/" + fig_file_name + ".pickle"
    fig.savefig(fig_file_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

    f = open(pickle_file_name, 'wb')
    pickle.dump((empirical_probs, empirical_probs_sampled_standard_list, true_probs, all_sampled_states, log_Z_estimate, all_samples_of_log_Z), f)
    f.close() 

    return gumbel_tv_distance, standard_tv_distance


if __name__ == "__main__":
    test_sampling_correctness(ITERS=100000, probs_to_use='rand')
    # test_sampling_correctness(ITERS=100000, probs_to_use='0_1')
    # test_sampling_correctness(ITERS=100000, probs_to_use='step')
