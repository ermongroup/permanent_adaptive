import numpy as np
import perform_tracking
import generate_data
import plotting
import os
import sys
from pymatgen.optimization import linear_assignment
from permanent import permanent as rysers_permanent

from IPython.utils import io
import pickle
import matplotlib
import matplotlib.pyplot as plt
COLORMAP = plt.cm.gist_ncar

def find_min_cost(cost_matrix):
    assert((cost_matrix >= 0).all()), matrix

    lin_assign = linear_assignment.LinearAssignment(cost_matrix)
    solution = lin_assign.solution
    association_list = zip([i for i in range(len(solution))], solution)

    minimum_cost = 0
    for (row,col) in association_list:
        minimum_cost += np.asscalar(cost_matrix[row][col])
    return (association_list, minimum_cost)

def run_experiment_over_parameter_set(num_time_steps, num_targets, initial_position_means, initial_velocity_means, initial_position_variance, initial_vel_variance, measurement_variance, spring_constants, dt):
    # (all_states, all_measurements, gen_params) = generate_data.get_parameters_and_data(num_time_steps, state_space, measurement_space,
    #   markov_order, num_targets)

    # (all_states, all_measurements, gen_params) = generate_data.get_parameters_and_data_targets_identical_plus_noise(num_time_steps, state_space, hidden_state_space, observed_state_space, measurement_space,
    #   markov_order, num_targets)

    (all_states, all_measurements, gen_params) = generate_data.get_parameters_and_data(num_time_steps, num_targets, \
        initial_position_means, initial_velocity_means, initial_position_variance, initial_vel_variance, measurement_variance, spring_constants, dt)



    # experiment_name = 'velocityInMeas_%dtargets_seed=%d_10k' % (num_targets, SEED)
    # experiment_name = 'varySpringConstants309_%dtargets_seed=%d' % (num_targets, SEED)
    # experiment_name = 'moreDEBUGCHECK_%dtargets_seed=%d' % (num_targets, SEED)

    # experiment_name = 'fixImportanceWeightParticleFilter_noResample_noSlackTightening_estimatePermanent_%dtargets_seed=%d' % (num_targets, SEED)
    experiment_name = 'icml_fixImportanceWeightParticleFilter_noResample_DEBUG_estimatePermanent_shorter_spreadOutINit%dtargets_seed=%d' % (num_targets, SEED)

   # experiment_name = 'fixImportanceWeightParticleFilter_noResample_estimatePermanent_%dtargets_seed=%d' % (num_targets, SEED)
    # experiment_name = 'fixImportanceWeightParticleFilter1ResampleALot_%dtargets_seed=%d' % (num_targets, SEED)
    experiment_folder = './' + experiment_name + '/'
    if not os.path.isdir(experiment_folder):
        os.mkdir(experiment_folder)

    f = open(experiment_folder + 'input_data.pickle', 'w')
    pickle.dump((gen_params, all_measurements), f)
    f.close()  

    gt_likelihood, gt_all_log_likelihoods = perform_tracking.get_gt_association_likelihood(gen_params, all_measurements)
    f = open(experiment_folder + 'log_likelihoods.txt', 'a')
    std_out = sys.stdout
    sys.stdout = f
    print 'Q:'
    print gen_params.q_matrix
    print "gen_params.num_time_steps:", gen_params.num_time_steps
    print "gen_params.num_targets:", gen_params.num_targets
    print "gen_params.initial_position_means:", gen_params.initial_position_means
    print "gen_params.initial_velocity_means:", gen_params.initial_velocity_means
    print "gen_params.initial_position_variance:", gen_params.initial_position_variance
    print "gen_params.initial_vel_variance:", gen_params.initial_vel_variance
    print "gen_params.measurement_variance:", gen_params.measurement_variance
    print "gen_params.spring_constants:", gen_params.spring_constants
    print "gen_params.r_matrix:", gen_params.r_matrix    
    sys.stdout = std_out
    f.write('ground truth log_likelihood = %f, should also be %f\n' % (gt_likelihood, gt_all_log_likelihoods[-1]))
    f.close()

    ######################## PLOT GROUND TRUTH TRAJECTORIES IN ONE FIGURE ########################
    for target_idx in range(gen_params.num_targets):
        xs = all_states[target_idx]
        zs = all_measurements[target_idx]
        # print "states:", [x[1] for x in xs]
        plt.plot([x[0] for x in xs], label='states %d' % target_idx, marker='+', linestyle="None")
        # print "measurements:", zs
        # plt.plot([z[0] for z in zs], label='measurements %d' % target_idx, marker='x', linestyle="None")
#     plt.ylabel('some numbers')
    plt.legend()
    plt.title('Ground Truth Trajectories')
    plt.ylabel('position')                
    plt.xlabel('time step')
    plt.savefig(experiment_folder  + 'ground_truth_trajectories')
    plt.show()
    plt.close()


    # for (n_particles, method) in [(10, 'MHT'), (10, 'exact_sampling')]:
    # for (n_particles, method) in [(10, 'exact_sampling'), (10, 'MHT'), (100, 'MHT'), (20, 'exact_sampling'), (500, 'MHT'), (50, 'exact_sampling'), (100, 'exact_sampling'), (1000, 'MHT'), (5000, 'MHT'), (500, 'exact_sampling'), (2000, 'exact_sampling')]:
    # for (n_particles, method) in [(100, 'exact_sampling'), (90, 'exact_sampling'), (100, 'exact_sampling'), (10, 'MHT'), (100, 'MHT'), (500, 'MHT'), (1000, 'MHT'), (5000, 'MHT'), (10000, 'exact_sampling'), (20000, 'exact_sampling'), (20, 'exact_sampling'), (50, 'exact_sampling'),]:
    #for (n_particles, method) in [(10, 'exact_sampling'), (10, 'MHT'), (100, 'MHT'), (500, 'MHT'), (1000, 'MHT'), (5000, 'MHT'), (10000, 'MHT'), (20000, 'MHT'), (20, 'exact_sampling')]:
    
    # for (n_particles, method) in [(11, 'MHT'), (100, 'MHT')]:
    # for (n_particles, method) in [(10, 'exact_sampling'), (11, 'exact_sampling'), (12, 'exact_sampling'), (13, 'exact_sampling'), (100, 'exact_sampling'), (200, 'exact_sampling'), (500, 'exact_sampling'), (1000, 'exact_sampling'), (10000, 'exact_sampling')]:
    list_of_particle_counts = []
    list_of_log_likelihoods = []
    list_of_mean_squared_errors = []
    # for (n_particles, method) in [(11, 'exact_sampling'),(13, 'exact_sampling'),(15, 'exact_sampling'),\
    #                               (50, 'exact_sampling'),(70, 'exact_sampling'),(100, 'exact_sampling'),(150, 'exact_sampling'),(200, 'exact_sampling'),(400, 'exact_sampling')]:
    for (n_particles, method) in [(500, 'exact_sampling'),(1000, 'exact_sampling'),(2000, 'exact_sampling'),(5000, 'exact_sampling'),(10000, 'exact_sampling'),(20000, 'exact_sampling')]:

    # for (n_particles, method) in [(11, 'sequential_proposal_SMC'),(13, 'sequential_proposal_SMC'),(15, 'sequential_proposal_SMC'),\
    #                               (50, 'sequential_proposal_SMC'),(70, 'sequential_proposal_SMC'),(100, 'sequential_proposal_SMC'),(150, 'sequential_proposal_SMC'),(200, 'sequential_proposal_SMC'),(400, 'sequential_proposal_SMC'),\
    #                               (500, 'sequential_proposal_SMC'),(1000, 'sequential_proposal_SMC'),(2000, 'sequential_proposal_SMC'),(5000, 'sequential_proposal_SMC'),(10000, 'sequential_proposal_SMC'),(20000, 'sequential_proposal_SMC')]:

    # for (n_particles, method) in [(10, 'MHT'),(50, 'MHT'),(100, 'MHT'),\
    #                               (500, 'MHT'),(1000, 'MHT'),(2000, 'MHT'),(5000, 'MHT'),(10000, 'MHT'),(20000, 'MHT')]:


    # for (n_particles, method) in [(10, 'MHT'), (100, 'MHT'), (200, 'MHT'), (500, 'MHT'), (1000, 'MHT'), (10000, 'MHT')]:
    # for (n_particles, method) in [(10, 'sequential_proposal_SMC'), (100, 'sequential_proposal_SMC'), (200, 'sequential_proposal_SMC'), (500, 'sequential_proposal_SMC'), (1000, 'sequential_proposal_SMC'), (10000, 'sequential_proposal_SMC')]:
    # for (n_particles, method) in [(11, 'sequential_proposal_SMC'), (12, 'sequential_proposal_SMC'), (13, 'sequential_proposal_SMC'), (14, 'sequential_proposal_SMC')]:
  
    # for (n_particles, method) in [(1, 'gt_assoc')]:
    # for (n_particles, method) in [(1, 'gt_assoc'), (10, 'MHT'), (100, 'MHT')]:

    # for (n_particles, method) in [(100, 'MHT'), (1000, 'MHT'), (5000, 'MHT'), (10000, 'exact_sampling'), (20000, 'exact_sampling'), (20, 'exact_sampling'), (50, 'exact_sampling'),]:
    # for (n_particles, method) in [(10, 'exact_sampling'), (100, 'exact_sampling'), (1000, 'exact_sampling'), (100000, 'exact_sampling')]:
    # for (n_particles, method) in [(10, 'MHT'), (100, 'MHT'), (1000, 'MHT'), (100000, 'MHT')]:
    # for (n_particles, method) in [(10000, 'exact_sampling')]:
        for use_group_particles in [False]:
            cur_experiment = "%s_particles=%d_use_group_particles=%s" % (method, n_particles, use_group_particles)
            print("cur_experiment:", cur_experiment)
            # with io.capture_output() as captured:
            (all_target_posteriors, all_target_priors, most_probable_particle, all_log_likelihoods, log_likelihoods_from_most_probable_particles) = perform_tracking.run_tracking(all_measurements, tracking_method=method, generative_parameters=gen_params, n_particles=n_particles, use_group_particles=use_group_particles)

            most_probable_particle_log_prob = most_probable_particle.log_importance_weight_normalization + np.log(most_probable_particle.importance_weight)

            # f = open(experiment_folder + '%s_results.pickle'%cur_experiment, 'w')
            # pickle.dump((all_target_posteriors, all_target_priors, most_probable_particle, most_probable_particle_log_prob), f)
            # f.close()




            ######################## Match Inferred and ground truth trajectories ########################
            trajectory_costs = np.ones((gen_params.num_targets, gen_params.num_targets))*np.inf
            for inferred_trajectory_idx in range(gen_params.num_targets):
                for gt_trajectory_idx in range(gen_params.num_targets):
                    inf_traj = all_target_posteriors[inferred_trajectory_idx]
                    gt_traj = [x[0] for x in all_states[gt_trajectory_idx]]
                    cur_cost = 0
                    assert(len(inf_traj) == len(gt_traj))
                    for idx, inf_location in enumerate(inf_traj):
                        cur_cost += (inf_location - gt_traj[idx])**2
                    trajectory_costs[inferred_trajectory_idx, gt_trajectory_idx] = cur_cost
            assert(not (trajectory_costs == np.inf).any())
            (association_list, minimum_cost) = find_min_cost(trajectory_costs)
            mean_squared_error = minimum_cost/(gen_params.num_targets*num_time_steps)


            list_of_particle_counts.append(n_particles)
            list_of_log_likelihoods.append(most_probable_particle_log_prob)
            list_of_mean_squared_errors.append(mean_squared_error)

            f = open(experiment_folder + '%s_results_2.pickle'%method, 'w')
            pickle.dump((list_of_particle_counts, list_of_log_likelihoods, list_of_mean_squared_errors, gt_likelihood), f)
            f.close()              

            f = open(experiment_folder + 'log_likelihoods.txt', 'a')
            f.write(cur_experiment + ' log_likelihood = %f, mean_squared_error = %f\n' % (most_probable_particle_log_prob, mean_squared_error))
            f.close()

            ######################## PLOT DATA ########################
            #plot log likelihoods
            PLOT_ADJUSTED_LOG_LIKELIHOODS = True
            if PLOT_ADJUSTED_LOG_LIKELIHOODS:
                #to make plot more legible, shift so the largest log-likelihood at each timestep is 0
                max_log_likelihoods = []
                for cur_timestep_log_likelihoods in all_log_likelihoods:
                    max_log_likelihoods.append(max(cur_timestep_log_likelihoods))

                adjusted_all_log_likelihoods = []
                for time_step, cur_timestep_log_likelihoods in enumerate(all_log_likelihoods):
                    adjusted_all_log_likelihoods.append([l - max_log_likelihoods[time_step] for l in cur_timestep_log_likelihoods])


                for i in range(gen_params.num_time_steps//10):
                    plt.plot(range(i*10,i*10+10), [cur_timestep_log_likelihoods for cur_timestep_log_likelihoods in adjusted_all_log_likelihoods[i*10:i*10+10]], marker='x', linestyle="None", color='blue')
                    for cur_particle_log_likelihoods in log_likelihoods_from_most_probable_particles:
                        # print "cur_particle_log_likelihoods:", cur_particle_log_likelihoods
                        adjusted_cur_particle_log_likelihoods = []
                        for time_step, log_likelihood in enumerate(cur_particle_log_likelihoods):
                            adjusted_cur_particle_log_likelihoods.append(log_likelihood - max_log_likelihoods[time_step])
                        plt.plot(range(i*10,i*10+10), adjusted_cur_particle_log_likelihoods[i*10:i*10+10], marker='x', linestyle='-', color='blue')
                        # print "adjusted_cur_particle_log_likelihoods:", adjusted_cur_particle_log_likelihoods
                        # print
                        
                    plt.title('%s log likelihoods' % (method))
                    plt.ylabel('log likelihood')                
                    plt.xlabel('time step')
                    plt.savefig(experiment_folder + cur_experiment + '_log_likelihoods_%d' % i)
                    plt.close()

            else:
                plt.plot([cur_timestep_log_likelihoods for cur_timestep_log_likelihoods in all_log_likelihoods], marker='x', linestyle="None", color='blue')
                for cur_particle_log_likelihoods in log_likelihoods_from_most_probable_particles:
                    plt.plot(cur_particle_log_likelihoods, marker='x', linestyle='-', color='blue')

                plt.title('%s log likelihoods' % (method))
                plt.ylabel('log likelihood')                
                plt.xlabel('time step')
                plt.savefig(experiment_folder + cur_experiment + '_log_likelihoods')
                plt.close()

            if PLOT_ADJUSTED_LOG_LIKELIHOODS:

                #ground truth log likelihoods
                adjusted_gt_log_likelihoods = []
                for time_step, log_likelihood in enumerate(gt_all_log_likelihoods):
                    print "gt_log_likelihood:", log_likelihood, "max_log_likelihoods[time_step]:", max_log_likelihoods[time_step]   
                    adjusted_gt_log_likelihoods.append(log_likelihood - max_log_likelihoods[time_step])
                plt.plot(adjusted_gt_log_likelihoods, marker='x', linestyle='-', color='green')
                plt.title('%s gt log likelihoods' % (method))
                plt.ylabel('gt log likelihood - largest inferred log likelihood')                
                plt.xlabel('time step')
                plt.savefig(experiment_folder + cur_experiment + '_gt_log_likelihoods')
                plt.close()                
            print "gt_all_log_likelihoods:", gt_all_log_likelihoods
            print "log_likelihoods_from_most_probable_particles:", log_likelihoods_from_most_probable_particles
            #plot trajectories
            for target_idx in range(gen_params.num_targets):
                inferred_trajectory_idx = association_list[target_idx][0]
                gt_trajectory_idx = association_list[target_idx][1]
                xs = all_target_posteriors[inferred_trajectory_idx]
                # print "states:", [x[0,0] for x in xs]
                plt.plot(xs, label='inferred states', marker='+', linestyle="None")
                plt.plot([x[0] for x in all_states[gt_trajectory_idx]], label='true states', marker='x', linestyle="None")
                plt.plot([z[0] for z in all_measurements[gt_trajectory_idx]], label='measurements', marker='+', linestyle="None")
            #     plt.ylabel('some numbers')
                plt.title('%s target %d (k=%f), inferred_target %d (k=%f)' % (method, gt_trajectory_idx, spring_constants[gt_trajectory_idx], inferred_trajectory_idx, spring_constants[inferred_trajectory_idx]))
                plt.legend()
                # plt.show()
                plt.savefig(experiment_folder + cur_experiment + 'target_%d' % target_idx)
                plt.close()

def plot_log_likelihoods(num_targets):
    experiment_name = 'icml_fixImportanceWeightParticleFilter_noResample_DEBUG_estimatePermanent_shorter_spreadOutINit%dtargets_seed=%d' % (num_targets, SEED)
    experiment_folder = './' + experiment_name + '/'

    for method in ['exact_sampling', 'sequential_proposal_SMC', 'MHT']:
        f = open(experiment_folder + '%s_results.pickle'%method, 'r')
        (list_of_particle_counts, list_of_log_likelihoods, list_of_mean_squared_errors, gt_likelihood) = pickle.load(f)
        f.close()              

        if method == 'exact_sampling':
            f = open(experiment_folder + '%s_results.pickle'%method, 'r')
            (list_of_particle_counts1, list_of_log_likelihoods1, list_of_mean_squared_errors1, gt_likelihood1) = pickle.load(f)
            f.close()              
            list_of_particle_counts.extend(list_of_particle_counts1)
            list_of_log_likelihoods.extend(list_of_log_likelihoods1)
            list_of_mean_squared_errors.extend(list_of_mean_squared_errors1)


        if method == 'exact_sampling':
            label = 'Optimal Proposal Distribution'
            marker='x'
        elif method == 'sequential_proposal_SMC':
            label = 'Sequential Proposal Distribution'
            marker='+'
        else:
            label = 'MHT'
            marker='2'
        plt.semilogx(list_of_particle_counts, list_of_log_likelihoods, label=label, marker=marker, linestyle="None", markersize=15)

    # matplotlib.rcParams.update({'font.size': 30})
    plt.axhline(y=gt_likelihood, linewidth=1, color='b', label='Ground Truth')
    plt.ylabel('Log-Likelihood', fontsize=19)
    plt.xlabel('Particle Count', fontsize=19)
    plt.title('Maximum Inferred Log-Likelihood', fontsize=22)
    plt.legend(loc='lower right',prop={'size': 15},# bbox_to_anchor=(0.5, -.1),
              fancybox=False, shadow=False, ncol=1, numpoints = 1)
    plt.tick_params(labelsize='large')
    # plt.show()
    plt.savefig(experiment_folder + 'all_log_likelihoods', bbox_inches = "tight")
    plt.close()    

def plot_mean_squared_errors(num_targets):
    experiment_name = 'icml_fixImportanceWeightParticleFilter_noResample_DEBUG_estimatePermanent_shorter_spreadOutINit%dtargets_seed=%d' % (num_targets, SEED)
    experiment_folder = './' + experiment_name + '/'

    for method in ['exact_sampling', 'sequential_proposal_SMC', 'MHT']:
        f = open(experiment_folder + '%s_results.pickle'%method, 'r')
        (list_of_particle_counts, list_of_log_likelihoods, list_of_mean_squared_errors, gt_likelihood) = pickle.load(f)
        f.close() 

        if method == 'exact_sampling':
            f = open(experiment_folder + '%s_results_2.pickle'%method, 'r')
            (list_of_particle_counts1, list_of_log_likelihoods1, list_of_mean_squared_errors1, gt_likelihood1) = pickle.load(f)
            f.close()              
            list_of_particle_counts.extend(list_of_particle_counts1)
            list_of_log_likelihoods.extend(list_of_log_likelihoods1)
            list_of_mean_squared_errors.extend(list_of_mean_squared_errors1)

        print method
        print list_of_log_likelihoods
        print list_of_particle_counts

        if method == 'exact_sampling':
            label = 'Optimal Proposal Distribution'
            marker='x'
        elif method == 'sequential_proposal_SMC':
            label = 'Sequential Proposal Distribution'
            marker='+'
        else:
            label = 'MHT'             
            marker='2'
        plt.semilogx(list_of_particle_counts, list_of_mean_squared_errors, label=label, marker=marker, linestyle="None", markersize=15)

    # matplotlib.rcParams.update({'font.size': 10})
    plt.ylabel('Mean Squared Error', fontsize=19)
    plt.xlabel('Particle Count', fontsize=19)
    plt.title('Target Position Mean Squared Error', fontsize=22)
    plt.legend(loc='upper right',prop={'size': 15},# bbox_to_anchor=(0.5, -.1),
              fancybox=False, shadow=False, ncol=1, numpoints = 1)
    plt.tick_params(labelsize='large')    
    # plt.show()
    plt.savefig(experiment_folder + 'all_mean_squared_errors', bbox_inches = "tight")
    plt.close()      

def replot_previous_experiment_data():
    experiment_name = 'test_experiment'
    experiment_folder = './' + experiment_name + '/'

    f = open(experiment_folder + 'input_data.pickle', 'r')
    (gen_params, all_measurements) = pickle.load(f)
    f.close()  

    # for (n_particles, method) in [(10, 'MHT'), (10, 'exact_sampling')]:
    for (n_particles, method) in [(10, 'MHT'), (10, 'exact_sampling'), (100, 'MHT'), (20, 'exact_sampling'), (1000, 'MHT'), (50, 'exact_sampling'), (10000, 'MHT'), (100, 'exact_sampling')]:
        cur_experiment = "%s_particles=%d" % (method, n_particles)

        f = open(experiment_folder + '%s_results.pickle'%cur_experiment, 'r')
        (all_target_posteriors, all_target_priors, most_probable_particle, most_probable_particle_log_prob) = pickle.load(f)
        f.close()
        
        print(cur_experiment, most_probable_particle.importance_weight)
        
    #     for time_step in [5, 11, 17]:
    #         for target_idx in range(NUM_TARGETS):
    #             plotting.bar_plot(all_target_priors[target_idx][time_step], ylim=(0,1.1), title='Prior Distribution over Target States', c=COLORMAP(target_idx/NUM_TARGETS))
    #             plt.axvline(all_states[target_idx][time_step], lw=1.5, c=COLORMAP(target_idx/NUM_TARGETS))
    #         plt.tight_layout()
    #         plt.savefig(experiment_folder + cur_experiment + 'Priors_timeStep%d.pdf'%time_step)
    #         plt.close()

    #         for target_idx in range(NUM_TARGETS):
    #             plotting.bar_plot(all_target_posteriors[target_idx][time_step], ylim=(0,1.1), title='Posterior Distribution over Target States', c=COLORMAP(target_idx/NUM_TARGETS))
    #             plt.axvline(all_states[target_idx][time_step], lw=1.5, c=COLORMAP(target_idx/NUM_TARGETS))
    #         plt.tight_layout()
    #         plt.savefig(experiment_folder + cur_experiment + 'Posteriors_timeStep%d.pdf'%time_step)



def calc_permanent_rysers(matrix):
    '''
    Exactly calculate the permanent of the given matrix user Ryser's method (faster than calc_permanent)
    '''
    N = matrix.shape[0]
    assert(N == matrix.shape[1])
    #this looks complicated because the method takes and returns a complex matrix,
    #we are only dealing with real matrices so set complex component to 0
    return np.real(rysers_permanent(1j*np.zeros((N,N)) + matrix))


if __name__ == "__main__":
    # a = np.array([[1.27242553e-002, 0.00000000e+000, 1.20707341e-235, 2.24392756e-003, 0.00000000e+000, 0.00000000e+000, 8.86430921e-080, 0.00000000e+000, 0.00000000e+000, 1.00000000e+000, 1.25701570e-231, 6.93121164e-135, 0.00000000e+000, 0.00000000e+000, 5.07717856e-060,],
    #               [0.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.83366339e-187, 1.23217822e-018, 0.00000000e+000, 2.65897325e-048, 0.00000000e+000, 3.65830115e-004, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
    #               [0.00000000e+000, 2.54058400e-003, 1.00000000e+000, 1.33104113e-208, 1.61514466e-012, 0.00000000e+000, 1.14077851e-058, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
    #               [4.87942899e-102, 1.59550433e-080, 8.12878494e-048, 4.13344397e-041, 2.99143543e-118, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 5.20284487e-091, 2.33907152e-132, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
    #               [0.00000000e+000, 1.24020249e-049, 1.92123914e-016, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 4.03972131e-122, 0.00000000e+000, 5.20511265e-001, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 7.03378714e-239, 0.00000000e+000, 0.00000000e+000,],
    #               [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.18174177e-075, 0.00000000e+000, 0.00000000e+000,],
    #               [1.04036746e-135, 2.81243741e-052, 1.68696669e-032, 3.32552644e-057, 9.78142923e-095, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 2.49679797e-068, 6.82868495e-170, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
    #               [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000,],
    #               [8.62446203e-119, 5.66515157e-067, 3.04660780e-040, 5.76203465e-049, 7.16441832e-107, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 1.72867231e-079, 5.54502739e-151, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
    #               [1.00000000e+000, 0.00000000e+000, 6.62524543e-221, 1.00000000e+000, 0.00000000e+000, 0.00000000e+000, 7.26395036e-071, 0.00000000e+000, 0.00000000e+000, 1.00000000e+000, 5.83978724e-248, 7.38080757e-150, 0.00000000e+000, 0.00000000e+000, 7.35824599e-070,],
    #               [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.00000000e+000, 7.78723940e-102, 0.00000000e+000, 0.00000000e+000, 2.70628319e-195,],
    #               [7.85276999e-184, 0.00000000e+000, 0.00000000e+000, 2.05402150e-126, 0.00000000e+000, 0.00000000e+000, 3.16028051e-315, 0.00000000e+000, 0.00000000e+000, 2.04697504e-143, 5.17740865e-026, 1.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.76727893e-016,],
    #               [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 6.40859143e-246, 9.22033486e-017, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 0.00000000e+000,],
    #               [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 2.60681320e-017, 0.00000000e+000,],
    #               [1.95942725e-085, 0.00000000e+000, 0.00000000e+000, 6.63968497e-063, 0.00000000e+000, 0.00000000e+000, 9.20492366e-212, 0.00000000e+000, 0.00000000e+000, 1.27874164e-059, 3.24625780e-076, 3.61484419e-017, 0.00000000e+000, 0.00000000e+000, 1.00000000e+000,],])

    # print "exact permanent = ", calc_permanent_rysers(a)
    # sleep(3)
    SEED = 0
    np.random.seed(SEED)

    # num_time_steps = 6
    # state_space = np.array((20,20))
    # measurement_space = np.array((20))
    # markov_order = 1
    # num_targets = 20

    # num_time_steps = 4
    hidden_state_space = np.array((2))
    observed_state_space = np.array((20))
    # state_space = np.array((2, 20))
    # measurement_space = np.array((20))
    # markov_order = 1
    # num_targets = 15  

    num_targets = 10

    plot_log_likelihoods(num_targets=num_targets)
    plot_mean_squared_errors(num_targets=num_targets)
    sleep(3)

    # run_experiment_over_parameter_set(num_time_steps=20, num_targets=3,\
    #                                   initial_position_variance=0, initial_vel_variance=5000, measurement_variance=.01, spring_k=15, dt=.1)
            

    # run_experiment_over_parameter_set(num_time_steps=20, num_targets=num_targets,\
    #                                   initial_position_means=[40*np.random.rand() for i in range(num_targets)],\
    #                                   # initial_position_means=[20*np.random.rand() for i in range(num_targets)],\
    #                                   initial_position_variance=6, initial_vel_variance=30, measurement_variance=1,\
    #                                   spring_constants=[100*np.random.rand() for i in range(num_targets)], dt=.1)

    # run_experiment_over_parameter_set(num_time_steps=50, num_targets=num_targets,\
    #                                   initial_position_means=[40*np.random.rand() for i in range(num_targets)],\
    #                                   # initial_position_means=[20*np.random.rand() for i in range(num_targets)],\
    #                                   initial_position_variance=20, initial_vel_variance=30, measurement_variance=1,\
    #                                   spring_constants=[100*np.random.rand() for i in range(num_targets)], dt=.01)

    #previous experiment before getting stuff runnig before ICML
    # run_experiment_over_parameter_set(num_time_steps=50, num_targets=num_targets,\
    #                                   initial_position_means=[10*np.random.rand() for i in range(num_targets)],\
    #                                   # initial_position_means=[20*np.random.rand() for i in range(num_targets)],\
    #                                   initial_velocity_means=[100*np.random.rand() for i in range(num_targets)],\
    #                                   initial_position_variance=.20, initial_vel_variance=.30, measurement_variance=1,\
    #                                   spring_constants=[1000*np.random.rand() for i in range(num_targets)], dt=.01)


    run_experiment_over_parameter_set(num_time_steps=20, num_targets=num_targets,\
                                      initial_position_means=[10*np.random.rand() for i in range(num_targets)],\
                                      # initial_position_means=[20*np.random.rand() for i in range(num_targets)],\
                                      initial_velocity_means=[100*np.random.rand() for i in range(num_targets)],\
                                      initial_position_variance=10.0, initial_vel_variance=.30, measurement_variance=1,\
                                      spring_constants=[2000*np.random.rand() for i in range(num_targets)], dt=.01)



    # run_experiment_over_parameter_set(num_time_steps=10, num_targets=num_targets,\
    #                                   # initial_position_means=[40*np.random.rand() for i in range(num_targets)],\
    #                                   initial_position_means=[20*np.random.rand() for i in range(num_targets)],\
    #                                   initial_position_variance=6, initial_vel_variance=30, measurement_variance=1,\
    #                                   spring_constants=[50*np.random.rand() for i in range(num_targets)], dt=.1)



