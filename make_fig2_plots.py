import numpy as np
import pickle
import math
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.use('Agg') #prevent error running remotely

import matplotlib.pyplot as plt

def plot_runtime_vs_N(pickle_file_paths=['./number_of_times_partition_called_for_each_n.pickle'], pickle_file_paths2=None,\
                      plot_filename=None):
    n_vals_mean1 = []
    log_n_vals_mean1 = []
    log_run_time_vals_mean1 = []
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
            if np.mean(runtimes_list) > 260.032737398: #match max runtime of adapart with Law for block diagonal
                continue

            all_n_vals1.extend([n for i in range(len(runtimes_list))])
            all_run_time_vals1.extend(runtimes_list)
            # all_number_of_times_partition_called_vals.extend(number_of_times_partition_called_list)
            log_n_vals_mean1.append(math.log(n))
            n_vals_mean1.append(n)
            log_run_time_vals_mean1.append(math.log(np.mean(runtimes_list)))
            run_time_vals_mean1.append(np.mean(runtimes_list))
            # number_of_times_partition_called_vals_mean.append(math.log(np.mean(number_of_times_partition_called_list)))
            # number_of_times_partition_called_vals_mean.append(np.mean(number_of_times_partition_called_list))
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(n_vals_mean1, log_run_time_vals_mean1, 'r+', label='AdaPart' , markersize=10)
    # ax.plot(all_n_vals1, all_run_time_vals1, 'r+', label='run_time_vals_mean' , markersize=10)


    if pickle_file_paths2 is not None:
        n_vals_mean2 = []
        log_run_time_vals_mean2 = []
        run_time_vals_mean2 = []
        log_n_vals_mean2 = []

        for pickle_file_path in pickle_file_paths2:
            f = open(pickle_file_path, 'rb')
            number_of_times_partition_called_for_each_n = pickle.load(f)
            f.close()

            for n, (runtimes_list, all_samples_of_log_Z, exact_log_Z, permanent_UBs) in number_of_times_partition_called_for_each_n.items():
                if n < 5:
                    continue
                # if np.mean(runtimes_list) > 260.032737398: #match max runtime of Law with adapart for uniform
                #     continue

                n_vals_mean2.append(n)
                log_n_vals_mean2.append(math.log(n))
                log_run_time_vals_mean2.append(math.log(np.mean(runtimes_list)))
                run_time_vals_mean2.append(np.mean(runtimes_list))

        ax.plot(n_vals_mean2, log_run_time_vals_mean2, 'yx', label='Law' , markersize=10)

    print np.max(log_run_time_vals_mean2)
    matplotlib.rcParams.update({'font.size': 20})
    POLY_FIT = False #fit with some polynomials
    if POLY_FIT:
        xp = np.linspace(5, 180, 200)
        p5 = np.exp(-20)*xp**5
        p4 = np.exp(-9.5)*xp**2

        # p5 = np.exp(-28)*xp**8
        # p4 = np.exp(-9.5)*xp**2

        p6 = p4 + p5
        ax.plot(xp, np.log(p4), '--', label=r'e^{-9.5} n^2')
        ax.plot(xp, np.log(p5), '--', label=r'$e^{-20} n^5$')
        ax.plot(xp, np.log(p6), '-', label=r'$e^{-9.5} n^2 + e^{-20} n^5$')

    # ax.plot(n_vals_mean, number_of_times_partition_called_vals_mean, 'gx', label='number_of_times_partition_called_vals_mean' , markersize=10)
    plt.title('Block Diagonal, K=10', fontsize=24)
    plt.xlabel('n (matrix dimension)')
    plt.ylabel('log(runtime) (log(seconds))')
    # Put a legend below current axis
    lgd = ax.legend(loc='lower right', prop={'size': 12},# bbox_to_anchor=(0.5, -.11),
              fancybox=False, shadow=False, ncol=1, numpoints = 1)
    plt.setp(lgd.get_title(),fontsize='xx-small')

    fig.savefig(plot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent_estimation_scaling_plots_diagMatrix', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent_estimation_scaling_plots_uniformMatrix', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent_estimation_scaling_plots_01Matrix', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    plt.close()

    fig = plt.figure()
    ax = plt.subplot(111)
    # ax.loglog(n_vals_mean1, run_time_vals_mean1, 'r+', label='AdaPart' , markersize=10)
    ax.loglog((n_vals_mean1), (run_time_vals_mean1), 'r+', label='AdaPart' , markersize=10)
    if pickle_file_paths2 is not None:
        # ax.loglog(n_vals_mean2, run_time_vals_mean2, 'yx', label='Law' , markersize=10)
        ax.loglog((n_vals_mean2), (run_time_vals_mean2), 'yx', label='Law' , markersize=10)

    print np.max(run_time_vals_mean2)
    print np.max(run_time_vals_mean1)
    plt.title('Block Diagonal, K=10', fontsize=24)
    plt.xlabel('n (matrix dimension)')
    plt.ylabel('Runtime (seconds)')
    # Put a legend below current axis
    # fig.savefig('permanent_estimation_scaling_plots_logN_diagMatrix', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent_estimation_scaling_plots_logN_uniformMatrix', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent_estimation_scaling_plots_logN_01Matrix', bbox_extra_artists=(lgd,), bbox_inches='tight')    

    matplotlib.rcParams.update({'font.size': 15})
    POLY_FIT = False #fit with some polynomials
    if POLY_FIT:
        xp = np.linspace(5, 180, 200)
        # p6 = .05*xp**6

        p5 = np.exp(-20)*xp**5
        p4 = np.exp(-9.5)*xp**2

        # p5 = np.exp(-28)*xp**8
        # p4 = np.exp(-9.5)*xp**2

        p6 = p4 + p5
        ax.loglog((xp), (p4), '--', label=r'e^{-9.5} n^2')
        ax.loglog((xp), (p5), '--', label=r'$e^{-20} n^5$')
        ax.loglog((xp), (p6), '-', label=r'$e^{-9.5} n^2 + e^{-20} n^5$')
        # ax.loglog(xp, np.log(p4(xp)), '--')
    plt.setp(lgd.get_title(),fontsize='xx-small')
    lgd = ax.legend(loc='lower right', prop={'size': 17},# bbox_to_anchor=(0.5, -.1),
              fancybox=False, shadow=False, ncol=2, numpoints = 1)


    fig.savefig(plot_filename + 'logN', bbox_extra_artists=(lgd,), bbox_inches='tight')    

    plt.close()

    return 


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

        ax.plot(n_vals_mean, run_time_vals_mean, 'yx', label='run_time_vals_mean 2' , markersize=10)
        # ax.plot(all_n_vals, all_run_time_vals, 'r+', label='run_time_vals_mean' , markersize=10)
        ax.plot(n_vals_mean, number_of_times_partition_called_vals_mean, 'mx', label='number_of_times_partition_called_vals_mean 2' , markersize=10)

    # fig.savefig('permanent_estimation_scaling_plots', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    fig.savefig('permanent_estimation_scaling_plots_diagMatrix', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent sampling scaling single gumbel perm UB', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent sampling scaling single gumbel, pick partition order perm UB', bbox_extra_artists=(lgd,), bbox_inches='tight')    
    # fig.savefig('permanent sampling scaling single gumbel, close to 01 matrix perm UB', bbox_extra_artists=(lgd,), bbox_inches='tight')    

    plt.close()


if __name__ == "__main__":
    ITERS = 5

    # plot_runtime_vs_N(pickle_file_paths = ['./neurips_plots/ourUBbound_findBestRowFastTri_number_of_times_partition_called_for_each_n_%diters.pickle' % (ITERS)],
    #                   pickle_file_paths2 =['./neurips_plots/nestingProvedUB_noRowSearchFastTri_number_of_times_partition_called_for_each_n_%diters.pickle' % (ITERS)],
    #                    plot_filename='./neurips_plots/compareNoSearchNestingUB_soules_FastTri_1')
    plot_runtime_vs_N(pickle_file_paths = ['./neurips_plots/ourUBbound_findBestRowFastTri_number_of_times_partition_called_for_each_n_%ditersDiagMatrix.pickle' % (ITERS)],
                      pickle_file_paths2 =['./neurips_plots/nestingProvedUB_noRowSearchFastTri_number_of_times_partition_called_for_each_n_%ditersDiagMatrix.pickle' % (ITERS)],
                       plot_filename='./neurips_plots/compareNoSearchNestingUB_soules_FastTriDiagMatrix_1')


