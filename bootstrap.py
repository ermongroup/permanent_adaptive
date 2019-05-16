from __future__ import division
import numpy as np

def test_gumbel_mean_concentration(samples):
    mean_off_by_more_than_point1 = []
    for idx in range(100):
        # print "idx =", idx
        gumbel_samples = np.random.gumbel(size=samples)
        mean = np.mean(gumbel_samples)
        mean -= np.euler_gamma
        # if abs(mean) > 0.0005624488538291672:
        # if abs(mean) > 0.000000000111179:
        if abs(mean) > 0.002:            
            mean_off_by_more_than_point1.append(1)
        else:
            mean_off_by_more_than_point1.append(0)
    # print "gumbel mean =", mean
    print "fraction mean_off_by_more_than_point1 (for mean)=", np.mean(mean_off_by_more_than_point1)


def test_gumbel_MLE_concentration(num_gumbel_samples, additive_log_error=.1):
    mle_off_by_more_than_point1 = []
    fraction_positive = []
    for idx in range(100000):
        # print "idx =", idx
        gumbel_samples = np.random.gumbel(size=num_gumbel_samples)
        mle_mu_hat = -np.log(np.mean(np.exp(-gumbel_samples)))
        # print "mle_mu_hat:", mle_mu_hat
        # print 'mean:', np.mean(gumbel_samples)
        # print 'mean - gamma:', np.mean(gumbel_samples) - np.euler_gamma
        # print
        # mean -= np.euler_gamma
        # if abs(mean) > 0.0005624488538291672:
        # if abs(mle_mu_hat) > 0.000000000111179:
        # if abs(mle_mu_hat) > 0.000111179:        
        if abs(mle_mu_hat) > additive_log_error:        
            mle_off_by_more_than_point1.append(1)
        else:
            mle_off_by_more_than_point1.append(0)
        if mle_mu_hat > 0:
            fraction_positive.append(1)
        else:
            fraction_positive.append(0)
    # print "gumbel mean =", mean
    print "fraction mle_off_by_more_than %f (for MLE) =" % additive_log_error, np.mean(mle_off_by_more_than_point1)
    # print "fraction mle larger than 0 =", np.mean(fraction_positive)

    # print "closed form, 2 gumbels =", 1 - .5*np.exp(-2*np.exp(-0.1))

def test_binomial_concentration(num_gumbel_samples, p=.5, additive_log_error=.1):
    binomial_off_by_more_than_1point1 = []
    accepted_samples_list = []
    for idx in range(100000):
        accepted_samples = 0
        total_samples = 0
        while accepted_samples < num_gumbel_samples:
            cur_num_samples = int((num_gumbel_samples - accepted_samples)/p)
            total_samples += cur_num_samples
            # print "total_samples:", total_samples
            accepted_samples += np.random.binomial(n=cur_num_samples, p=p)
        accepted_samples_list.append(accepted_samples)
        p_hat = accepted_samples/total_samples
        # binom_samples = (samples/p)
        # fraction = np.random.binomial(n=samples, p=p)/samples
        # p_hat = np.random.binomial(n=binom_samples, p=p)/binom_samples
        # print "p_hat/p =", p_hat/p        
        # if p_hat/p > 1.1 or p_hat/p < .9:
        # if p_hat/p > np.exp(additive_log_error) or p_hat/p < np.exp(-additive_log_error):
        if np.abs(np.log(p_hat/p)) > additive_log_error:
            binomial_off_by_more_than_1point1.append(1)
        else:
            binomial_off_by_more_than_1point1.append(0)

    print "np.mean(accepted_samples_list):", np.mean(accepted_samples_list)
    print "binomial fraction of log estiamtes off by more than %f =" % additive_log_error, np.mean(binomial_off_by_more_than_1point1)


def binomial_concentration_chernoff(p, n, additive_log_error=.1):
    '''
    return chernoff bound on additive log estimate being off by more than additive_log_error for n trials of bernoulli with success p
    '''
    mu = p*n
    epsilon = additive_log_error

    delta_plus = np.exp(epsilon) - 1
    assert(delta_plus > 0)
    upper_tail_prob = np.exp(-mu*(delta_plus**2)/(2 + delta_plus))

    delta_minus = 1 - np.exp(-epsilon)
    assert(delta_minus > 0 and delta_minus < 1)
    lower_tail_prob = np.exp(-mu*(delta_minus**2)/2)

    return (upper_tail_prob + lower_tail_prob)

def binomial_concentration_hoeffding(p, n, additive_log_error=.1):
    '''
    return chernoff bound on additive log estimate being off by more than additive_log_error for n trials of bernoulli with success p
    '''
    epsilon = additive_log_error

    epsilon_minus = p*(1 - np.exp(-epsilon))
    lower_tail_prob = np.exp(-2*n*epsilon_minus**2)


    epsilon_plus = p*(np.exp(-epsilon) - 1)
    upper_tail_prob = np.exp(-2*n*epsilon_plus**2)

    return (upper_tail_prob + lower_tail_prob)    

if __name__ == "__main__":
    # test_gumbel_mean_concentration(100000)

    num_gumbel_samples = 1000
    additive_log_error = .1
    p=.000000000000001
    test_gumbel_MLE_concentration(num_gumbel_samples=num_gumbel_samples, additive_log_error=additive_log_error)

    test_binomial_concentration(num_gumbel_samples=num_gumbel_samples, p=p, additive_log_error=additive_log_error)

    print "chernoff bound on probability of log estimate off by more than %f =" % additive_log_error, binomial_concentration_chernoff(p=p, n=num_gumbel_samples/p, additive_log_error=additive_log_error)
    print "hoeffding bound on probability of log estimate off by more than %f =" % additive_log_error, binomial_concentration_hoeffding(p=p, n=num_gumbel_samples/p, additive_log_error=additive_log_error)

