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


def test_gumbel_MLE_concentration(samples):
    mle_off_by_more_than_point1 = []
    fraction_positive = []
    for idx in range(1000):
        # print "idx =", idx
        gumbel_samples = np.random.gumbel(size=samples)
        mle = -np.log(np.mean(np.exp(-gumbel_samples)))
        print "mle:", mle
        print 'mean:', np.mean(gumbel_samples)
        print 'mean - gamma:', np.mean(gumbel_samples) - np.euler_gamma
        print
        # mean -= np.euler_gamma
        # if abs(mean) > 0.0005624488538291672:
        # if abs(mle) > 0.000000000111179:
        # if abs(mle) > 0.000111179:        
        if abs(mle) > 0.1:        
            mle_off_by_more_than_point1.append(1)
        else:
            mle_off_by_more_than_point1.append(0)
        if mle > 0:
            fraction_positive.append(1)
        else:
            fraction_positive.append(0)
    # print "gumbel mean =", mean
    print "fraction mle_off_by_more_than_point1 (for MLE) =", np.mean(mle_off_by_more_than_point1)
    print "fraction mle larger than 0 =", np.mean(fraction_positive)


def test_binomial_concentration(samples, p=.5):
    mle_off_by_more_than_1point1 = []
    for idx in range(1000):
        fraction = np.random.binomial(n=samples, p=p)/samples
        # print "fraction/p =", fraction/p
        if fraction/p > 1.1 or fraction/p < .9:
            mle_off_by_more_than_1point1.append(1)
        else:
            mle_off_by_more_than_1point1.append(0)

    print "binomial fration mle_off_by_more_than_1point1 =", np.mean(mle_off_by_more_than_1point1)


if __name__ == "__main__":
    # test_gumbel_mean_concentration(100000)
    test_gumbel_MLE_concentration(400)
    test_binomial_concentration(400, p=.1)