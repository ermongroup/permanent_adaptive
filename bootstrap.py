import numpy as np

def test_gumbel_mean_concentration(samples):
    mean_off_by_more_than_point1 = []
    for idx in range(10000):
        # print "idx =", idx
        gumbel_samples = np.random.gumbel(size=samples)
        mean = np.mean(gumbel_samples)
        mean -= np.euler_gamma
        # if abs(mean) > 0.0005624488538291672:
        if abs(mean) > 0.4:
            mean_off_by_more_than_point1.append(1)
        else:
            mean_off_by_more_than_point1.append(0)
    # print "gumbel mean =", mean
    print "fraction mean_off_by_more_than_point1 (for mean)=", np.mean(mean_off_by_more_than_point1)


def test_gumbel_MLE_concentration(samples):
    mle_off_by_more_than_point1 = []
    for idx in range(10000):
        # print "idx =", idx
        gumbel_samples = np.random.gumbel(size=samples)
        mle = -np.log(np.mean(np.exp(-gumbel_samples)))
        # mean -= np.euler_gamma
        # if abs(mean) > 0.0005624488538291672:
        if abs(mle) > 0.4:
            mle_off_by_more_than_point1.append(1)
        else:
            mle_off_by_more_than_point1.append(0)
    # print "gumbel mean =", mean
    print "fraction mle_off_by_more_than_point1 (for MLE) =", np.mean(mle_off_by_more_than_point1)


if __name__ == "__main__":
    test_gumbel_mean_concentration(50)
    test_gumbel_MLE_concentration(33)