import scipy.stats as stats

def binomial_hypothesis_test(m_1, n_1, m_0, n_0):
    """Bayesian hypothesis testing with binomial likelihood and beta prior.

    Args:
        m_1 (int): Number of successes in the first group.
        n_1 (int): Number of trials in the first group.
        m_0 (int): Number of successes in the second group.
        n_0 (int): Number of trials in the second group.

    Returns:
        float: p-value (probability of observing such or more extreme data given the null hypothesis).
    
    We model each group as a sequence of Bernoulli trials with unknown success probability.
    We treat the success probability as a random variable p with a beta(1, 1) (uniform) prior.

    The likelihood of the data given the success probability is binomial.
    The posterior distribution of the success probability is beta with parameters alpha = m + 1 and beta = n - m + 1.

    We marginalize over the posterior distribution of the success probability to obtain the predictive distribution of the data.
    The predictive distribution is a beta-binomial distribution.

    The null hypothesis is that the success probability is the same in both groups.
    We calculate the probability of observing such or more extreme data given the null hypothesis.
    If p_1 > p_0 then we calculate the probability of observing m_1 or more successes in n_1 trials
    given m_0 successes in n_0 trials.

    """
    if n_0 == 0 or n_1 == 0:
        print("Number of trials in each group must be greater than 0.")
        return -1

    # Map estimate of p_1 under beta posterior
    p_0 = m_0 / n_0
    p_1 = m_1 / n_1

    # Predictive distribution of the data given the null hypothesis
    pred_dist = stats.betabinom(n=n_1, a=m_0 + 1, b=n_0 - m_0 + 1)

    # Probability of observing such or more extreme data given the null hypothesis
    if p_1 > p_0:
        # -1 because cdf is P(X <= x) and we want P(X >= x)
        p_value = 1 - pred_dist.cdf(m_1 - 1)
    else:
        p_value = pred_dist.cdf(m_1)

    return p_value