# encoding: utf-8
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp


import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
plotBlue = sns.color_palette()[0]

np.random.seed(3)

######### NO. 1 single nomal distribution

N = 1000
X1 = np.random.normal(4, 12, N)
f, axes = plt.subplots(nrows=2, sharex=True)
axes[0].set_xlim(-50, 50)
axes[0].scatter(X1, np.zeros(N), marker='x', c=plotBlue)
axes[1].hist(X1, bins=50)
plt.show()




######### NO. 2 take mean and standard deviation

sample_mean = X1.mean()
sample_sigma = X1.std()

print("Sample Mean: %f", sample_mean)
print("Sample Standard Deviation: %f", sample_sigma)


########## NO.3 estimate for the distribution from mean and standard deviation
base = np.linspace(-50, 50, 100)
normal = sp.stats.norm.pdf(base, sample_mean, sample_sigma)
lower_bound = sample_mean - (2.58 * sample_sigma)
upper_bound = sample_mean + (2.58 * sample_sigma)
anomalous = np.logical_or(base < [lower_bound]*100, base > [upper_bound]*100)

plt.plot(base, normal)
plt.fill_between(base, normal, where=anomalous, color=[1, 0, 0, 0.4])
plt.xlim(-50, 50)
plt.show()
print('Lower Bound:', lower_bound)
print('Upper Bound:', upper_bound)


########### NO.4 see if they're anomalous
plt.scatter(X1, np.zeros(N), marker='x', c=plotBlue)
plt.xlim(-50, 50)
plt.scatter(-29, 0, marker='x', color='red', s=150, linewidths=3)
plt.scatter(17, 0, marker='x', color='green', s=150, linewidths=3)
plt.axvline(lower_bound, ymin=.25, ymax=.75, color='red', linewidth=1)
plt.axvline(upper_bound, ymin=.25, ymax=.75, color='red', linewidth=1)
plt.show()


########### NO.5 no normaly distribution
X2 = X1[X1 > 0]
plt.hist(X2, bins=30)
plt.xlim(-50, 50)
plt.show()


########### NO.6 try to model as normal distribution
sample_mean = X2.mean()
sample_sigma = X2.std()
base = np.linspace(-50, 50, 100)
normal = sp.stats.norm.pdf(base, sample_mean, sample_sigma)
lower_bound = sample_mean - (2.58 * sample_sigma)
upper_bound = sample_mean + (2.58 * sample_sigma)
anomalous = np.logical_or(base < [lower_bound]*100, base > [upper_bound]*100)

plt.hist(X2, bins=30, normed=True, zorder=1)
plt.fill_between(base, normal, where=anomalous, color=[1, 0, 0, 0.4], zorder=2)
plt.plot(base, normal, color='black', zorder=3)
plt.xlim(-50, 50)
plt.show()
print('Lower Bound:', lower_bound)
print('Upper Bound:', upper_bound)


############# NO.7 transfrom toroughly normal
X3 = X2 ** 0.55
plt.hist(X3, bins=30)
plt.show()


############# NO.8 multi variable (independent)
N = 1000
X1 = np.random.normal(4, 12, N)
X2 = np.random.normal(9, 5, N)
plt.scatter(X1, X2, c=plotBlue)
plt.show()

############# NO.9 multi variable predicted
x1_sample_mean = X1.mean()
x2_sample_mean = X2.mean()
x1_sample_sigma = X1.std()
x2_sample_sigma = X2.std()
print('Sample Mean 1:', x1_sample_mean)
print('Sample Mean 2:', x2_sample_mean)
print('Sample Standard Deviation 1:', x1_sample_sigma)
print('Sample Standard Deviation 2:', x2_sample_sigma)

############# NO.9 multi variable heat map of expectation
delta = 0.025
x1 = np.arange(-40, 50, delta)
x2 = np.arange(-40, 50, delta)
x, y = np.meshgrid(x1, x2)
z = plt.mlab.bivariate_normal(x, y, x1_sample_sigma, x2_sample_sigma, x1_sample_mean, x2_sample_mean)
plt.contourf(x, y, z, cmap='Blues_r')
thinned_points = np.array([n in np.random.choice(N, 300) for n in range(N)])
plt.scatter(X1[thinned_points], X2[thinned_points], c='gray')
plt.show()


############ NO.10 joing propability
def positive_support_normal(mean, sigma, n):
    xs = np.random.normal(mean, sigma, n)
    for i, num in enumerate(xs):
        while num < 0:
            num = np.random.normal(mean[i], sigma)
        xs[i] = num
    return xs

N = 1000

mu_cons = 10
sigma_cons = 6
sigma_latency = 20
beta = 3

cons = positive_support_normal(np.array([mu_cons]*N), sigma_cons, N)
latency = positive_support_normal(beta * cons, sigma_latency, N)
ax = sns.jointplot('cons', 'latency', pd.DataFrame({'cons': cons, 'latency': latency}))

############## NO.11 stan
import pystan

model_code = """
    data {
        int<lower=0> N;             // number of observations
        vector<lower=0>[N] latency; // observed latency
        vector<lower=0>[N] cons;    // observed concurrent connections

        real latency_test;          // test for event probabilities
        real cons_test;             // test for event probabilities
    }

    parameters {
        real mu_cons;               // mean of concurrent connections
        real sigma_cons;            // sigma of concurrent connections
        real sigma_latency;         // sigma of latencies
        real beta;                  // linear coefficient for connections in latency
    }

    model {
        // declare some weak prior beliefs about our parameters
        mu_cons ~ normal(100, 100);
        sigma_cons ~ normal(100, 100);
        sigma_latency ~ normal(100, 100);
        beta ~ uniform(-100, 100);

        // both these distributions are truncated below at 0.0
        for (i in 1:N) {
            cons[i] ~ normal(mu_cons, sigma_cons) T[0.0,];

            // latency is linearly related to cons
            latency[i] ~ normal(beta * cons[i], sigma_latency) T[0.0,];
        }
    }

    generated quantities {
        real prob_cons;
        real prob_latency;
        real prob;

        prob_cons <- 2 * (.5 - abs(.5 - normal_cdf(cons_test, mu_cons, sigma_cons)));
        prob_latency <- 2 * (.5 - abs(.5 - normal_cdf(latency_test, beta * cons_test, sigma_latency)));
        prob <- prob_cons * prob_latency;
    }
"""

data = {
    'N': N,
    'latency': latency,
    'cons': cons,
    'latency_test': 3.,
    'cons_test': 3.,
}

fit = pystan.stan(model_code=model_code, data=data)
fit.plot(['mu_cons', 'sigma_cons', 'beta', 'sigma_latency'])
print('The left column of graphs represent the posterior distributions of parameters.')
print('The right column of graphs plot the samples drawn from those distributions.')
plt.show()
