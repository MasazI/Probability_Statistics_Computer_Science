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