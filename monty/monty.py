#encoding: utf-8
'''
monty
'''

import numpy as np
import matplotlib.pyplot as plt


np.random.seed()
N = 10000
BIN_N = 4

x = np.random.uniform(0.0, 1.0, N)
bins = np.linspace(min(x), max(x), BIN_N)
bin_indices = np.digitize(x, bins)

y = np.random.uniform(0.0, 1.0, N)
user_bins = np.linspace(min(y), max(y), BIN_N)
user_bin_indices = np.digitize(y, user_bins)

#plt.hist(x, bins=bins)
#plt.show()

no_change_correct = 0.0
for correct_index, first_user_index in zip(bin_indices, user_bin_indices):
    if correct_index == first_user_index:
        no_change_correct += 1
print("no change accuracy: %f" % ((no_change_correct / N) * 100.0))


change_correct = 0.0
for correct_index, first_user_index in zip(bin_indices, user_bin_indices):
    if correct_index != first_user_index:
        change_correct += 1
print("change accuracy: %f" % ((change_correct / N) * 100.0))
