#encoding: utf-8
'''
財産をくじで倍にするならどんな風に投資する？
'''

import math
import matplotlib.pyplot as plt
from numpy.random import binomial


P = 0.99
first_experiment = []
for i in xrange(100):
    money = 10000
    # ベルヌーイ分布に従うサンプルを1000個出力(0.7の確率で1を出力)
    samples = binomial(n=1, p=0.7, size=1000)
    for sample in samples:
        if sample == 1:
            money = money * P * 2
        else:
            money = money * (1.0 - P) * 2
    if money != 0:
        first_experiment.append(math.log(money, 10.0))
    else:
        first_experiment.append(money)
plt.bar(range(100), first_experiment)
plt.show()


P = 0.7
first_experiment = []
for i in xrange(100):
    money = 10000
    # ベルヌーイ分布に従うサンプルを1000個出力(0.7の確率で1を出力)
    samples = binomial(n=1, p=0.7, size=1000)
    for sample in samples:
        if sample == 1:
            money = money * P * 2
        else:
            money = money * (1.0 - P) * 2
    if money != 0:
        first_experiment.append(math.log(money, 10.0))
    else:
        first_experiment.append(money)
plt.bar(range(100), first_experiment)
plt.show()