#encoding: utf-8

import matplotlib.pyplot as plt
from numpy.random import binomial
from collections import defaultdict

# ベルヌーイ分布に従うサンプルを1000個出力(0.1の確率で1を出力)
samples = binomial(n=1, p=0.1, size=1000)

print samples

# 1を交通事故(accident)とみなした時、事故が1度おきるとその後続けて起きると言われるが本当かどうか？
term = 0
dict = defaultdict(int)
for sample in samples:
    if sample == 1:
        previous = dict[term]
        dict[term] = previous + 1
        term = 0
    else:
        term += 1

# 事故の間隔のヒストグラムを表示
X = []
Y = []
for key, value in dict.iteritems():
    X.append(key)
    Y.append(value)

plt.bar(X, Y)
plt.show()
