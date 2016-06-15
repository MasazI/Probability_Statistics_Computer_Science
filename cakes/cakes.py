# encoding:utf-8

import random
import matplotlib.pyplot as plt
from collections import defaultdict

def carve():
    knife = (random.random(), random.random())
    y_brothers = max(knife) - min(knife)
    o_brothres = 1 - y_brothers
    return round(y_brothers, 1)

def experiment():
    dict = defaultdict(int)
    for i in xrange(10000):
        result = carve()
        previous = dict[result]
        dict[result] = previous + 1  

    X = []
    Y = []
    for key, value in dict.iteritems():
        print key
        print value
        X.append(key)
        Y.append(value)

    plt.bar(X, Y)
    plt.show()

if __name__ == '__main__':
    experiment()
