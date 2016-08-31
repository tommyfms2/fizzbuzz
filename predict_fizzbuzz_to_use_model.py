
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.serializers
from chainer.datasets import tuple_dataset
from chainer import Chain, Variable, optimizers
from chainer import training
from chainer.training import extensions
import sys
import random


class MLP(Chain):
    def __init__(self, n_in, n_unit, n_out):
        super(MLP, self).__init__(
            l1 = L.Linear(n_in, n_unit),
            l2 = L.Linear(n_unit, n_unit),
            l3 = L.Linear(n_unit, n_out),
            )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

    def predict(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

def to_input_binary(n, binary_size):
    fmt = '{0:%sb}' % (binary_size)
    res = list(fmt.format(n))
    res = [0 if x== ' ' else float(x) for x in res]
    return res

def predict(model, num, binary_size):
    print(num)
    test_data = to_input_binary( num, binary_size )
    print(test_data)
    x = Variable(np.asarray([to_input_binary(num, binary_size)], np.float32))
    x.data[0] = x.data[0]/255
    print(x.data[0])
    y = model.predictor(x)
    print(y.data[0])
    #return to_str(num, y.data[0])

def main():
    parse = argparse.ArgumentParser(description='Chainer model test')
    parse.add_argument('--model','-m', default='my.model')
    parse.add_argument('--optimizer','-o', default='my.state')
    parse.add_argument('--maxnum','-n', type=int, default=100)
    parse.add_argument('--unit','-u',type=int, default=1000)
    args = parse.parse_args()

    binary_size = len(bin(args.maxnum))-2
    model = L.Classifier(MLP(binary_size, args.unit, 4))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    chainer.serializers.load_npz(args.model, model)
    chainer.serializers.load_npz(args.optimizer, optimizer)


    for i in range(1, args.maxnum):
        pred = predict(model, i, binary_size)
        print(pred)
    
    

if __name__ == '__main__':
    main()
