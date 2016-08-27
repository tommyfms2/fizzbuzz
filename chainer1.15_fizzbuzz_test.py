
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
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
            l3 = L.Linear(n_unit, n_unit),
            l4 = L.Linear(n_unit, n_unit),
            l5 = L.Linear(n_unit, n_out),
            )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h1))
        h4 = F.relu(self.l4(h1))
        return self.l5(h2)

def to_input_binary(n, binary_size):
    fmt = '{0:%sb}' % (binary_size)
    res = list(fmt.format(n))
    res = [0 if x== ' ' else float(x) for x in res]
    return res

def to_output_binary(n):
    if n % 15 == 0:
        return 0
    elif n % 5 == 0:
        return 1
    elif n % 3 == 0:
        return 2
    else:
        return 3

def to_str(i, pred):
    if pred == 0:
        return 'FizzBuzz'
    elif pred == 1:
        return 'Buzz'
    elif pred == 2:
        return 'Fizz'
    else:
        return str(i)


def main():
    parse = argparse.ArgumentParser(description='Chainer Tom example: pre mode')
    parse.add_argument('--batchsize','-b',type=int, default=100,
                       help='Number of images in each mini batch')
    parse.add_argument('--maxnum','-n',type=int, default=100,
                       help="Number of max num")
    parse.add_argument('--epoch','-e',type=int, default=20,
                       help='Number of sweeps over the dataset to train')
    parse.add_argument('--gpu','-g',type=int, default=-1,
                       help='GPU ID(negative value indicates CPU)')
    parse.add_argument('--out','-o', default='result',
                       help='Directory to output the result')
    parse.add_argument('--resume','-r',default='',
                       help='Resume the training from snapshot')
    parse.add_argument('--unit','-u',type=int, default=1000,
                       help='Number of units')
    args = parse.parse_args()
    
    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    binary_size = len(bin(args.maxnum))-2
    model = L.Classifier(MLP(binary_size, args.unit, 4))
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    nums = list(range(1, args.maxnum))
    random.shuffle(nums)
    nptrain_data = np.ones( (args.maxnum, binary_size) )
    nptrain_label_data = np.ones(args.maxnum)
    i = 0
    for n in nums:
        nptrain_data[i] = to_input_binary(n, binary_size)
        nptrain_label_data[i] = to_output_binary(n)
        i = i + 1

    nums_test = list(range(1, args.maxnum))
    random.shuffle(nums_test)
    nptest_data = np.ones( (args.maxnum, binary_size) )
    nptest_label_data = np.ones(args.maxnum)
    i = 0
    for n in nums_test:
        nptest_data[i] = to_input_binary(n, binary_size)
        nptest_label_data[i] = to_output_binary(n)
        i = i + 1

    # how come this isnt work?
    #nptrain_data = nptrain_data.astype(np.float32)/255
    nptrain_data = nptrain_data.astype(np.float32)
    nptrain_label_data = nptrain_label_data.astype(np.int32)
    #nptest_data = nptest_data.astype(np.float32)/255
    nptest_data = nptest_data.astype(np.float32)
    nptest_label_data = nptest_label_data.astype(np.int32)
    train = tuple_dataset.TupleDataset(nptrain_data, nptrain_label_data)
    test  = tuple_dataset.TupleDataset(nptest_data, nptest_label_data)

    print(train[0])
    # this is original
    #train2, test2 = chainer.datasets.get_mnist()
    

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter  = chainer.iterators.SerialIterator(test,  args.batchsize,
                                          repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    trainer.extend(extensions.dump_graph('main/loss'))
    
    trainer.extend(extensions.snapshot())
    
    trainer.extend(extensions.LogReport())
    
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    
    trainer.extend(extensions.ProgressBar())

    trainer.run()

if __name__ == '__main__':
    main()













