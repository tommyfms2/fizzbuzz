
import argparse
from chainer import Chain, Variable, optimizers
import chainer.links as L
import chainer.functions as F
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', type=int, default=100)
parser.add_argument('-n', '--max-num', type=int, default=100)
args = parser.parse_args()

binary_size = len(bin(args.max_num)) - 2
in_num = binary_size
hidden_num = 10
out_num = 2

model = Chain(l1 = L.Linear(in_num, hidden_num),
              l2 = L.Linear(hidden_num, hidden_num),
              l3 = L.Linear(hidden_num, out_num))


def to_input_binary(n, binary_size):
    fmt = '{0:%sb}' % (binary_size)
    res = list(fmt.format(n))
    res = [0 if x == ' ' else float(x) for x in res]
    return res

def to_output_binary(n):
    if n % 15 == 0:
        return [0, 0]
    elif n % 5 == 0:
        return [0, 1]
    elif n % 3 == 0:
        return [1, 0]
    else:
        return [1, 1]

def to_str(i, pred):
    pred = list(map(lambda x: int(round(x)), pred))
    if pred == [0, 0]:
        return 'FizzBuzz'
    elif pred == [0, 1]:
        return 'Buzz'
    elif pred == [1, 0]:
        return 'Fizz'
    else:
        return str(i)

def train(model, optimizer, epoch, max_num, binary_size):
    for _ in range(epoch):
        nums = list(range(1, max_num))
        random.shuffle(nums)
        for n in nums:
            x = Variable(np.asarray([to_input_binary(n, binary_size)], np.float32))
            t = Variable(np.asarray([to_output_binary(n)], np.float32))
            
            optimizer.zero_grads()
            h = F.relu(model.l1(x))
            h = F.relu(model.l2(h))
            y = model.l3(h)
            loss = F.mean_squared_error(y,t)
            loss.backward()
            optimizer.update()
            

def predict(model, num, binary_size):
    x = Variable(np.asarray([to_input_binary(num, binary_size)], np.float32))
    h = F.relu(model.l1(x))
    h = F.relu(model.l2(h))
    y = model.l3(h)
    return to_str(num, y.data[0])



optimizer = optimizers.Adam()
optimizer.setup(model)

train(model, optimizer, args.epoch, args.max_num, binary_size)

for i in range(1, args.max_num):
    pred = predict(model, i, binary_size)
    print(pred)
