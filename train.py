from __future__ import print_function

import json

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils
import model

print("Package versions:")
print("pandas==%s" % pd.__version__)
print("numpy==%s" % np.__version__)
print("torch==%s" % torch.__version__)
print("matplotlib==%s" % matplotlib.__version__)

print("Package git commit hashes:")
print("pandas==%s" % pd.__git_version__)
print("numpy==%s" % np.__git_revision__)
print("torch==%s" % torch.version.git_version)
print("matplotlib==%s" % json.loads(matplotlib._version.version_json)['full-revisionid'])

seed = 42

if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load data and make training set
    data = torch.load('traindata.pt')
    train_x, train_y, test_x, test_y = utils.make_training_and_testing_set(data, percent_train=97.0)

    # build the model
    seq = model.Sequence()
    seq.double()

    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)

    # begin to train
    for i in range(15):
        print('STEP: ', i)


        def closure():
            optimizer.zero_grad()
            out = seq(train_x)
            loss = criterion(out, train_y)
            print('loss:', loss.item())
            loss.backward()
            return loss


        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_x, future=future)
            loss = criterion(pred[:, :-future], test_y)
            print('test loss:', loss.item())
            y = pred.detach().numpy()
        # draw the result
        plt.figure(figsize=(30, 10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)


        def draw(yi, color):
            plt.plot(np.arange(train_x.size(1)), yi[:train_x.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(train_x.size(1), train_x.size(1) + future), yi[train_x.size(1):], color + ':',
                     linewidth=2.0)


        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf' % i)
        plt.close()
