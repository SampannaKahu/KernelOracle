from __future__ import print_function

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot, make_dot_from_trace

import utils
import model

matplotlib.use('Agg')

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
device = utils.DEVICE
torch.cuda.empty_cache()

if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load and preprocess data.
    df = pd.read_csv("data/scheduling_data_out_ab_nginx.csv")
    df = utils.preprocess_data(df=df)
    dataa = df.to_numpy()
    dataa = dataa.reshape(985, 28, -1)  # TODO: Stop hard-coding this.
    dataa = np.transpose(dataa, axes=(0, 2, 1))

    assert not df.isnull().values.any(), "Dataset contains a NaN value. Aborting."

    assert df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).all(), \
        "At least 1 value in the dataframe is non-numeric."

    train_x, train_y, test_x, test_y = utils.make_training_and_testing_set(dataa, percent_train=97.0)
    train_x, train_y, test_x, test_y = train_x.to(device), train_y.to(device), test_x.to(device), test_y.to(device)

    # build the model
    seq = model.Sequence2(in_dim=28, out_dim=28, hidden_dim=51).to(device)
    seq.double()

    criterion = nn.MSELoss().to(device)
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)

    # begin to train
    for i in range(15):
        print('STEP: ', i)


        def closure():
            optimizer.zero_grad()
            out = seq(train_x).to(device)
            _loss = criterion(out, train_y).to(device)
            print('loss:', _loss.item())
            _loss.backward()
            return _loss


        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_x, future=future)
            loss = criterion(pred[:, :-future], test_y)
            print('test loss:', loss.item())
            y = pred.cpu().numpy()
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
