import torch
import numpy as np


def make_training_and_testing_set(data, percent_train=90.):
    num_train = int(float(percent_train) * data.shape[0] / 100)
    num_test = data.shape[0] - num_train

    _train_x = torch.from_numpy(data[num_test:, :-1])
    _train_y = torch.from_numpy(data[num_test:, 1:])
    _test_x = torch.from_numpy(data[:num_test, :-1])
    _test_y = torch.from_numpy(data[:num_test, 1:])
    return _train_x, _train_y, _test_x, _test_y
