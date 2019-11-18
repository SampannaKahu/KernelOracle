import torch
import torch.nn as nn
import utils

device = utils.DEVICE

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


class Sequence2(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(Sequence2, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.lstm1 = nn.LSTMCell(self.in_dim, self.hidden_dim)
        self.lstm2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, input_tensor, future=0):
        outputs = []
        h_t = torch.zeros(input_tensor.size(0), self.hidden_dim, dtype=torch.double).to(device)
        c_t = torch.zeros(input_tensor.size(0), self.hidden_dim, dtype=torch.double).to(device)
        h_t2 = torch.zeros(input_tensor.size(0), self.hidden_dim, dtype=torch.double).to(device)
        c_t2 = torch.zeros(input_tensor.size(0), self.hidden_dim, dtype=torch.double).to(device)

        # for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
        # - Use this -------> torch.chunk(input_tensor, input_tensor.size(0), dim=0)[0].size()

        # The batch size needs to be smaller because the GPU can't store all the outputs for all the batches.
        chunks = torch.chunk(input_tensor, input_tensor.size(1), dim=1)
        for i, input_t in enumerate(chunks):
            input_t = torch.reshape(input_t, (input_t.size()[0], input_t.size()[2]))
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            # output = torch.reshape(output, (input_t.size()[0], input_t.size()[2]))
            outputs += [output]
        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs
