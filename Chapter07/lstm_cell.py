import math
import typing

import torch


class LSTMCell(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int):
        """
        :param input_size: input vector size
        :param hidden_size: cell state vector size
        """

        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # combine all gates in a single matrix multiplication
        self.x_fc = torch.nn.Linear(input_size, 4 * hidden_size)
        self.h_fc = torch.nn.Linear(hidden_size, 4 * hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        """Xavier initialization """
        size = math.sqrt(3.0 / self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-size, size)

    def forward(self,
                x_t: torch.Tensor,
                hidden: typing.Tuple[torch.Tensor, torch.Tensor] = (None, None)) \
            -> typing.Tuple[torch.Tensor, torch.Tensor]:
        h_t_1, c_t_1 = hidden  # t_1 is equivalent to t-1

        # in case of more than 2-dimensional input
        # flatten the tensor (similar to numpy.reshape)
        x_t = x_t.view(-1, x_t.size(1))
        h_t_1 = h_t_1.view(-1, h_t_1.size(1))
        c_t_1 = c_t_1.view(-1, c_t_1.size(1))

        # compute the activations of all gates simultaneously
        gates = self.x_fc(x_t) + self.h_fc(h_t_1)

        # split the input to the 4 separate gates
        i_t, f_t, candidate_c_t, o_t = gates.chunk(4, 1)

        # compute the activations for all gates
        i_t, f_t, candidate_c_t, o_t = \
            i_t.sigmoid(), f_t.sigmoid(), candidate_c_t.tanh(), o_t.sigmoid()

        # choose new state based on the input and forget gates
        c_t = torch.mul(f_t, c_t_1) + torch.mul(i_t, candidate_c_t)

        # compute the cell output
        h_t = torch.mul(o_t, c_t.tanh())

        return h_t, c_t
