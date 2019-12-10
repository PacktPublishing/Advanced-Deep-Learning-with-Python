import torch


class GRUCell(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int):
        """
        :param input_size: input vector size
        :param hidden_size: cell state vector size
        """

        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # x to reset gate r
        self.x_r_fc = torch.nn.Linear(input_size, hidden_size)

        # x to update gate z
        self.x_z_fc = torch.nn.Linear(input_size, hidden_size)

        # x to candidate state h'(t)
        self.x_h_fc = torch.nn.Linear(input_size, hidden_size)

        # network output/state h(t-1) to reset gate r
        self.h_r_fc = torch.nn.Linear(hidden_size, hidden_size)

        # network output/state h(t-1) to update gate z
        self.h_z_fc = torch.nn.Linear(hidden_size, hidden_size)

        # network state h(t-1) passed through the reset gate r towards candidate state h(t)
        self.hr_h_fc = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self,
                x_t: torch.Tensor,
                h_t_1: torch.Tensor = None) \
            -> torch.Tensor:

        # compute update gate vector
        z_t = torch.sigmoid(self.x_z_fc(x_t) + self.h_z_fc(h_t_1))

        # compute reset gate vector
        r_t = torch.sigmoid(self.x_r_fc(x_t) + self.h_r_fc(h_t_1))

        # compute candidate state
        candidate_h_t = torch.tanh(self.x_h_fc(x_t) + self.hr_h_fc(torch.mul(r_t, h_t_1)))

        # compute cell output
        h_t = torch.mul(z_t, h_t_1) + torch.mul(1 - z_t, candidate_h_t)

        return h_t
