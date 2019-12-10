import numpy as np
import torch
from gru_cell import GRUCell
from lstm_cell import LSTMCell

EPOCHS = 10  # training epochs
TRAINING_SAMPLES = 10000  # training dataset size
BATCH_SIZE = 16  # mini batch size
TEST_SAMPLES = 1000  # test dataset size
SEQUENCE_LENGTH = 20  # binary sequence length
HIDDEN_UNITS = 20  # hidden units of the LSTM cell


class LSTMModel(torch.nn.Module):
    """LSTM model with a single output layer connected to the lstm cell output"""

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size

        # Our own LSTM implementation
        self.lstm = LSTMCell(input_size, hidden_size)

        # Fully-connected output layer
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Start with empty network output and cell state to initialize the sequence
        c_t = torch.zeros((x.size(0), self.hidden_size)).to(x.device)
        h_t = torch.zeros((x.size(0), self.hidden_size)).to(x.device)

        # Iterate over all sequence elements across all sequences of the mini-batch
        for seq in range(x.size(1)):
            h_t, c_t = self.lstm(x[:, seq, :], (h_t, c_t))

        # Final output layer
        return self.fc(h_t)


class GRUModel(torch.nn.Module):
    """LSTM model with a single output layer connected to the lstm cell output"""

    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size

        # Our own GRU implementation
        self.gru = GRUCell(input_size, hidden_size)

        # Fully-connected output layer
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Start with empty network output and cell state to initialize the sequence
        h_t = torch.zeros((x.size(0), self.hidden_size)).to(x.device)

        # Iterate over all sequence elements across all sequences of the mini-batch
        for seq in range(x.size(1)):
            h_t = self.gru(x[:, seq, :], h_t)

        # Final output layer
        return self.fc(h_t)


def generate_dataset(sequence_length: int, samples: int):
    """
    Generate training/testing datasets
    :param sequence_length: length of the binary sequence
    :param samples: number of samples
    """

    sequences = list()
    labels = list()
    for i in range(samples):
        a = np.random.randint(sequence_length) / sequence_length
        sequence = list(np.random.choice(2, sequence_length, p=[a, 1 - a]))
        sequences.append(sequence)
        labels.append(int(np.sum(sequence)))

    sequences = np.array(sequences)
    labels = np.array(labels, dtype=np.int8)

    result = torch.utils.data.TensorDataset(
        torch.from_numpy(sequences).float().unsqueeze(-1),
        torch.from_numpy(labels).float())

    return result


def train_model(model, loss_function, optimizer, data_loader):
    # set model to training mode
    model.train()

    current_loss = 0.0
    current_acc = 0

    # iterate over the training data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        model.zero_grad()
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # forward
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(outputs.round() == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Train Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))


def test_model(model, loss_function, data_loader):
    # set model in evaluation mode
    model.eval()

    current_loss = 0.0
    current_acc = 0

    # iterate over  the validation data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, labels)

        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(outputs.round() == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Test Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))

    return total_loss, total_acc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="LSTM/GRU count 1s in binary sequence")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-lstm', action='store_true', help="LSTM")
    group.add_argument('-gru', action='store_true', help="GRU")
    args = parser.parse_args()

    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Generate training and testing datasets
    train = generate_dataset(SEQUENCE_LENGTH, TRAINING_SAMPLES)
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

    test = generate_dataset(SEQUENCE_LENGTH, TEST_SAMPLES)
    test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

    # Instantiate LSTM or GRU model
    # input of size 1 for digit of the sequence
    # number of hidden units
    # regression model output size (number of ones)
    if args.lstm:
        model = LSTMModel(input_size=1,
                          hidden_size=HIDDEN_UNITS,
                          output_size=1)
    elif args.gru:
        model = GRUModel(input_size=1,
                         hidden_size=HIDDEN_UNITS,
                         output_size=1)

    # Transfer the model to the GPU
    model = model.to(device)

    # loss function (we use MSELoss because of the regression)
    loss_function = torch.nn.MSELoss()

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Train
    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, EPOCHS))

        train_model(model, loss_function, optimizer, train_loader)
        test_model(model, loss_function, test_loader)
