"""
MIT License

This example is based on https://github.com/harvardnlp/annotated-transformer
Copyright (c) 2018 Alexander Rush
Copyright (c) 2019 Ivan Vasilev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import copy
import math

import numpy as np
import torch


def attention(query, key, value, mask=None, dropout=None):
    """Scaled Dot Product Attention"""
    d_k = query.size(-1)

    # 1) and 2) Compute the alignment scores with scaling
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3) Compute the attention scores (softmax)
    p_attn = torch.nn.functional.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    # 4) Apply the attention scores over the values
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: number of heads
        :param d_model: query/key/value vector length
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        # Create 4 fully connected layers
        # 3 for the query/key/value projections
        # 1 to concatenate the outputs of all heads
        self.fc_layers = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        batch_samples = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        projections = list()
        for l, x in zip(self.fc_layers, (query, key, value)):
            projections.append(
                l(x).view(batch_samples, -1, self.h, self.d_k).transpose(1, 2)
            )

        query, key, value = projections

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value,
                                 mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(batch_samples, -1, self.h * self.d_k)

        return self.fc_layers[-1](x)


def clones(module: torch.nn.Module, n: int):
    """
    Produce N identical copies of module in a ModuleList
    :param module: The module to be copied.
        The module itself is not part of the output module list
     :param n: Number of copies
    """
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class PositionwiseFFN(torch.nn.Module):
    """Implements FFN equation from the paper"""

    def __init__(self, d_model: int, d_ff: int, dropout=0.1):
        super(PositionwiseFFN, self).__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.nn.functional.relu(self.w_1(x))))


class Embeddings(torch.nn.Module):
    """Encoder/Decoder input embeddings"""

    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.lut = torch.nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class SublayerConnection(torch.nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(torch.nn.Module):
    """Construct a layer normalization module (See citation for details)."""

    def __init__(self, features: int, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = torch.nn.Parameter(torch.ones(features))
        self.b_2 = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)],
                                        requires_grad=False)
        return self.dropout(x)


class EncoderBlock(torch.nn.Module):
    """Encoder block with self-attention and residual connections"""

    def __init__(self,
                 size: int,
                 self_attn: MultiHeadedAttention,
                 ffn: PositionwiseFFN,
                 dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attn = self_attn
        self.ffn = ffn

        # Create 2 sub-layer connections
        # 1 for the self-attention
        # 1 for the FFN
        self.sublayers = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Self-attention, followed by FFN + residual connections"""
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayers[1](x, self.ffn)


class Encoder(torch.nn.Module):
    """Transformer encoder with a stack of N blocks"""

    def __init__(self, block: EncoderBlock, N: int):
        super(Encoder, self).__init__()
        self.blocks = clones(block, N)
        self.norm = LayerNorm(block.size)

    def forward(self, x, mask):
        """Iterate over all blocks and normalize"""
        for layer in self.blocks:
            x = layer(x, mask)

        return self.norm(x)


class DecoderBlock(torch.nn.Module):
    """One decoder block, composed of self-attention, encoder-attention, and FFN"""

    def __init__(self,
                 size: int,
                 self_attn: MultiHeadedAttention,
                 encoder_attn: MultiHeadedAttention,
                 ffn: PositionwiseFFN,
                 dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.encoder_attn = encoder_attn
        self.ffn = ffn

        # Create 3 sub-layer connections
        # 1 for the self-attention
        # 1 for the encoder attention
        # 1 for the FFN
        self.sublayers = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, encoder_states, source_mask, target_mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayers[1](x, lambda x: self.encoder_attn(x, encoder_states, encoder_states, source_mask))
        return self.sublayers[2](x, self.ffn)


class Decoder(torch.nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, block: DecoderBlock, N: int, vocab_size: int):
        super(Decoder, self).__init__()
        self.blocks = clones(block, N)
        self.norm = LayerNorm(block.size)
        self.projection = torch.nn.Linear(block.size, vocab_size)

    def forward(self, x, encoder_states, source_mask, target_mask):
        for layer in self.blocks:
            x = layer(x, encoder_states, source_mask, target_mask)

        x = self.norm(x)

        return torch.nn.functional.log_softmax(self.projection(x), dim=-1)


class EncoderDecoder(torch.nn.Module):
    """A Encoder-Decoder architecture"""

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 source_embeddings: torch.nn.Sequential,
                 target_embeddings: torch.nn.Sequential):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embeddings = source_embeddings
        self.target_embeddings = target_embeddings

    def forward(self, source, target, source_mask, target_mask):
        """Take in and process masked src and target sequences."""
        encoder_output = self.encoder(
            x=self.source_embeddings(source),
            mask=source_mask)

        return self.decoder(
            x=self.target_embeddings(target),
            encoder_states=encoder_output,
            source_mask=source_mask,
            target_mask=target_mask)


def build_model(source_vocabulary: int,
                target_vocabulary: int,
                N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Build the full transformer model"""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFFN(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        encoder=Encoder(EncoderBlock(d_model, c(attn), c(ff), dropout), N),
        decoder=Decoder(DecoderBlock(d_model, c(attn), c(attn),
                                     c(ff), dropout), N, target_vocabulary),
        source_embeddings=torch.nn.Sequential(
            Embeddings(d_model, source_vocabulary), c(position)),
        target_embeddings=torch.nn.Sequential(
            Embeddings(d_model, target_vocabulary), c(position)))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    return model


class RandomDataset(torch.utils.data.Dataset):
    """Random data copy dataset"""

    def __init__(self, V, total_samples, sample_length):
        self.samples = list()

        sample = dict()
        for i in range(total_samples):
            data = torch.from_numpy(np.random.randint(1, V, size=(sample_length,)))
            data[0] = 1
            source = torch.autograd.Variable(data, requires_grad=False)
            target = torch.autograd.Variable(data, requires_grad=False)

            sample['source'] = source
            sample['target'] = target[:-1]
            sample['target_y'] = target[1:]
            sample['source_mask'] = (source != 0).unsqueeze(-2)
            sample['target_mask'] = self.make_std_mask(sample['target'], 0)
            sample['tokens_count'] = (sample['target_y'] != 0).data.sum()

            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @staticmethod
    def make_std_mask(target, pad):
        """Create a mask to hide padding and future words."""
        target_mask = (target != pad)
        target_mask = target_mask & torch.autograd.Variable(
            RandomDataset.subsequent_mask(target.size(-1)).type_as(target_mask.data))

        return target_mask

    @staticmethod
    def subsequent_mask(size):
        """Mask out subsequent positions."""
        attn_shape = (size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0


def train_model(model, loss_function, optimizer, data_loader):
    # set model to training mode
    model.train()

    current_loss = 0.0
    counter = 0

    # iterate over the training data
    for i, batch in enumerate(data_loader):
        with torch.set_grad_enabled(True):
            out = model.forward(batch['source'], batch['target'],
                                batch['source_mask'], batch['target_mask'])

            loss = loss_function(out.contiguous().view(-1, out.size(-1)),
                                 batch['target_y'].contiguous().view(-1))

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # statistics
            current_loss += loss
            counter += 1

            if counter % 5 == 0:
                print("Batch: %d; Loss: %f" % (i + 1, current_loss / counter))
                current_loss = 0.0
                counter = 0


if __name__ == '__main__':
    V = 11
    BATCH_SIZE = 50
    train_set = RandomDataset(11, BATCH_SIZE * 1000, 10)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=BATCH_SIZE)

    model = build_model(V, V, N=2)
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = torch.nn.CrossEntropyLoss()

    train_model(model, loss_function, optimizer, train_loader)
