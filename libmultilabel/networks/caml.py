import os
from math import floor

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from ..networks.base import BaseModel


class CAML(BaseModel):
    def __init__(self, config, embed_vecs):
        super(CAML, self).__init__(config, embed_vecs)

        emb_dim = embed_vecs.shape[1]

        self.convs = nn.ModuleList()
        for filter_size in config.filter_sizes:
            conv = nn.Conv1d(
                in_channels=emb_dim,
                out_channels=config.num_filter_per_size,
                kernel_size=filter_size,
                padding=(filter_size // 2),
            )
            xavier_uniform_(conv.weight)
            self.convs.append(conv)
        total_conv_size = len(self.convs) * config.num_filter_per_size

        # Context vectors for computing attention as in 2.2
        self.U = nn.Linear(total_conv_size, config.num_classes)
        xavier_uniform_(self.U.weight)

        # Final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(total_conv_size, config.num_classes)
        xavier_uniform_(self.final.weight)

    def forward(self, text):
        # Get embeddings and apply dropout
        h = self.embedding(text)
        h = self.embed_drop(h)
        h = h.transpose(1,2)

        h_list = []
        for conv in self.convs:
            h_sub = conv(h) # (batch_size, num_filter, length)
            h_list.append(h_sub)

        if len(self.convs) > 1:
            h = torch.cat(h_list, 1)
        else:
            h = h_list[0]
        # Max-pooling and monotonely increasing non-linearities commute. Here
        # we apply the activation function after max-pooling for better
        # efficiency.
        h = self.activation(h) # (batch_size, tot_num_filter, length)

        # Apply attention
        alpha = torch.softmax(self.U.weight.matmul(h), dim=2) # (batch_size, num_class, length)

        # Document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(h.transpose(1, 2)) # (batch_size, num_class, tot_num_filter)

        # similarity
        x = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        return {'logits': x, 'attention': alpha}
