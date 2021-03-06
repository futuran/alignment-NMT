"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.position_ffn import ActivationFunction
from onmt.utils.misc import sequence_mask


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0,
                 pos_ffn_activation_fn=ActivationFunction.relu):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout,
                                                    pos_ffn_activation_fn)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, adj, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=adj, attn_type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions,
                 pos_ffn_activation_fn=ActivationFunction.relu):
        super(TransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions,
                pos_ffn_activation_fn=pos_ffn_activation_fn)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
            pos_ffn_activation_fn=opt.pos_ffn_activation_fn,
        )

    # 20211005 tamura
    def forward(self, src, adj, sep_id, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        # 20211005 tamura

        device = src.device

        batch_size = src.shape[1]
        adj_size = lengths[0]

        coo = adj[:,1:].T.clone().detach()   # 2 * num_of_coo
        coo[0,:] += adj[:,0] * adj_size
        coo = torch.sparse_coo_tensor(coo, torch.ones(coo.shape[1], device=device), (batch_size*adj_size,adj_size)).to(device)
        true_adj = coo.to_dense().view(batch_size, adj_size, adj_size)

        # a,b : adj_size * batch_size * 1
        a = torch.where(src == sep_id, torch.ones_like(src), torch.zeros_like(src))
        b = torch.arange(0, adj_size).repeat(1, batch_size, 1).T.to(device)
        c = torch.sum((a*b).view(-1,batch_size), dim=0)

        #def eye_like(tensor):
        #    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))

        #true_adj = torch.zeros_like(true_adj).to(device)
        #true_adj = eye_like(true_adj).to(device)

        for i, sl in enumerate(c):
            true_adj[i][:sl,:sl] = torch.ones((sl,sl))
            true_adj[i][sl,:] = torch.ones((1,adj_size))
            true_adj[i][:,sl] = torch.ones(adj_size)
            #true_adj[i][sl+1:,sl+1:] = torch.ones((adj_size - sl - 1,adj_size - sl - 1))

            for j in range(adj_size):
                true_adj[i][j,j] = 1

        true_adj = (1-true_adj).to(torch.bool).to(device)

        # 20211005 tamura end

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        mask = ~sequence_mask(lengths).unsqueeze(1)
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            # 20211007 tamura
            out = layer(out, true_adj, mask)
            # 20211007 tamura end
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
