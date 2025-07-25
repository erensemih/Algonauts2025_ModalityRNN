import torch
import torch.nn as nn
from typing import List


class LSTMRegressor(nn.Module):
    """
    Configurable RNN-based model with separate modality-specific LSTMs,
    a post-aggregation RNN (LSTM or GRU), and multiple parallel regression heads.

    Args:
        modality_dims (List[int]): list of input feature dims per modality
        hidden_size (int): hidden dimension of each modality LSTM
        num_layers (int): number of stacked layers in each modality LSTM
        post_hidden (int): hidden size of the post-aggregation RNN
        post_layers (int): number of layers in the post-aggregation RNN
        out_features (int): number of output channels
        dropout (float): dropout probability between layers
        modality_bidirectional (bool): whether to use bidirectional modality LSTMs
        post_bidirectional (bool): whether to use bidirectional post RNN
        model_type (str): 'lstm_lstm' to use LSTM for post-RNN,
                          'lstm_gru' to use GRU for post-RNN
    """
    def __init__(
        self,
        modality_dims: List[int],
        hidden_size: int,
        num_layers: int,
        post_hidden: int,
        post_layers: int,
        out_features: int,
        dropout: float = 0.0,
        modality_bidirectional: bool = True,
        post_bidirectional: bool = True,
        model_type: str = 'lstm_lstm',
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.modality_bidirectional = modality_bidirectional
        self.post_bidirectional = post_bidirectional
        self.model_type = model_type.lower()

        # modality-specific LSTMs
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=d,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=modality_bidirectional
            ) for d in modality_dims
        ])

        # compute LSTM output dim
        num_dirs_mod = 2 if modality_bidirectional else 1
        lstm_output_dim = hidden_size * num_dirs_mod

        # choose post-RNN type
        RNN = nn.LSTM if self.model_type == 'lstm_lstm' else nn.GRU

        # post-aggregation RNN
        self.post_rnn = RNN(
            input_size=lstm_output_dim,
            hidden_size=post_hidden,
            num_layers=post_layers,
            batch_first=True,
            dropout=dropout if post_layers > 1 else 0.0,
            bidirectional=post_bidirectional
        )

        # compute post output dim
        num_dirs_post = 2 if post_bidirectional else 1
        head_input_dim = post_hidden * num_dirs_post

        # define four parallel regression heads
        self.heads = nn.ModuleList([
            nn.Linear(head_input_dim, out_features)
            for _ in range(4)
        ])

    def forward(self, x: torch.Tensor, subject):
        """
        x:      (batch, F, T)
        subject: int or sequence/tensor of length batch, values in {1,2,3,5}
        returns: (batch, out_features, T)
        """
        # prepare subject indices 1->0,2->1,3->2,5->3
        if isinstance(subject, (list, tuple)):
            subject = torch.tensor(subject, device=x.device)
        elif torch.is_tensor(subject):
            subject = subject.to(x.device)
        else:
            subject = torch.tensor([subject], device=x.device)
        subject = subject.view(-1)
        subject = torch.where(subject == 5, subject - 1, subject)
        subject = subject - 1

        # time-first for RNN (batch, T, features)
        x = x.permute(0, 2, 1)

        # split and encode each modality
        chunks = torch.split(x, self.modality_dims, dim=2)
        lstm_outs = [l(chunk)[0] for l, chunk in zip(self.lstm_layers, chunks)]
        avg_out = torch.mean(torch.stack(lstm_outs, dim=0), dim=0)

        # post-aggregation RNN
        post_out, _ = self.post_rnn(avg_out)

        # apply all heads: stack as (batch, 4, T, out_features)
        head_stack = torch.stack([h(post_out) for h in self.heads], dim=1)

        # select per-sample head
        batch_idx = torch.arange(x.size(0), device=x.device)
        sel = head_stack[batch_idx, subject]

        # return as (batch, out_features, T)
        return sel.permute(0, 2, 1)
