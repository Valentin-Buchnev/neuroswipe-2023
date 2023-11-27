import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, n_channels=256, n_residual_blocks=3, **kwargs):
        super().__init__(**kwargs)

        self.conv0 = torch.nn.Sequential(
            torch.nn.Conv1d(2, n_channels // 8, 9, padding=4),
            torch.nn.BatchNorm1d(n_channels // 8),
            torch.nn.Conv1d(n_channels // 8, n_channels // 4, 7, padding=3),
            torch.nn.BatchNorm1d(n_channels // 4),
            torch.nn.Conv1d(n_channels // 4, n_channels // 2, 5, padding=2),
            torch.nn.BatchNorm1d(n_channels // 2),
            torch.nn.Conv1d(n_channels // 2, n_channels, 3, padding=1),
            torch.nn.BatchNorm1d(n_channels),
        )
        self.rnn = torch.nn.LSTM(n_channels, n_channels, num_layers=3, batch_first=True, bidirectional=True)

        self.attn = torch.nn.Linear(2 * n_channels, 1)

        self.head = torch.nn.Sequential(
            torch.nn.Linear(2 * n_channels, n_channels),
            torch.nn.Tanh(),
            torch.nn.Linear(n_channels, 2 * 100),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv0(x)
        x, _ = self.rnn(x.transpose(1, 2))
        x_attn = self.attn(x).squeeze(-1)
        x_attn = F.softmax(x_attn, dim=-1)
        x = torch.sum(x_attn.unsqueeze(-1) * x, dim=1)
        x = x.reshape((x.shape[0], -1))
        return self.head(x)
