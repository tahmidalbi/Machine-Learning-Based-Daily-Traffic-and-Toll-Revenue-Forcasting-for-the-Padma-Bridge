import torch
import torch.nn as nn


class SequenceRegressor(nn.Module):
    """
    Same architecture framework can create either:

        LSTM
        GRU

    Only the recurrent unit changes.
    """

    def __init__(
        self,
        input_size,
        future_size,
        model_type="LSTM",
        hidden_size=64,
        num_layers=1,
        dropout=0.20,
    ):
        super().__init__()

        self.model_type = model_type.upper()

        if self.model_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )

        elif self.model_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )

        else:
            raise ValueError(
                "model_type must be 'LSTM' or 'GRU'"
            )

        self.sequence_norm = nn.LayerNorm(
            hidden_size
        )

        self.sequence_dropout = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # Optional branch for target-date known information
        # ----------------------------------------------------

        self.future_size = future_size

        if future_size > 0:
            self.future_branch = nn.Sequential(
                nn.Linear(future_size, 16),
                nn.GELU(),
                nn.Dropout(dropout)
            )

            combined_size = hidden_size + 16

        else:
            self.future_branch = None
            combined_size = hidden_size

        # ----------------------------------------------------
        # Regression head
        # ----------------------------------------------------

        self.head = nn.Sequential(
            nn.Linear(combined_size, 32),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1)
        )


    def forward(
        self,
        sequence,
        future_features=None
    ):
        # sequence:
        # batch x days x features

        rnn_output, _ = self.rnn(sequence)

        # representation from final observed day
        sequence_repr = rnn_output[:, -1, :]

        sequence_repr = self.sequence_norm(
            sequence_repr
        )

        sequence_repr = self.sequence_dropout(
            sequence_repr
        )

        if (
            self.future_branch is not None
            and future_features is not None
        ):
            future_repr = self.future_branch(
                future_features
            )

            combined = torch.cat(
                [
                    sequence_repr,
                    future_repr
                ],
                dim=1
            )

        else:
            combined = sequence_repr

        prediction = self.head(combined)

        return prediction