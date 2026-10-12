"""The architecture in Chapter 3, Table 3.2, and a CNN ablation."""
import torch
from torch import nn


class MalariaModel(nn.Module):
    def __init__(self, architecture="cnn_lstm", dropout=0.5):
        super().__init__()
        if architecture not in {"cnn_lstm", "cnn"}:
            raise ValueError("architecture must be cnn_lstm or cnn")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.architecture = architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
        )
        if architecture == "cnn_lstm":
            self.lstm = nn.LSTM(30 * 64, 128, batch_first=True)
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, 1))
        else:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, 1)
            )

    @staticmethod
    def row_sequence(features):
        # N,C,H,W -> N,H,W,C -> N,H,(W*C). A timestep is one spatial row.
        return features.permute(0, 2, 3, 1).contiguous().flatten(2)

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 128, 128):
            raise ValueError("Expected N x 3 x 128 x 128 RGB tensors")
        features = self.features(images)
        if self.architecture == "cnn_lstm":
            _, (hidden, _) = self.lstm(self.row_sequence(features))
            features = hidden[-1]
        # Train with BCEWithLogitsLoss; apply sigmoid only during scoring.
        return self.classifier(features).squeeze(1)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("class_mapping") != {"Uninfected": 0, "Parasitized": 1}:
        raise ValueError("Unsupported checkpoint or class mapping")
    model = MalariaModel(checkpoint["architecture"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint
