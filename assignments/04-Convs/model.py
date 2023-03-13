import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(self, num_channels: int, num_classes: int) -> None:
        super().__init__()
        self.flatten = torch.nn.Flatten()

        # part1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(4,4), stride=2)

        # print(part1.weight.shape)

        self.CNN_stack = torch.nn.Sequential(
            # 3x32x32
            torch.nn.Conv2d(
                in_channels=num_channels, out_channels=16, kernel_size=(5, 5)
            ),
            torch.nn.ReLU(),
            # Now 16x28x28
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
            torch.nn.ReLU(),
            # now 16x14x14
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5)),
            torch.nn.ReLU(),
            # Now 32x10x10
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
            torch.nn.ReLU(),
            # now 32x5x5
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=32 * 5 * 5, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.CNN_stack(x)
        return outputs
