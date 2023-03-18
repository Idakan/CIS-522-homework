import torch


class Model(torch.nn.Module):
    """
    CNN Model
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Initialize our model, define the CNN layers
        """
        super().__init__()

        self.CNN_stack = torch.nn.Sequential(
            # 3x32x32
            torch.nn.Conv2d(
                in_channels=num_channels, out_channels=16, kernel_size=(5, 5)
            ),
            torch.nn.ReLU(),
            # Now 16x28x28
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=16 * 14 * 14, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the model on our tensor
        """
        outputs = self.CNN_stack(x)
        return outputs
