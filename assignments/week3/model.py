from typing import Callable
import torch


class MLP(torch.nn.Module):
    """
    Create a MLP class implementing torch.nn.Module principles

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU(),
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.actv = activation
        self.layers = torch.nn.ModuleList()
        for i in range(hidden_count):
            next_num_inputs = hidden_size
            self.layers += [torch.nn.Linear(input_size, next_num_inputs)]
            input_size = next_num_inputs

            # Create final layer

        self.out = torch.nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor()) -> torch.nn.Linear:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers:
            x = self.actv(layer(x))

        # Get activations of each layer
        return self.out(x)
