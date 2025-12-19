import torch
import torch.nn as nn


# label smoothing
class LossWrapper:
    """
    Wrapper class for LabelSmoothing loss. This is for convenience.
    Input params: 
        - generator (Generator): generator model. Pass the `model.generator` module. 
        - criterion (nn.Module): loss criterion. Pass LabelSmoothing criterion. 
    """

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, model_output, target, norm):
        """
        Forward pass for loss wrapper. Reshapes tensors to appropriate dimensions, computes loss, and normalize.
        Input params: 
            - model_output (Tensor): output from model. 
            - target (Tensor): target labels. 
            - norm (int): normalization factor. 
        Returns: 
            - normalized_loss (Tensor): normalized loss. 
        """
        x = self.generator(model_output)
        normalized_loss = (
                self.criterion(
                    x.contiguous().view(-1, x.size(-1)),
                    target.contiguous().view(-1)
                ) / norm
        )
        return normalized_loss


class LabelSmoothing(nn.Module):
    """
    Module for label smoothing. 
    Input params: 
        - vocab_size (int): size of the vocabulary. 
        - padding_idx (int): padding index. 
        - smoothing (float): smoothing factor; some small float value
    """

    def __init__(self, vocab_size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, model_output, target):
        """
        Forward pass for label smoothing.
        Input params: 
            - model_output (Tensor): output from model.
            - target (Tensor): target labels.
        Returns: 
            - loss (Tensor): loss. 
        
        Steps: 
        1. Create `smoothed_dist` tensor as per the formula to be the same shape as `model_output` data. 
        2. Zero out the padding tokens. 
        3. Compute and return the loss. 
        
        The following functions can help you:
        - Use `model_output.data.clone()` to create a copy of the model output tensor.
        - tensor.fill_(value)
        - tensor.scatter_(dim, index, src)
        - tensor.index_fill_(dim, index, value)
        """

        # TODO: Implement the forward pass for label smoothing.
        # YOUR CODE STARTS HERE
        with torch.no_grad():
            # base prob for non-target, non-pad positions
            smoothing_value = self.smoothing / (self.vocab_size - 1)

            smoothed = model_output.data.clone().fill_(smoothing_value)
            # zero out PAD column everywhere
            smoothed[:, self.padding_idx] = 0.0

            # put confidence on the target index
            smoothed.scatter_(1, target.unsqueeze(1), self.confidence)

            # make rows whose target==PAD all zeros (no loss contribution)
            pad_rows = (target == self.padding_idx)
            smoothed[pad_rows] = 0.0

        loss = self.criterion(model_output, smoothed)
        # YOUR CODE ENDS HERE
        return loss 
