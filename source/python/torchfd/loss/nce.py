import math
import torch

from .utils import BaseVariationalBoundLoss


class InfoNCELoss(BaseVariationalBoundLoss):
    """
    Noise-Contrastive Estimation variational lower bound for
    the mutual information.

    References
    ----------
    .. [1] Oord A., Li Y. and Vinyals O. "Representation Learning with
           Contrastive Predictive Coding". arXiv:1807.03748
    """

    is_lower_bound = True
    
    def __init__(self):
        super().__init__()

    def forward(self, T_joint: torch.tensor, T_product: torch.tensor) -> torch.tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        T_joint : torch.tensor
            Critic network value on all samples from the batch.
        T_product : torch.tensor
            Critic network value on all possible pairs of samples from the batch.
        """

        batch_size = T_joint.shape[0]
        T_product = T_product.view((batch_size, batch_size))
        
        return torch.mean(torch.logsumexp(T_product - T_joint, dim=-2)) - math.log(batch_size)