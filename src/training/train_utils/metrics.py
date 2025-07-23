import torch
import torch.nn as nn
from scipy.stats import pearsonr
import numpy as np


def compute_encoding_accuracy(fmri_val, fmri_val_pred, subject, modality):
    encoding_accuracy = np.zeros((fmri_val.shape[1]), dtype=np.float32)
    for p in range(len(encoding_accuracy)):
        encoding_accuracy[p] = pearsonr(fmri_val[:, p],
                                        fmri_val_pred[:, p])[0]    

    return encoding_accuracy


class PearsonCorrLoss(nn.Module):
    def __init__(self,
                 eps: float = 1e-6,
                 lambda_mse: float = 0.2):
        """
        Composite loss = − Corr + λ * MSE

        Args:
          eps: numerical stabilizer for denom
          lambda_mse: weight on the MSE term
        """
        super().__init__()
        self.eps = eps
        self.lambda_mse = lambda_mse

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (T, F)
        returns: scalar loss
        """
        # --- Pearson part ---
        pred_mean = pred.mean(dim=0, keepdim=True)    # (1, F)
        target_mean = target.mean(dim=0, keepdim=True)  # (1, F)
        p_centered = pred - pred_mean                  # (T, F)
        t_centered = target - target_mean              # (T, F)

        numerator = torch.sum(p_centered * t_centered, dim=0)  # (F,)
        denom = torch.sqrt(
            torch.sum(p_centered**2, dim=0) *
            torch.sum(t_centered**2, dim=0)
        ).clamp(min=self.eps)                                  # (F,)

        corr = numerator / denom                              # (F,)
        corr_loss = -corr.mean()                              # scalar

        # --- MSE part ---
        mse_loss = torch.mean((pred - target)**2)

        # --- Composite ---
        loss = corr_loss + self.lambda_mse * mse_loss
        return loss
