import numpy as np
import torch
import torch.nn as nn


def ebno_to_noise_std(
    ebno_db: float, bits_per_symbol: int, num_complex_dims: int = 1
) -> float:
  ebno_linear = 10.0 ** (ebno_db / 10.0)
  code_rate = bits_per_symbol / (2.0 * num_complex_dims)
  noise_variance = 1.0 / (2.0 * code_rate * ebno_linear)
  return np.sqrt(noise_variance)


class AWGNChannel(nn.Module):

  def __init__(self):
    super(AWGNChannel, self).__init__()

  def forward(self, x: torch.Tensor, noise_std: float) -> torch.Tensor:
    noise = torch.randn_like(x) * noise_std
    return x + noise


class RayleighChannel(nn.Module):

  def __init__(self):
    super(RayleighChannel, self).__init__()

  def forward(self, x: torch.Tensor, noise_std: float, return_csi: bool = False):
    batch_size = x.shape[0]
    device = x.device
    h_r = torch.randn(batch_size, 1, device=device) * np.sqrt(0.5)
    h_i = torch.randn(batch_size, 1, device=device) * np.sqrt(0.5)

    x_i = x[:, 0:1]
    x_q = x[:, 1:2]

    y_i = h_r * x_i - h_i * x_q
    y_q = h_i * x_i + h_r * x_q

    y_faded = torch.cat([y_i, y_q], dim=1)
    noise = torch.randn_like(y_faded) * noise_std
    y_received = y_faded + noise

    if return_csi:
      h_matrix = torch.cat([h_r, h_i], dim=1)
      return y_received, h_matrix
    return y_received