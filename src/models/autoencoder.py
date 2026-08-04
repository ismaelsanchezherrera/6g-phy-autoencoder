import torch
import torch.nn as nn
import torch.nn.functional as F


class Transmitter(nn.Module):
  """Transmisor Neural (Encoder) con normalización de energía constante."""

  def __init__(self, num_symbols: int = 16, channel_dims: int = 2):
    super(Transmitter, self).__init__()
    self.M = num_symbols
    self.channel_dims = channel_dims

    # Red Densa para aprender la representación en la capa física (Constelación)
    self.fc1 = nn.Linear(self.M, 32)
    self.fc2 = nn.Linear(32, 32)
    self.fc3 = nn.Linear(32, self.channel_dims)

  def forward(self, s_onehot: torch.Tensor) -> torch.Tensor:
    x = F.relu(self.fc1(s_onehot))
    x = F.relu(self.fc2(x))
    x_raw = self.fc3(x)

    # Capa de Normalización de Potencia Media: E[||x||^2] = 1
    energy = torch.mean(torch.sum(x_raw**2, dim=1, keepdim=True))
    x_norm = x_raw / torch.sqrt(energy)

    return x_norm


class Receiver(nn.Module):
  """Receptor Neural (Decoder) con soporte opcional para información de estado del canal (CSI)."""

  def __init__(
      self,
      num_symbols: int = 16,
      channel_dims: int = 2,
      use_csi: bool = False,
  ):
    super(Receiver, self).__init__()
    self.use_csi = use_csi
    input_dim = channel_dims * 2 if use_csi else channel_dims

    self.fc1 = nn.Linear(input_dim, 32)
    self.fc2 = nn.Linear(32, 32)
    self.fc3 = nn.Linear(32, num_symbols)

  def forward(self, y: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
    if self.use_csi and h is not None:
      inp = torch.cat([y, h], dim=1)
    else:
      inp = y

    x = F.relu(self.fc1(inp))
    x = F.relu(self.fc2(x))
    logits = self.fc3(x)
    return logits


class EndToEndAutoencoder(nn.Module):
  """Autoencoder Completo de Capa Física 6G."""

  def __init__(
      self,
      num_symbols: int = 16,
      channel_dims: int = 2,
      use_csi: bool = False,
  ):
    super(EndToEndAutoencoder, self).__init__()
    self.M = num_symbols
    self.transmitter = Transmitter(num_symbols, channel_dims)
    self.receiver = Receiver(num_symbols, channel_dims, use_csi)

  def encode(self, symbols: torch.Tensor) -> torch.Tensor:
    one_hot = F.one_hot(symbols, num_classes=self.M).float()
    return self.transmitter(one_hot)

  def decode(self, y: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
    return self.receiver(y, h)