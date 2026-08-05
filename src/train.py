import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.channel.channel import AWGNChannel, RayleighChannel, ebno_to_noise_std
from src.models.autoencoder import EndToEndAutoencoder


def train_awgn(
    model: EndToEndAutoencoder,
    epochs: int = 40,
    batch_size: int = 1024,
    lr: float = 0.001,
    ebno_db_min: float = 3.0,
    ebno_db_max: float = 10.0,
):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  channel = AWGNChannel().to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  criterion = nn.CrossEntropyLoss()

  history = []
  bits_per_symbol = int(np.log2(model.M))
  print(f"--- Entrenando Autoencoder en Canal AWGN ({device}) ---")

  for epoch in range(1, epochs + 1):
    model.train()
    symbols = torch.randint(0, model.M, (batch_size * 20,), device=device)
    dataset = TensorDataset(symbols)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    for batch_s in loader:
      s = batch_s[0]
      optimizer.zero_grad()

      # Variar Eb/N0 dinámicamente en cada batch para mayor robustez
      ebno_db = torch.empty(1).uniform_(ebno_db_min, ebno_db_max).item()
      noise_std = ebno_to_noise_std(
          ebno_db, bits_per_symbol=bits_per_symbol
      )

      x_norm = model.encode(s)
      y = channel(x_norm, noise_std)
      logits = model.decode(y)

      loss = criterion(logits, s)
      loss.backward()
      optimizer.step()

      total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    history.append(avg_loss)

    if epoch % 10 == 0 or epoch == 1:
      print(f"Época [{epoch}/{epochs}] - Loss Cross-Entropy: {avg_loss:.5f}")

  return history


def train_rayleigh(
    model: EndToEndAutoencoder,
    epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 0.001,
    ebno_db_min: float = 5.0,
    ebno_db_max: float = 15.0,
):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  channel = RayleighChannel().to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  criterion = nn.CrossEntropyLoss()

  history = []
  bits_per_symbol = int(np.log2(model.M))
  print(
      f"\n--- Entrenando Autoencoder en Canal Rayleigh + CSI ({device}) ---"
  )

  for epoch in range(1, epochs + 1):
    model.train()
    symbols = torch.randint(0, model.M, (batch_size * 20,), device=device)
    dataset = TensorDataset(symbols)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    for batch_s in loader:
      s = batch_s[0]
      optimizer.zero_grad()

      ebno_db = torch.empty(1).uniform_(ebno_db_min, ebno_db_max).item()
      noise_std = ebno_to_noise_std(
          ebno_db, bits_per_symbol=bits_per_symbol
      )

      x_norm = model.encode(s)
      y, h = channel(x_norm, noise_std, return_csi=True)
      logits = model.decode(y, h)

      loss = criterion(logits, s)
      loss.backward()
      optimizer.step()

      total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    history.append(avg_loss)

    if epoch % 10 == 0 or epoch == 1:
      print(f"Época [{epoch}/{epochs}] - Loss Cross-Entropy: {avg_loss:.5f}")

  return history


if __name__ == "__main__":
  os.makedirs("models_checkpoints", exist_ok=True)
  M = 16

  # 1. Entrenamiento AWGN
  model_awgn = EndToEndAutoencoder(
      num_symbols=M, channel_dims=2, use_csi=False
  )
  history_awgn = train_awgn(model_awgn, epochs=40)
  torch.save(
      model_awgn.state_dict(), "models_checkpoints/autoencoder_awgn.pth"
  )
  print("Modelo AWGN guardado en models_checkpoints/autoencoder_awgn.pth")

  # 2. Entrenamiento Rayleigh
  model_rayleigh = EndToEndAutoencoder(
      num_symbols=M, channel_dims=2, use_csi=True
  )
  history_rayleigh = train_rayleigh(model_rayleigh, epochs=50)
  torch.save(
      model_rayleigh.state_dict(), "models_checkpoints/autoencoder_rayleigh.pth"
  )
  print(
      "Modelo Rayleigh guardado en"
      " models_checkpoints/autoencoder_rayleigh.pth"
  )