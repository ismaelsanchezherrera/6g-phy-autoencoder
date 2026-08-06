import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc
import torch

from src.channel.channel import AWGNChannel, RayleighChannel, ebno_to_noise_std
from src.models.autoencoder import EndToEndAutoencoder


def plot_learned_constellation(
    model: EndToEndAutoencoder,
    title: str = "Constelación Aprendida (M=16)",
    save_path: str = "docs/constellation_learned.png",
):
  """Extrae las posiciones I/Q aprendidas por el Encoder y las grafica."""
  model.eval()
  device = next(model.parameters()).device

  symbols = torch.arange(0, model.M, device=device)
  with torch.no_grad():
    encoded_symbols = model.encode(symbols).cpu().numpy()

  plt.figure(figsize=(6, 6))
  plt.scatter(
      encoded_symbols[:, 0],
      encoded_symbols[:, 1],
      color="red",
      s=80,
      label="Puntos NN (I/Q)",
  )

  for i in range(model.M):
    plt.annotate(
        f" S{i}", (encoded_symbols[i, 0], encoded_symbols[i, 1]), fontsize=9
    )

  circle = plt.Circle(
      (0, 0),
      1.0,
      color="gray",
      fill=False,
      linestyle="--",
      label="Límite Potencia Promedio (E=1)",
  )
  plt.gca().add_patch(circle)

  plt.axhline(0, color="black", linewidth=0.5)
  plt.axvline(0, color="black", linewidth=0.5)
  plt.xlim([-2, 2])
  plt.ylim([-2, 2])
  plt.title(title)
  plt.xlabel("Componente En Fase (I)")
  plt.ylabel("Componente En Cuadratura (Q)")
  plt.grid(True)
  plt.legend()
  plt.tight_layout()

  if save_path:
    dir_name = os.path.dirname(save_path)
    if dir_name:
      os.makedirs(dir_name, exist_ok=True)
    plt.savefig(save_path)

  plt.show()


def int_to_bits(
    tensor_symbols: torch.Tensor, bits_per_symbol: int
) -> torch.Tensor:
  """Convierte un tensor de símbolos enteros a una matriz de bits (0s y 1s)."""
  mask = 2 ** torch.arange(
      bits_per_symbol - 1, -1, -1, device=tensor_symbols.device
  )
  return tensor_symbols.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def evaluate_ber(
    model: EndToEndAutoencoder,
    channel_type: str = "awgn",
    ebno_db_range: list = range(0, 14, 2),
    num_samples: int = 200000,
) -> tuple:
  """Calcula la BER mediante simulación Monte Carlo sobre una serie de valores Eb/N0."""
  model.eval()
  device = next(model.parameters()).device
  bits_per_symbol = int(np.log2(model.M))

  channel = (
      AWGNChannel().to(device)
      if channel_type == "awgn"
      else RayleighChannel().to(device)
  )
  ber_results = []

  with torch.no_grad():
    for ebno_db in ebno_db_range:
      symbols_tx = torch.randint(0, model.M, (num_samples,), device=device)
      bits_tx = int_to_bits(symbols_tx, bits_per_symbol)

      noise_std = ebno_to_noise_std(ebno_db, bits_per_symbol)
      x_norm = model.encode(symbols_tx)

      if channel_type == "awgn":
        y = channel(x_norm, noise_std)
        logits = model.decode(y)
      else:
        y, h = channel(x_norm, noise_std, return_csi=True)
        logits = model.decode(y, h)

      symbols_rx = torch.argmax(logits, dim=1)
      bits_rx = int_to_bits(symbols_rx, bits_per_symbol)

      bit_errors = torch.sum(bits_tx != bits_rx).item()
      total_bits = num_samples * bits_per_symbol
      ber = bit_errors / total_bits

      ber_results.append(ber)
      print(f"[{channel_type.upper()}] Eb/N0 = {ebno_db} dB | BER = {ber:.6f}")

  return list(ebno_db_range), ber_results


def theoretical_qpsk_ber(ebno_db_array):
  """Calcula la BER teórica para QPSK ideal en canal AWGN."""
  ebno_linear = 10.0 ** (np.array(ebno_db_array) / 10.0)
  return 0.5 * erfc(np.sqrt(ebno_linear))


def plot_ber_curves(
    ebno_axis,
    ber_autoencoder,
    ber_theoretical=None,
    label="Autoencoder 6G",
    save_path: str = "docs/ber_performance_curve.png",
):
  """Grafica las curvas BER vs Eb/N0 en escala semilogarítmica."""
  plt.figure(figsize=(8, 5))
  plt.semilogy(ebno_axis, ber_autoencoder, "o-", linewidth=2, label=label)

  if ber_theoretical is not None:
    plt.semilogy(
        ebno_axis,
        ber_theoretical,
        "k--",
        linewidth=2,
        label="Referencia Teórica QPSK (AWGN)",
    )

  plt.title("Rendimiento Bit Error Rate (BER) vs Eb/N0")
  plt.xlabel("Eb/N0 (dB)")
  plt.ylabel("Tasa de Error de Bit (BER)")
  plt.ylim([1e-5, 1e-0])
  plt.grid(True, which="both", linestyle="--", alpha=0.5)
  plt.legend()
  plt.tight_layout()

  if save_path:
    dir_name = os.path.dirname(save_path)
    if dir_name:
      os.makedirs(dir_name, exist_ok=True)
    plt.savefig(save_path)

  plt.show()


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Cargar modelo AWGN preentrenado
  model_awgn = EndToEndAutoencoder(
      num_symbols=16, channel_dims=2, use_csi=False
  ).to(device)
  model_awgn.load_state_dict(
      torch.load(
          "models_checkpoints/autoencoder_awgn.pth", map_location=device
      )
  )

  # Visualizar y guardar Constelación Aprendida
  plot_learned_constellation(
      model_awgn,
      title="Constelación Reconfigurada Aprendida por la IA (M=16)",
      save_path="docs/constellation_learned.png",
  )

  # Calcular y Graficar BER
  ebno_axis, ber_nn = evaluate_ber(
      model_awgn, channel_type="awgn", ebno_db_range=range(0, 12, 2)
  )
  ber_qpsk = theoretical_qpsk_ber(ebno_axis)
  plot_ber_curves(
      ebno_axis,
      ber_nn,
      ber_qpsk,
      label="Autoencoder (16-Símbolos)",
      save_path="docs/ber_performance_curve.png",
  )