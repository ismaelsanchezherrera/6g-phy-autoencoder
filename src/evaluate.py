import os
import matplotlib.pyplot as plt
import torch
from src.models.autoencoder import EndToEndAutoencoder


def plot_learned_constellation(
    model: EndToEndAutoencoder,
    title: str = "Constelación Aprendida (M=16)",
    save_path: str = None,
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

  plt.show()


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model_awgn = EndToEndAutoencoder(
      num_symbols=16, channel_dims=2, use_csi=False
  ).to(device)
  model_awgn.load_state_dict(
      torch.load(
          "models_checkpoints/autoencoder_awgn.pth", map_location=device
      )
  )

  plot_learned_constellation(
      model_awgn,
      "Constelación Reconfigurada Aprendida por la IA (M=16)",
      save_path="docs/constellation_learned.png",
  )