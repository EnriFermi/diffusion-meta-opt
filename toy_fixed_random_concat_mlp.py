from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange


def main() -> None:
    seed = 42
    d_in = 64
    d_random = 4096
    hidden_dim = 512
    num_steps = 5000
    lr = 1e-3

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target = torch.randn(d_in, device=device)
    fixed_random = torch.randn(d_random, device=device)
    x = torch.cat([target, fixed_random], dim=0).unsqueeze(0)
    y = target.unsqueeze(0)

    model = nn.Sequential(
        nn.Linear(d_in + d_random, d_in),
        nn.ReLU(),
        nn.Linear(d_in, d_in),
        nn.ReLU(),
        nn.Linear(d_in, d_in),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    progress = trange(num_steps, desc="train", dynamic_ncols=True)
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()

        progress.set_postfix(loss=f"{loss.item():.6f}")

    with torch.no_grad():
        final_pred = model(x)
        final_loss = F.mse_loss(final_pred, y).item()

    print(f"device={device}")
    print(f"final_loss={final_loss:.6f}")
    print(f"target[:8]={target[:8].detach().cpu()}")
    print(f"pred[:8]={final_pred[0, :8].detach().cpu()}")


if __name__ == "__main__":
    main()
