import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import inception_v3, Inception_V3_Weights


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


class CrossSelfBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.q_ln = nn.LayerNorm(dim)
        self.kv_ln = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self_ln = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp_ln = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, q, kv):
        qn = self.q_ln(q)
        kvn = self.kv_ln(kv)
        q = q + self.cross_attn(qn, kvn, kvn, need_weights=False)[0]
        qn = self.self_ln(q)
        q = q + self.self_attn(qn, qn, qn, need_weights=False)[0]
        q = q + self.mlp(self.mlp_ln(q))
        return q


class ConvUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.skip(x), inplace=True)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        token_dim: int = 512,
        n_latent_tokens: int = 32,
        grid_size: int = 8,
        n_blocks: int = 10,
        n_heads: int = 8,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.n_latent_tokens = n_latent_tokens
        self.grid_size = grid_size
        self.n_queries = grid_size * grid_size

        self.latent_proj = nn.Linear(latent_dim, n_latent_tokens * token_dim)
        self.latent_pos = nn.Parameter(torch.randn(1, n_latent_tokens, token_dim) * 0.02)
        self.query_tokens = nn.Parameter(torch.randn(1, self.n_queries, token_dim) * 0.02)
        self.query_pos = nn.Parameter(torch.randn(1, self.n_queries, token_dim) * 0.02)

        self.blocks = nn.ModuleList([CrossSelfBlock(token_dim, n_heads) for _ in range(n_blocks)])
        self.to_grid = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
        )

        self.up16 = ConvUpBlock(token_dim, 384)
        self.up32 = ConvUpBlock(384, 256)
        self.refine = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, z):
        b = z.shape[0]
        latent_tokens = self.latent_proj(z).view(b, self.n_latent_tokens, self.token_dim)
        latent_tokens = latent_tokens + self.latent_pos
        q = self.query_tokens.expand(b, -1, -1) + self.query_pos
        for blk in self.blocks:
            q = blk(q, latent_tokens)
        q = self.to_grid(q)
        x = q.transpose(1, 2).reshape(b, self.token_dim, self.grid_size, self.grid_size)
        x = self.up16(x)
        x = self.up32(x)
        x = self.refine(x)
        return self.out_conv(x)


class InceptionFID(nn.Module):
    def __init__(self):
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        net = inception_v3(weights=weights, aux_logits=True, transform_input=False)
        net.fc = nn.Identity()
        net.eval()
        self.net = net

    @torch.no_grad()
    def forward(self, x):
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x + 1.0) * 0.5
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        x = (x - mean) / std
        return self.net(x)


def compute_stats(features: torch.Tensor):
    mu = features.mean(dim=0)
    xc = features - mu
    sigma = (xc.T @ xc) / (features.shape[0] - 1)
    return mu, sigma


def sqrtm_psd(a: torch.Tensor):
    evals, evecs = torch.linalg.eigh(a)
    evals = evals.clamp_min(0).sqrt()
    return (evecs * evals.unsqueeze(0)) @ evecs.T


def fid_from_stats(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    s1_sqrt = sqrtm_psd(sigma1)
    middle = s1_sqrt @ sigma2 @ s1_sqrt
    covmean = sqrtm_psd(middle)
    return (
        diff.dot(diff)
        + torch.trace(sigma1)
        + torch.trace(sigma2)
        - 2.0 * torch.trace(covmean)
    ).item()


@torch.no_grad()
def collect_real_features(fid_model, loader, device):
    feats = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        feats.append(fid_model(x).cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def collect_fake_features(fid_model, generator, latent_dim, n_samples, batch_size, device):
    feats = []
    done = 0
    while done < n_samples:
        b = min(batch_size, n_samples - done)
        z = torch.randn(b, latent_dim, device=device)
        x = generator(z)
        feats.append(fid_model(x).cpu())
        done += b
    return torch.cat(feats, dim=0)


def get_real_stats(cache_path, data_root, split, batch_size, num_workers, fid_model, device):
    if os.path.exists(cache_path):
        blob = torch.load(cache_path, map_location="cpu")
        return blob["mu"], blob["sigma"]

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(data_root, train=(split == "train"), transform=tf, download=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    feats = collect_real_features(fid_model, dl, device)
    mu, sigma = compute_stats(feats)
    torch.save({"mu": mu, "sigma": sigma}, cache_path)
    return mu, sigma


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_fake", type=int, default=10000)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--cache_dir", type=str, default="./fid_cache")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    device = torch.device(args.device)

    gen = Decoder(args.latent_dim).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    gen.load_state_dict(ckpt["dec"])
    gen.eval()

    fid_model = InceptionFID().to(device)

    cache_path = os.path.join(args.cache_dir, f"cifar10_{args.split}_inception_stats.pt")
    mu_real, sigma_real = get_real_stats(
        cache_path,
        args.data_root,
        args.split,
        args.batch_size,
        args.num_workers,
        fid_model,
        device,
    )

    fake_feats = collect_fake_features(
        fid_model,
        gen,
        args.latent_dim,
        args.n_fake,
        args.batch_size,
        device,
    )
    mu_fake, sigma_fake = compute_stats(fake_feats)

    fid = fid_from_stats(mu_real, sigma_real, mu_fake, sigma_fake)
    print(f"FID ({args.split}, n_fake={args.n_fake}): {fid:.4f}")


if __name__ == "__main__":
    main()