import argparse
import os
import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torchvision.models import resnet18


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class CFG:
    data_root: str = "./data"
    out_dir: str = "./runs/cifar_stochastic_latent"
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    mode: str = "train_all"
    stage1_ckpt: str = ""
    stage2_ckpt: str = ""

    batch_size: int = 256
    latent_dim: int = 64

    stage1_epochs: int = 100
    stage2_epochs: int = 120
    lr_stage1: float = 3e-4
    lr_stage2: float = 3e-4
    wd: float = 1e-4

    lambda_pos: float = 1.0
    lambda_neg: float = 1.0
    lambda_gauss: float = 5.0
    neg_margin: float = 4.0
    sw_projections: int = 64

    recon_loss: str = "l1"
    stage2_n_aug: int = 3
    stage2_aug_weight: float = 0.0

    sample_n: int = 64
    log_every: int = 100
    save_every: int = 10


def norm_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def strong_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.35, 0.35, 0.35, 0.10),
        ], p=0.9),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=15, translate=(0.10, 0.10), scale=(0.85, 1.15), shear=8),
        ], p=0.8),
        transforms.RandomApply([
            transforms.RandomPerspective(distortion_scale=0.18, p=1.0),
        ], p=0.35),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)),
        ], p=0.25),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class CIFARStage1Pairs(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool):
        self.base = datasets.CIFAR10(root, train=train, transform=None, download=True)
        self.tf = strong_transform()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        img, y = self.base[idx]
        return self.tf(img), self.tf(img), y


class CIFARStage2Views(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool, n_aug: int):
        self.base = datasets.CIFAR10(root, train=train, transform=None, download=True)
        self.plain_tf = norm_transform()
        self.aug_tf = strong_transform()
        self.n_aug = n_aug

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        img, y = self.base[idx]
        x_plain = self.plain_tf(img)
        x_augs = torch.stack([self.aug_tf(img) for _ in range(self.n_aug)], dim=0)
        return x_plain, x_augs, y


def make_loaders(cfg: CFG):
    ds_train_stage1 = CIFARStage1Pairs(cfg.data_root, train=True)
    ds_train_stage2 = CIFARStage2Views(cfg.data_root, train=True, n_aug=cfg.stage2_n_aug)
    ds_train_plain = datasets.CIFAR10(cfg.data_root, train=True, transform=norm_transform(), download=True)
    ds_test_plain = datasets.CIFAR10(cfg.data_root, train=False, transform=norm_transform(), download=True)

    dl_train_stage1 = DataLoader(ds_train_stage1, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    dl_train_stage2 = DataLoader(ds_train_stage2, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    dl_train_plain = DataLoader(ds_train_plain, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    dl_test_plain = DataLoader(ds_test_plain, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    return dl_train_stage1, dl_train_stage2, dl_train_plain, dl_test_plain


class StochasticEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Identity()
        self.backbone = net
        self.trunk = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(512, latent_dim)
        self.log_std = nn.Linear(512, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self.backbone(x))
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-5.0, 2.0)
        return mu, log_std


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.skip(x), inplace=True)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, token_dim: int = 512, n_latent_tokens: int = 32, grid_size: int = 8, n_blocks: int = 10, n_heads: int = 8):
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
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


def reparameterize(mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    return mu + torch.exp(log_std) * torch.randn_like(mu)


def diag_gaussian_w2_sq(mu1: torch.Tensor, log_std1: torch.Tensor, mu2: torch.Tensor, log_std2: torch.Tensor) -> torch.Tensor:
    s1 = torch.exp(log_std1)
    s2 = torch.exp(log_std2)
    return (mu1 - mu2).pow(2).sum(dim=1) + (s1 - s2).pow(2).sum(dim=1)


def pair_loss(mu1: torch.Tensor, log_std1: torch.Tensor, mu2: torch.Tensor, log_std2: torch.Tensor, margin: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b = mu1.shape[0]
    pos = diag_gaussian_w2_sq(mu1, log_std1, mu2, log_std2).mean()

    mu = torch.cat([mu1, mu2], dim=0)
    sig = torch.cat([torch.exp(log_std1), torch.exp(log_std2)], dim=0)
    mu_d2 = (mu[:, None, :] - mu[None, :, :]).pow(2).sum(dim=2)
    sig_d2 = (sig[:, None, :] - sig[None, :, :]).pow(2).sum(dim=2)
    w2 = mu_d2 + sig_d2
    w2.fill_diagonal_(float("inf"))
    pair_idx = torch.cat([torch.arange(b, 2 * b, device=mu.device), torch.arange(0, b, device=mu.device)], dim=0)
    w2[torch.arange(2 * b, device=mu.device), pair_idx] = float("inf")
    neg_d = w2.min(dim=1).values.sqrt()
    neg = F.relu(margin - neg_d).pow(2).mean()
    return pos + neg, pos, neg


def sliced_wasserstein_to_gaussian(z: torch.Tensor, n_proj: int) -> torch.Tensor:
    n, d = z.shape
    g = torch.randn_like(z)
    u = torch.randn(n_proj, d, device=z.device, dtype=z.dtype)
    u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-8)
    proj_z = z @ u.T
    proj_g = g @ u.T
    proj_z = proj_z.sort(dim=0).values
    proj_g = proj_g.sort(dim=0).values
    return (proj_z - proj_g).pow(2).mean()


def recon_loss(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    return F.mse_loss(pred, target) if kind == "l2" else F.l1_loss(pred, target)


@torch.no_grad()
def save_image_grid(x: torch.Tensor, path: str, nrow: int = 8) -> None:
    utils.save_image((x.clamp(-1, 1) + 1.0) * 0.5, path, nrow=nrow)


@torch.no_grad()
def eval_latent_stats(enc: nn.Module, loader: DataLoader, device: torch.device, n_proj: int = 32, max_knn_points: int = 1024):
    enc.eval()
    zs = []
    sigmas = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        mu, log_std = enc(x)
        z = reparameterize(mu, log_std)
        zs.append(z.cpu())
        sigmas.append(torch.exp(log_std).cpu())
    z = torch.cat(zs, dim=0)
    sigma = torch.cat(sigmas, dim=0)
    n, d = z.shape
    mean = z.mean(dim=0)
    zc = z - mean
    cov = (zc.T @ zc) / max(n - 1, 1)
    diag = cov.diag()
    off = cov - torch.diag(diag)
    r2 = z.pow(2).sum(dim=1)

    u = torch.randn(n_proj, d)
    u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-8)
    s = z @ u.T
    proj_mean_abs = s.mean(dim=0).abs().mean().item()
    proj_var_err = (s.var(dim=0, unbiased=False) - 1.0).abs().mean().item()
    s_center = s - s.mean(dim=0, keepdim=True)
    s_std = s.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    s_norm = s_center / s_std
    proj_skew_abs = s_norm.pow(3).mean(dim=0).abs().mean().item()
    proj_kurt_excess_abs = (s_norm.pow(4).mean(dim=0) - 3.0).abs().mean().item()

    m = min(max_knn_points, n)
    idx = torch.randperm(n)[:m]
    dist = torch.cdist(z[idx], z[idx])
    dist.fill_diagonal_(float("inf"))
    nn1 = dist.min(dim=1).values

    return {
        "mean_abs": mean.abs().mean().item(),
        "diag_err": (diag - 1.0).abs().mean().item(),
        "diag_mean": diag.mean().item(),
        "diag_std": diag.std().item(),
        "offdiag_abs_mean": off.abs().mean().item(),
        "r2_mean": r2.mean().item(),
        "r2_var": r2.var(unbiased=False).item(),
        "r2_target_mean": float(d),
        "r2_target_var": float(2 * d),
        "proj_mean_abs": proj_mean_abs,
        "proj_var_err": proj_var_err,
        "proj_skew_abs": proj_skew_abs,
        "proj_kurt_excess_abs": proj_kurt_excess_abs,
        "nn1_mean": nn1.mean().item(),
        "nn1_std": nn1.std(unbiased=False).item(),
        "sigma_mean": sigma.mean().item(),
        "sigma_std": sigma.std(unbiased=False).item(),
    }


@torch.no_grad()
def save_recons(enc: nn.Module, dec: nn.Module, loader: DataLoader, device: torch.device, out_path: str):
    enc.eval()
    dec.eval()
    x, _ = next(iter(loader))
    x = x[:64].to(device)
    mu, _ = enc(x)
    xr = dec(mu)
    save_image_grid(torch.cat([x, xr], dim=0), out_path, nrow=8)


@torch.no_grad()
def save_samples(dec: nn.Module, latent_dim: int, device: torch.device, out_path: str, n: int = 64):
    dec.eval()
    z = torch.randn(n, latent_dim, device=device)
    x = dec(z)
    save_image_grid(x, out_path, nrow=8)


def maybe_load_stage1_for_resume(cfg: CFG, enc: nn.Module, device: torch.device) -> int:
    ckpt_path = cfg.stage1_ckpt or os.path.join(cfg.out_dir, "stage1.pt")
    if not os.path.exists(ckpt_path):
        return 1
    ckpt = torch.load(ckpt_path, map_location=device)
    enc.load_state_dict(ckpt["enc"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    print(f"loaded stage1 checkpoint: {ckpt_path}; resuming from epoch {start_epoch}")
    return start_epoch


def maybe_load_stage2_for_resume(cfg: CFG, dec: nn.Module, device: torch.device) -> int:
    ckpt_path = cfg.stage2_ckpt or os.path.join(cfg.out_dir, "stage2.pt")
    if not os.path.exists(ckpt_path):
        return 1
    ckpt = torch.load(ckpt_path, map_location=device)
    dec.load_state_dict(ckpt["dec"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    print(f"loaded stage2 checkpoint: {ckpt_path}; resuming from epoch {start_epoch}")
    return start_epoch


def load_stage1_if_needed(cfg: CFG, enc: nn.Module, device: torch.device):
    ckpt_path = cfg.stage1_ckpt or os.path.join(cfg.out_dir, "stage1.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    enc.load_state_dict(ckpt["enc"])
    print(f"loaded stage1 checkpoint: {ckpt_path}")


def load_stage2_if_needed(cfg: CFG, dec: nn.Module, device: torch.device):
    ckpt_path = cfg.stage2_ckpt or os.path.join(cfg.out_dir, "stage2.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    dec.load_state_dict(ckpt["dec"])
    print(f"loaded stage2 checkpoint: {ckpt_path}")


def train_stage1(cfg: CFG, enc: nn.Module, loader: DataLoader, test_loader: DataLoader, device: torch.device, start_epoch: int = 1):
    enc.train()
    opt = torch.optim.AdamW(enc.parameters(), lr=cfg.lr_stage1, weight_decay=cfg.wd)
    os.makedirs(cfg.out_dir, exist_ok=True)

    if start_epoch > cfg.stage1_epochs:
        print(f"stage1 already finished: start_epoch={start_epoch} > stage1_epochs={cfg.stage1_epochs}")
        return

    for epoch in range(start_epoch, cfg.stage1_epochs + 1):
        enc.train()
        meter_pair = meter_pos = meter_neg = meter_gauss = 0.0
        steps = 0

        for it, (x1, x2, _) in enumerate(loader, start=1):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            mu1, log_std1 = enc(x1)
            mu2, log_std2 = enc(x2)
            z1 = reparameterize(mu1, log_std1)
            z2 = reparameterize(mu2, log_std2)

            pair, pos, neg = pair_loss(mu1, log_std1, mu2, log_std2, cfg.neg_margin)
            z = torch.cat([z1, z2], dim=0)
            gauss = sliced_wasserstein_to_gaussian(z, cfg.sw_projections)
            loss = cfg.lambda_pos * pos + cfg.lambda_neg * neg + cfg.lambda_gauss * gauss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            meter_pair += pair.item()
            meter_pos += pos.item()
            meter_neg += neg.item()
            meter_gauss += gauss.item()
            steps += 1

            if it % cfg.log_every == 0:
                print(
                    f"[stage1][epoch {epoch:03d}][iter {it:04d}] "
                    f"pair={meter_pair/steps:.4f} pos={meter_pos/steps:.4f} "
                    f"neg={meter_neg/steps:.4f} gauss={meter_gauss/steps:.4f}"
                )

        stats = eval_latent_stats(enc, test_loader, device)
        print(
            f"[stage1][epoch {epoch:03d}] "
            f"pair={meter_pair/steps:.4f} pos={meter_pos/steps:.4f} "
            f"neg={meter_neg/steps:.4f} gauss={meter_gauss/steps:.4f} | "
            f"mean_abs={stats['mean_abs']:.4f} diag_err={stats['diag_err']:.4f} "
            f"diag_mean={stats['diag_mean']:.4f} diag_std={stats['diag_std']:.4f} "
            f"offdiag_abs_mean={stats['offdiag_abs_mean']:.4f} | "
            f"r2_mean={stats['r2_mean']:.2f}/{stats['r2_target_mean']:.2f} "
            f"r2_var={stats['r2_var']:.2f}/{stats['r2_target_var']:.2f} | "
            f"proj_mean_abs={stats['proj_mean_abs']:.4f} proj_var_err={stats['proj_var_err']:.4f} "
            f"proj_skew_abs={stats['proj_skew_abs']:.4f} proj_kurt_excess_abs={stats['proj_kurt_excess_abs']:.4f} | "
            f"nn1_mean={stats['nn1_mean']:.4f} nn1_std={stats['nn1_std']:.4f} | "
            f"sigma_mean={stats['sigma_mean']:.4f} sigma_std={stats['sigma_std']:.4f}"
        )

        if epoch % cfg.save_every == 0 or epoch == cfg.stage1_epochs:
            torch.save({"enc": enc.state_dict(), "cfg": vars(cfg), "epoch": epoch}, os.path.join(cfg.out_dir, "stage1.pt"))


def train_stage2(cfg: CFG, enc: nn.Module, dec: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: torch.device, start_epoch: int = 1):
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()
    dec.train()
    opt = torch.optim.AdamW(dec.parameters(), lr=cfg.lr_stage2, weight_decay=cfg.wd)
    os.makedirs(cfg.out_dir, exist_ok=True)

    if start_epoch > cfg.stage2_epochs:
        print(f"stage2 already finished: start_epoch={start_epoch} > stage2_epochs={cfg.stage2_epochs}")
        return

    for epoch in range(start_epoch, cfg.stage2_epochs + 1):
        dec.train()
        meter = meter_plain = meter_aug = 0.0
        steps = 0

        for it, (x_plain, x_augs, _) in enumerate(train_loader, start=1):
            x_plain = x_plain.to(device, non_blocking=True)
            x_augs = x_augs.to(device, non_blocking=True)
            b, k, c, h, w = x_augs.shape
            x_aug_flat = x_augs.view(b * k, c, h, w)

            with torch.no_grad():
                mu_plain, _ = enc(x_plain)
                z_plain = mu_plain
                mu_aug, _ = enc(x_aug_flat)
                z_aug = mu_aug

            xr_plain = dec(z_plain)
            xr_aug = dec(z_aug)
            loss_plain = recon_loss(xr_plain, x_plain, cfg.recon_loss)
            loss_aug = recon_loss(xr_aug, x_aug_flat, cfg.recon_loss)
            loss = loss_plain + cfg.stage2_aug_weight * loss_aug

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            meter += loss.item()
            meter_plain += loss_plain.item()
            meter_aug += loss_aug.item()
            steps += 1

            if it % cfg.log_every == 0:
                print(f"[stage2][epoch {epoch:03d}][iter {it:04d}] recon={meter/steps:.4f} plain={meter_plain/steps:.4f} aug={meter_aug/steps:.4f}")

        print(f"[stage2][epoch {epoch:03d}] recon={meter/steps:.4f} plain={meter_plain/steps:.4f} aug={meter_aug/steps:.4f}")

        if epoch % cfg.save_every == 0 or epoch == cfg.stage2_epochs:
            torch.save({"dec": dec.state_dict(), "cfg": vars(cfg), "epoch": epoch}, os.path.join(cfg.out_dir, "stage2.pt"))
            save_recons(enc, dec, test_loader, device, os.path.join(cfg.out_dir, f"recons_e{epoch:03d}.png"))
            save_samples(dec, cfg.latent_dim, device, os.path.join(cfg.out_dir, f"samples_e{epoch:03d}.png"), n=cfg.sample_n)


def parse_args() -> CFG:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./runs/cifar_stochastic_latent")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--stage1_epochs", type=int, default=100)
    p.add_argument("--stage2_epochs", type=int, default=300)
    p.add_argument("--lr_stage1", type=float, default=3e-4)
    p.add_argument("--lr_stage2", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--lambda_pos", type=float, default=1.0)
    p.add_argument("--lambda_neg", type=float, default=1.0)
    p.add_argument("--lambda_gauss", type=float, default=5.0)
    p.add_argument("--neg_margin", type=float, default=4.0)
    p.add_argument("--sw_projections", type=int, default=64)
    p.add_argument("--recon_loss", type=str, default="l1", choices=["l1", "l2"])
    p.add_argument("--stage2_n_aug", type=int, default=3)
    p.add_argument("--stage2_aug_weight", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--sample_n", type=int, default=64)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--mode", type=str, default="train_all", choices=["train_all", "train_stage1", "train_stage2", "sample"])
    p.add_argument("--stage1_ckpt", type=str, default="")
    p.add_argument("--stage2_ckpt", type=str, default="")
    return CFG(**vars(p.parse_args()))


def main():
    cfg = parse_args()
    seed_everything(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device(cfg.device)
    print(f"device={device}")

    dl_train_stage1, dl_train_stage2, dl_train_plain, dl_test_plain = make_loaders(cfg)
    enc = StochasticEncoder(cfg.latent_dim).to(device)
    dec = Decoder(cfg.latent_dim).to(device)

    if cfg.mode == "train_stage1":
        start_epoch_stage1 = maybe_load_stage1_for_resume(cfg, enc, device)
        train_stage1(cfg, enc, dl_train_stage1, dl_test_plain, device, start_epoch=start_epoch_stage1)
        return

    if cfg.mode == "train_stage2":
        load_stage1_if_needed(cfg, enc, device)
        start_epoch_stage2 = maybe_load_stage2_for_resume(cfg, dec, device)
        train_stage2(cfg, enc, dec, dl_train_stage2, dl_test_plain, device, start_epoch=start_epoch_stage2)
        return

    if cfg.mode == "train_all":
        start_epoch_stage1 = maybe_load_stage1_for_resume(cfg, enc, device)
        train_stage1(cfg, enc, dl_train_stage1, dl_test_plain, device, start_epoch=start_epoch_stage1)
        load_stage1_if_needed(cfg, enc, device)
        start_epoch_stage2 = maybe_load_stage2_for_resume(cfg, dec, device)
        train_stage2(cfg, enc, dec, dl_train_stage2, dl_test_plain, device, start_epoch=start_epoch_stage2)
        return

    if cfg.mode == "sample":
        load_stage2_if_needed(cfg, dec, device)
        save_samples(dec, cfg.latent_dim, device, os.path.join(cfg.out_dir, "samples_only.png"), n=cfg.sample_n)
        return


if __name__ == "__main__":
    main()
