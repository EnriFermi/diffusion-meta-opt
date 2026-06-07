from __future__ import annotations

from .types import *

class BigVAELatentTensorStore(nn.Module):
    def __init__(
        self,
        initial_tensors: dict[str, torch.Tensor],
        *,
        big_vae: BigWeightVAE,
        latent_init: str,
        latent_space: str | None = None,
        latent_parameterization: str = "euclidean",
        random_init_std: float,
        latent_noise_std: float,
        decode_policy: str,
        tile_T_patches: int,
        tile_d_out: int,
        latent_diffusion_prior: Any | None = None,
        latent_diffusion_prior_steps: int = 50,
        latent_diffusion_prior_sampler: str = "ddim",
        latent_diffusion_prior_eta: float = 0.0,
        big_vae_decoder_flow: nn.Module | None = None,
        encoder_context_rows: int = 64,
        encoder_context_std: float = 1.0,
        encoder_batch_size: int = 16,
    ) -> None:
        super().__init__()
        init_mode = str(latent_init).strip().lower()
        if init_mode not in {"base", "random", "encoded", "diffusion_prior"}:
            raise ValueError(
                "big_vae_latent_init must be 'base', 'random', 'encoded' or 'diffusion_prior', "
                f"got {latent_init!r}"
            )
        policy = str(decode_policy).strip().lower()
        if policy not in {"weights", "all"}:
            raise ValueError(f"big_vae_decode must be 'weights' or 'all', got {decode_policy!r}")
        if int(tile_T_patches) <= 0:
            raise ValueError(f"big_vae_tile_T_patches must be > 0, got {tile_T_patches}")
        if int(tile_d_out) <= 0:
            raise ValueError(f"big_vae_tile_d_out must be > 0, got {tile_d_out}")

        self.big_vae = big_vae
        self.big_vae.eval()
        for param in self.big_vae.parameters():
            param.requires_grad_(False)

        self._name_to_key: dict[str, str] = {}
        self._key_to_name: dict[str, str] = {}
        self._specs: dict[str, TensorMatrixSpec] = {}
        self._tile_specs: dict[str, BigVAEDecodeTileSpec] = {}
        self._tensor_key_to_tile_keys: dict[str, list[str]] = {}
        self._groups: dict[tuple[int, int, int], list[str]] = {}
        self._direct_name_to_key: dict[str, str] = {}
        self._tile_cond_patch: dict[str, torch.Tensor] = {}
        self.latent_slots = nn.ParameterDict()
        self.latent_radii = nn.ModuleDict()
        self.direct_tensors = nn.ModuleDict()
        self.latent_init_mode = init_mode
        self.latent_parameterization = str(latent_parameterization).strip().lower()
        if self.latent_parameterization not in {"euclidean", "sphere"}:
            raise ValueError(
                "latent_parameterization must be one of {'euclidean', 'sphere'}, "
                f"got {latent_parameterization!r}"
            )
        resolved_latent_space = str(latent_space).strip().lower() if latent_space is not None else ""
        if not resolved_latent_space:
            resolved_latent_space = "decoder_z" if init_mode == "diffusion_prior" else "encoder_slots"
        if resolved_latent_space not in {"encoder_slots", "decoder_z"}:
            raise ValueError(
                "latent_space must be one of {'encoder_slots', 'decoder_z'}, "
                f"got {latent_space!r}"
            )
        if init_mode == "diffusion_prior" and resolved_latent_space != "decoder_z":
            raise ValueError("big_vae_latent_init='diffusion_prior' requires latent_space='decoder_z'")
        self.latent_space = resolved_latent_space
        self.patch_size = int(self.big_vae.cfg.patch_size)
        self.tile_T_patches = int(tile_T_patches)
        self.tile_d_in = int(self.patch_size) * int(self.tile_T_patches)
        self.tile_d_out = int(tile_d_out)
        self.use_distribution_encoder = bool(getattr(self.big_vae, "use_distribution_encoder", False))
        self.d_dist = int(self.big_vae.cfg.distribution.d_dist)
        self.encoder_context_rows = max(1, int(encoder_context_rows))
        self.encoder_context_std = float(encoder_context_std)
        self.encoder_batch_size = max(1, int(encoder_batch_size))
        self.random_init_std = float(random_init_std)
        self.latent_noise_std = float(latent_noise_std)
        self.latent_diffusion_prior = latent_diffusion_prior
        self.latent_diffusion_prior_steps = max(1, int(latent_diffusion_prior_steps))
        self.latent_diffusion_prior_sampler = str(latent_diffusion_prior_sampler).strip().lower()
        self.latent_diffusion_prior_eta = float(latent_diffusion_prior_eta)
        self.big_vae_decoder_flow = big_vae_decoder_flow
        if self.big_vae_decoder_flow is not None:
            if self.latent_space != "decoder_z":
                raise ValueError("big_vae_decoder_flow requires latent_space='decoder_z'")
            self.big_vae_decoder_flow.to(device=self.big_vae.latent_base.device)
            self.big_vae_decoder_flow.eval()
            for param in self.big_vae_decoder_flow.parameters():
                param.requires_grad_(False)
        if self.latent_diffusion_prior is not None:
            self.latent_diffusion_prior.to(device=self.big_vae.latent_base.device)
            self.latent_diffusion_prior.eval()
            for param in self.latent_diffusion_prior.parameters():
                param.requires_grad_(False)
        if self.latent_init_mode == "diffusion_prior":
            if self.latent_diffusion_prior is None:
                raise ValueError(
                    "big_vae_latent_init='diffusion_prior' requires latent_diffusion_prior checkpoint/model"
                )
            if not self.use_distribution_encoder:
                raise ValueError(
                    "big_vae_latent_init='diffusion_prior' requires BigVAE distribution encoder to be enabled"
                )

        base_latents = self.big_vae.latent_base.detach().clone()
        for idx, (name, initial) in enumerate(initial_tensors.items()):
            tensor_key = f"p{idx:04d}"
            if not self._should_decode_with_big_vae(name=name, tensor=initial, policy=policy):
                self._direct_name_to_key[name] = tensor_key
                self.direct_tensors[tensor_key] = DirectTensor(initial)
                continue

            spec = make_tensor_matrix_spec(
                initial,
                output_first_dim=bool(initial.ndim >= 2 and str(name).endswith(".weight")),
            )
            rows, cols = spec.matrix_shape
            self._name_to_key[name] = tensor_key
            self._key_to_name[tensor_key] = name
            self._specs[tensor_key] = spec
            self._tensor_key_to_tile_keys[tensor_key] = []

            pending_segments: list[BigVAETileSegment] = []
            tile_index = 0

            def flush_tile() -> None:
                nonlocal pending_segments, tile_index
                if not pending_segments:
                    return
                tile_key = f"{tensor_key}_t{tile_index:04d}"
                tile_index += 1
                self._tile_specs[tile_key] = BigVAEDecodeTileSpec(
                    d_in=int(self.tile_d_in),
                    d_out=int(self.tile_d_out),
                    T=int(self.tile_T_patches),
                    segments=list(pending_segments),
                )
                self._tensor_key_to_tile_keys[tensor_key].append(tile_key)
                group_key = (int(self.tile_d_in), int(self.tile_d_out), int(self.tile_T_patches))
                self._groups.setdefault(group_key, []).append(tile_key)

                if init_mode in {"base", "encoded", "diffusion_prior"}:
                    latent = base_latents.clone()
                elif init_mode == "random":
                    latent = base_latents.clone() + torch.randn_like(base_latents) * self.random_init_std
                else:
                    latent = base_latents.clone()
                if float(latent_noise_std) > 0.0:
                    latent = latent + torch.randn_like(latent) * float(latent_noise_std)
                self.latent_slots[tile_key] = nn.Parameter(latent.detach().clone())
                self.latent_radii[tile_key] = FrozenBufferTensor(
                    torch.zeros((), device=latent.device, dtype=latent.dtype)
                )
                self._load_materialized_latent_slot_(
                    tile_key,
                    latent,
                    update_radius=True,
                    value_is_decoder_latent=True,
                )
                pending_segments = []

            current_tile_rows = 0
            for col_start in range(0, int(cols), int(self.tile_d_out)):
                col_len = min(int(self.tile_d_out), int(cols) - int(col_start))
                for row_start in range(0, int(rows), int(self.patch_size)):
                    row_len = min(int(self.patch_size), int(rows) - int(row_start))
                    if current_tile_rows > 0 and current_tile_rows + int(row_len) > int(self.tile_d_in):
                        flush_tile()
                        current_tile_rows = 0
                    pending_segments.append(
                        BigVAETileSegment(
                            tensor_name=name,
                            tensor_key=tensor_key,
                            tile_row_start=int(current_tile_rows),
                            row_start=int(row_start),
                            row_len=int(row_len),
                            col_start=int(col_start),
                            col_len=int(col_len),
                        )
                    )
                    current_tile_rows += int(row_len)
                    if current_tile_rows >= int(self.tile_d_in):
                        flush_tile()
                        current_tile_rows = 0
            flush_tile()

        if init_mode == "encoded":
            self._initialize_latents_from_encoder(initial_tensors)

    @staticmethod
    def _should_decode_with_big_vae(*, name: str, tensor: torch.Tensor, policy: str) -> bool:
        if policy == "all":
            return True
        # BigVAE was trained on 2D layer weight matrices. Keep conv kernels,
        # pos_embed, bias and LayerNorm tensors direct by default.
        return tensor.ndim == 2 and str(name).endswith(".weight")

    def decode_group_count(self) -> int:
        return int(len(self._groups))

    def decoded_tile_count(self) -> int:
        return int(len(self._tile_specs))

    def decoded_tensor_names(self) -> list[str]:
        return list(self._name_to_key.keys())

    def tile_decode_shape(self) -> tuple[int, int, int]:
        return int(self.tile_d_in), int(self.tile_d_out), int(self.tile_T_patches)

    def latent_numel(self) -> int:
        return int(sum(param.numel() for param in self.latent_slots.values()))

    @staticmethod
    def _stable_norm(value: torch.Tensor) -> torch.Tensor:
        flat = value.reshape(-1)
        tiny = max(float(torch.finfo(value.dtype).eps), 1e-12)
        return flat.norm().clamp_min(tiny)

    def materialize_latent_slot(self, key: str) -> torch.Tensor:
        latent = self.latent_slots[key]
        if self.latent_parameterization != "sphere":
            return latent
        radius = self.latent_radii[key]().to(device=latent.device, dtype=latent.dtype)
        return latent * (radius / self._stable_norm(latent))

    def _to_decoder_adapter_latent(self, value: torch.Tensor) -> torch.Tensor:
        if self.big_vae_decoder_flow is None:
            return value
        if self.latent_space != "decoder_z":
            raise ValueError("big_vae_decoder_flow requires latent_space='decoder_z'")
        flat_value = value.reshape(1, -1).to(dtype=torch.float32)
        adapter_flat = self.big_vae_decoder_flow(flat_value)[0]
        return adapter_flat.to(device=value.device, dtype=value.dtype).reshape_as(value)

    @torch.no_grad()
    def materialized_latent_slots_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            str(key): self.materialize_latent_slot(str(key)).detach().cpu().contiguous()
            for key in self.latent_slots.keys()
        }

    @torch.no_grad()
    def _load_materialized_latent_slot_(
        self,
        key: str,
        value: torch.Tensor,
        *,
        update_radius: bool,
        value_is_decoder_latent: bool = False,
    ) -> None:
        target = self.latent_slots[key]
        value = value.detach().to(device=target.device, dtype=target.dtype).contiguous()
        if value_is_decoder_latent:
            value = self._to_decoder_adapter_latent(value)
        target.copy_(value)
        if self.latent_parameterization == "sphere" and update_radius:
            radius = value.reshape(-1).norm()
            self.latent_radii[key].copy_(radius.reshape(()))

    @torch.no_grad()
    def load_materialized_latent_slots_state_dict(
        self,
        state: dict[str, torch.Tensor],
        *,
        strict: bool = True,
        update_radii: bool = True,
    ) -> None:
        missing = [str(key) for key in self.latent_slots.keys() if key not in state]
        unexpected = [str(key) for key in state.keys() if key not in self.latent_slots]
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"materialized latent slot state mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
            )
        for key, value in state.items():
            if key not in self.latent_slots:
                continue
            self._load_materialized_latent_slot_(str(key), value, update_radius=bool(update_radii))

    def decoded_numel(self) -> int:
        total = 0
        for spec in self._specs.values():
            numel = 1
            for dim in spec.original_shape:
                numel *= int(dim)
            total += int(numel)
        for module in self.direct_tensors.values():
            if isinstance(module, DirectTensor):
                total += int(module.value.numel())
        return int(total)

    def big_vae_decoded_numel(self) -> int:
        total = 0
        for spec in self._specs.values():
            numel = 1
            for dim in spec.original_shape:
                numel *= int(dim)
            total += int(numel)
        return int(total)

    def latent_init_diversity(self) -> dict[str, float]:
        if not self.latent_slots:
            return {"count": 0.0, "across_layer_std_mean": 0.0, "max_pair_delta": 0.0}
        stacked = torch.stack(
            [self.materialize_latent_slot(str(key)).detach().float().cpu() for key in self.latent_slots.keys()],
            dim=0,
        )
        if int(stacked.shape[0]) <= 1:
            return {"count": float(stacked.shape[0]), "across_layer_std_mean": 0.0, "max_pair_delta": 0.0}
        centered = stacked - stacked.mean(dim=0, keepdim=True)
        return {
            "count": float(stacked.shape[0]),
            "across_layer_std_mean": float(stacked.std(dim=0, unbiased=False).mean().item()),
            "max_pair_delta": float(centered.abs().max().item()),
        }

    def target_matrix(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        key = self._name_to_key[name]
        return tensor_to_matrix(tensor, self._specs[key])

    def _encoder_context(self, batch_size: int, d_in: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        rows = torch.arange(self.encoder_context_rows, device=device, dtype=torch.float32).view(1, -1, 1)
        cols = torch.arange(int(d_in), device=device, dtype=torch.float32).view(1, 1, -1)
        batch_phase = torch.arange(int(batch_size), device=device, dtype=torch.float32).view(-1, 1, 1) * 0.173
        context = torch.sin((rows + 1.0) * (cols + 1.0) * 0.017 + batch_phase)
        return (context * float(self.encoder_context_std)).to(dtype=dtype)

    @staticmethod
    def _decoded_tensor_order_key(name: str) -> tuple[int, int, str]:
        raw = str(name)
        if raw == "patch_embed.weight":
            return (0, 0, raw)
        if raw == "patch_embed.bias":
            return (0, 1, raw)
        if raw == "cls_token":
            return (0, 2, raw)
        if raw == "pos_embed":
            return (0, 3, raw)
        if raw.startswith("blocks."):
            parts = raw.split(".")
            if len(parts) >= 4 and parts[1].isdigit():
                block_idx = int(parts[1])
                suffix = ".".join(parts[2:])
                within_block_order = {
                    "norm1.weight": 0,
                    "norm1.bias": 1,
                    "attn.qkv.weight": 2,
                    "attn.qkv.bias": 3,
                    "attn.proj.weight": 4,
                    "attn.proj.bias": 5,
                    "norm2.weight": 6,
                    "norm2.bias": 7,
                    "mlp.fc1.weight": 8,
                    "mlp.fc1.bias": 9,
                    "mlp.fc2.weight": 10,
                    "mlp.fc2.bias": 11,
                }
                return (1 + block_idx, within_block_order.get(suffix, 100), raw)
        final_order = {
            "norm.weight": 0,
            "norm.bias": 1,
            "head.weight": 2,
            "head.bias": 3,
        }
        if raw in final_order:
            return (10_000, final_order[raw], raw)
        return (20_000, 0, raw)

    def decoded_tensor_execution_order(self) -> list[str]:
        return sorted(self._name_to_key.keys(), key=self._decoded_tensor_order_key)

    def _subsample_activation_rows(self, activation_matrix: torch.Tensor) -> torch.Tensor:
        if activation_matrix.ndim != 2:
            raise ValueError(f"activation_matrix must be [R,d], got {tuple(activation_matrix.shape)}")
        row_count = int(activation_matrix.shape[0])
        if row_count <= int(self.encoder_context_rows):
            return activation_matrix.contiguous()
        indices = torch.linspace(
            0,
            row_count - 1,
            steps=int(self.encoder_context_rows),
            device=activation_matrix.device,
            dtype=torch.float32,
        ).round().to(dtype=torch.long)
        return activation_matrix.index_select(0, indices).contiguous()

    @staticmethod
    def _tile_d_out_mask(tile: BigVAEDecodeTileSpec, *, device: torch.device) -> torch.Tensor:
        d_out_mask = torch.zeros(int(tile.d_out), device=device, dtype=torch.bool)
        used_cols = 0
        for segment in tile.segments:
            used_cols = max(used_cols, int(segment.col_len))
        d_out_mask[:used_cols] = True
        return d_out_mask

    def initialize_from_diffusion_prior_autoregressive(
        self,
        model: nn.Module,
        *,
        calibration_images: torch.Tensor,
    ) -> None:
        if self.latent_init_mode != "diffusion_prior":
            return
        if self.latent_diffusion_prior is None:
            raise RuntimeError("latent_diffusion_prior is required for diffusion_prior initialization")
        if calibration_images.ndim != 4:
            raise ValueError(f"calibration_images must be [B,C,H,W], got {tuple(calibration_images.shape)}")
        if not self._name_to_key:
            return

        first_latent = next(iter(self.latent_slots.values()))
        device = first_latent.device
        dtype = first_latent.dtype
        images = calibration_images.to(device=device, dtype=dtype, non_blocking=False)

        with torch.no_grad():
            for tensor_name in self.decoded_tensor_execution_order():
                X_full = getattr(model, "_orig_mod", model).collect_parameter_input_matrix(tensor_name, images)
                X_full = self._subsample_activation_rows(X_full.to(device=device, dtype=dtype))
                tensor_key = self._name_to_key[tensor_name]
                tile_keys = self._tensor_key_to_tile_keys.get(tensor_key, [])
                if not tile_keys:
                    continue

                tile_batch = len(tile_keys)
                X = torch.zeros(tile_batch, int(X_full.shape[0]), int(self.tile_d_in), device=device, dtype=dtype)
                x_mask = torch.ones(tile_batch, int(X_full.shape[0]), device=device, dtype=torch.bool)
                d_in_mask = torch.zeros(tile_batch, int(self.tile_d_in), device=device, dtype=torch.bool)
                batch_layer_names = [str(tensor_name)] * tile_batch

                for item_idx, key in enumerate(tile_keys):
                    tile = self._tile_specs[key]
                    for segment in tile.segments:
                        tile_row_start = int(segment.tile_row_start)
                        tile_row_end = tile_row_start + int(segment.row_len)
                        row_start = int(segment.row_start)
                        row_end = row_start + int(segment.row_len)
                        X[item_idx, :, tile_row_start:tile_row_end] = X_full[:, row_start:row_end]
                        d_in_mask[item_idx, tile_row_start:tile_row_end] = True

                (
                    T,
                    _d_in_pad,
                    patch_mask,
                    _structural_patch_mask,
                    _dist_var_by_patch,
                    dist_patch_by_patch,
                    dist_var_pooled,
                ) = self.big_vae._encode_distribution_context(
                    X,
                    x_mask=x_mask,
                    d_in_mask=d_in_mask,
                )
                expected_T = int(self.tile_T_patches)
                if int(T) != expected_T:
                    raise RuntimeError(
                        f"BigVAE encoder T mismatch for tensor {tensor_name}: expected T={expected_T}, got T={T}"
                    )
                if dist_patch_by_patch is None:
                    raise RuntimeError("distribution encoder must provide dist_patch_by_patch for diffusion prior")

                cond_global = build_cond_global_from_dist_var_pooled(
                    dist_var_pooled=dist_var_pooled,
                    patch_mask=patch_mask,
                )
                metadata_cond = build_layer_metadata_condition_vector(
                    device=device,
                    dtype=dtype,
                    use_layer_type_conditioning=bool(self.latent_diffusion_prior.cfg.use_layer_type_conditioning),
                    use_layer_depth_conditioning=bool(self.latent_diffusion_prior.cfg.use_layer_depth_conditioning),
                    depth_fourier_dim=int(self.latent_diffusion_prior.cfg.layer_depth_fourier_dim),
                    depth_scale=float(self.latent_diffusion_prior.cfg.layer_depth_scale),
                    layer_names=batch_layer_names,
                )
                if metadata_cond is not None:
                    if cond_global is None:
                        cond_global = metadata_cond
                    else:
                        cond_global = torch.cat(
                            [
                                cond_global.to(device=device, dtype=dtype),
                                metadata_cond.to(device=device, dtype=dtype),
                            ],
                            dim=-1,
                        )

                sampled = self.latent_diffusion_prior.sample_latents(
                    cond_patch=dist_patch_by_patch,
                    cond_global=cond_global,
                    patch_mask=patch_mask,
                    num_steps=self.latent_diffusion_prior_steps,
                    sampler_type=self.latent_diffusion_prior_sampler,
                    eta=self.latent_diffusion_prior_eta,
                )
                sampled = sampled.view(tile_batch, *first_latent.shape)
                for item_idx, key in enumerate(tile_keys):
                    sampled_item = sampled[item_idx].detach().to(device=device, dtype=dtype)
                    self._tile_cond_patch[key] = dist_patch_by_patch[item_idx].detach().to(device=device, dtype=dtype)
                    if self.latent_noise_std > 0.0:
                        sampled_item = sampled_item + torch.randn_like(sampled_item) * self.latent_noise_std
                    self._load_materialized_latent_slot_(key, sampled_item, update_radius=True)

    def _initialize_latents_from_encoder(self, initial_tensors: dict[str, torch.Tensor]) -> None:
        if not self.latent_slots:
            return
        first_latent = next(iter(self.latent_slots.values()))
        device = first_latent.device
        dtype = first_latent.dtype
        matrix_cache: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for (d_in, d_out, expected_T), keys in self._groups.items():
                for start in range(0, len(keys), self.encoder_batch_size):
                    batch_keys = keys[start : start + self.encoder_batch_size]
                    batch = int(len(batch_keys))
                    W = torch.zeros(batch, int(d_in), int(d_out), device=device, dtype=dtype)
                    d_in_mask = torch.zeros(batch, int(d_in), device=device, dtype=torch.bool)
                    d_out_mask = torch.zeros(batch, int(d_out), device=device, dtype=torch.bool)
                    batch_layer_names: list[str] = []

                    for item_idx, key in enumerate(batch_keys):
                        tile = self._tile_specs[key]
                        batch_layer_names.append(str(tile.segments[0].tensor_name) if tile.segments else str(key))
                        for segment in tile.segments:
                            matrix = matrix_cache.get(segment.tensor_name)
                            if matrix is None:
                                source = initial_tensors[segment.tensor_name].to(device=device, dtype=dtype)
                                matrix = tensor_to_matrix(source, self._specs[segment.tensor_key])
                                matrix_cache[segment.tensor_name] = matrix

                            tile_row_start = int(segment.tile_row_start)
                            tile_row_end = tile_row_start + int(segment.row_len)
                            row_start = int(segment.row_start)
                            row_end = row_start + int(segment.row_len)
                            col_start = int(segment.col_start)
                            col_end = col_start + int(segment.col_len)
                            W[item_idx, tile_row_start:tile_row_end, : int(segment.col_len)] = matrix[
                                row_start:row_end,
                                col_start:col_end,
                            ]
                            d_in_mask[item_idx, tile_row_start:tile_row_end] = True
                            d_out_mask[item_idx, : int(segment.col_len)] = True

                    X = self._encoder_context(batch, int(d_in), device=device, dtype=dtype)
                    X = X * d_in_mask.to(dtype=dtype).unsqueeze(1)
                    x_mask = torch.ones(batch, self.encoder_context_rows, device=device, dtype=torch.bool)
                    (
                        T,
                        d_in_pad,
                        patch_mask,
                        _structural_patch_mask,
                        dist_var_by_patch,
                        dist_patch_by_patch,
                        dist_var_pooled,
                    ) = self.big_vae._encode_distribution_context(
                        X,
                        x_mask=x_mask,
                        d_in_mask=d_in_mask,
                    )
                    if int(T) != int(expected_T):
                        raise RuntimeError(
                            f"BigVAE encoder T mismatch for group {(d_in, d_out, expected_T)}: got T={T}"
                        )
                    latents = self.big_vae._encode_latent_slots(
                        W,
                        T=int(T),
                        d_in_pad=int(d_in_pad),
                        patch_mask=patch_mask,
                        d_out_mask=d_out_mask,
                        dist_var_by_patch=dist_var_by_patch,
                        dist_patch_by_patch=dist_patch_by_patch,
                        dist_var_pooled=dist_var_pooled,
                    )
                    for item_idx, key in enumerate(batch_keys):
                        encoded = latents[item_idx].detach().to(device=device, dtype=dtype)
                        if dist_patch_by_patch is not None:
                            cond_patch = dist_patch_by_patch[item_idx].detach().to(device=device, dtype=dtype)
                            self._tile_cond_patch[key] = cond_patch
                        if self.latent_noise_std > 0.0:
                            encoded = encoded + torch.randn_like(encoded) * self.latent_noise_std
                        self._load_materialized_latent_slot_(
                            key,
                            encoded,
                            update_radius=True,
                            value_is_decoder_latent=self.latent_space == "decoder_z",
                        )

    def _initialize_latents_from_diffusion_prior(self, initial_tensors: dict[str, torch.Tensor]) -> None:
        if not self.latent_slots:
            return
        if self.latent_diffusion_prior is None:
            raise RuntimeError("latent_diffusion_prior is required for diffusion_prior initialization")
        first_latent = next(iter(self.latent_slots.values()))
        device = first_latent.device
        dtype = first_latent.dtype
        matrix_cache: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for (d_in, d_out, expected_T), keys in self._groups.items():
                for start in range(0, len(keys), self.encoder_batch_size):
                    batch_keys = keys[start : start + self.encoder_batch_size]
                    batch = int(len(batch_keys))
                    W = torch.zeros(batch, int(d_in), int(d_out), device=device, dtype=dtype)
                    d_in_mask = torch.zeros(batch, int(d_in), device=device, dtype=torch.bool)
                    d_out_mask = torch.zeros(batch, int(d_out), device=device, dtype=torch.bool)
                    batch_layer_names: list[str] = []

                    for item_idx, key in enumerate(batch_keys):
                        tile = self._tile_specs[key]
                        batch_layer_names.append(str(tile.segments[0].tensor_name) if tile.segments else str(key))
                        for segment in tile.segments:
                            matrix = matrix_cache.get(segment.tensor_name)
                            if matrix is None:
                                source = initial_tensors[segment.tensor_name].to(device=device, dtype=dtype)
                                matrix = tensor_to_matrix(source, self._specs[segment.tensor_key])
                                matrix_cache[segment.tensor_name] = matrix

                            tile_row_start = int(segment.tile_row_start)
                            tile_row_end = tile_row_start + int(segment.row_len)
                            row_start = int(segment.row_start)
                            row_end = row_start + int(segment.row_len)
                            col_start = int(segment.col_start)
                            col_end = col_start + int(segment.col_len)
                            W[item_idx, tile_row_start:tile_row_end, : int(segment.col_len)] = matrix[
                                row_start:row_end,
                                col_start:col_end,
                            ]
                            d_in_mask[item_idx, tile_row_start:tile_row_end] = True
                            d_out_mask[item_idx, : int(segment.col_len)] = True

                    X = self._encoder_context(batch, int(d_in), device=device, dtype=dtype)
                    X = X * d_in_mask.to(dtype=dtype).unsqueeze(1)
                    x_mask = torch.ones(batch, self.encoder_context_rows, device=device, dtype=torch.bool)
                    (
                        T,
                        _d_in_pad,
                        patch_mask,
                        _structural_patch_mask,
                        _dist_var_by_patch,
                        dist_patch_by_patch,
                        _dist_var_pooled,
                    ) = self.big_vae._encode_distribution_context(
                        X,
                        x_mask=x_mask,
                        d_in_mask=d_in_mask,
                    )
                    if int(T) != int(expected_T):
                        raise RuntimeError(
                            f"BigVAE encoder T mismatch for group {(d_in, d_out, expected_T)}: got T={T}"
                        )
                    if dist_patch_by_patch is None:
                        raise RuntimeError("distribution encoder must provide dist_patch_by_patch for diffusion prior")
                    cond_global = build_cond_global_from_dist_var_pooled(
                        dist_var_pooled=_dist_var_pooled,
                        patch_mask=patch_mask,
                    )
                    metadata_cond = build_layer_metadata_condition_vector(
                        device=device,
                        dtype=dtype,
                        use_layer_type_conditioning=bool(
                            self.latent_diffusion_prior.cfg.use_layer_type_conditioning
                        ),
                        use_layer_depth_conditioning=bool(
                            self.latent_diffusion_prior.cfg.use_layer_depth_conditioning
                        ),
                        depth_fourier_dim=int(self.latent_diffusion_prior.cfg.layer_depth_fourier_dim),
                        depth_scale=float(self.latent_diffusion_prior.cfg.layer_depth_scale),
                        layer_names=batch_layer_names,
                    )
                    if metadata_cond is not None:
                        if cond_global is None:
                            cond_global = metadata_cond
                        else:
                            cond_global = torch.cat(
                                [
                                    cond_global.to(device=device, dtype=dtype),
                                    metadata_cond.to(device=device, dtype=dtype),
                                ],
                                dim=-1,
                            )
                    sampled = self.latent_diffusion_prior.sample_latents(
                        cond_patch=dist_patch_by_patch,
                        cond_global=cond_global,
                        patch_mask=patch_mask,
                        num_steps=self.latent_diffusion_prior_steps,
                        sampler_type=self.latent_diffusion_prior_sampler,
                        eta=self.latent_diffusion_prior_eta,
                    )
                    sampled = sampled.view(batch, *first_latent.shape)
                    for item_idx, key in enumerate(batch_keys):
                        sampled_item = sampled[item_idx].detach().to(device=device, dtype=dtype)
                        cond_patch = dist_patch_by_patch[item_idx].detach().to(device=device, dtype=dtype)
                        self._tile_cond_patch[key] = cond_patch
                        if self.latent_noise_std > 0.0:
                            sampled_item = sampled_item + torch.randn_like(sampled_item) * self.latent_noise_std
                        self._load_materialized_latent_slot_(
                            key,
                            sampled_item,
                            update_radius=True,
                            value_is_decoder_latent=True,
                        )

    def decoded_matrix(self, name: str) -> torch.Tensor:
        return self.decode_all_matrices()[name]

    def decode_all_matrices(self) -> dict[str, torch.Tensor]:
        if not self.latent_slots:
            return {}

        first_latent = next(iter(self.latent_slots.values()))
        result: dict[str, torch.Tensor] = {}
        for tensor_key, spec in self._specs.items():
            name = self._key_to_name[tensor_key]
            result[name] = torch.zeros(
                spec.matrix_shape,
                device=first_latent.device,
                dtype=first_latent.dtype,
            )

        for (d_in, d_out, T), keys in self._groups.items():
            latents = torch.stack([self.materialize_latent_slot(str(key)) for key in keys], dim=0)
            batch = int(latents.shape[0])
            d_in_pad = int(T) * self.patch_size
            device = latents.device
            patch_mask = torch.zeros(batch, int(T), device=device, dtype=torch.bool)
            d_in_mask = torch.zeros(batch, int(d_in), device=device, dtype=torch.bool)
            d_out_mask = torch.zeros(batch, int(d_out), device=device, dtype=torch.bool)
            for item_idx, key in enumerate(keys):
                tile = self._tile_specs[key]
                used_rows = 0
                used_cols = 0
                for segment in tile.segments:
                    segment_row_end = int(segment.tile_row_start) + int(segment.row_len)
                    d_in_mask[item_idx, int(segment.tile_row_start) : segment_row_end] = True
                    used_rows = max(used_rows, segment_row_end)
                    used_cols = max(used_cols, int(segment.col_len))
                valid_patches = int(math.ceil(float(used_rows) / float(self.patch_size)))
                patch_mask[item_idx, :valid_patches] = True
                d_out_mask[item_idx, :used_cols] = True
            dist_patch = None
            if self.use_distribution_encoder:
                if all(key in self._tile_cond_patch for key in keys):
                    dist_patch = torch.stack([self._tile_cond_patch[key].to(device=device, dtype=latents.dtype) for key in keys], dim=0)
                else:
                    dist_patch = torch.zeros(batch, int(T), self.d_dist, device=device, dtype=latents.dtype)
            if self.latent_space == "decoder_z":
                decoder_latents = latents
                if self.big_vae_decoder_flow is not None:
                    decoder_latents_flat = self.big_vae_decoder_flow.inverse(
                        latents.reshape(batch, -1).to(dtype=torch.float32)
                    )[0]
                    decoder_latents = decoder_latents_flat.to(device=latents.device, dtype=latents.dtype).reshape_as(latents)
                decoded = self.big_vae._decode_from_decoder_latent(
                    decoder_latents,
                    dist_patch_by_patch=dist_patch,
                    patch_mask=patch_mask,
                    d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask,
                    d_in=int(d_in),
                    d_out=int(d_out),
                    d_in_pad=d_in_pad,
                    T=int(T),
                )[0]
            else:
                decoded = self.big_vae._decode_from_latent_slots(
                    latents,
                    dist_patch_by_patch=dist_patch,
                    patch_mask=patch_mask,
                    d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask,
                    d_in=int(d_in),
                    d_out=int(d_out),
                    d_in_pad=d_in_pad,
                    T=int(T),
                )[0]
            for item_idx, key in enumerate(keys):
                tile = self._tile_specs[key]
                for segment in tile.segments:
                    tile_row_start = int(segment.tile_row_start)
                    tile_row_end = tile_row_start + int(segment.row_len)
                    result[segment.tensor_name][
                        int(segment.row_start) : int(segment.row_start) + int(segment.row_len),
                        int(segment.col_start) : int(segment.col_start) + int(segment.col_len),
                    ] = decoded[item_idx, tile_row_start:tile_row_end, : int(segment.col_len)]
        return result

    def decode_all_tensors(self) -> dict[str, torch.Tensor]:
        return {
            name: matrix_to_tensor(matrix, self._specs[self._name_to_key[name]])
            for name, matrix in self.decode_all_matrices().items()
        }

    def tensor(self, name: str) -> torch.Tensor:
        direct_key = self._direct_name_to_key.get(name)
        if direct_key is not None:
            return self.direct_tensors[direct_key]()
        key = self._name_to_key[name]
        return matrix_to_tensor(self.decoded_matrix(name), self._specs[key])
