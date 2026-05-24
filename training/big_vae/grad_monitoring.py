from __future__ import annotations

from training.big_vae.collector_monitoring import *
from training.big_vae.grad_plots import *
from training.big_vae.grad_stats import *

__all__ = [
    '_layer_key_from_param_name',
    '_grad_stat_group_prefixes',
    '_grad_group_name_for_param',
    '_parameter_count_summary',
    '_collect_params_by_grad_group',
    '_grad_l2_norm_for_params',
    '_clip_grad_norm_with_optional_foreach',
    '_clip_return_to_float',
    'compute_grad_stats',
    '_collect_nonfinite_grad_report',
    'collect_grad_rms_per_layer',
    'collect_param_rms_per_layer',
    '_append_grad_layer_rms_csv',
    '_save_grad_rms_layer_plot',
    '_save_grad_rms_layer_heatmap',
    '_build_collector_status_snapshot',
    '_collector_tracked_children',
    '_get_encoder_conditioning_alpha_values',
    '_get_patch_tokenizer_block_alpha_stats',
    '_get_patch_latent_variance_stats',
]
