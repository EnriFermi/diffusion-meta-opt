from __future__ import annotations

from dataset.big_vae_offline_parts.metadata import *
from dataset.big_vae_offline_parts.chunk_io import *
from dataset.big_vae_offline_parts.offline_dataset import *
from dataset.big_vae_offline_parts.presliced_utils import *
from dataset.big_vae_offline_parts.presliced_dataset import *
from dataset.big_vae_offline_parts.builder import *

__all__ = [
    "OFFLINE_BIG_VAE_FORMAT_VERSION",
    "PRESLICED_BIG_VAE_FORMAT_VERSION",
    "_DEPTH_PATTERNS",
    "_BALANCED_SAMPLING_GROUP_KEY_ALIASES",
    "resolve_big_vae_curriculum_targets",
    "resolve_offline_target_size_bytes",
    "estimate_stage_slice_capacity",
    "_prepare_cpu_sample_tensor",
    "_sample_dataset_names",
    "_primary_dataset_name",
    "infer_layer_type",
    "infer_layer_depth",
    "_shape_key",
    "_pow2_bucket",
    "_source_key",
    "_weight_path",
    "_chunk_index_path",
    "_normalize_sampling_group_keys",
    "_normalize_offline_sampling_mode",
    "_chunk_record_index_entry",
    "_atomic_write_json",
    "_load_json_payload",
    "_directory_size_bytes",
    "BigVAEOfflineDatasetWriter",
    "OfflineBigVAEDataset",
    "_preslicing_cfg",
    "_preslicing_source_root",
    "_preslicing_root",
    "_file_sha1",
    "_source_offline_dataset_fingerprint",
    "resolve_presliced_big_vae_spec",
    "_presliced_manifest_path",
    "_presliced_manifest_matches",
    "_presliced_pad_x",
    "_presliced_pad_matrix",
    "_presliced_index_groups",
    "PreslicedBigVAEWriter",
    "PreslicedBigVAEDataset",
    "ensure_presliced_big_vae_dataset",
    "presliced_big_vae_data_pipeline",
    "offline_big_vae_data_pipeline",
    "build_big_vae_offline_dataset",
]
