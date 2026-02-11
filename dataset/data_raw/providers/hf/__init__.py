"""HF adapters package for raw dataset virtualization."""

from importlib import import_module

_ADAPTER_MODULES = [
    "dataset.data_raw.providers.hf.bdd100k",
    "dataset.data_raw.providers.hf.cc12m",
    "dataset.data_raw.providers.hf.coco2017",
    "dataset.data_raw.providers.hf.flickr30k",
    "dataset.data_raw.providers.hf.mapillary_vistas_v2",
    "dataset.data_raw.providers.hf.relaion400m",
    "dataset.data_raw.providers.hf.scene_parse_150",
    "dataset.data_raw.providers.hf.visual_genome",
]


def register_all_adapters() -> None:
    for module_name in _ADAPTER_MODULES:
        import_module(module_name)


__all__ = ["register_all_adapters"]
