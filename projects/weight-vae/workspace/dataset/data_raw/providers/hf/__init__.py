"""HF adapters package for raw dataset virtualization."""

from importlib import import_module

_ADAPTER_MODULES = [
    "dataset.data_raw.providers.hf.bigearthnet",
    "dataset.data_raw.providers.hf.bdd100k",
    "dataset.data_raw.providers.hf.cc12m",
    "dataset.data_raw.providers.hf.chexpert",
    "dataset.data_raw.providers.hf.coco2017",
    "dataset.data_raw.providers.hf.cord_v2",
    "dataset.data_raw.providers.hf.doclaynet_v11",
    "dataset.data_raw.providers.hf.docvqa_1200",
    "dataset.data_raw.providers.hf.dtd_textures",
    "dataset.data_raw.providers.hf.eurosat_rgb",
    "dataset.data_raw.providers.hf.flickr30k",
    "dataset.data_raw.providers.hf.food101",
    "dataset.data_raw.providers.hf.funsd",
    "dataset.data_raw.providers.hf.mapillary_vistas_v2",
    "dataset.data_raw.providers.hf.openimages_v7",
    "dataset.data_raw.providers.hf.oxford_pets",
    "dataset.data_raw.providers.hf.patchcamelyon",
    "dataset.data_raw.providers.hf.pascal_voc_2012",
    "dataset.data_raw.providers.hf.relaion400m",
    "dataset.data_raw.providers.hf.rvl_cdip",
    "dataset.data_raw.providers.hf.scene_parse_150",
    "dataset.data_raw.providers.hf.stanford_cars",
    "dataset.data_raw.providers.hf.sun397",
    "dataset.data_raw.providers.hf.visual_genome",
    "dataset.data_raw.providers.hf.wider_face",
]


def register_all_adapters() -> None:
    for module_name in _ADAPTER_MODULES:
        import_module(module_name)


__all__ = ["register_all_adapters"]
