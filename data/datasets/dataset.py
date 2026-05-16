import os
import random
from logging import LoggerAdapter

import torch
from torch.utils.data import Dataset, IterableDataset
from yacs.config import CfgNode

from data.datasets.video.toy_coloring import ToyColoringDataset


def _try_import(module_path: str, *names: str):
    import importlib

    try:
        module = importlib.import_module(module_path)
    except (ImportError, ModuleNotFoundError):
        return {name: None for name in names}
    return {name: getattr(module, name, None) for name in names}


_dataset_classes = {}
_dataset_classes.update(_try_import("data.datasets.video.openvid", "OpenVid1MDataset"))
_dataset_classes.update(_try_import("data.datasets.video.openvid_flowception", "OpenVid1MFlowception"))
_dataset_classes.update(_try_import("data.datasets.video.kinetics_flowception", "KineticsDatasetFlowception"))
_dataset_classes.update(_try_import("data.datasets.video.custom_subjectness", "CustomSubjectnessFlowception"))
_dataset_classes.update(
    _try_import(
        "data.datasets.video.custom_subjectness_aug",
        "CustomSubjectnessFlowceptionAug",
    )
)
_dataset_classes.update(_try_import("data.datasets.video.taichi", "TaichiDatasetFlowception"))
_dataset_classes.update(_try_import("data.datasets.video.taichi_cache", "TaichiInMemoryFlowception"))
_dataset_classes.update(_try_import("data.datasets.video.re10k", "Re10kMP4DatasetFlowception"))
_dataset_classes.update(_try_import("data.datasets.video.vchitect_flowception", "VChitectTarFlowception"))
_dataset_classes.update(_try_import("data.datasets.video.youcook2", "YouCook2Flowception"))
_dataset_classes.update(_try_import("data.datasets.video.youcook2_iter", "YouCook2IterFlowception"))
_dataset_classes.update(_try_import("data.datasets.custom_webdataset", "CustomWebDatasetProcessor"))


SUPPORTED_DATASETS = {
    "toy_coloring",
    "openvid1m",
    "openvid1m_flowception",
    "kinetics_flowception",
    "custom_subjectness_flowception",
    "custom_subjectness_flowception_aug",
    "taichi_flowception",
    "taichi_cache_flowception",
    "re10k_flowception",
    "vchitect2_flowception",
    "youcook2",
    "youcook2_iter",
    "custom_webdataset",
    "custom_webdataset_aes",
}


class ResampledShards(IterableDataset):
    """Yield WebDataset shard URLs with a light reshuffle helper."""

    def __init__(self, urls):
        super().__init__()
        self.urls = list(urls)
        random.shuffle(self.urls)

    def __iter__(self):
        for url in self.urls:
            yield {"url": url}

    def reshuffle_shards(self, local_idx=0):
        import time

        generator = torch.Generator().manual_seed(int(time.time()) + local_idx)
        order = torch.randperm(len(self.urls), generator=generator)
        self.urls = [self.urls[i] for i in order]


def _load_paths():
    with open("data/datasets/paths.yaml") as dataf:
        return CfgNode.load_cfg(dataf)


def _path_section(data_paths: CfgNode, section: str, cluster: str):
    if not hasattr(data_paths, section):
        raise KeyError(f"Missing {section} in data/datasets/paths.yaml")
    section_cfg = getattr(data_paths, section)
    cluster_key = cluster.upper()
    if not hasattr(section_cfg, cluster_key):
        raise KeyError(f"Missing {section}.{cluster_key} in data/datasets/paths.yaml")
    return getattr(section_cfg, cluster_key)


def _dataset_class(name: str):
    cls = _dataset_classes.get(name)
    if cls is None:
        raise ImportError(
            f"Dataset class {name} could not be imported. "
            "Install the dataset's optional dependencies or remove this dataset from DATA.DATASET."
        )
    return cls


def _split_cfg_value(value: str) -> list[str]:
    if value is None:
        return []
    values = [item.strip() for item in str(value).split(",")]
    return [item for item in values if item]


def get_all_datasets(cfg: CfgNode, logger: LoggerAdapter, num_gpus: int, seed: int = 0, start_epoch: int = 0):
    dataset_names = _split_cfg_value(cfg.DATA.DATASET)
    dataset_roots = _split_cfg_value(cfg.DATA.DATA_ROOT) or [""]
    dataset_vars = _split_cfg_value(cfg.DATA.DATASET_VARIANT.upper()) or ["ORIGINAL"]

    if len(dataset_roots) == 1 and len(dataset_names) > 1:
        dataset_roots = dataset_roots * len(dataset_names)
    if len(dataset_vars) == 1 and len(dataset_names) > 1:
        dataset_vars = dataset_vars * len(dataset_names)

    train_datasets = []
    for dataset_name, dataset_root, dataset_var in zip(
        dataset_names, dataset_roots, dataset_vars, strict=False
    ):
        train_dataset, _ = get_dataset(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            dataset_var=dataset_var,
            cfg=cfg,
            logger=logger,
            num_gpus=num_gpus,
            seed=seed,
            start_epoch=start_epoch,
        )
        train_datasets.append(train_dataset)

    val_dataset_name = cfg.DATA.VAL_DATASET.lower() if cfg.DATA.VAL_DATASET else dataset_names[0]
    val_dataset_root = cfg.DATA.VAL_DATA_ROOT if cfg.DATA.VAL_DATA_ROOT else dataset_roots[0]
    val_dataset_var = (
        cfg.DATA.VAL_DATASET_VARIANT.upper() if cfg.DATA.VAL_DATASET_VARIANT else dataset_vars[0]
    )
    if not cfg.DATA.VAL_DATASET:
        logger.info(f"No validation dataset specified. Defaulting to {val_dataset_name}.")

    _, val_dataset = get_dataset(
        dataset_name=val_dataset_name,
        dataset_root=val_dataset_root,
        dataset_var=val_dataset_var,
        cfg=cfg,
        logger=logger,
        num_gpus=num_gpus,
        seed=seed,
        start_epoch=start_epoch,
    )

    return train_datasets, val_dataset


def get_extra_datasets(
    cfg: CfgNode, logger: LoggerAdapter, num_gpus: int, seed: int = 0, start_epoch: int = 0
):
    dataset_names = _split_cfg_value(cfg.DATA.EXTRA_DATASETS)
    dataset_roots = _split_cfg_value(cfg.DATA.DATA_ROOT) or [""]
    if len(dataset_roots) == 1 and len(dataset_names) > 1:
        dataset_roots = dataset_roots * len(dataset_names)

    train_datasets = []
    for dataset_name, dataset_root in zip(dataset_names, dataset_roots, strict=False):
        train_dataset, _ = get_dataset(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            dataset_var=cfg.DATA.DATASET_VARIANT,
            cfg=cfg,
            logger=logger,
            num_gpus=num_gpus,
            seed=seed,
            start_epoch=start_epoch,
        )
        train_datasets.append(train_dataset)
    return train_datasets


def _toy_coloring(cfg: CfgNode):
    train_dataset = ToyColoringDataset(
        num_frames=cfg.DATA.MAX_FRAMES,
        min_length=15,
        max_length=cfg.DATA.MAX_FRAMES,
        height=3,
        width=3,
        num_start_frames=cfg.FLOWCEPTION.NUM_START_FRAMES,
        latent_downsample=cfg.MODEL.VAE.TEMPORAL_FACTOR,
        padding_idx=cfg.DATA.PADDING_INDEX,
    )
    val_dataset = ToyColoringDataset(
        num_frames=cfg.DATA.MAX_FRAMES,
        min_length=15,
        max_length=cfg.DATA.MAX_FRAMES,
        height=3,
        width=3,
        num_start_frames=cfg.FLOWCEPTION.NUM_START_FRAMES,
        latent_downsample=cfg.MODEL.VAE.TEMPORAL_FACTOR,
        padding_idx=cfg.DATA.PADDING_INDEX,
        length=256,
    )
    return train_dataset, val_dataset


def _custom_webdataset(
    cfg: CfgNode, paths: CfgNode, num_gpus: int, start_epoch: int, aes_filter: bool = False
):
    import webdataset as wds

    processor_cls = _dataset_class("CustomWebDatasetProcessor")
    root = paths.ORIGINAL.IMG_ROOT
    captions_dir_list = [paths[f"CAPTIONS_{i}"] for i in [0, 1]]
    entropy_dir = paths.ENTROPY

    processor = processor_cls(
        crop=True,
        flip=True,
        img_size=(cfg.SOLVER.IM_SIZE, cfg.SOLVER.IM_SIZE),
        crop_scale=cfg.DATA.CROP_SCALE,
        recap_ratio=cfg.DATA.RECAPTION.RECAPTIONED_RATIO,
        captions_dir=captions_dir_list,
        root_dir=root,
        aes_cond=cfg.DATA.AESTHETIC_COND,
        flip_cond=cfg.DATA.FLIP_COND,
        blur_cond=cfg.DATA.BLUR_COND,
        entropy_dir=entropy_dir,
        explicit_aspect_ratio=cfg.DATA.POWER_COSINE,
    )

    tarfiles = []
    for dirpath, _, filenames in os.walk(root):
        tarfiles.extend(os.path.join(dirpath, name) for name in filenames if name.endswith(".tar"))

    if not tarfiles:
        raise FileNotFoundError(f"No .tar shards found under {root}")

    epoch_length = 380000000 // max(num_gpus * cfg.SOLVER.BATCH_SIZE, 1)
    pipeline_steps = [
        ResampledShards(tarfiles),
        wds.detshuffle(epoch=start_epoch, seed=0),
        wds.tarfile_to_samples(handler=wds.ignore_and_continue),
        wds.split_by_node,
        wds.split_by_worker,
        wds.shuffle(1000),
        wds.map(processor.transform, handler=wds.ignore_and_continue),
    ]
    if aes_filter:
        min_score = cfg.DATA.ENTROPY_THRESHOLD

        def filtering(sample):
            return sample.condition["aesthetic_score"] > min_score

        pipeline_steps.append(wds.select(filtering))

    pipeline_steps.append(
        wds.batched(cfg.SOLVER.BATCH_SIZE, collation_fn=torch.utils.data.default_collate, partial=False)
    )
    dataset = wds.DataPipeline(*pipeline_steps, repetitions=10**7).with_epoch(epoch_length)

    return dataset, dataset


def get_dataset(
    dataset_name: str,
    dataset_root: str,
    dataset_var: str,
    cfg: CfgNode,
    logger: LoggerAdapter,
    num_gpus: int,
    seed: int = 0,
    start_epoch: int = 0,
):
    del dataset_root, dataset_var, seed
    dataset_name = dataset_name.lower()
    cluster = cfg.DATA.CLUSTER.upper()
    logger.info(f"Loading dataset: {dataset_name}")

    if dataset_name == "toy_coloring":
        return _toy_coloring(cfg)

    data_paths = _load_paths()

    if dataset_name == "openvid1m":
        paths = _path_section(data_paths, "OPENVID1M", cluster)
        dataset_cls = _dataset_class("OpenVid1MDataset")
        train_dataset = dataset_cls(
            annotations_dir=paths.ROOT,
            width=cfg.SOLVER.IM_SIZE,
            height=cfg.SOLVER.IM_SIZE,
            num_frames=cfg.DATA.MAX_FRAMES + 8,
            min_motion_score=cfg.DATA.VIDEO.MIN_MOTION_SCORE,
        )
        val_dataset = dataset_cls(
            annotations_dir=paths.ROOT,
            width=cfg.SOLVER.IM_SIZE,
            height=cfg.SOLVER.IM_SIZE,
            num_frames=cfg.DATA.MAX_FRAMES,
            min_motion_score=cfg.DATA.VIDEO.MIN_MOTION_SCORE,
        )
    elif dataset_name == "openvid1m_flowception":
        paths = _path_section(data_paths, "OPENVID1M", cluster)
        dataset_cls = _dataset_class("OpenVid1MFlowception")
        train_dataset = dataset_cls(
            annotations_dir=paths.ROOT,
            width=cfg.SOLVER.IM_SIZE,
            height=cfg.SOLVER.IM_SIZE,
            num_frames=cfg.DATA.MAX_FRAMES + cfg.MODEL.VAE.TEMPORAL_FACTOR,
            min_motion_score=cfg.DATA.VIDEO.MIN_MOTION_SCORE,
            sampling_fps=cfg.DATA.VIDEO.SAMPLING_FPS,
            num_start_frames=cfg.FLOWCEPTION.NUM_START_FRAMES,
            latent_downsample=cfg.MODEL.VAE.TEMPORAL_FACTOR,
        )
        val_dataset = train_dataset
    elif dataset_name == "kinetics_flowception":
        paths = _path_section(data_paths, "KINETICS", cluster)
        dataset_cls = _dataset_class("KineticsDatasetFlowception")
        train_dataset = dataset_cls(
            csv_path=paths.CSV_PATH,
            videos_root=paths.VID_ROOT,
            width=cfg.SOLVER.IM_SIZE,
            height=cfg.SOLVER.IM_SIZE,
            num_frames=cfg.DATA.MAX_FRAMES + cfg.MODEL.VAE.TEMPORAL_FACTOR,
            sampling_fps=cfg.DATA.VIDEO.SAMPLING_FPS,
            num_start_frames=cfg.FLOWCEPTION.NUM_START_FRAMES,
            latent_downsample=cfg.MODEL.VAE.TEMPORAL_FACTOR,
        )
        val_dataset = train_dataset
    elif dataset_name == "custom_subjectness_flowception":
        paths = _path_section(data_paths, "CUSTOM_SUBJECTNESS", cluster)
        dataset_cls = _dataset_class("CustomSubjectnessFlowception")
        train_dataset = dataset_cls(
            annotations_dir=paths.ROOT,
            width=cfg.SOLVER.IM_SIZE,
            height=cfg.SOLVER.IM_SIZE,
            num_frames=cfg.DATA.MAX_FRAMES + cfg.MODEL.VAE.TEMPORAL_FACTOR,
            min_motion_score=cfg.DATA.VIDEO.MIN_MOTION_SCORE,
            sampling_fps=cfg.DATA.VIDEO.SAMPLING_FPS,
            num_start_frames=cfg.FLOWCEPTION.NUM_START_FRAMES,
            latent_downsample=cfg.MODEL.VAE.TEMPORAL_FACTOR,
            min_entropy=cfg.DATA.ENTROPY_THRESHOLD,
            min_subjectness=cfg.DATA.MIN_SUBJECTNESS,
            min_size_ratio=cfg.DATA.MIN_SIZE_RATIO,
            max_size_ratio=cfg.DATA.MAX_SIZE_RATIO,
        )
        val_dataset = train_dataset
    elif dataset_name == "custom_subjectness_flowception_aug":
        paths = _path_section(data_paths, "CUSTOM_SUBJECTNESS", cluster)
        dataset_cls = _dataset_class("CustomSubjectnessFlowceptionAug")
        train_dataset = dataset_cls(
            annotations_dir=paths.ROOT,
            width=cfg.SOLVER.IM_SIZE,
            height=cfg.SOLVER.IM_SIZE,
            num_frames=cfg.DATA.MAX_FRAMES + cfg.MODEL.VAE.TEMPORAL_FACTOR,
            min_motion_score=cfg.DATA.VIDEO.MIN_MOTION_SCORE,
            sampling_fps=cfg.DATA.VIDEO.SAMPLING_FPS,
            num_start_frames=cfg.FLOWCEPTION.NUM_START_FRAMES,
            latent_downsample=cfg.MODEL.VAE.TEMPORAL_FACTOR,
            min_entropy=cfg.DATA.ENTROPY_THRESHOLD,
            min_subjectness=cfg.DATA.MIN_SUBJECTNESS,
            min_size_ratio=cfg.DATA.MIN_SIZE_RATIO,
            max_size_ratio=cfg.DATA.MAX_SIZE_RATIO,
        )
        val_dataset = train_dataset
    elif dataset_name == "taichi_flowception":
        paths = _path_section(data_paths, "TAICHI", cluster)
        dataset_cls = _dataset_class("TaichiDatasetFlowception")
        train_dataset = dataset_cls(
            annotations_dir=paths.ANNOT_PT,
            width=cfg.SOLVER.IM_SIZE,
            height=cfg.SOLVER.IM_SIZE,
            num_frames=cfg.DATA.MAX_FRAMES + 8,
            min_motion_score=cfg.DATA.VIDEO.MIN_MOTION_SCORE,
            sampling_fps=cfg.DATA.VIDEO.SAMPLING_FPS,
            num_start_latents=cfg.FLOWCEPTION.NUM_START_FRAMES,
            num_context_latents=cfg.MODEL.VIDEO.CONTEXT_LATENTS,
        )
        val_dataset = train_dataset
    elif dataset_name == "taichi_cache_flowception":
        paths = _path_section(data_paths, "TAICHI", cluster)
        dataset_cls = _dataset_class("TaichiInMemoryFlowception")
        train_dataset = dataset_cls(
            annotations_dir=paths.ANNOT_PT,
            width=cfg.SOLVER.IM_SIZE,
            height=cfg.SOLVER.IM_SIZE,
            num_frames=cfg.DATA.MAX_FRAMES + 8,
            min_motion_score=cfg.DATA.VIDEO.MIN_MOTION_SCORE,
            sampling_fps=cfg.DATA.VIDEO.SAMPLING_FPS,
            num_start_frames=cfg.FLOWCEPTION.NUM_START_FRAMES,
        )
        val_dataset = train_dataset
    elif dataset_name == "re10k_flowception":
        paths = _path_section(data_paths, "RE10K", cluster)
        dataset_cls = _dataset_class("Re10kMP4DatasetFlowception")
        train_dataset = dataset_cls(
            video_dir=paths.VIDEOS_DIR,
            index_path=paths.VIDEO_PATHS,
            width=cfg.SOLVER.IM_SIZE,
            height=cfg.SOLVER.IM_SIZE,
            num_frames=cfg.DATA.MAX_FRAMES + 8,
            sampling_fps=cfg.DATA.VIDEO.SAMPLING_FPS,
            num_start_latents=cfg.FLOWCEPTION.NUM_START_FRAMES,
        )
        val_dataset = train_dataset
    elif dataset_name == "vchitect2_flowception":
        paths = _path_section(data_paths, "VCHITECT2", cluster)
        dataset_cls = _dataset_class("VChitectTarFlowception")
        train_dataset = dataset_cls(
            annotations_json=paths.ANNOT_JSON,
            index_db=paths.INDEX_DB,
            width=cfg.SOLVER.IM_SIZE,
            height=cfg.SOLVER.IM_SIZE,
            num_frames=cfg.DATA.MAX_FRAMES + 8,
            min_motion_score=cfg.DATA.VIDEO.MIN_MOTION_SCORE,
            sampling_fps=cfg.DATA.VIDEO.SAMPLING_FPS,
            num_start_frames=cfg.FLOWCEPTION.NUM_START_FRAMES,
            latent_downsample=8,
            max_retries=20,
            drop_missing=False,
        )
        val_dataset = train_dataset
    elif dataset_name == "youcook2":
        paths = _path_section(data_paths, "YOUCOOK2", cluster)
        dataset_cls = _dataset_class("YouCook2Flowception")
        train_dataset = dataset_cls(
            annotations=paths.ANNOT_DIR,
            vid_root=paths.ROOT,
            width=cfg.SOLVER.IM_SIZE,
            height=cfg.SOLVER.IM_SIZE,
            num_frames=cfg.DATA.MAX_FRAMES + 8,
            sampling_fps=cfg.DATA.VIDEO.SAMPLING_FPS,
            native_fps=24,
            num_start_frames=cfg.FLOWCEPTION.NUM_START_FRAMES,
        )
        val_dataset = train_dataset
    elif dataset_name == "youcook2_iter":
        paths = _path_section(data_paths, "YOUCOOK2", cluster)
        dataset_cls = _dataset_class("YouCook2IterFlowception")
        train_dataset = dataset_cls(
            annotations=paths.ANNOT_DIR,
            vid_root=paths.ROOT,
            width=cfg.SOLVER.IM_SIZE,
            height=cfg.SOLVER.IM_SIZE,
            num_frames=cfg.DATA.MAX_FRAMES + 8,
            sampling_fps=cfg.DATA.VIDEO.SAMPLING_FPS,
            native_fps=24,
            num_start_frames=cfg.FLOWCEPTION.NUM_START_FRAMES,
        )
        val_dataset = train_dataset
    elif dataset_name == "custom_webdataset":
        paths = _path_section(data_paths, "CUSTOM_WEBDATASET", cluster)
        train_dataset, val_dataset = _custom_webdataset(
            cfg, paths=paths, num_gpus=num_gpus, start_epoch=start_epoch, aes_filter=False
        )
    elif dataset_name == "custom_webdataset_aes":
        paths = _path_section(data_paths, "CUSTOM_WEBDATASET", cluster)
        train_dataset, val_dataset = _custom_webdataset(
            cfg, paths=paths, num_gpus=num_gpus, start_epoch=start_epoch, aes_filter=True
        )
    else:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}")

    return train_dataset, val_dataset
