# 经过校验的 NPZ 数据集、数据加载器及不使用 pickle 的检查点存储。

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset

from .problem import STATIC_KEYS, Instance, validate_instance


def _npz_path(path) -> Path:
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"Expected a .npz path: {path}")
    return path


def file_sha256(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dataset_identity(manifest):
    # 按数据内容校验续训身份，允许内容相同的数据集迁移到其他目录。
    if manifest is None:
        return None
    return {
        split: {key: entry[key] for key in ("sha256", "count", "n_w", "n_s", "n_c")}
        for split, entry in manifest["splits"].items()
    }


def save_instances(data: TensorDict | Instance, path, *, metadata=None) -> None:
    path = _npz_path(path)
    td = data.tensors if isinstance(data, Instance) else data
    validate_instance(td)
    values = {key: td[key].detach().cpu().numpy() for key in STATIC_KEYS if key in td}
    values["metadata_json"] = np.array(json.dumps(metadata or {}, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **values)


def load_instances(path) -> tuple[TensorDict, dict]:
    path = _npz_path(path).resolve(strict=True)
    with np.load(path, allow_pickle=False) as archive:
        unknown = set(archive.files) - set(STATIC_KEYS) - {"metadata_json"}
        if unknown:
            raise ValueError(f"Unknown dataset fields: {sorted(unknown)}")
        values = {
            key: torch.from_numpy(archive[key].copy()) for key in STATIC_KEYS if key in archive
        }
        metadata = json.loads(str(archive["metadata_json"])) if "metadata_json" in archive else {}
    if "customer_demand" not in values or values["customer_demand"].ndim != 2:
        raise ValueError("customer_demand must have shape [B,C]")
    td = TensorDict(values, [values["customer_demand"].shape[0]])
    instance = Instance(td)
    if not isinstance(metadata, dict):
        raise ValueError("Dataset metadata must be an object")
    metadata = {
        **metadata,
        "path": str(path),
        "sha256": file_sha256(path),
        "count": td.batch_size[0],
        **dict(zip(("n_w", "n_s", "n_c"), instance.dimensions, strict=True)),
    }
    return td, metadata


def load_dataset_directory(directory, splits: Sequence[str] = ("train", "val", "test")):
    directory = Path(directory).resolve(strict=True)
    data, manifest = {}, {}
    for split in splits:
        data[split], manifest[split] = load_instances(directory / split / "instances.npz")
    if len({Instance(value).dimensions for value in data.values()}) != 1:
        raise ValueError("Dataset split dimensions differ")
    for index, first in enumerate(splits):
        for second in splits[index + 1 :]:
            a, b = manifest[first], manifest[second]
            if a["sha256"] == b["sha256"]:
                raise ValueError(f"Identical files for {first} and {second}")
            if "seed" in a and "seed" in b and a["seed"] == b["seed"]:
                raise ValueError(f"Overlapping generator seeds in {first} and {second}")

            # 元数据不能证明划分互不重叠，需要比较实际数值行。
            def hashes(td):
                return {
                    hashlib.sha256(
                        b"".join(td[key][row].numpy().tobytes() for key in STATIC_KEYS[:7])
                    ).digest()
                    for row in range(td.batch_size[0])
                }

            if hashes(data[first]) & hashes(data[second]):
                raise ValueError(f"Overlapping instances in {first} and {second}")
    return data, {"directory": str(directory), "splits": manifest}


class NPZDataset(Dataset):
    def __init__(self, path):
        self.data, self.metadata = load_instances(path)

    def __len__(self):
        return self.data.batch_size[0]

    def __getitem__(self, index):
        return self.data[index]


def make_dataloader(dataset: NPZDataset, *, batch_size: int, shuffle: bool, seed: int):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=torch.stack,
        generator=torch.Generator().manual_seed(seed),
    )


def save_checkpoint(payload: dict, path) -> None:
    # 以张量和基础类型树保存优化器及随机数状态，不使用 pickle。
    path = _npz_path(path)
    arrays = {}

    def encode(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            key = f"tensor_{len(arrays)}"
            arrays[key] = (
                tensor.view(torch.uint16).numpy()
                if tensor.dtype == torch.bfloat16
                else tensor.numpy()
            )
            return {"kind": "tensor", "key": key, "dtype": str(tensor.dtype)}
        if isinstance(value, dict):
            return {"kind": "dict", "items": [[encode(k), encode(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple)):
            return {
                "kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [encode(v) for v in value],
            }
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")

    manifest = {"archive_version": 1, "payload": encode(payload)}
    arrays["manifest_json"] = np.array(json.dumps(manifest, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_checkpoint(path) -> dict:
    path = _npz_path(path).resolve(strict=True)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"]))
        if manifest["archive_version"] != 1:
            raise ValueError("Unsupported checkpoint archive version")

        def decode(value):
            if not isinstance(value, dict):
                return value
            kind = value["kind"]
            if kind == "tensor":
                tensor = torch.from_numpy(archive[value["key"]].copy())
                if value["dtype"] == "torch.bfloat16":
                    return tensor.view(torch.bfloat16)
                if str(tensor.dtype) != value["dtype"]:
                    raise ValueError("Checkpoint tensor dtype does not match its manifest")
                return tensor
            if kind == "dict":
                return {decode(k): decode(v) for k, v in value["items"]}
            if kind in ("list", "tuple"):
                items = [decode(item) for item in value["items"]]
                return tuple(items) if kind == "tuple" else items
            raise ValueError(f"Unknown checkpoint node: {kind}")

        payload = decode(manifest["payload"])
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary")
    return payload
