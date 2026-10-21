# 保留原始实现中的科研回归测试。

import pytest
import torch
from test_problem import deterministic_cpu as deterministic_cpu
from test_problem import tiny_instance as tiny_instance
from test_problem import tiny_solution as tiny_solution

from ss_tscflp.generator import TSCFLPGenerator
from ss_tscflp.problem import (
    validate_instance,
)


def test_npz_round_trip_and_deterministic_loader(tmp_path):
    from ss_tscflp.dataset import NPZDataset, load_instances, make_dataloader, save_instances

    data = TSCFLPGenerator(2, 3, 5, seed=42)([7])
    path = tmp_path / "space and unicode 数据" / "instances.npz"
    save_instances(data, path, metadata={"seed": 42})
    loaded, metadata = load_instances(path)
    assert metadata["seed"] == 42 and metadata["count"] == 7
    assert all(torch.equal(data[key], loaded[key]) for key in data.keys())
    dataset = NPZDataset(path)
    first = list(make_dataloader(dataset, batch_size=3, shuffle=True, seed=8))
    second = list(make_dataloader(dataset, batch_size=3, shuffle=True, seed=8))
    assert [len(batch) for batch in first] == [3, 3, 1]
    assert all(
        torch.equal(a["customer_demand"], b["customer_demand"])
        for a, b in zip(first, second, strict=True)
    )


@pytest.mark.parametrize("case", ["missing", "object", "float64", "shape", "unknown"])
def test_npz_rejects_bad_input(tmp_path, tiny_instance, case):
    import numpy as np

    from ss_tscflp.dataset import load_instances

    path = tmp_path / "bad.npz"
    if case == "missing":
        with pytest.raises(FileNotFoundError):
            load_instances(path)
        return
    values = {key: value.numpy() for key, value in tiny_instance.items()}
    if case == "object":
        values["customer_demand"] = np.array([[object()]], dtype=object)
    elif case == "float64":
        values["cost_sc"] = values["cost_sc"].astype(np.float64)
    elif case == "shape":
        values["cost_sc"] = values["cost_sc"][:, :, :2]
    else:
        values["unexpected"] = np.array([1])
    np.savez(path, **values)
    with pytest.raises(ValueError):
        load_instances(path)


def test_split_overlap(tmp_path, tiny_instance):
    from ss_tscflp.dataset import load_dataset_directory, save_instances

    for index, split in enumerate(("train", "val")):
        save_instances(tiny_instance, tmp_path / split / "instances.npz", metadata={"seed": index})
    with pytest.raises(ValueError, match="Overlapping instances"):
        load_dataset_directory(tmp_path, ("train", "val"))


def test_checkpoint_round_trip(tmp_path):
    import json

    import numpy as np

    from ss_tscflp.dataset import load_checkpoint, save_checkpoint

    payload = {
        "optimizer": {0: {"step": torch.tensor(2.0), "exp_avg": torch.arange(4.0)}},
        "rng": torch.get_rng_state(),
        "integer": torch.tensor([1, -1]),
        "half": torch.tensor([1.25], dtype=torch.bfloat16),
        "empty": (),
        "settings": [True, None, "text", 1.5],
    }
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(payload, path)
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind != "O" for key in archive.files)
        assert json.loads(str(archive["manifest_json"]))["archive_version"] == 1
    restored = load_checkpoint(path)
    assert restored["empty"] == () and restored["settings"] == payload["settings"]
    assert torch.equal(restored["optimizer"][0]["exp_avg"], payload["optimizer"][0]["exp_avg"])
    for key in ("rng", "integer", "half"):
        assert torch.equal(restored[key], payload[key])
        assert restored[key].dtype == payload[key].dtype


def test_config_path_resolution(tmp_path, monkeypatch):
    import argparse

    from ss_tscflp.common import parse_config

    config = tmp_path / "settings" / "run.yaml"
    config.parent.mkdir()
    config.write_text(
        "data: ../data/input.npz\noutput: ../results\ndevice: cpu\n", encoding="utf-8"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    parser = argparse.ArgumentParser()
    for key in ("data", "output", "device"):
        parser.add_argument("--" + key)
    args = parse_config(parser, ["--config", str(config), "--output", "cli-output"])
    assert args["data"] == str(tmp_path / "data" / "input.npz")
    assert args["output"] == "cli-output"


def test_explicit_cuda_never_falls_back_to_cpu(monkeypatch):
    from ss_tscflp.common import validate_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        validate_device("cuda:0")
    with pytest.raises(ValueError, match="explicit"):
        validate_device("")


@pytest.mark.parametrize("sizes", [(5, 10, 20), (1, 5, 100), (1, 1, 1), (8, 2, 7), (2, 8, 2)])
def test_generator_capacity_invariants(sizes):
    gen = TSCFLPGenerator(*sizes, seed=7)
    td = gen([64])
    validate_instance(td)
    assert td.batch_size == torch.Size([64])
    total = td["customer_demand"].sum(-1)
    for name in ("station_capacity", "warehouse_capacity"):
        ratio = td[name].sum(-1) / total
        assert ((ratio >= 1.3 - 1e-6) & (ratio <= 1.8 + 1e-6)).all()
    assert (
        td["customer_demand"].max(-1).values <= td["station_capacity"].max(-1).values + 1e-5
    ).all()
    assert (
        td["station_capacity"].max(-1).values <= td["warehouse_capacity"].max(-1).values + 1e-5
    ).all()
    assert ((td["customer_demand"] >= 1) & (td["customer_demand"] <= 10)).all()
    assert ((td["warehouse_fixed_cost"] >= 50) & (td["warehouse_fixed_cost"] <= 100)).all()
    assert ((td["station_fixed_cost"] >= 20) & (td["station_fixed_cost"] <= 50)).all()
    assert torch.allclose(td["cost_ws"], 10 * torch.cdist(td["warehouse_loc"], td["station_loc"]))
    assert torch.allclose(td["cost_sc"], 10 * torch.cdist(td["station_loc"], td["customer_loc"]))
    station_base = gen._partition_loads(td["customer_demand"], sizes[1])
    warehouse_base = gen._partition_loads(station_base, sizes[0])
    assert (station_base <= td["station_capacity"] + 1e-5).all()
    assert (warehouse_base <= td["warehouse_capacity"] + 1e-5).all()


def test_generator_seed_and_stream():
    first, second = TSCFLPGenerator(seed=99), TSCFLPGenerator(seed=99)
    a, b = first([8]), second([8])
    assert all(torch.equal(a[key], b[key]) for key in a.keys())
    assert not torch.equal(a["customer_demand"], first([8])["customer_demand"])


@pytest.mark.parametrize("kwargs", [{"n_w": 0}, {"n_c": -1}, {"cost_factor": -1}])
def test_generator_rejects_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        TSCFLPGenerator(**kwargs)


def test_generator_rejects_bad_schema(tiny_instance):
    tiny_instance["cost_sc"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite float32"):
        validate_instance(tiny_instance)
