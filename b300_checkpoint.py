"""Offline GLM5-Next BF16 checkpoint transformations for the B300 training loop."""
from __future__ import annotations

import argparse
import errno
import fcntl
import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from b300_candidate import B300_NOMINAL_BYTES
from b300_lineage import WeightSnapshot, stable_digest as digest

INDEX = "model.safetensors.index.json"
TEXT_MODULE = re.compile(r"model\.language_model\.layers\.([0-9]+)\.(self_attn\.o_proj|mlp\.down_proj)")
ADAPTER_KEY = re.compile(r"base_model\.model\.(.+)\.lora_([AB])\.weight")


def read_json(path):
    return json.loads(path.read_bytes())


def copy_shard(source, target):
    # A filesystem reflink is independent on first write. Fall back across filesystems.
    try:
        with source.open("rb") as src, target.open("xb") as dst:
            fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())  # Linux FICLONE
    except OSError as error:
        if error.errno not in {errno.ENOTTY, errno.EINVAL, errno.EOPNOTSUPP,
                               errno.EXDEV, errno.ENOSYS, errno.EPERM}:
            raise
        shutil.copyfile(source, target)


def checkpoint(model):
    config = read_json(model / "config.json")
    text = config["text_config"]
    if (config.get("model_type") != "glm5_next" or config.get("quantization_config")
            or text.get("quantization_config")
            or text.get("dtype", text.get("torch_dtype")) != "bfloat16"):
        raise ValueError("Expected the unquantized GLM5-Next BF16 checkpoint")
    index = read_json(model / INDEX)
    if not index["weight_map"] or index["metadata"]["total_size"] <= 0:
        raise ValueError("Empty checkpoint index")
    for shard in set(index["weight_map"].values()):
        if Path(shard).name != shard or not shard.endswith(".safetensors") or not (model / shard).is_file():
            raise ValueError(f"Invalid or missing checkpoint shard: {shard}")
    return config, index


def write_checkpoint(model, output, config, index, edits, metadata, source):
    """Rewrite only touched shards, and publish the completed directory by rename."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    if output == model or model in output.parents:
        raise ValueError("Place the output outside the base checkpoint")
    weights = index["weight_map"]
    missing = edits.keys() - weights.keys()
    if missing:
        raise ValueError(f"Unknown target tensors: {sorted(missing)[:3]}")
    changed = {weights[name] for name in edits}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Serialize competing publishers, including after an interrupted process.
    with (output.parent / ("." + output.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".checkpoint-") as tmp:
            dest = Path(tmp) / "new"
            dest.mkdir()
            delta_bytes = 0
            for shard in sorted(set(weights.values())):
                if shard not in changed:
                    copy_shard(model / shard, dest / shard)
                    continue
                with safe_open(str(model / shard), framework="pt", device="cpu") as src:
                    tensors = {name: src.get_tensor(name) for name in src.keys()}
                    if {name for name in edits if weights[name] == shard} - tensors.keys():
                        raise ValueError(f"Index disagrees with {shard}")
                    for name in tensors:
                        if name in edits:
                            old = tensors[name]
                            updated = edits[name](old).contiguous()
                            delta_bytes += updated.nbytes - old.nbytes
                            tensors[name] = updated
                    save_file(tensors, str(dest / shard), metadata=src.metadata() or {"format": "pt"})
                    del tensors
            for path in model.iterdir():
                if (path.is_file() and path.name not in {"config.json", INDEX, "transformation.json"}
                        and path.suffix in {".json", ".jinja", ".txt", ".tiktoken", ".model"}):
                    shutil.copyfile(path, dest / path.name)
            index["metadata"]["total_size"] += delta_bytes
            for name, data in (("config.json", config), (INDEX, index), ("transformation.json", metadata)):
                (dest / name).write_text(json.dumps(data, indent=2) + "\n")
            source.check()
            dest.rename(output)


def merge(model, run, out):
    import torch
    from safetensors.torch import load_file

    source = WeightSnapshot(model)
    config, index = checkpoint(model)
    record = read_json(run / "run.json")
    if (record.get("status") != "trained"
            or record.get("config_sha256") != digest(model / "config.json")
            or record.get("weight_index_sha256") != digest(model / INDEX)):
        raise ValueError("The training run does not identify this base checkpoint as its source")
    source.verify_weights(index, record)
    adapter = run / "artifacts" / "adapter"
    settings = read_json(adapter / "adapter_config.json")
    if (settings.get("peft_type") != "LORA" or settings.get("bias", "none") != "none"
            or any(settings.get(k) for k in ("use_dora", "fan_in_fan_out", "modules_to_save",
                    "rank_pattern", "alpha_pattern", "lora_bias", "layer_replication",
                    "target_parameters", "trainable_token_indices", "use_qalora", "alora_invocation_tokens"))):
        raise ValueError("Unsupported adapter variant for the GLM training recipe")
    rank, alpha = settings["r"], settings["lora_alpha"]
    if type(rank) is not int or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid LoRA rank or alpha")
    scale = alpha / (math.sqrt(rank) if settings.get("use_rslora") else rank)
    adapter_path = adapter / "adapter_model.safetensors"
    pairs = defaultdict(dict)
    for key, tensor in load_file(str(adapter_path)).items():
        match = ADAPTER_KEY.fullmatch(key)
        if not match or not TEXT_MODULE.fullmatch(match[1]):
            raise ValueError(f"Unexpected PEFT tensor {key}")
        pairs[match[1] + ".weight"][match[2]] = tensor
    if not pairs:
        raise ValueError("Empty adapter")

    def update(a, b):
        def apply(w):
            if (w.dtype != torch.bfloat16 or w.ndim != 2 or a.shape != (rank, w.shape[1])
                    or b.shape != (w.shape[0], rank)):
                raise ValueError("Adapter does not match BF16 base dimensions")
            result = torch.addmm(w.float(), b.float(), a.float(), alpha=scale).to(w.dtype)
            if not torch.isfinite(result).all():
                raise ValueError("Nonfinite adapter update")
            return result
        return apply

    edits = {}
    for key, pair in pairs.items():
        if pair.keys() != {"A", "B"}:
            raise ValueError(f"Incomplete adapter at {key}")
        edits[key] = update(pair["A"], pair["B"])
    write_checkpoint(model, out, config, index, edits, {"operation": "merge", "run": str(run),
                     "base": str(model), "adapter_sha256": digest(adapter_path), "modules": len(edits)}, source)


def growth_headroom(model, export, fit, source, index):
    # The trusted evaluator's report is bound to the source checkpoint through the export.
    manifest = read_json(export / "export-manifest.json")
    result = read_json(fit)
    deployed = manifest["candidate"]
    measured = result["candidate"]
    if (manifest["config_sha256"] != digest(model / "config.json")
            or manifest["index_sha256"] != digest(model / INDEX) or result.get("status") != "passed"
            or result["model_sha256"] != deployed["sha256"] or measured["sha256"] != deployed["sha256"]
            or measured["size_bytes"] != deployed["size_bytes"] or result["parallel_slots"] != 1):
        raise ValueError("Growth requires a passing one-card fit for this exported parent")
    source.verify_weights(index, manifest)
    mib = 1024 ** 2
    card = result["gpu"]["total_mib"] * mib
    peak = result["sampled_device_peak_mib_including_reservation"] * mib
    if not 0 < peak <= card or not 0 < measured["size_bytes"] < card:
        raise ValueError("Invalid parent GPU measurement")
    weight_room = min(card, B300_NOMINAL_BYTES) - measured["reserve_bytes"] - measured["size_bytes"]
    return min(card - peak, weight_room)


def grow(model, export, fit, out, extra, seed):
    import torch

    source = WeightSnapshot(model)
    config, index = checkpoint(model)
    text = config["text_config"]
    width, hidden = text["intermediate_size"], text["hidden_size"]
    dense = [i for i, kind in enumerate(text["mlp_layer_types"]) if kind == "dense"]
    if not dense or dense != list(range(text["first_k_dense_replace"])) or width <= 0 or hidden <= 0:
        raise ValueError("Unsupported GLM dense-layer layout")
    if extra <= 0 or extra % 256 or (width + extra) % 256:
        raise ValueError("Choose a positive width increment aligned to 256 for Q4_K_M")
    params = 3 * len(dense) * extra * hidden
    estimate = 4 * params  # even a float32 materialization of the new weights
    headroom = growth_headroom(model, export, fit, source, index)
    if estimate > headroom:
        raise ValueError(f"Growth proposal needs {estimate} bytes of headroom, available {headroom}")
    edits = {}

    def widen(layer, projection):
        def apply(w):
            output = projection == "down_proj"
            axis = 1 if output else 0
            shape = (hidden, width) if output else (width, hidden)
            if w.dtype != torch.bfloat16 or tuple(w.shape) != shape:
                raise ValueError("Unexpected dense weight shape or dtype")
            if output:
                added = w.new_zeros(hidden, extra)
            else:
                # Independent gate/up draws give the new SwiGLU channels distinct feature pairs.
                rng = torch.Generator().manual_seed(seed + 2 * layer + (projection == "up_proj"))
                rows = torch.randint(width, (extra,), generator=rng)
                added = w.index_select(0, rows)
            return torch.cat((w, added), dim=axis)
        return apply

    for layer in dense:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.language_model.layers.{layer}.mlp.{projection}.weight"
            edits[key] = widen(layer, projection)
    text["intermediate_size"] += extra
    write_checkpoint(model, out, config, index, edits, {"operation": "grow", "base": str(model),
                     "parent_fit": str(fit), "added_parameters": params, "added_width": extra,
                     "estimated_extra_weight_bytes": estimate, "parent_headroom_bytes": headroom,
                     "seed": seed, "requires_new_fit": True}, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    merge_cli = sub.add_parser("merge", help="Merge the trained adapter into its original BF16 parent")
    merge_cli.add_argument("--run", type=Path, required=True)
    grow_cli = sub.add_parser("grow", help="Widen the dense GLM MLPs inside the parent's measured budget")
    grow_cli.add_argument("--export", type=Path, required=True)
    grow_cli.add_argument("--fit", type=Path, required=True)
    grow_cli.add_argument("--extra-width", type=int, required=True)
    grow_cli.add_argument("--seed", type=int, default=17)
    for cmd in (grow_cli, merge_cli):
        cmd.add_argument("--model", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model, out = args.model.resolve(strict=True), args.out.resolve()
    try:
        if args.action == "merge":
            merge(model, args.run.resolve(strict=True), out)
        else:
            grow(model, args.export.resolve(strict=True), args.fit.resolve(strict=True),
                 out, args.extra_width, args.seed)
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(out)


if __name__ == "__main__":
    main()
