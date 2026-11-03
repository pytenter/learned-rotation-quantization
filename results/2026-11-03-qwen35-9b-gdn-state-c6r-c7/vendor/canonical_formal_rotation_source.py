#!/usr/bin/env python3
"""Canonical HF/manual Qwen3.5-9B GDN C5/C6 frozen-rotation runner.

This is an additive deployment copy of the frozen baseline runner.  It preserves
sampling, cache, and outer post-decode INT8-C128 QDQ semantics while adding only
direct loading of already-materialized per-layer FP32 Delta matrices.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

CANONICAL_AIME_REPO = Path("/data/zypan/worktrees/aime26-sglang-rotation-v1")
CANONICAL_AIME_CODE = CANONICAL_AIME_REPO / "experiments/aime26"
if str(CANONICAL_AIME_CODE) not in sys.path:
    sys.path.insert(0, str(CANONICAL_AIME_CODE))
from aime26_common import TASK, ids_sha256, load_frozen_dataset, raw_messages


REPO = Path("/data/zypan/worktrees/hadamard-init-dense-oracle-v1-qwen")
DATA_ROOT = Path(os.environ.get("GDN_DATA_ROOT", "/data/zypan"))
MODEL_PATH = Path(os.environ.get("QWEN_MODEL_PATH", DATA_ROOT / "modelscope_models/Qwen3.5-9B"))
TRANSFORMERS_SRC = Path(os.environ.get("QWEN_TRANSFORMERS_SRC", DATA_ROOT / "transformers-qwen35"))
STATE_QUANT_SRC = Path(os.environ.get("QWEN_STATE_QUANT_SRC", DATA_ROOT / "experiments/qwen35_gdn_quant"))
DATASET = CANONICAL_AIME_REPO / "artifacts/aime26_v1/formal/dataset/aime26_frozen.jsonl"
OUT_ROOT = Path(os.environ.get(
    "QWEN_C5_C6_OUT_ROOT",
    REPO / "experiments/qwen_gdn/QWEN_C5_C6_AIME26_CANONICAL60_V1/outputs",
))
GDN_LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30)
METHODS = {
    "c5": "QWEN_INT8_C128_KEY_C5",
    "c6": "QWEN_INT8_C128_KEY_C6",
}
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
PRESENCE_PENALTY = 1.5
REPETITION_PENALTY = 1.0
MAX_NEW_TOKENS = 81920
EPS = 1e-12
PROTOCOL_VERSION = "GDN_KDA_AIME26_OFFICIAL_SAMPLING_81920_2SEED_FORMAL_V2"
FORMAL_WORKERS = 4
TOP_K_AUDIT_VERSION = "HF_TOPK_TIE_AWARE_V2"
TOP_K_IMPLEMENTATION = "transformers.TopKLogitsWarper"
LEGACY_TOP_K_AUDIT_COMMIT = "0dc24e7d6f616d91681213e6090bcd5eeb4af642"


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def runtime_version():
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor):
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def output_payload_sha256(row):
    payload = dict(row)
    payload.pop("output_payload_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class FrozenRotationLayer(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        if value.device.type != "cpu" or value.dtype != torch.float32 or tuple(value.shape) != (128, 128):
            raise RuntimeError("frozen rotation matrix must be CPU FP32 [128,128]")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("nonfinite frozen rotation matrix")
        self.register_buffer("frozen_matrix", value.contiguous().clone())

    def matrix(self):
        return self.frozen_matrix


class FrozenRotationBank(torch.nn.Module):
    def __init__(self, layer_ids, matrices):
        super().__init__()
        self.layer_ids = tuple(int(x) for x in layer_ids)
        if self.layer_ids != GDN_LAYERS or set(matrices) != {str(x) for x in GDN_LAYERS}:
            raise RuntimeError("frozen rotation layer mapping mismatch")
        self.layers = torch.nn.ModuleDict({str(layer): FrozenRotationLayer(matrices[str(layer)]) for layer in GDN_LAYERS})

    def layer(self, layer_id):
        return self.layers[str(layer_id)]


def load_frozen_rotation(rotation_file, manifest_file, device, expected_condition):
    rotation_file = Path(rotation_file).resolve()
    manifest_file = Path(manifest_file).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    rotation_sha = file_sha256(rotation_file)
    manifest_sha = file_sha256(manifest_file)
    if manifest.get("condition") != expected_condition.upper():
        raise RuntimeError("deployment manifest condition mismatch")
    if manifest.get("file_sha256") != rotation_sha:
        raise RuntimeError("frozen rotation file SHA mismatch")
    if manifest.get("loading_rule") != "load frozen rotation directly; do not reconstruct R from theta on inference host":
        raise RuntimeError("non-canonical frozen loading rule")
    payload = torch.load(rotation_file, map_location="cpu", weights_only=False)
    if payload.get("condition") != expected_condition.upper() or tuple(payload.get("layer_ids", ())) != GDN_LAYERS:
        raise RuntimeError("frozen rotation payload identity mismatch")
    if payload.get("matrix_dtype") != "torch.float32" or payload.get("matrix_shape") != [128, 128]:
        raise RuntimeError("frozen rotation payload dtype/shape mismatch")
    matrices = payload.get("matrices", {})
    for layer in GDN_LAYERS:
        value = matrices.get(str(layer))
        if not torch.is_tensor(value) or value.dtype != torch.float32 or tuple(value.shape) != (128, 128):
            raise RuntimeError(f"layer {layer} frozen matrix structure mismatch")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"layer {layer} frozen matrix nonfinite")
        if tensor_sha256(value) != manifest.get("per_layer_sha256", {}).get(str(layer)):
            raise RuntimeError(f"layer {layer} frozen matrix SHA mismatch")
    bank = FrozenRotationBank(payload["layer_ids"], matrices).to(device)
    return bank, {
        "condition": expected_condition.upper(),
        "rotation_file": str(rotation_file),
        "rotation_sha256": rotation_sha,
        "deployment_manifest": str(manifest_file),
        "deployment_manifest_sha256": manifest_sha,
        "theta_checkpoint_sha256": payload.get("theta_checkpoint_sha256"),
        "layer_count": len(matrices),
        "direct_frozen_load": True,
        "theta_reconstruction": False,
        "orientation": payload.get("orientation"),
    }


def setup_seed(seed):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hadamard(n=128, dtype=torch.float32, device=None):
    matrix = torch.ones((1, 1), dtype=dtype, device=device)
    while matrix.shape[0] < n:
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    return matrix / math.sqrt(n)


def normalize_qk(value):
    return value * torch.rsqrt((value * value).sum(-1, keepdim=True) + 1e-6)


class QwenKeyHadamardPatch:
    """Reference-closed q/k transform immediately before the GDN core."""

    def __init__(self, model):
        self.model = model
        self.saved = []
        self.handles = []
        self.current_layer = None
        self.calls = 0
        self.layers = set()
        self.phase_counts = {"chunk_prefill": 0, "recurrent_decode": 0}

    def _transform(self, value, use_norm):
        value = normalize_qk(value) if use_norm else value
        return value.float() @ hadamard(128, dtype=torch.float32, device=value.device)

    def _wrap(self, operator, function):
        def wrapped(query, key, value, *args, **kwargs):
            use_norm = bool(kwargs.pop("use_qk_l2norm_in_kernel", False))
            query = self._transform(query, use_norm)
            key = self._transform(key, use_norm)
            core, state = function(query, key, value, *args, use_qk_l2norm_in_kernel=False, **kwargs)
            self.calls += 1
            self.layers.add(int(self.current_layer))
            phase = "recurrent_decode" if operator == "recurrent" else "chunk_prefill"
            self.phase_counts[phase] += 1
            return core.to(value.dtype), state

        return wrapped

    def __enter__(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as qmod

        layers = self.model.model.layers if hasattr(self.model.model, "layers") else self.model.model.language_model.layers
        for index, layer in enumerate(layers):
            module = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
            if module is not None:
                self.handles.append(module.register_forward_pre_hook(lambda _m, _a, i=index: setattr(self, "current_layer", i)))
        for name, operator in (("torch_recurrent_gated_delta_rule", "recurrent"), ("torch_chunk_gated_delta_rule", "chunk")):
            if hasattr(qmod, name):
                original = getattr(qmod, name)
                self.saved.append((qmod, name, original))
                setattr(qmod, name, self._wrap(operator, original))
        return self

    def __exit__(self, *_):
        for module, name, original in self.saved:
            setattr(module, name, original)
        for handle in self.handles:
            handle.remove()


class QwenFrozenRotationPatch(QwenKeyHadamardPatch):
    """Canonical key-side H128 followed by a directly loaded frozen Delta."""

    def __init__(self, model, bank, deployment_identity):
        super().__init__(model)
        self.bank = bank
        self.deployment_identity = dict(deployment_identity)
        self.direct_frozen_load = True
        self.theta_reconstruction = False

    def _transform(self, value, use_norm):
        value = normalize_qk(value) if use_norm else value
        layer = int(self.current_layer)
        if layer not in GDN_LAYERS:
            raise RuntimeError(f"frozen rotation invoked on unmapped GDN layer {layer}")
        delta = self.bank.layer(layer).matrix()
        return value.float().matmul(hadamard(128, dtype=torch.float32, device=value.device)).matmul(delta)


def get_state(cache, layer_idx):
    if hasattr(cache, "recurrent_states"):
        state = cache.recurrent_states[layer_idx]
        if isinstance(state, dict):
            return state[0]
        if isinstance(state, (tuple, list)):
            return state[0]
        return state
    states = getattr(cache.layers[layer_idx], "recurrent_states")
    return states[0] if not isinstance(states, dict) else states[0]


def quantize_c128(cache, audit):
    for layer_idx in GDN_LAYERS:
        state = get_state(cache, layer_idx)
        if list(state.shape[-3:]) != [32, 128, 128]:
            raise RuntimeError(f"unexpected GDN state shape at layer {layer_idx}: {list(state.shape)}")
        source = state.detach().float()
        scale = source.abs().amax(dim=-2, keepdim=True).clamp_min(EPS) / 127.0
        codes = torch.round(source / scale).clamp(-127, 127)
        dequant = codes * scale
        if not bool(torch.isfinite(dequant).all().item()):
            raise FloatingPointError(f"nonfinite C128 dequant at layer {layer_idx}")
        state.copy_(dequant.to(state.dtype))
        audit["layers"].add(layer_idx)
        audit["scale_shapes"][str(layer_idx)] = list(scale.shape)
        audit["code_min"] = min(audit["code_min"], int(codes.min().item()))
        audit["code_max"] = max(audit["code_max"], int(codes.max().item()))
    audit["calls"] += 1


def load_model_and_tokenizer():
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    for path in (str(STATE_QUANT_SRC), str(TRANSFORMERS_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


def render_input(tokenizer, problem):
    messages = raw_messages(problem)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    return messages, input_ids


def build_warpers():
    from transformers.generation import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

    return LogitsProcessorList(
        [TemperatureLogitsWarper(TEMPERATURE), TopKLogitsWarper(TOP_K), TopPLogitsWarper(TOP_P)]
    )


def sample_next(warpers, logits, generated_token_mask, generator):
    scores = logits.float()
    if PRESENCE_PENALTY:
        scores = scores - generated_token_mask.to(scores.dtype) * PRESENCE_PENALTY
    empty_context = torch.empty((scores.shape[0], 0), dtype=torch.long, device=scores.device)
    warped = warpers(empty_context, scores)
    survivors = int(torch.isfinite(warped).sum(dim=-1).max().item())
    probs = torch.softmax(warped, dim=-1)
    token = torch.multinomial(probs, num_samples=1, generator=generator)
    return token, survivors


def build_top_k_audit(topk_survivors):
    """Describe the executed HF top-k path without treating cutoff ties as failure."""
    maximum = max(topk_survivors) if topk_survivors else None
    return {
        "audit_version": TOP_K_AUDIT_VERSION,
        "implementation": TOP_K_IMPLEMENTATION,
        "requested_top_k": TOP_K,
        "tie_policy": "include_scores_equal_to_kth_cutoff",
        "top_k_applied": bool(topk_survivors),
        "max_survivors": maximum,
        "tie_expansion_observed": bool(maximum is not None and maximum > TOP_K),
    }


def top_k_audit_valid(row):
    audit = row.get("sampling_audit", {})
    maximum = audit.get("max_survivors")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
        return False
    if row.get("top_k") != TOP_K:
        return False
    if row.get("effective_generation_config", {}).get("top_k") != TOP_K:
        return False
    if audit.get("audit_version") == TOP_K_AUDIT_VERSION:
        return (
            audit.get("implementation") == TOP_K_IMPLEMENTATION
            and audit.get("requested_top_k") == TOP_K
            and audit.get("tie_policy") == "include_scores_equal_to_kth_cutoff"
            and audit.get("top_k_applied") is True
        )
    # The legacy flag incorrectly required survivors <= K. At this exact commit,
    # source provenance proves TopKLogitsWarper(TOP_K) executed; >K only means
    # BF16 logits tied at the kth cutoff, matching Transformers native semantics.
    return row.get("git_commit") == LEGACY_TOP_K_AUDIT_COMMIT


def stop_ids(tokenizer):
    values = set()
    if tokenizer.eos_token_id is not None:
        values.add(int(tokenizer.eos_token_id))
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            values.add(int(token_id))
    return values


def generate_one(model, tokenizer, item, seed, method_key, physical_gpu, worker_id, canonical_index,
                 rotation_bank, deployment_identity, formal_config_sha256):
    method = METHODS[method_key]
    setup_seed(seed)
    device = next(model.parameters()).device
    messages, input_ids = render_input(tokenizer, item["problem"])
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    generator = torch.Generator(device=device).manual_seed(seed)
    warpers = build_warpers()
    stops = stop_ids(tokenizer)
    quant_audit = {"calls": 0, "layers": set(), "scale_shapes": {}, "code_min": 0, "code_max": 0}
    tokens = []
    topk_survivors = []
    runtime_error = None
    nonfinite = False
    patch = QwenFrozenRotationPatch(model, rotation_bank, deployment_identity)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    try:
        with patch as hadamard_audit, torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            past = output.past_key_values
            if not bool(torch.isfinite(output.logits).all().item()):
                nonfinite = True
            else:
                generated_token_mask = torch.zeros_like(output.logits[:, -1, :], dtype=torch.bool)
                current, survivors = sample_next(warpers, output.logits[:, -1, :], generated_token_mask, generator)
                generated_token_mask.scatter_(1, current, True)
                tokens.append(int(current.item()))
                topk_survivors.append(survivors)
                while len(tokens) < MAX_NEW_TOKENS and int(current.item()) not in stops:
                    output = model(input_ids=current, past_key_values=past, use_cache=True)
                    past = output.past_key_values
                    if method_key != "fp_state":
                        quantize_c128(past, quant_audit)
                    if not bool(torch.isfinite(output.logits).all().item()):
                        nonfinite = True
                        break
                    current, survivors = sample_next(warpers, output.logits[:, -1, :], generated_token_mask, generator)
                    generated_token_mask.scatter_(1, current, True)
                    tokens.append(int(current.item()))
                    topk_survivors.append(survivors)
            learned_rotation_record = {
                "calls": hadamard_audit.calls,
                "layers": sorted(hadamard_audit.layers),
                "phase_counts": hadamard_audit.phase_counts,
                **deployment_identity,
            }
    except Exception:
        runtime_error = traceback.format_exc()
        learned_rotation_record = {
            "calls": getattr(patch, "calls", 0),
            "layers": sorted(getattr(patch, "layers", [])),
            "phase_counts": getattr(patch, "phase_counts", {}),
            **deployment_identity,
        }
    latency = time.time() - started
    ended_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    response = tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ""
    finish_reason = "runtime_error" if runtime_error else "nonfinite" if nonfinite else "eos" if tokens and tokens[-1] in stops else "length"
    commit = git_commit()
    versions = runtime_version()
    record = {
        "task": TASK,
        "protocol_version": PROTOCOL_VERSION,
        "model": "Qwen3.5-9B",
        "architecture": "GDN",
        "runtime": "canonical_manual",
        "method": method,
        "configuration": method,
        "problem_id": item["problem_id"],
        "problem_idx": int(item["problem_idx"]),
        "seed": int(seed),
        "sampling_seed": int(seed),
        "problem": item["problem"],
        "gold_answer": item["answer"],
        "raw_messages": messages,
        "input_ids_hash": ids_sha256(input_ids[0].tolist()),
        "formal_config_sha256": formal_config_sha256,
        "thinking": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "min_p": MIN_P,
        "presence_penalty": PRESENCE_PENALTY,
        "repetition_penalty": REPETITION_PENALTY,
        "presence_penalty_scope": "generated_output_tokens_only",
        "max_new_tokens": MAX_NEW_TOKENS,
        "effective_generation_config": {
            "thinking": True,
            "do_sample": True,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "min_p": MIN_P,
            "presence_penalty": PRESENCE_PENALTY,
            "repetition_penalty": REPETITION_PENALTY,
            "presence_penalty_scope": "generated_output_tokens_only",
            "max_new_tokens": MAX_NEW_TOKENS,
            "sampling_seed": int(seed),
        },
        "state_quantization": "none" if method_key == "fp_state" else "INT8 symmetric C128 scale=[B,H,1,V] post-decode QDQ",
        "rotation": "key-side direct frozen H128@Delta after q/k normalization before GDN",
        "rotation_sha256": deployment_identity["rotation_sha256"],
        "deployment_manifest_sha256": deployment_identity["deployment_manifest_sha256"],
        "direct_frozen_load": True,
        "theta_reconstruction": False,
        "response": response,
        "generated_token_ids": tokens,
        "generated_token_count": len(tokens),
        "extracted_answer": None,
        "correct": None,
        "scoring_status": "DEFERRED_UNTIL_ALL_C5_C6_OUTPUTS_FROZEN",
        "input_tokens": int(input_ids.shape[-1]),
        "output_tokens": len(tokens),
        "finish_reason": finish_reason,
        "eos_seen": finish_reason == "eos",
        "truncated": finish_reason == "length",
        "runtime_error": runtime_error,
        "nonfinite": bool(nonfinite),
        "latency_s": latency,
        "start_utc": started_utc,
        "end_utc": ended_utc,
        "retry_count": 0,
        "tokens_per_s": len(tokens) / max(latency, 1e-9),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "gpu": physical_gpu,
        "worker_id": int(worker_id),
        "num_workers": FORMAL_WORKERS,
        "canonical_index": int(canonical_index),
        "formal_identity": {
            "model": "Qwen3.5-9B",
            "configuration": method,
            "problem_id": item["problem_id"],
            "seed": int(seed),
            "protocol_version": PROTOCOL_VERSION,
        },
        "git_commit": commit,
        "runtime_version": versions,
        "sampling_audit": build_top_k_audit(topk_survivors),
        "quantizer_audit": {
            "calls": quant_audit["calls"],
            "layers": sorted(quant_audit["layers"]),
            "scale_shapes": quant_audit["scale_shapes"],
            "qrange_observed": [quant_audit["code_min"], quant_audit["code_max"]],
            "prefill_quantized": False,
        },
        "hadamard_audit": {"calls": 0, "layers": [], "phase_counts": {}},
        "learned_rotation_audit": learned_rotation_record,
    }
    record["output_payload_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return record


def output_path(stage, method_key, worker_id=None):
    name = {"c5": "C5.jsonl", "c6": "C6.jsonl"}[method_key]
    if stage == "smoke":
        return OUT_ROOT / "smoke" / name
    if worker_id is None:
        raise ValueError("formal output requires worker_id")
    return OUT_ROOT / "formal" / "shards" / f"worker{worker_id}" / name


def is_valid_record(row, method_key):
    method = METHODS[method_key]
    required_identity = row.get("formal_identity", {})
    return (
        row.get("protocol_version") == PROTOCOL_VERSION
        and row.get("model") == "Qwen3.5-9B"
        and row.get("method") == method
        and required_identity.get("model") == "Qwen3.5-9B"
        and required_identity.get("configuration") == method
        and required_identity.get("problem_id") == row.get("problem_id")
        and int(required_identity.get("seed", -1)) == int(row.get("seed", -2))
        and required_identity.get("protocol_version") == PROTOCOL_VERSION
        and row.get("runtime_error") is None
        and row.get("nonfinite") is False
        and isinstance(row.get("response"), str)
        and bool(row.get("response"))
        and isinstance(row.get("generated_token_ids"), list)
        and bool(row.get("generated_token_ids"))
        and row.get("generated_token_count") == len(row.get("generated_token_ids", []))
        and row.get("output_tokens") == len(row.get("generated_token_ids", []))
        and row.get("finish_reason") in {"eos", "length"}
        and isinstance(row.get("eos_seen"), bool)
        and row.get("eos_seen") == (row.get("finish_reason") == "eos")
        and row.get("truncated") == (row.get("finish_reason") == "length")
        and (row.get("finish_reason") != "length" or row.get("generated_token_count") == MAX_NEW_TOKENS)
        and row.get("max_new_tokens") == MAX_NEW_TOKENS
        and row.get("effective_generation_config", {}).get("max_new_tokens") == MAX_NEW_TOKENS
        and row.get("effective_generation_config", {}).get("temperature") == TEMPERATURE
        and row.get("effective_generation_config", {}).get("top_p") == TOP_P
        and row.get("effective_generation_config", {}).get("top_k") == TOP_K
        and row.get("effective_generation_config", {}).get("min_p") == MIN_P
        and row.get("effective_generation_config", {}).get("presence_penalty") == PRESENCE_PENALTY
        and row.get("effective_generation_config", {}).get("repetition_penalty") == REPETITION_PENALTY
        and row.get("effective_generation_config", {}).get("presence_penalty_scope") == "generated_output_tokens_only"
        and row.get("effective_generation_config", {}).get("thinking") is True
        and row.get("effective_generation_config", {}).get("do_sample") is True
        and int(row.get("effective_generation_config", {}).get("sampling_seed", -1)) == int(row.get("seed", -2))
        and row.get("scoring_status") == "DEFERRED_UNTIL_ALL_C5_C6_OUTPUTS_FROZEN"
        and row.get("correct") is None
        and row.get("extracted_answer") is None
        and row.get("direct_frozen_load") is True
        and row.get("theta_reconstruction") is False
        and row.get("learned_rotation_audit", {}).get("direct_frozen_load") is True
        and row.get("learned_rotation_audit", {}).get("theta_reconstruction") is False
        and row.get("learned_rotation_audit", {}).get("rotation_sha256") == row.get("rotation_sha256")
        and row.get("learned_rotation_audit", {}).get("layers") == list(GDN_LAYERS)
        and isinstance(row.get("formal_config_sha256"), str)
        and len(row.get("formal_config_sha256")) == 64
        and row.get("retry_count") == 0
        and isinstance(row.get("start_utc"), str)
        and isinstance(row.get("end_utc"), str)
        and row.get("output_payload_sha256") == output_payload_sha256(row)
    )


def read_done(path, method_key):
    if not path.exists():
        return set()
    done = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"corrupt JSON at {path}:{line_number}") from exc
        if is_valid_record(row, method_key):
            identity = (row["model"], row["method"], row["problem_id"], int(row["seed"]), row["protocol_version"])
            if identity in done:
                raise RuntimeError(f"duplicate valid formal identity in {path}: {identity}")
            done.add(identity)
    return done


def validate_records(rows, method_key, expected, expected_jobs=None):
    method = METHODS[method_key]
    reasons = []
    if len(rows) != expected:
        reasons.append(f"record_count={len(rows)} expected={expected}")
    if any(row["method"] != method for row in rows):
        reasons.append("method mismatch")
    if any(row["runtime_error"] for row in rows):
        reasons.append("runtime error")
    if any(row["nonfinite"] for row in rows):
        reasons.append("nonfinite")
    if any(not is_valid_record(row, method_key) for row in rows):
        reasons.append("invalid completed record")
    identities = [
        (row["model"], row["method"], row["problem_id"], int(row["seed"]), row["protocol_version"])
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        reasons.append("duplicate valid formal identity")
    if expected_jobs is not None:
        actual = {(row["problem_id"], int(row["seed"])) for row in rows}
        if actual != set(expected_jobs):
            reasons.append("shard coverage mismatch")
    if any(not top_k_audit_valid(row) for row in rows):
        reasons.append("top_k audit failed")
    if method_key != "fp_state":
        if any(row["generated_token_count"] > 1 and row["quantizer_audit"]["calls"] <= 0 for row in rows):
            reasons.append("C128 not triggered")
        for row in rows:
            shapes = row["quantizer_audit"]["scale_shapes"].values()
            if row["generated_token_count"] > 1 and (not shapes or any(shape[-2:] != [1, 128] for shape in shapes)):
                reasons.append("C128 scale shape mismatch")
                break
    for row in rows:
        audit = row.get("learned_rotation_audit", {})
        if (audit.get("calls", 0) <= 0 or audit.get("layers") != list(GDN_LAYERS)
                or audit.get("direct_frozen_load") is not True
                or audit.get("theta_reconstruction") is not False
                or row.get("rotation_sha256") != audit.get("rotation_sha256")):
            reasons.append("frozen learned rotation not triggered exactly on all GDN layers")
            break
    return {"gate": "PASS" if not reasons else "FAIL", "method": method, "record_count": len(rows), "reasons": reasons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "formal", "verify"), required=True)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--num-workers", type=int, default=FORMAL_WORKERS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rotation-file")
    parser.add_argument("--deployment-manifest")
    parser.add_argument("--formal-config-sha256")
    args = parser.parse_args()
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if physical_gpu not in {"0", "1", "2", "3"}:
        raise SystemExit("set CUDA_VISIBLE_DEVICES to one physical GPU 0..3")
    if args.stage != "smoke":
        if args.num_workers != FORMAL_WORKERS or args.worker_id not in range(FORMAL_WORKERS):
            raise SystemExit(f"formal requires --num-workers {FORMAL_WORKERS} and --worker-id 0..{FORMAL_WORKERS - 1}")
        if args.stage == "formal" and not args.resume:
            raise SystemExit("formal execution requires --resume to prevent accidental shard overwrite")
    if args.stage in {"smoke", "formal"} and (not args.rotation_file or not args.deployment_manifest):
        raise SystemExit("C5/C6 execution requires --rotation-file and --deployment-manifest")
    if args.stage == "formal" and (not args.formal_config_sha256 or len(args.formal_config_sha256) != 64):
        raise SystemExit("formal execution requires --formal-config-sha256")
    worker_id = 0 if args.worker_id is None else args.worker_id
    path = output_path("smoke" if args.stage == "smoke" else "formal", args.method, worker_id)
    if args.stage == "verify":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        all_rows = load_frozen_dataset(DATASET)
        all_jobs = [(row, seed) for row in all_rows for seed in (1, 2)]
        method_offset = list(METHODS).index(args.method) * len(all_jobs)
        assigned = [(row["problem_id"], seed) for local_index, (row, seed) in enumerate(all_jobs) if (method_offset + local_index) % FORMAL_WORKERS == worker_id]
        result = validate_records(rows, args.method, len(assigned), assigned)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["gate"] == "PASS" else 2)
    rows = load_frozen_dataset(DATASET)
    if args.stage == "smoke":
        rows = rows[:3]
    all_jobs = [(row, seed) for row in rows for seed in (1, 2)]
    method_offset = list(METHODS).index(args.method) * len(all_jobs)
    if args.stage == "formal":
        jobs = [
            (row, seed, method_offset + local_index)
            for local_index, (row, seed) in enumerate(all_jobs)
            if (method_offset + local_index) % args.num_workers == worker_id
        ]
    else:
        jobs = [(row, seed, local_index) for local_index, (row, seed) in enumerate(all_jobs)]
    path.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(path, args.method) if args.resume else set()
    mode = "a" if args.resume else "w"
    model, tokenizer = load_model_and_tokenizer()
    device = next(model.parameters()).device
    rotation_bank, deployment_identity = load_frozen_rotation(
        args.rotation_file, args.deployment_manifest, device, args.method)
    with path.open(mode, encoding="utf-8") as handle:
        for index, (item, seed, canonical_index) in enumerate(jobs, 1):
            key = ("Qwen3.5-9B", METHODS[args.method], item["problem_id"], seed, PROTOCOL_VERSION)
            if key in done:
                continue
            print(f"[{time.strftime('%F %T')}] {index}/{len(jobs)} {METHODS[args.method]} {item['problem_id']} seed={seed}", flush=True)
            record = generate_one(
                model, tokenizer, item, seed, args.method, physical_gpu, worker_id, canonical_index,
                rotation_bank, deployment_identity, args.formal_config_sha256)
            if not is_valid_record(record, args.method):
                attempts_path = path.with_suffix(".attempts.jsonl")
                with attempts_path.open("a", encoding="utf-8") as attempts:
                    attempts.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                    attempts.flush()
                    os.fsync(attempts.fileno())
                raise RuntimeError(f"formal unit failed validation: {key}")
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    all_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_jobs = [(item["problem_id"], seed) for item, seed, _ in jobs]
    result = validate_records(all_rows, args.method, len(jobs), expected_jobs)
    summary_path = path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["gate"] == "PASS" else 2)


if __name__ == "__main__":
    main()
