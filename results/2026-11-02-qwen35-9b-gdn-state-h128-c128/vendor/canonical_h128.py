import math,os,random,sys
from pathlib import Path
import torch
EPS=1e-12
GDN_LAYERS=tuple(i for i in range(32) if i%4!=3)
MODEL_PATH=Path("/data/zypan/modelscope_models/Qwen3.5-9B")
TRANSFORMERS_SRC=Path(__file__).resolve().parents[1]/"runtime/transformers-qwen35"
STATE_QUANT_SRC=Path(__file__).resolve().parents[1]/"runtime/qwen35_gdn_quant"

def setup_seed(seed):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

_HADAMARD_CACHE={}

def hadamard(n=128, dtype=torch.float32, device=None):
    key=(n,dtype,str(device))
    if key in _HADAMARD_CACHE:return _HADAMARD_CACHE[key]
    matrix = torch.ones((1, 1), dtype=dtype, device=device)
    while matrix.shape[0] < n:
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    result=matrix / math.sqrt(n)
    _HADAMARD_CACHE[key]=result
    return result

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
