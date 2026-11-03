"""Frozen evaluation H128 -> Delta, preserving both FP32 GEMMs and their order."""
import ast
import hashlib
import json
import math
from pathlib import Path
import types
import torch
from vendor.canonical_h128 import hadamard, normalize_qk
from vendor.prior_core import LAYERS

FORMAL_SHA = '77b1e1f6dad8a0dd53a27cf5a47c02be86c26d2cf30732604ba6d3747e178f69'
FROZEN_SHAS = {'C6R': 'a886588b3613f66e6635e5874dbfaeaee26b850e05b4a1524da0bef952d22da6',
               'C7': '57e7a11714405806385064f07438e5e2ac6b5952dd421606332b894cef3ee6e8'}
METHODS = ('Native', 'Hadamard', 'C6R', 'C7')

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def canonical_rotation_code(root):
    """Compile original pure loading/rotation definitions, never the generator."""
    path = Path(root) / 'vendor/canonical_formal_rotation_source.py'
    source = path.read_text()
    assert file_sha(path) == FORMAL_SHA
    wanted = {'file_sha256', 'tensor_sha256', 'FrozenRotationLayer', 'FrozenRotationBank',
              'load_frozen_rotation', 'hadamard', 'normalize_qk',
              'QwenKeyHadamardPatch', 'QwenFrozenRotationPatch'}
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    namespace = {'torch': torch, 'Path': Path, 'json': json, 'hashlib': hashlib,
                 'math': math, 'GDN_LAYERS': LAYERS}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return types.SimpleNamespace(**namespace)

class Rotations:
    def __init__(self, root, device='cpu'):
        self.root = Path(root)
        self.device = torch.device(device)
        self.h = hadamard(device=self.device)
        self.original = canonical_rotation_code(root)
        assert torch.equal(self.h, self.original.hadamard(device=self.device))
        self.banks, self.load_identity, self.original_patches = {}, {}, {}
        for method in ('C6R', 'C7'):
            path = self.root / 'rotations' / method / 'frozen_rotation_fp32.pt'
            manifest = self.root / 'provenance' / (method.lower() + '_rotation_manifest.json')
            assert file_sha(path) == FROZEN_SHAS[method]
            bank, identity = self.original.load_frozen_rotation(path, manifest, self.device, method)
            self.banks[method] = bank
            self.load_identity[method] = identity
            self.original_patches[method] = self.original.QwenFrozenRotationPatch(None, bank, identity)

    def delta(self, method, layer):
        return self.banks[method].layer(layer).matrix()

    def qk(self, method, x, layer, use_norm):
        assert method in METHODS and layer in LAYERS
        y = normalize_qk(x) if use_norm else x
        y = y.float()
        if method == 'Native':
            return y
        y = y.matmul(self.h)
        if method in self.banks:
            y = y.matmul(self.delta(method, layer))
        return y

    def original_qk(self, method, x, layer, use_norm):
        patch = self.original_patches[method]
        patch.current_layer = layer
        return patch._transform(x, use_norm)

    def to_basis(self, method, state, layer):
        if method == 'Native':
            return state
        y = torch.matmul(self.h.T, state)
        if method in self.banks:
            y = torch.matmul(self.delta(method, layer).T, y)
        return y

    def from_basis(self, method, state, layer):
        if method == 'Native':
            return state
        y = state
        if method in self.banks:
            y = torch.matmul(self.delta(method, layer), y)
        return torch.matmul(self.h, y)

    def matrix(self, method, layer, dtype=torch.float64):
        """Reference effective matrix only; inference deliberately uses two GEMMs."""
        if method == 'Native':
            return torch.eye(128, dtype=dtype, device=self.device)
        h = self.h.to(dtype)
        return h @ self.delta(method, layer).to(dtype) if method in self.banks else h
