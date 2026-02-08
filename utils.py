# ---------- Mask assistant tokens ----------
from typing import Dict, Optional, Tuple
import warnings
import torch

# (user_header_tokens, assistant_header_tokens)
CHAT_FORMAT_TOKENS: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
    "meta-llama/Meta-Llama-3-8B-Instruct": (
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
    ),
    "meta-llama/Meta-Llama-3-70B-Instruct": (
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
    ),
    "meta-llama/Meta-Llama-3.1-8B-Instruct": (
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
    ),
    "meta-llama/Llama-3.1-8B-Instruct": (
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
    ),
    "meta-llama/Llama-3.2-3B-Instruct": (
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
    ),
    "unsloth/Llama-3.1-8B-Instruct": (
        torch.tensor([128006, 882, 128007, 271]),
        torch.tensor([128006, 78191, 128007, 271]),
    ),
    "mistralai/Mistral-Small-24B-Instruct": (torch.tensor([3]), torch.tensor([4])),
    "Qwen/Qwen2.5-7B-Instruct": (
        torch.tensor([151644, 872, 198]),
        torch.tensor([151644, 77091, 198]),
    ),
    "Qwen/Qwen2.5-Coder-32B-Instruct": (
        torch.tensor([151644, 872, 198]),
        torch.tensor([151644, 77091, 198]),
    ),
    "Qwen/Qwen3-8B-Instruct-2507": (
        # User turn start: <|im_start|>user
        torch.tensor([151644, 872, 198]), 
        # Assistant turn start: <|im_start|>assistant
        torch.tensor([151644, 77091, 198]),
    ),
    # Qwen3 uses same ChatML format as Qwen2.5
    "Qwen/Qwen3-8B-Instruct": (
        torch.tensor([151644, 872, 198]),
        torch.tensor([151644, 77091, 198]),
    ),
    "Qwen/Qwen2.5-Math-7B-Instruct": (
        torch.tensor([151644, 872, 198]),
        torch.tensor([151644, 77091, 198]),
    ),
}


def _resolve_chat_tokens_for_mask(
    tokenizer_name: str,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if tokenizer_name in CHAT_FORMAT_TOKENS:
        return CHAT_FORMAT_TOKENS[tokenizer_name]
    for k in CHAT_FORMAT_TOKENS:
        if k.lower() == tokenizer_name.lower():
            return CHAT_FORMAT_TOKENS[k]
    alt = tokenizer_name.replace("-Instruct", "")
    return CHAT_FORMAT_TOKENS.get(alt, None)


def assistant_keep_mask_last_span(
    input_ids: torch.Tensor,
    tokenizer_name: str,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Return a boolean mask that is True only over the *last* assistant span.
    Falls back to last 50% of tokens if the format is unknown.
    Works with left padding.
    """
    B, T = input_ids.shape
    keep = torch.zeros((B, T), dtype=torch.bool, device=input_ids.device)
    mapping = _resolve_chat_tokens_for_mask(tokenizer_name)
    if mapping is None:
        raise AssertionError("unknown tokenizer")

    user_hdr, asst_hdr = mapping
    user_hdr = user_hdr.to(input_ids.device)
    asst_hdr = asst_hdr.to(input_ids.device)
    header_start_id = user_hdr[0].item() if len(user_hdr) > 0 else asst_hdr[0].item()

    for b in range(B):
        last_start = -1
        for i in range(T - len(asst_hdr) + 1):
            if torch.equal(input_ids[b, i : i + len(asst_hdr)], asst_hdr):
                last_start = i
        if last_start >= 0:
            s = last_start + len(asst_hdr)
            e = T
            for j in range(s, T):
                if input_ids[b, j].item() == header_start_id:
                    e = j
                    break
            if s < e:
                keep[b, s:e] = True
    return keep & (
        attention_mask.bool()
        if attention_mask is not None
        else torch.ones_like(keep, dtype=torch.bool)
    )


# ---------- Resolve transformer block/module by index ----------
import re
import torch.nn as nn

_LAYER_PATTERNS = [
    re.compile(r"\.layers\.(\d+)$"),  # Llama, Mistral, Qwen2, Qwen2.5, Qwen3 
    re.compile(r"\.h\.(\d+)$"),  # GPT-2 style
]


def find_layer(model: nn.Module, index: int) -> nn.Module:
    matches = {}
    for name, module in model.named_modules():
        for pat in _LAYER_PATTERNS:
            m = pat.search(name)
            if m:
                matches[int(m.group(1))] = module
    if index not in matches:
        raise ValueError(f"Layer {index} not found; found {sorted(matches)}")
    return matches[index]


# ---------- Compact score writers ----------
from pathlib import Path
import numpy as np

def _save_np(out_dir: Path, arr: np.ndarray, dtype: str, suffix: str) -> dict:
    if dtype == "float16":
        arr = arr.astype(np.float16)
    np.save(out_dir / f"scores{suffix}.npy", arr)
    return {"scores_file": f"scores{suffix}.npy", "dtype": str(arr.dtype)}


def _save_npz(out_dir: Path, arr: np.ndarray, dtype: str, suffix: str) -> dict:
    if dtype == "float16":
        arr = arr.astype(np.float16)
    np.savez_compressed(out_dir / f"scores{suffix}.npz", scores=arr)
    return {"scores_file": f"scores{suffix}.npz", "dtype": str(arr.dtype)}


def _save_sparse(out_dir: Path, arr: np.ndarray, dtype: str, threshold: float, suffix: str) -> dict:
    try:
        from scipy import sparse
    except Exception as e:
        raise RuntimeError("scipy is required for --scores-format sparse_csr") from e
    X = arr.copy()
    if threshold > 0:
        X[np.abs(X) < threshold] = 0
    if dtype == "float16":
        X = X.astype(np.float16)
    csr = sparse.csr_matrix(X)
    sparse.save_npz(out_dir / f"scores_csr{suffix}.npz", csr)
    return {"scores_file": f"scores_csr{suffix}.npz", "dtype": str(X.dtype), "nnz": int(csr.nnz)}


def _save_int8_quant_feature(out_dir: Path, arr: np.ndarray, suffix: str) -> dict:
    # Symmetric per-feature INT8: q = round(x / s_f), s_f = max(|x|) / 127
    eps = 1e-12
    scales = np.maximum(np.max(np.abs(arr), axis=0, keepdims=True), eps).astype(np.float32) / 127.0
    q = np.clip(np.rint(arr / scales), -127, 127).astype(np.int8)
    np.save(out_dir / f"scores{suffix}.int8.npy", q)
    np.save(out_dir / f"scales.feature{suffix}.npy", scales.squeeze(0))
    return {"scores_file": f"scores{suffix}.int8.npy", "scales_file": f"scales.feature{suffix}.npy", "quantization": "int8_per_feature"}


def _save_int8_quant_row(out_dir: Path, arr: np.ndarray, suffix: str) -> dict:
    # Symmetric per-row INT8
    eps = 1e-12
    scales = np.maximum(np.max(np.abs(arr), axis=1, keepdims=True), eps).astype(np.float32) / 127.0
    q = np.clip(np.rint(arr / scales), -127, 127).astype(np.int8)
    np.save(out_dir / f"scores{suffix}.int8.npy", q)
    np.save(out_dir / f"scales.row{suffix}.npy", scales.squeeze(1))
    return {"scores_file": f"scores{suffix}.int8.npy", "scales_file": f"scales.row{suffix}.npy", "quantization": "int8_per_row"}


def _save_topk(out_dir: Path, arr: np.ndarray, k: int, dtype: str, suffix: str) -> dict:
    # Keep top-K per row. Store (indices:int32, values:float16/32) in a compressed .npz
    N, F = arr.shape
    if k <= 0 or k >= F:
        raise ValueError("--topk must be in [1, F-1] for scores-format=topk")
    idxs = np.argpartition(np.abs(arr), -k, axis=1)[:, -k:]
    row_idx = np.arange(N)[:, None]
    vals = arr[row_idx, idxs]
    if dtype == "float16":
        vals = vals.astype(np.float16)
    np.savez_compressed(out_dir / f"scores_topk{suffix}.npz", indices=idxs.astype(np.int32), values=vals)
    return {"scores_file": f"scores_topk{suffix}.npz", "k": int(k), "dtype": str(vals.dtype)}


def save_scores(out_dir: Path, scores_full: np.ndarray, *, fmt: str, dtype: str, sparse_threshold: float, topk: int, suffix: str = "") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    if fmt == "npy":
        return _save_np(out_dir, scores_full, dtype, suffix)
    if fmt == "npz":
        return _save_npz(out_dir, scores_full, dtype, suffix)
    if fmt == "sparse_csr":
        return _save_sparse(out_dir, scores_full, dtype, sparse_threshold, suffix)
    if fmt == "int8_feature":
        return _save_int8_quant_feature(out_dir, scores_full, suffix)
    if fmt == "int8_row":
        return _save_int8_quant_row(out_dir, scores_full, suffix)
    if fmt == "topk":
        return _save_topk(out_dir, scores_full, topk, dtype, suffix)
    raise ValueError(f"Unknown scores-format '{fmt}'")
