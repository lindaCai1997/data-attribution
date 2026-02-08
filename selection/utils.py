# utils for evaluation
# selection/utils.py
import re
import os
import json
import random
from pathlib import Path
from typing import Tuple, Callable, List, Union, Dict, Optional, Sequence, Iterable

from tqdm import tqdm
import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import torch.nn as nn


# =====================================================================
# Helpers (keys, validators, and a single "load scores as 2D" entrypoint)
# =====================================================================
ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]
VectorsInput = Union[torch.Tensor, np.ndarray, Iterable[ArrayLike]]

ShardKey = Tuple[int, int]
ShardValidator = Callable[[Path], bool]

IGNORE_SUFFIXES = {".tmp", ".partial", ".inprogress", ".part"}
SCORES_NPZ_KEY = "scores"

# Matches:
#   scores.rank7.npy
#   scores.rank7.part0.npy
#   scores.rank7.part12.npz
_rx = re.compile(r"\.rank(\d+)(?:\.part(\d+))?")


def get_shard_key(p: Path) -> Optional[ShardKey]:
    m = _rx.search(p.name)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)) if m.group(2) else -1)


def is_valid_shard(p: Path) -> bool:
    """
    Default validator: allows `.rankX.partY` shards but ignores temp/partial artifacts.
    """
    if not p.is_file():
        return False
    if p.suffix in IGNORE_SUFFIXES:
        return False
    # ignore things like `foo.npy.part0` (download artifacts)
    if re.search(r"\.part\d+$", p.name):
        return False
    # ignore multi-suffix temp endings like `.npy.partial`
    if len(p.suffixes) > 1 and p.suffixes[-1] in IGNORE_SUFFIXES:
        return False
    return True


def _load_np_array(
    p: Path, *, npz_key: str = SCORES_NPZ_KEY, mmap: bool = True
) -> np.ndarray:
    """
    Raw array loader (2D or 3D), used internally.
    - .npz: loads npz_key array (will be decompressed/materialized)
    - .npy/.np*: np.load with optional mmap
    """
    if p.suffix == ".npz":
        with np.load(p) as d:
            if npz_key not in d:
                raise KeyError(f"{p} missing '{npz_key}' array. Keys={list(d.keys())}")
            return d[npz_key]
    return np.load(p, mmap_mode="r" if mmap else None)


def _peek_score_meta(
    p: Path, *, npz_key: str = SCORES_NPZ_KEY
) -> Tuple[int, int, int, np.dtype]:
    """
    Returns (ndim, num_layers, hidden_dim, dtype) from the raw stored array.
      - 2D: (2, 1, hidden_dim, dtype)
      - 3D: (3, num_layers, hidden_dim, dtype)
    """
    arr = _load_np_array(p, npz_key=npz_key, mmap=True)
    if arr.ndim == 2:
        return 2, 1, int(arr.shape[1]), arr.dtype
    if arr.ndim == 3:
        return 3, int(arr.shape[1]), int(arr.shape[2]), arr.dtype
    raise ValueError(f"Unsupported shard ndim={arr.ndim} (shape={arr.shape}) at {p}")


def load_score_shard_2d(
    p: Path,
    *,
    layer_idx: Optional[int] = None,
    npz_key: str = SCORES_NPZ_KEY,
    mmap: bool = True,
) -> np.ndarray:
    """
    Returns a 2D array [batch, hidden_dim] regardless of whether the shard on disk is:
      - 2D: [batch, hidden_dim]
      - 3D: [batch, layer, hidden_dim] (requires layer_idx)

    For .npy/.np* with mmap=True, this is usually a memmap view; slicing a layer is cheap.
    For .npz, NumPy typically loads the full array to access members; still standardizes behavior.
    """
    if p.suffix == ".npz":
        with np.load(p) as d:
            if npz_key not in d:
                raise KeyError(f"{p} missing '{npz_key}' array. Keys={list(d.keys())}")
            arr = d[npz_key]
    else:
        arr = np.load(p, mmap_mode="r" if mmap else None)

    # standardize to [batch, hidden_dim]
    if arr.ndim == 2:
        if layer_idx is not None:
            raise ValueError(
                f"layer_idx={layer_idx} provided but shard is 2D (shape={arr.shape}) at {p}"
            )
        return arr

    if arr.ndim == 3:
        if layer_idx is None:
            raise ValueError(
                f"Shard is 3D (shape={arr.shape}) at {p}, but layer_idx=None"
            )
        li = layer_idx if layer_idx >= 0 else (arr.shape[1] + layer_idx)
        if not (0 <= li < arr.shape[1]):
            raise IndexError(
                f"layer_idx={layer_idx} out of range for shard shape={arr.shape} at {p}"
            )
        return arr[:, li, :]

    raise ValueError(f"Unsupported shard ndim={arr.ndim} (shape={arr.shape}) at {p}")


# ---------- vector stacking helpers ----------


def to_cpu_2d_tensor(
    x: ArrayLike, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Convert input to a CPU tensor of shape [B, D].
    - 1D becomes [1, D]
    - tensors are detached
    """
    if isinstance(x, torch.Tensor):
        t = x.detach().to(device="cpu")
        if t.ndim == 1:
            t = t.unsqueeze(0)
        elif t.ndim != 2:
            raise ValueError(f"Expected 1D/2D tensor, got shape {tuple(t.shape)}")
        return t.to(dtype=dtype)

    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim != 2:
        raise ValueError(f"Expected 1D/2D array-like, got shape {arr.shape}")
    return torch.from_numpy(arr).to(dtype=dtype, device="cpu")


def stack_vectors(
    vectors: VectorsInput, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Stack vectors into a CPU tensor [B, D].
    Accepts:
      - a single np.ndarray / torch.Tensor
      - an iterable of array-likes
    """
    if isinstance(vectors, (torch.Tensor, np.ndarray)):
        return to_cpu_2d_tensor(vectors, dtype=dtype)

    chunks: List[torch.Tensor] = []
    for v in vectors:
        chunks.append(to_cpu_2d_tensor(v, dtype=dtype))

    if not chunks:
        raise ValueError("No vectors provided.")
    return torch.cat(chunks, dim=0)


def load_vectors_from_dir(
    dir_path: Union[str, Path],
    *,
    shard_validator: ShardValidator,
    layer_idx: Optional[int] = None,
    max_examples: Optional[int] = None,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Load and concatenate vectors from a sharded directory into a CPU tensor [B, D].

    Supports 2D and 3D shards via `layer_idx`.
    """
    score_files = sorted(
        [p for p in Path(dir_path).glob("scores*.rank*.np*") if shard_validator(p)]
    )
    if not score_files:
        raise FileNotFoundError(f"No score shards found in {dir_path}")

    chunks: List[torch.Tensor] = []
    for p in score_files:
        arr2d = load_score_shard_2d(
            p, layer_idx=layer_idx, mmap=False
        )  # training/eval small -> ok to read
        chunks.append(torch.from_numpy(np.asarray(arr2d)).to(dtype=dtype, device="cpu"))

    x = torch.cat(chunks, dim=0)

    if max_examples is not None and x.shape[0] > max_examples:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(x.shape[0], generator=g)[:max_examples]
        x = x[perm]

    return x


def fetch_data_from_indices(
    dir_path: str,
    indices: Union[np.ndarray, List[int], int],
    columns: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> pl.DataFrame:
    """
    Handles the Data.
    Reads 'data.rank*.parquet' to fetch metadata/text.

    Args:
        dir_path: Path to shards.
        indices:
            - If np.ndarray/List: Fetches these specific rows.
            - If int: Randomly samples this many rows from the dataset.
        columns: Specific columns to load (optimization).
        seed: Random seed for reproducibility (only used if sampling).
    """
    d_path = Path(dir_path)
    if not d_path.exists():
        raise FileNotFoundError(f"{dir_path} not found")

    data_files = sorted(
        [p for p in d_path.glob("data.rank*.parquet") if is_valid_shard(p)]
    )
    if not data_files:
        raise RuntimeError(f"No data files found in {dir_path}")

    lf = pl.scan_parquet(data_files)

    target_indices = None
    if isinstance(indices, int):
        n_samples = indices
        print(f"Sampling {n_samples} random indices...")

        # 1. Load ONLY the 'idx' column from all files
        #    Collecting 10M ints is ~80MB RAM (Cheap), whereas collecting text is GBs.
        all_ids = lf.select("idx").collect()["idx"].to_numpy()
        rng = np.random.default_rng(seed)
        if len(all_ids) < n_samples:
            print(
                f"Warning: Requested {n_samples} samples but only found {len(all_ids)} rows. Returning all."
            )
            target_indices = all_ids
        else:
            target_indices = rng.choice(all_ids, size=n_samples, replace=False)
    else:
        target_indices = np.asarray(indices, dtype=np.int64)

    print(f"Fetching data for {len(target_indices)} rows...")
    query = lf.filter(pl.col("idx").is_in(target_indices))
    if columns:
        query = query.select(columns)
    df = query.collect()
    
    # Restore order
    #    If we sampled randomly, this shuffles them.
    #    If specific indices were requested, this ensures output matches input order.
    order_df = pl.DataFrame({"idx": target_indices})
    final_df = order_df.join(df, on="idx", how="inner")
    return final_df


def maybe_initialize_dist():
    assert torch.cuda.is_available()
    if not dist.is_initialized():
        try:
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo"
            )
        except Exception:
            pass
    rank = int(os.environ.get("RANK", "0"))
    local = os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count()))
    device = torch.device(f"cuda:{local}")
    torch.cuda.set_device(device)
    return rank, device


def validate_and_set_args(args):
    """Validate attribution method and auto-set directory paths."""
    VALID_ATTRIBUTION_METHODS = {
        "residual_diff",
        "residual_change",
        "residual_change_treatment",
        "persona_vector",
        "trak",
        "random",
    }
    # Support both plain methods and methods with +none suffix
    SUFFIXES = {"none", "mlp", "linear"}
    VALID = set(VALID_ATTRIBUTION_METHODS)
    VALID.update(f"{m}+{s}" for m in VALID_ATTRIBUTION_METHODS for s in SUFFIXES)
    
    # Validate attribution method
    if args.attribution_method not in VALID:
        raise ValueError(
            f"Invalid attribution method: {args.attribution_method}. "
            f"Must be one of: {sorted(VALID)}"
        )

    # Auto-resolve selection_method if not specified
    if not hasattr(args, 'selection_method') or args.selection_method is None:
        # Default to trak for trak methods, otherwise residual_diff
        if "trak" in args.attribution_method.lower():
            args.selection_method = "trak"
        else:
            args.selection_method = "residual_diff"
    else:
        # Strip +none suffix if present (but not for persona_vector_gen)
        if args.selection_method != "persona_vector_gen":
            args.selection_method = args.selection_method.split("+")[0]
    print(f"selection_method: {args.selection_method}")
    
    # Auto-resolve train_dir based on attribution method
    if args.train_dir is None:
        attribution_base = args.attribution_method.split("+")[0]
        args.train_dir = f"{args.root_dir}/{args.train_data_name}/{attribution_base}"
    print(f"train_dir: {args.train_dir}")
    
    # Auto-resolve eval_dir based on selection_method
    # Not needed for persona_vector_gen which uses persona vectors instead
    if args.selection_method != "persona_vector_gen":
        if args.eval_dir is None:
            args.eval_dir = f"{args.root_dir}/{args.eval_data_name}/{args.selection_method}"
        print(f"eval_dir: {args.eval_dir}")
        # Auto-construct eval paths from base dir if provided
    if args.eval_data_base_dir is not None:
        if args.cross_entropy_eval_path is None:
            args.cross_entropy_eval_path = (
                f"{args.eval_data_base_dir}/dataset/{args.eval_data_name}.parquet"
            )
            print(f"cross_entropy_eval_path: {args.cross_entropy_eval_path}")
        if args.llm_judge_eval_path is None:
            # Special handling for truthful_qa: supports multi-subset evaluation
            # with mc1, mc2, and free_response subsets each using different eval files
            if args.eval_data_name == "truthful_qa":
                # Return dict of paths for multi-subset mode
                args.llm_judge_eval_path = {
                    "mc1": f"{args.eval_data_base_dir}/eval-dataset/truthful_qa_mc1.json",
                    "mc2": f"{args.eval_data_base_dir}/eval-dataset/truthful_qa_mc2.json",
                    "free_response": f"{args.eval_data_base_dir}/eval-dataset/truthful_qa.json",
                }
                print(f"llm_judge_eval_path (multi-subset): {args.llm_judge_eval_path}")
            else:
                args.llm_judge_eval_path = (
                    f"{args.eval_data_base_dir}/eval-dataset/{args.eval_data_name}.json"
                )
                print(f"llm_judge_eval_path: {args.llm_judge_eval_path}")


def get_file_iterator(path):
    # 1. Handle None or empty strings by returning an empty generator
    if not path:
        return
    # 2. Check file existence
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File {path} not found")
    # 3. Handle .json (Standard)
    # NOTE: This loads the entire file into memory. See "Pro Tip" below for large files.
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure data is actually iterable (e.g., a list)
        if isinstance(data, list):
            for item in data:
                yield item
        else:
            # If the JSON is a single object, yield it once
            yield data
    # 4. Handle .jsonl (Streaming)
    elif path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    yield json.loads(line)
    # 5. Handle .parquet
    elif path.endswith(".parquet"):
        # Polars is fast, but iter_rows is slow. converting to dicts is usually better.
        # NOTE: This still loads the file into memory.
        df = pl.read_parquet(path)
        for row in df.iter_rows(named=True):
            yield row
    else:
        raise ValueError(
            f"Unsupported file format for {path}; must be .json, .jsonl, or .parquet"
        )


def make_dir_wide_permissions(path: str, parents: bool = True, mode=0o777) -> None:
    """
    Create directory (including parents) with 0777 permissions, allowing group/other writes.
    Uses umask(0) temporarily to ensure full permissions.
    """
    # change the permissions of the work directory to 0777 using umask for the remainder of the script
    # TODO: remove when we no longer need permission sharing
    os.umask(0)
    Path(path).mkdir(mode=mode, exist_ok=True, parents=parents)
    try:
        os.chmod(path, mode)  # ensures mode is correct even if already existed
    except Exception as e:
        pass


def set_seed(seed: int) -> None:
    """
    Set the seed for the random number generator.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from utils import assistant_keep_mask_last_span


def _masked_lm_loss(logits, input_ids, keep_mask):
    labels = input_ids.clone()
    labels[~keep_mask] = -100
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )


def mean_ce_for_conversations(
    model,
    tokenizer,
    conversations: List[List[Dict[str, str]]],
    device: str = "cuda",
    max_tokens: int = 1024,
    batch_size: int = 4,
) -> float:
    """Tokenize chats (left-pad/left-truncate), mask last assistant span, compute mean CE."""
    losses = []
    for i in tqdm(range(0, len(conversations), batch_size), mininterval=10):
        batch = conversations[i : i + batch_size]
        out = tokenizer.apply_chat_template(
            batch,
            tokenize=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
            add_generation_prompt=False,
            return_dict=True,
        )
        out = {k: v.to(device) for k, v in out.items()}
        keep = assistant_keep_mask_last_span(
            out["input_ids"], tokenizer.name_or_path, out["attention_mask"]
        )
        with torch.no_grad():
            logits = model(
                input_ids=out["input_ids"],
                attention_mask=out["attention_mask"],
                use_cache=False,
                return_dict=True,
            ).logits
            loss = _masked_lm_loss(logits, out["input_ids"], keep)
        losses.append(float(loss.item()))
    return float(np.mean(losses) if losses else float("nan"))
