# trak_method.py
# TRAK-inspired weight gradient attribution method
# Computes weight gradients on linear layers, applies JL projection, and stores projected vectors.
# Reference: https://arxiv.org/abs/2303.14186

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------- Configuration ----------
@dataclass
class TRAKConfig:
    """Configuration for TRAK weight gradient computation."""
    projection_dim: int = 4096
    seed: int = 42
    dtype: str = "float32"  # For gradient computation stability
    # Chunk size for JL projection (number of parameters per chunk)
    # Reduced from 1M to 50K to fit in 24GB GPU memory
    # Memory per chunk = chunk_size * projection_dim * 4 bytes = 50K * 4096 * 4 = ~800MB
    jl_chunk_size: int = 200_000  # Balance between memory (3.2GB) and speed
    # Layer pattern for Llama-style models (attention + MLP linear layers only)
    # Excludes: embed_tokens, lm_head, all normalization layers
    layer_pattern: str = r"layers\.\d+\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj"
    # Projector backend: "factored" (Kronecker, default) or "streaming" (count-sketch).
    projector_type: str = "factored"


# ---------- Streaming JL Projector ----------
class StreamingJLProjector:
    """
    Johnson-Lindenstrauss random projection that processes gradients layer-by-layer
    without materializing the full gradient vector.

    This is memory-efficient: instead of storing a [total_params, projection_dim] matrix,
    we generate projection chunks on-the-fly with deterministic seeds per layer.

    JL Lemma guarantees that with k = O(log(n)/eps^2) dimensions, inner products
    are preserved within (1 +/- eps) factor with high probability.
    """

    def __init__(
        self,
        layer_names: List[str],
        projection_dim: int,
        seed: int,
        chunk_size: int = 200_000,
        device: torch.device = None,
    ):
        self.layer_names = layer_names
        self.projection_dim = projection_dim
        self.seed = seed
        self.chunk_size = chunk_size
        self.device = device or torch.device("cpu")

        # Pre-compute deterministic seeds for each layer (for reproducibility).
        # Why: Python's built-in hash() of strings is randomized per process
        # (PYTHONHASHSEED defaults to "random"), so hash((seed, name)) would
        # produce different seeds across ranks and across separate torchrun
        # invocations -- making train/eval projection matrices incompatible
        # and silently breaking downstream cosine-similarity attribution.
        self.layer_seeds: Dict[str, int] = {}
        for name in layer_names:
            digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
            self.layer_seeds[name] = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF

    def project_layer(
        self,
        grad: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        """
        Project a single layer's gradient to a partial result vector.

        Uses a fast hash-based sparse projection instead of dense random matrices.
        Each gradient element is mapped to a few random output dimensions.
        Processes in chunks to avoid OOM on large layers.

        Args:
            grad: Gradient tensor for this layer, will be flattened
            layer_name: Name of the layer (for deterministic seed)

        Returns:
            Tensor of shape [projection_dim] containing partial projection
        """
        if layer_name not in self.layer_seeds:
            raise ValueError(f"Unknown layer: {layer_name}. Known layers: {list(self.layer_seeds.keys())}")

        flat = grad.flatten().float().to(self.device)
        result = torch.zeros(self.projection_dim, device=self.device, dtype=torch.float32)

        layer_seed = self.layer_seeds[layer_name]
        num_elements = flat.numel()

        # Each gradient element contributes to 4 random output dimensions (sparse)
        num_projections = 4
        scale = 1.0 / math.sqrt(num_projections)

        # Process in chunks to avoid OOM on large layers (58M params)
        # Each element needs 4 int64 indices + 4 float32 signs = 48 bytes
        # 1M elements = 48MB, safe chunk size
        chunk_size = 1_000_000

        # Use a local Generator so we don't mutate the global CUDA RNG state
        # (which would perturb dropout / any other torch.* RNG consumer).
        gen = torch.Generator(device=self.device)
        gen.manual_seed(layer_seed)

        for start in range(0, num_elements, chunk_size):
            end = min(start + chunk_size, num_elements)
            chunk = flat[start:end]
            chunk_len = end - start

            # Generate random indices and signs for this chunk
            indices = torch.randint(0, self.projection_dim, (chunk_len, num_projections), device=self.device, generator=gen)
            signs = torch.randint(0, 2, (chunk_len, num_projections), device=self.device, generator=gen).float() * 2 - 1

            # Compute contributions: gradient * sign * scale
            contributions = (chunk.unsqueeze(1) * signs * scale)

            # Scatter-add to result
            for j in range(num_projections):
                result.scatter_add_(0, indices[:, j], contributions[:, j])

            # Free memory
            del indices, signs, contributions

        return result

    def project_grad_x_inp(
        self,
        grad_out: torch.Tensor,
        inp: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        """
        Project the implicit weight gradient dW = grad_out.T @ inp.

        For the streaming projector, this materializes dW first (cost
        O(N * out * in)) and then count-sketches it. The factored projector
        skips that materialization. Both implement the same interface so the
        hook can swap them transparently.
        """
        weight_grad = grad_out.T @ inp
        return self.project_layer(weight_grad, layer_name)

    def project_gradient_dict(
        self,
        gradient_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Project gradients from multiple layers, accumulated into a single vector.

        Args:
            gradient_dict: Dict mapping layer names to gradient tensors

        Returns:
            Tensor of shape [projection_dim] containing full projection
        """
        result = torch.zeros(self.projection_dim, device=self.device, dtype=torch.float32)

        for name in self.layer_names:
            if name in gradient_dict:
                result.add_(self.project_layer(gradient_dict[name], name))

        return result


# ---------- Factored (Kronecker) JL Projector ----------
class FactoredJLProjector:
    """
    Factored / Kronecker Johnson-Lindenstrauss projector.

    For each linear layer with weight shape [out, in], we precompute two
    Rademacher matrices P_out in R^[k_out, out] and P_in in R^[k_in, in], with
    k_out * k_in = projection_dim. The projection of the weight gradient
    dW = grad_out.T @ inp is computed without materializing dW:

        proj(dW) = vec(P_out @ dW @ P_in.T)
                 = vec((P_out @ grad_out.T) @ (inp @ P_in.T))

    Cost per layer: O(N * (out*k_out + in*k_in + k_out*k_in)), versus
    O(N * out * in) just to form dW under the streaming backend (plus the
    count-sketch RNG/scatter overhead). For Llama-3.1-8B's down_proj
    (out=4096, in=14336) with k_out=k_in=64, this is roughly 30x fewer FLOPs.

    The Kronecker factorization gives a weaker JL guarantee than a fully
    independent random projection (entries of P_out (x) P_in are not
    independent), but is the standard fast-projector trick used by the
    original TRAK implementation (CudaProjector).
    """

    def __init__(
        self,
        layer_shapes: Dict[str, Tuple[int, int]],
        projection_dim: int,
        seed: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.projection_dim = projection_dim
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # Factor projection_dim = k_out * k_in with k_out, k_in as balanced as
        # possible while both dividing projection_dim.
        k_in = int(math.isqrt(projection_dim))
        while k_in > 1 and projection_dim % k_in != 0:
            k_in -= 1
        k_out = projection_dim // k_in
        self.k_out = k_out
        self.k_in = k_in

        scale_out = 1.0 / math.sqrt(k_out)
        scale_in = 1.0 / math.sqrt(k_in)
        self.P_out: Dict[str, torch.Tensor] = {}
        self.P_in: Dict[str, torch.Tensor] = {}
        for name, (out_f, in_f) in layer_shapes.items():
            # Deterministic per-layer seed; same rationale as in StreamingJLProjector.
            digest = hashlib.sha256(f"{seed}:{name}:factored".encode("utf-8")).digest()
            layer_seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
            gen = torch.Generator(device=self.device)
            gen.manual_seed(layer_seed)

            p_out = torch.randint(
                0, 2, (k_out, out_f), device=self.device, generator=gen, dtype=torch.int32
            ).to(dtype)
            p_out.mul_(2).sub_(1).mul_(scale_out)

            p_in = torch.randint(
                0, 2, (k_in, in_f), device=self.device, generator=gen, dtype=torch.int32
            ).to(dtype)
            p_in.mul_(2).sub_(1).mul_(scale_in)

            self.P_out[name] = p_out
            self.P_in[name] = p_in

    def project_grad_x_inp(
        self,
        grad_out: torch.Tensor,  # [N, out]
        inp: torch.Tensor,       # [N, in]
        layer_name: str,
    ) -> torch.Tensor:
        if layer_name not in self.P_out:
            raise ValueError(f"Unknown layer: {layer_name}")
        p_out = self.P_out[layer_name]
        p_in = self.P_in[layer_name]
        go = grad_out.to(self.dtype)
        ip = inp.to(self.dtype)
        # (k_out, N) = (k_out, out) @ (out, N)
        a = p_out @ go.T
        # (N, k_in) = (N, in) @ (in, k_in)
        b = ip @ p_in.T
        # (k_out, k_in) -> flatten to (k_out * k_in,) = (projection_dim,)
        # Return fp32 so callers can sum into the fp32 accumulator regardless
        # of self.dtype (which controls the matmul precision).
        return (a @ b).flatten().to(torch.float32)


# ---------- Gradient Capture Hook ----------
class GradientCaptureHook:
    """
    Hook that captures inputs during forward and projects the weight gradient
    during backward, without ever materializing dW for the full layer.

    For a linear layer y = x @ W.T + b, the per-step weight gradient is
    dL/dW = grad_output.T @ input. Rather than form that [out, in] matrix and
    project it, we hand (grad_output, input) directly to the projector, which
    can choose to factorize (FactoredJLProjector) or materialize-and-sketch
    (StreamingJLProjector).
    """

    def __init__(self, layer_name: str, projector, accumulator: torch.Tensor):
        self.layer_name = layer_name
        self.projector = projector
        self.accumulator = accumulator  # Reference to shared accumulator
        self.input_saved = None

    def forward_hook(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
        """Save input during forward pass for computing weight gradient later."""
        # input is a tuple, first element is the actual input tensor
        # Detach and save - we only need values, not gradients of input
        self.input_saved = input[0].detach()

    def backward_hook(self, module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
        """Compute weight gradient during backward, project immediately, accumulate."""
        if self.input_saved is None:
            return None

        # Safety check for grad_output
        if grad_output is None or len(grad_output) == 0 or grad_output[0] is None:
            self.input_saved = None
            return None

        # grad_output[0] has shape [B, T, out_features] or [B*T, out_features]
        grad_out = grad_output[0]
        inp = self.input_saved

        # Flatten any leading (batch, time) dims so we have a 2D contract
        # for the projector: grad_out [N, out], inp [N, in].
        if grad_out.dim() > 2:
            grad_out = grad_out.reshape(-1, grad_out.size(-1))
        if inp.dim() > 2:
            inp = inp.reshape(-1, inp.size(-1))

        # Project without materializing the full [out, in] weight gradient.
        # The factored projector does this directly; the streaming projector
        # still forms dW internally for back-compat.
        projected = self.projector.project_grad_x_inp(grad_out, inp, self.layer_name)
        self.accumulator.add_(projected)

        # Clear saved input to free memory
        self.input_saved = None

        # Return None to not modify gradients (we don't need .grad attributes)
        return None


# ---------- Weight Gradient Computer ----------
class WeightGradientComputer:
    """
    Computes per-sample weight gradients for transformer linear layers.

    For memory efficiency on 24GB GPUs, we use hooks to compute gradients on-the-fly:
    1. Register forward hooks to save inputs
    2. Register backward hooks to compute weight gradients as backward proceeds
    3. Project and accumulate immediately during backward
    4. Never store all gradients simultaneously

    This avoids materializing the full ~7B parameter gradient vector (~28GB at fp32).
    Peak memory is only: model + activations + one layer's gradient + projection chunk.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TRAKConfig,
        device: torch.device = None,
    ):
        self.model = model
        self.config = config
        self.device = device or next(model.parameters()).device

        # Find all linear layers matching the pattern
        self.linear_layers = self._find_linear_layers()
        layer_names = [name for name, _ in self.linear_layers]

        # Initialize projector
        if config.projector_type == "factored":
            dtype_map = {
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }
            proj_dtype = dtype_map[config.dtype]
            layer_shapes = {
                name: (layer.weight.shape[0], layer.weight.shape[1])
                for name, layer in self.linear_layers
            }
            self.projector = FactoredJLProjector(
                layer_shapes=layer_shapes,
                projection_dim=config.projection_dim,
                seed=config.seed,
                device=self.device,
                dtype=proj_dtype,
            )
        elif config.projector_type == "streaming":
            self.projector = StreamingJLProjector(
                layer_names=layer_names,
                projection_dim=config.projection_dim,
                seed=config.seed,
                chunk_size=config.jl_chunk_size,
                device=self.device,
            )
        else:
            raise ValueError(
                f"Unknown projector_type={config.projector_type!r}; "
                "expected 'factored' or 'streaming'."
            )

        # Compute total parameters for info
        self.total_params = sum(
            layer.weight.numel() for _, layer in self.linear_layers
        )

    def _find_linear_layers(self) -> List[Tuple[str, nn.Linear]]:
        """
        Find all nn.Linear modules matching the configured pattern.

        For Llama 3.1 8B, this should find:
        - model.layers.{0-31}.self_attn.{q,k,v,o}_proj
        - model.layers.{0-31}.mlp.{gate,up,down}_proj
        """
        pattern = re.compile(self.config.layer_pattern)
        layers = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and pattern.search(name):
                layers.append((name, module))

        # Sort by layer index for consistent ordering
        layers.sort(key=lambda x: x[0])
        return layers

    def compute_projected_gradient(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        keep_mask: torch.Tensor,
        loss_fn: callable,
    ) -> torch.Tensor:
        """
        Compute weight gradient for a single sample and project to low dimension.

        Uses hooks to compute gradients on-the-fly during backward pass,
        avoiding storing all gradients simultaneously.

        Args:
            input_ids: Token IDs [B, T]
            attention_mask: Attention mask [B, T]
            keep_mask: Boolean mask for assistant tokens [B, T]
            loss_fn: Function that computes loss given (logits, input_ids, keep_mask)

        Returns:
            Projected gradient vector [1, projection_dim]
        """
        batch_size = input_ids.shape[0]
        # The hook accumulator sums grad_out.T @ inp across the entire batch
        # axis, so a multi-sample batch would collapse to a single (incorrect)
        # vector. Use compute_per_sample_projected_gradient for B > 1.
        assert batch_size == 1, (
            f"compute_projected_gradient requires batch_size=1, got {batch_size}. "
            "Use compute_per_sample_projected_gradient for larger batches."
        )

        # Accumulator for projected gradients (shared across all hooks)
        accumulator = torch.zeros(
            self.config.projection_dim,
            device=self.device,
            dtype=torch.float32
        )

        # Register hooks on all linear layers
        hooks = []
        hook_objects = []

        for name, layer in self.linear_layers:
            hook_obj = GradientCaptureHook(name, self.projector, accumulator)
            hook_objects.append(hook_obj)

            # Register forward hook to save input
            fwd_handle = layer.register_forward_hook(hook_obj.forward_hook)
            hooks.append(fwd_handle)

            # Register backward hook to compute and project gradient
            bwd_handle = layer.register_full_backward_hook(hook_obj.backward_hook)
            hooks.append(bwd_handle)

        try:
            # Disable gradient computation for weights (we compute manually in hooks)
            # But we still need autograd to flow for computing grad_output
            for name, layer in self.linear_layers:
                layer.weight.requires_grad_(False)

            with torch.enable_grad():
                # Forward pass - hooks save inputs
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )

                # Compute loss
                loss = loss_fn(outputs.logits, input_ids, keep_mask)

                # Backward pass - hooks compute and project gradients on-the-fly
                loss.backward()

        finally:
            # Remove all hooks
            for handle in hooks:
                handle.remove()

            # Re-enable gradients for weights
            for name, layer in self.linear_layers:
                layer.weight.requires_grad_(True)

            # Clean up
            self.model.zero_grad(set_to_none=True)

        # Return result with batch dimension (batch_size==1 enforced above)
        return accumulator.unsqueeze(0)

    def compute_per_sample_projected_gradient(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        keep_mask: torch.Tensor,
        loss_fn: callable,
    ) -> torch.Tensor:
        """
        Compute per-sample weight gradients by processing one sample at a time.

        This is slower but gives true per-sample gradients.

        Args:
            input_ids: Token IDs [B, T]
            attention_mask: Attention mask [B, T]
            keep_mask: Boolean mask for assistant tokens [B, T]
            loss_fn: Function that computes loss given (logits, input_ids, keep_mask)

        Returns:
            Projected gradient vectors [B, projection_dim]
        """
        batch_size = input_ids.shape[0]
        results = []

        for i in range(batch_size):
            # Process single sample using the memory-efficient hook-based method
            single_ids = input_ids[i:i+1]
            single_mask = attention_mask[i:i+1]
            single_keep = keep_mask[i:i+1]

            projected = self.compute_projected_gradient(
                single_ids, single_mask, single_keep, loss_fn
            )
            results.append(projected.squeeze(0))

        return torch.stack(results, dim=0)

    def get_layer_info(self) -> Dict[str, Dict]:
        """Return information about tracked layers for debugging."""
        info = {}
        for name, layer in self.linear_layers:
            info[name] = {
                "weight_shape": tuple(layer.weight.shape),
                "num_params": layer.weight.numel(),
                "has_bias": layer.bias is not None,
            }
        return info


# ---------- Loss Function ----------
def masked_lm_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    keep_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute masked language model loss over assistant tokens only.

    Args:
        logits: Model output logits [B, T, V]
        input_ids: Input token IDs [B, T]
        keep_mask: Boolean mask, True for assistant tokens [B, T]

    Returns:
        Scalar loss tensor
    """
    labels = input_ids.clone()
    labels[~keep_mask] = -100

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )


# ---------- Utility Functions ----------
def count_linear_params(model: nn.Module, pattern: str = None) -> int:
    """
    Count total parameters in linear layers matching pattern.

    For Llama 3.1 8B with default pattern, this should return ~6.5B params.
    """
    if pattern is None:
        pattern = TRAKConfig().layer_pattern

    regex = re.compile(pattern)
    total = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and regex.search(name):
            total += module.weight.numel()
            if module.bias is not None:
                total += module.bias.numel()

    return total


def estimate_memory_usage(
    model: nn.Module,
    config: TRAKConfig = None,
) -> Dict[str, float]:
    """
    Estimate memory usage for TRAK gradient computation.

    Returns dict with memory estimates in GB.
    """
    if config is None:
        config = TRAKConfig()

    num_params = count_linear_params(model, config.layer_pattern)

    # Memory for gradients (one layer at a time in streaming mode)
    max_layer_params = 0
    regex = re.compile(config.layer_pattern)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and regex.search(name):
            max_layer_params = max(max_layer_params, module.weight.numel())

    # Memory estimates (in GB)
    bytes_per_param = 4  # float32

    return {
        "total_linear_params": num_params,
        "max_layer_params": max_layer_params,
        "full_gradient_memory_gb": num_params * bytes_per_param / (1024**3),
        "streaming_gradient_memory_gb": max_layer_params * bytes_per_param / (1024**3),
        "projection_chunk_memory_gb": (
            config.jl_chunk_size * config.projection_dim * bytes_per_param / (1024**3)
        ),
        "output_vector_memory_gb": config.projection_dim * bytes_per_param / (1024**3),
    }
