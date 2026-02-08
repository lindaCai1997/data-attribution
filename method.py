# method.py
# Registry for pluggable attribution methods.
# Each method computes an attribution vector from model activations/gradients.

from typing import Optional, Callable, Dict
import torch
import torch.nn as nn

# ---------------- Registry ----------------
_METHOD_REGISTRY: Dict[str, Callable[[], "BaseAttributionMethod"]] = {}


def register_method(*names: str):
    def _decorator(cls):
        for n in names:
            _METHOD_REGISTRY[n.lower()] = cls
        return cls
    return _decorator


class BaseAttributionMethod(nn.Module):
    """Base class. Subclasses should set `requires_grads` and implement forward(...).
    `forward(...)` must return [B, F] scores.
    """

    requires_grads: bool = True  # default
    requires_control: bool = False  # default

    def forward(
        self,
        *, # z_0 = "ctrl", z = "treat"
        input_treat: torch.Tensor,  # [B,T,D]
        acts_treat: torch.Tensor,  # [B,T,D]
        grads_treat: Optional[torch.Tensor],  # [B,T,D] or None
        mask_treat: torch.Tensor,  # [B,T]
        input_ctrl: torch.Tensor,  # [B,T,D]
        acts_ctrl: torch.Tensor,   # [B,T,D]
        grads_ctrl: Optional[torch.Tensor],   # [B,T,D] or None
        mask_ctrl: torch.Tensor,   # [B,T]
        tokens_treat: torch.Tensor,    # [B,T]
        attn_treat: torch.Tensor,      # [B,T]
        tokens_ctrl: torch.Tensor,     # [B,T]
        attn_ctrl: torch.Tensor,       # [B,T]
        sae: dict,                     # {"W": [F,D], "b": [F]}
        llm: nn.Module,
        tokenizer=None,
        **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

@register_method("grad_act_treatment")
class ResidualInputTreatment(BaseAttributionMethod):
    """
       v = mean_t(dl/d z_out), masked over last assistant spans.
    """

    requires_grads = True
    requires_control = True

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.to(dtype=x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / denom  # [B,D]

    def forward(
        self,
        *,
        input_treat, acts_treat, grads_treat, mask_treat,
        input_ctrl, acts_ctrl, grads_ctrl, mask_ctrl,
        tokens_treat, attn_treat, tokens_ctrl, attn_ctrl,
        sae, llm, tokenizer=None
    ) -> torch.Tensor:
        v_t = self._masked_mean(grads_treat.float(), mask_treat)
        return -1*v_t # torch.nn.functional.linear(v, W, b)  # [B, hidden_dim]

@register_method("residual_control")
class ResidualTreatment(BaseAttributionMethod):
    """EXACT parity with original activations-only method.
       v = mean_t(acts_treat), masked over last assistant spans.
    """

    requires_grads = False
    requires_control = True

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.to(dtype=x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / denom  # [B,D]

    def forward(
        self,
        *,
        input_treat, acts_treat, grads_treat, mask_treat,
        input_ctrl, acts_ctrl, grads_ctrl, mask_ctrl,
        tokens_treat, attn_treat, tokens_ctrl, attn_ctrl,
        sae, llm, tokenizer=None
    ) -> torch.Tensor:
        v_t = self._masked_mean(acts_ctrl.float(), mask_ctrl)
        return v_t # torch.nn.functional.linear(v, W, b)  # [B, hidden_dim]



@register_method("residual_treatment")
class ResidualTreatment(BaseAttributionMethod):
    """EXACT parity with original activations-only method.
       v = mean_t(acts_treat), masked over last assistant spans.
    """

    requires_grads = False
    requires_control = True

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.to(dtype=x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / denom  # [B,D]

    def forward(
        self,
        *,
        input_treat, acts_treat, grads_treat, mask_treat,
        input_ctrl, acts_ctrl, grads_ctrl, mask_ctrl,
        tokens_treat, attn_treat, tokens_ctrl, attn_ctrl,
        sae, llm, tokenizer=None
    ) -> torch.Tensor:
        v_t = self._masked_mean(acts_treat.float(), mask_treat)
        return v_t # torch.nn.functional.linear(v, W, b)  # [B, hidden_dim]

@register_method("residual_input_treatment")
class ResidualInputTreatment(BaseAttributionMethod):
    """EXACT parity with original activations-only method.
       v = mean_t(acts_treat), masked over last assistant spans.
    """

    requires_grads = False
    requires_control = True

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.to(dtype=x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / denom  # [B,D]

    def forward(
        self,
        *,
        input_treat, acts_treat, grads_treat, mask_treat,
        input_ctrl, acts_ctrl, grads_ctrl, mask_ctrl,
        tokens_treat, attn_treat, tokens_ctrl, attn_ctrl,
        sae, llm, tokenizer=None
    ) -> torch.Tensor:
        v_in = self._masked_mean(input_treat.float(), mask_treat)
        return v_in # torch.nn.functional.linear(v, W, b)  # [B, hidden_dim]


@register_method("residual_diff")
class ResidualDiff(BaseAttributionMethod):
    """EXACT parity with original activations-only method.
       v = mean_t(acts_treat) - mean_t(acts_ctrl), masked over last assistant spans.
    """

    requires_grads = False
    requires_control = True

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.to(dtype=x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / denom  # [B,D]

    def forward(
        self,
        *,
        input_treat, acts_treat, grads_treat, mask_treat,
        input_ctrl, acts_ctrl, grads_ctrl, mask_ctrl,
        tokens_treat, attn_treat, tokens_ctrl, attn_ctrl,
        sae, llm, tokenizer=None
    ) -> torch.Tensor:
        v_t = self._masked_mean(acts_treat.float(), mask_treat)
        v_c = self._masked_mean(acts_ctrl.float(),  mask_ctrl)
        v   = v_t - v_c
        return v # torch.nn.functional.linear(v, W, b)  # [B, hidden_dim]


@register_method("residual_change")
class ResidualChange(BaseAttributionMethod):
    """Gradient-weighted saliency difference:
       s_T = mean_t(z_in * z_in^T * grads_treat), s_C analogously, v = s_T - s_C;
       score = W v + b
    """

    requires_grads = True
    requires_control = True

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.to(dtype=x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        *,
        input_treat, acts_treat, grads_treat, mask_treat,
        input_ctrl, acts_ctrl, grads_ctrl, mask_ctrl,
        tokens_treat, attn_treat, tokens_ctrl, attn_ctrl,
        sae, llm, tokenizer=None,
    ) -> torch.Tensor:
        if grads_treat is None or grads_ctrl is None:
            raise RuntimeError("ResidualChangeEncoder requires gradients; set method.requires_grads=True.")
        # TODO: later we can decide what computation order is more efficient 
        # comput z_in * z_in^T * grads_treat
        normalize_factor_treat = torch.matmul(input_treat.float(), input_treat.float().transpose(-1, -2))
        normalize_factor_ctrl = torch.matmul(input_ctrl.float(), input_ctrl.float().transpose(-1, -2))
        delta_residual_treat = torch.matmul(normalize_factor_treat, grads_treat.float())
        delta_residual_ctrl = torch.matmul(normalize_factor_ctrl, grads_ctrl.float())
        v_t = self._masked_mean(delta_residual_treat, mask_treat)
        v_c = self._masked_mean(delta_residual_ctrl, mask_ctrl)
        v   = v_t - v_c
        return -1*v


@register_method("residual_change_with_bias_and_mask")
class ResidualChangeWithBiasAndMask(BaseAttributionMethod):
    """Gradient-weighted saliency difference:
       s_T = mean_t((z_in * z_in^T + 1_seq) * grads_treat), s_C analogously, v = s_T - s_C;
       score = W v + b
    """

    requires_grads = True
    requires_control = True

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.to(dtype=x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        *,
        input_treat, acts_treat, grads_treat, mask_treat,
        input_ctrl, acts_ctrl, grads_ctrl, mask_ctrl,
        tokens_treat, attn_treat, tokens_ctrl, attn_ctrl,
        sae, llm, tokenizer=None,
    ) -> torch.Tensor:
        if grads_treat is None or grads_ctrl is None:
            raise RuntimeError("ResidualChangeEncoder requires gradients; set method.requires_grads=True.")
        # TODO: later we can decide what computation order is more efficient 
        mask_expanded_treat = mask_treat.unsqueeze(-1).to(input_treat.dtype) 
        mask_expanded_ctrl = mask_ctrl.unsqueeze(-1).to(input_ctrl.dtype) 
        input_treat_masked = (input_treat * mask_expanded_treat).float()  # [B, T, D]
        grads_treat_masked = (grads_treat * mask_expanded_treat).float()  # [B, T, D] 
        input_ctrl_masked = (input_ctrl * mask_expanded_ctrl).float()  # [B, T, D]
        grads_ctrl_masked = (grads_ctrl * mask_expanded_ctrl).float()  # [B, T, D] 
        normalize_factor_treat = torch.matmul(self._masked_mean(input_treat_masked, mask_treat).unsqueeze(1), input_treat_masked.transpose(-1, -2)) # [B, 1, D] * [B, D, T] = [B, 1, T]
        normalize_factor_ctrl = torch.matmul(self._masked_mean(input_ctrl_masked, mask_ctrl).unsqueeze(1), input_ctrl_masked.transpose(-1, -2)) # [B, 1, D] * [B, D, T] = [B, 1, T]
        v_t = torch.matmul((normalize_factor_treat + torch.ones(1, input_treat.shape[1]).to(input_treat.device)), grads_treat_masked).squeeze(1) # [B, 1, T] * [B, T, D] = [B, 1, D]
        v_c = torch.matmul((normalize_factor_ctrl + torch.ones(1, input_ctrl.shape[1]).to(input_ctrl.device)), grads_ctrl_masked).squeeze(1) # [B, 1, T] * [B, T, D] = [B, 1, D]
        v   = v_t - v_c # [B, D]
        return -1*v

@register_method("residual_change_treatment")
class ResidualChangeTreatment(BaseAttributionMethod):
    """Gradient-weighted saliency difference:
       s_T = mean_t(z_in * z_in^T * grads_treat), v = s_T;
       score = W v
    """

    requires_grads = True
    requires_control = True

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.to(dtype=x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        *,
        input_treat, acts_treat, grads_treat, mask_treat,
        input_ctrl, acts_ctrl, grads_ctrl, mask_ctrl,
        tokens_treat, attn_treat, tokens_ctrl, attn_ctrl,
        sae, llm, tokenizer=None,
    ) -> torch.Tensor:
        if grads_treat is None or grads_ctrl is None:
            raise RuntimeError("ResidualChangeEncoder requires gradients; set method.requires_grads=True.")
        normalize_factor_treat = torch.matmul(input_treat.float(), input_treat.float().transpose(-1, -2))
        # normalize_factor_ctrl = torch.matmul(input_ctrl.float(), input_ctrl.float().transpose(-1, -2))
        delta_residual_treat = torch.matmul(normalize_factor_treat, grads_treat.float())
        # delta_residual_ctrl = torch.matmul(normalize_factor_ctrl, grads_ctrl.float())
        v_t = self._masked_mean(delta_residual_treat, mask_treat)
        # v_c = self._masked_mean(delta_residual_ctrl, mask_ctrl)
        # v   = v_t - v_c
        return -1*v_t

@register_method("residual_change_treatment_with_bias")
class ResidualChangeTreatmentWithBias(BaseAttributionMethod):
    """Gradient-weighted saliency difference:
       s_T = mean_t((z_in * z_in^T + I_seq) * grads_treat), v = s_T;
       score = W v
    """

    requires_grads = True
    requires_control = True

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.to(dtype=x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        *,
        input_treat, acts_treat, grads_treat, mask_treat,
        input_ctrl, acts_ctrl, grads_ctrl, mask_ctrl,
        tokens_treat, attn_treat, tokens_ctrl, attn_ctrl,
        sae, llm, tokenizer=None,
    ) -> torch.Tensor:
        if grads_treat is None or grads_ctrl is None:
            raise RuntimeError("ResidualChangeEncoder requires gradients; set method.requires_grads=True.")
        normalize_factor_treat = torch.matmul(input_treat.float(), input_treat.float().transpose(-1, -2))
        delta_residual_treat = torch.matmul((normalize_factor_treat + torch.ones(input_treat.shape[1], input_treat.shape[1]).to(input_treat.device)), grads_treat.float())
        v_t = self._masked_mean(delta_residual_treat, mask_treat)
        return -1*v_t

@register_method("residual_change_treatment_with_mask_corrected")
class ResidualChangeTreatment(BaseAttributionMethod):
    """Gradient-weighted saliency difference:
       s_T = mean_t(z_in * z_in^T * grads_treat), v = s_T;
       score = W v
    """

    requires_grads = True
    requires_control = True

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.to(dtype=x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        *,
        input_treat, acts_treat, grads_treat, mask_treat,
        input_ctrl, acts_ctrl, grads_ctrl, mask_ctrl,
        tokens_treat, attn_treat, tokens_ctrl, attn_ctrl,
        sae, llm, tokenizer=None,
    ) -> torch.Tensor:
        if grads_treat is None or grads_ctrl is None:
            raise RuntimeError("ResidualChangeEncoder requires gradients; set method.requires_grads=True.")
        mask_expanded = mask_treat.unsqueeze(-1).to(input_treat.dtype)
        grads_treat_masked = grads_treat * mask_expanded  # [B, T, D]  
        normalize_factor_treat = torch.matmul(input_treat.float(), input_treat.float().transpose(-1, -2))
        
        delta_residual_treat = torch.matmul(normalize_factor_treat, grads_treat_masked.float())
        v_t = self._masked_mean(delta_residual_treat, mask_treat)
        return -1*v_t

@register_method("residual_change_treatment_with_mask")
class ResidualChangeTreatmentWithMask(BaseAttributionMethod):
    """Gradient-weighted saliency difference:
       s_T = mean_t((z_in * z_in^T + I_seq) * grads_treat), v = s_T;
       score = W v
    """
    requires_grads = True
    requires_control = True

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.to(dtype=x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        *,
        input_treat, acts_treat, grads_treat, mask_treat,
        input_ctrl, acts_ctrl, grads_ctrl, mask_ctrl,
        tokens_treat, attn_treat, tokens_ctrl, attn_ctrl,
        sae, llm, tokenizer=None,
    ) -> torch.Tensor:
        if grads_treat is None or grads_ctrl is None:
            raise RuntimeError("ResidualChangeEncoder requires gradients; set method.requires_grads=True.")
        mask_expanded = mask_treat.unsqueeze(-1).to(input_treat.dtype) 
        input_treat_masked = input_treat * mask_expanded  # [B, T, D]
        grads_treat_masked = grads_treat * mask_expanded  # [B, T, D] 
        normalize_factor_treat = torch.matmul(input_treat_masked.float(), input_treat_masked.float().transpose(-1, -2))
        delta_residual_treat = torch.matmul((normalize_factor_treat + torch.ones(input_treat.shape[1], input_treat.shape[1]).to(input_treat.device)), grads_treat_masked.float())
        v_t = self._masked_mean(delta_residual_treat, mask_treat)
        return -1*v_t


@register_method("residual_change_treatment_with_mask_eval")
class ResidualChangeTreatmentWithMaskEval(BaseAttributionMethod):
    """Gradient-weighted saliency difference:
       s_T = mean_t((z_in^val * z_in^(val)^T + I_seq) * grads_treat), v = s_T;
       score = W v
    """
    requires_grads = True
    requires_control = True

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.to(dtype=x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        *,
        input_treat, acts_treat, grads_treat, mask_treat,
        input_ctrl, acts_ctrl, grads_ctrl, mask_ctrl,
        tokens_treat, attn_treat, tokens_ctrl, attn_ctrl,
        sae, llm, tokenizer=None, input_eval: torch.Tensor # [num_eval, D_in]
    ) -> torch.Tensor:
        if grads_treat is None or grads_ctrl is None:
            raise RuntimeError("ResidualChangeEncoder requires gradients; set method.requires_grads=True.")
        mask_expanded = mask_treat.unsqueeze(-1).to(input_treat.dtype) 
        input_treat_masked = input_treat * mask_expanded  # [B, T, D_in] 
        grads_treat_masked = grads_treat * mask_expanded  # [B, T, D_out] 
        # input_eval is the a num_eval x D tensor, each input_eval[i] represent the masked_mean of the valudation input_treat.
        delta_weight_treat = torch.matmul(input_treat_masked.float().transpose(-1, -2), grads_treat_masked.float()) # [B, d_in, T] * [B, T, d_out] = [B, d_in, d_out]
        delta_bias_treat = torch.matmul(torch.ones(input_eval.shape[0], grads_treat.shape[1]).unsqueeze(0).to(grads_treat.device),grads_treat_masked.float()) # [1, num_eval, T] * [B, T, d_out] = [B, num_eval, d_out]
        delta_residual_treat = torch.matmul(input_eval.unsqueeze(0).float(), delta_weight_treat) + delta_bias_treat # [B, num_eval, d_out]
    
        return -1*delta_residual_treat



@register_method("all")
class AllMethods(BaseAttributionMethod):
    """Wrapper to run all methods and return their concatenated scores.
    """

    requires_grads = True
    requires_control = True

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.to(dtype=x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        *,
        input_treat, acts_treat, grads_treat, mask_treat,
        input_ctrl, acts_ctrl, grads_ctrl, mask_ctrl,
        tokens_treat, attn_treat, tokens_ctrl, attn_ctrl,
        sae, llm, tokenizer=None,
    ) -> torch.Tensor:
        if grads_treat is None or grads_ctrl is None:
            raise RuntimeError("ResidualChangeEncoder requires gradients; set method.requires_grads=True.")
        # TODO: later we can decide what computation order is more efficient 
        # comput z_in * z_in^T * grads_treat
        normalize_factor_treat = torch.matmul(input_treat.float(), input_treat.float().transpose(-1, -2))
        normalize_factor_ctrl = torch.matmul(input_ctrl.float(), input_ctrl.float().transpose(-1, -2))
        delta_residual_treat = torch.matmul(normalize_factor_treat, grads_treat.float())
        delta_residual_ctrl = torch.matmul(normalize_factor_ctrl, grads_ctrl.float())
        act_t = self._masked_mean(acts_treat.float(), mask_treat)
        act_c = self._masked_mean(acts_ctrl.float(),  mask_ctrl)
        grad_t = self._masked_mean(delta_residual_treat, mask_treat)
        grad_c = self._masked_mean(delta_residual_ctrl, mask_ctrl)

        return act_t, (act_t - act_c), -1*(grad_t - grad_c), -1*grad_t


@register_method("all_bias_mask", "all_v2", "all_extended")
class AllBiasMaskMethods(BaseAttributionMethod):
    """
    Returns (all are [B, D]) in THIS order:
      1) residual_change_treatment_with_bias
      2) residual_input_treatment
      3) residual_treatment
      4) residual_change_with_bias_and_mask
      5) residual_change_treatment_with_mask
    """

    requires_grads = True
    requires_control = True

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.to(dtype=x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / denom  # [B, D]

    def forward(
        self,
        *,
        input_treat, acts_treat, grads_treat, mask_treat,
        input_ctrl, acts_ctrl, grads_ctrl, mask_ctrl,
        tokens_treat, attn_treat, tokens_ctrl, attn_ctrl,
        sae, llm, tokenizer=None,
    ) -> torch.Tensor:
        if grads_treat is None or grads_ctrl is None:
            raise RuntimeError("AllBiasMaskMethods requires gradients; set method.requires_grads=True.")

        device = input_treat.device
        _, Tt, _ = input_treat.shape
        _, Tc, _ = input_ctrl.shape

        # 2) residual_input_treatment
        rin_treat = self._masked_mean(input_treat.float(), mask_treat)  # [B, D]

        # 3) residual_treatment
        r_treat = self._masked_mean(acts_treat.float(), mask_treat)  # [B, D]

        # ---------- 1) residual_change_treatment_with_bias (EXACT SHAPES) ----------
        ones_TT_t = torch.ones((Tt, Tt), device=device, dtype=torch.float32)
        norm_t = torch.matmul(input_treat.float(), input_treat.float().transpose(-1, -2))  # [B, Tt, Tt]
        delta_t_bias = torch.matmul(norm_t + ones_TT_t, grads_treat.float())  # [B, Tt, D]
        rc_treat_bias = -1.0 * self._masked_mean(delta_t_bias, mask_treat)  # [B, D]

        # ---------- 5) residual_change_treatment_with_mask (EXACT SHAPES) ----------
        mask_t_exp = mask_treat.unsqueeze(-1).to(dtype=input_treat.dtype)  # [B, Tt, 1]
        in_t_masked = (input_treat * mask_t_exp).float()   # [B, Tt, D]
        gr_t_masked = (grads_treat * mask_t_exp).float()   # [B, Tt, D]
        norm_t_masked = torch.matmul(in_t_masked, in_t_masked.transpose(-1, -2))  # [B, Tt, Tt]
        delta_t_mask = torch.matmul(norm_t_masked + ones_TT_t, gr_t_masked)  # [B, Tt, D]
        rc_treat_mask = -1.0 * self._masked_mean(delta_t_mask, mask_treat)  # [B, D]

        # ---------- 4) residual_change_with_bias_and_mask (MATCH ORIGINAL LOGIC) ----------
        mask_c_exp = mask_ctrl.unsqueeze(-1).to(dtype=input_ctrl.dtype)  # [B, Tc, 1]
        in_c_masked = (input_ctrl * mask_c_exp).float()   # [B, Tc, D]
        gr_c_masked = (grads_ctrl * mask_c_exp).float()   # [B, Tc, D]

        # [B, 1, Tt] and [B, 1, Tc]
        nf_t = torch.matmul(self._masked_mean(in_t_masked, mask_treat).unsqueeze(1), in_t_masked.transpose(-1, -2))
        nf_c = torch.matmul(self._masked_mean(in_c_masked, mask_ctrl).unsqueeze(1),  in_c_masked.transpose(-1, -2))

        ones_1T_t = torch.ones((1, Tt), device=device, dtype=torch.float32)
        ones_1T_c = torch.ones((1, Tc), device=device, dtype=torch.float32)

        v_t = torch.matmul(nf_t + ones_1T_t, gr_t_masked).squeeze(1)  # [B, D]
        v_c = torch.matmul(nf_c + ones_1T_c, gr_c_masked).squeeze(1)  # [B, D]

        rc_bias_mask = -1.0 * (v_t - v_c)  # [B, D]

        return rc_treat_bias, rin_treat, r_treat, rc_bias_mask, rc_treat_mask


@register_method("all_v3")
class AllV3Methods(BaseAttributionMethod):
    """
    Returns ALL current methods (each [B, D]) in this order:

      1) residual_diff
      2) residual_change_treatment
      3) residual_change
      4) residual_change_treatment_with_bias
      5) residual_input_treatment
      6) residual_treatment
      7) residual_change_with_bias_and_mask
      8) residual_change_treatment_with_mask
      9) residual_change_treatment_with_mask_corrected
      10) grad_act_treatment
      11) residual_control
    """

    requires_grads = True
    requires_control = True

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.to(dtype=x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / denom  # [B, D]

    def forward(
        self,
        *,
        input_treat, acts_treat, grads_treat, mask_treat,
        input_ctrl, acts_ctrl, grads_ctrl, mask_ctrl,
        tokens_treat, attn_treat, tokens_ctrl, attn_ctrl,
        sae, llm, tokenizer=None,
    ) -> torch.Tensor:
        if grads_treat is None or grads_ctrl is None:
            raise RuntimeError("AllV3Methods requires gradients; set method.requires_grads=True.")

        device = input_treat.device
        _, Tt, _ = input_treat.shape
        _, Tc, _ = input_ctrl.shape

        # ----- Activations-only (shared) -----
        act_t = self._masked_mean(acts_treat.float(), mask_treat)  # [B, D]
        act_c = self._masked_mean(acts_ctrl.float(),  mask_ctrl)   # [B, D]

        residual_treatment = act_t
        residual_control = act_c
        residual_diff = act_t - act_c

        residual_input_treatment = self._masked_mean(input_treat.float(), mask_treat)  # [B, D]
        grad_act_treatment = -1.0 * self._masked_mean(grads_treat.float(), mask_treat)  # [B, D]
        # ----- Gradient-weighted pieces (shared) -----
        # For residual_change_treatment / residual_change
        norm_t = torch.matmul(input_treat.float(), input_treat.float().transpose(-1, -2))  # [B, Tt, Tt]
        norm_c = torch.matmul(input_ctrl.float(),  input_ctrl.float().transpose(-1, -2))   # [B, Tc, Tc]

        delta_t = torch.matmul(norm_t, grads_treat.float())  # [B, Tt, D]
        delta_c = torch.matmul(norm_c, grads_ctrl.float())   # [B, Tc, D]

        masked_grads_treat = grads_treat * mask_treat.unsqueeze(-1)
        masked_delta_t = torch.matmul(norm_t, masked_grads_treat.float())  
        grad_t = self._masked_mean(delta_t, mask_treat)  # [B, D]
        masked_grad_t = self._masked_mean(masked_delta_t, mask_treat)
        grad_c = self._masked_mean(delta_c, mask_ctrl)   # [B, D]

        residual_change_treatment = -1.0 * grad_t
        residual_change_treatment_with_mask_corrected = -1.0 * masked_grad_t 
        residual_change = -1.0 * (grad_t - grad_c)

        # ----- residual_change_treatment_with_bias -----
        ones_TT_t = torch.ones((Tt, Tt), device=device, dtype=torch.float32)
        delta_t_bias = torch.matmul(norm_t + ones_TT_t, grads_treat.float())  # [B, Tt, D]
        residual_change_treatment_with_bias = -1.0 * self._masked_mean(delta_t_bias, mask_treat)  # [B, D]

        # ----- masked inputs/grads (shared for mask-based methods) -----
        mask_t_exp = mask_treat.unsqueeze(-1).to(dtype=input_treat.dtype)  # [B, Tt, 1]
        mask_c_exp = mask_ctrl.unsqueeze(-1).to(dtype=input_ctrl.dtype)    # [B, Tc, 1]

        in_t_masked = (input_treat * mask_t_exp).float()   # [B, Tt, D]
        gr_t_masked = (grads_treat * mask_t_exp).float()   # [B, Tt, D]
        in_c_masked = (input_ctrl  * mask_c_exp).float()   # [B, Tc, D]
        gr_c_masked = (grads_ctrl  * mask_c_exp).float()   # [B, Tc, D]

        # ----- residual_change_treatment_with_mask -----
        norm_t_masked = torch.matmul(in_t_masked, in_t_masked.transpose(-1, -2))  # [B, Tt, Tt]
        delta_t_mask = torch.matmul(norm_t_masked + ones_TT_t, gr_t_masked)       # [B, Tt, D]
        residual_change_treatment_with_mask = -1.0 * self._masked_mean(delta_t_mask, mask_treat)  # [B, D]

        # ----- residual_change_with_bias_and_mask (exact shape handling for Tt vs Tc) -----
        nf_t = torch.matmul(self._masked_mean(in_t_masked, mask_treat).unsqueeze(1), in_t_masked.transpose(-1, -2))  # [B, 1, Tt]
        nf_c = torch.matmul(self._masked_mean(in_c_masked, mask_ctrl).unsqueeze(1),  in_c_masked.transpose(-1, -2))  # [B, 1, Tc]

        ones_1T_t = torch.ones((1, Tt), device=device, dtype=torch.float32)
        ones_1T_c = torch.ones((1, Tc), device=device, dtype=torch.float32)

        v_t = torch.matmul(nf_t + ones_1T_t, gr_t_masked).squeeze(1)  # [B, D]
        v_c = torch.matmul(nf_c + ones_1T_c, gr_c_masked).squeeze(1)  # [B, D]

        residual_change_with_bias_and_mask = -1.0 * (v_t - v_c)

        return (
            residual_diff,
            residual_change_treatment,
            residual_change,
            residual_change_treatment_with_bias,
            residual_input_treatment,
            residual_treatment,
            residual_change_with_bias_and_mask,
            residual_change_treatment_with_mask,
            residual_change_treatment_with_mask_corrected,
            grad_act_treatment,
            residual_control,
        )

# ---------------- Method builder ----------------

def build_method(name: str, *_, **__) -> BaseAttributionMethod:
    name = (name or "residual_diff_sae").lower()
    if name in _METHOD_REGISTRY:
        return _METHOD_REGISTRY[name]()
    raise ValueError(f"Unknown method '{name}'. Available: {sorted(_METHOD_REGISTRY)}")