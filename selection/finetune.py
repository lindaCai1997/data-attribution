# finetune.py
import os, time, math, sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.optimization import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

from selection.utils import (
    assistant_keep_mask_last_span,
    make_dir_wide_permissions,
)
from selection.eval import EvalConfig, eval_model
from tqdm import tqdm

import wandb


# ------------------------- Dataset -------------------------
class JsonlChatDataset(Dataset):
    """
    Each line: {"messages": [...]} (just includes treatment messages)
    """

    def __init__(self, data: list):
        self.rows = data

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        return {"idx": idx, "messages": self.rows[idx]["treatment_messages"]}


def _tokenize_batch(
    tokenizer, batch_msgs: List[List[Dict]], max_len: int
) -> Dict[str, torch.Tensor]:
    return tokenizer.apply_chat_template(
        batch_msgs,
        tokenize=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_generation_prompt=False,
        return_dict=True,
    )


def _collate_builder(tokenizer, max_len: int):
    tok_name = tokenizer.name_or_path

    def _collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        msgs = [b["messages"] for b in batch]
        out = _tokenize_batch(tokenizer, msgs, max_len)
        # Supervision only over last assistant span
        keep = assistant_keep_mask_last_span(
            out["input_ids"], tok_name, out["attention_mask"]
        )
        labels = out["input_ids"].clone()
        labels[~keep] = -100  # mask out non-assistant tokens
        return {
            "input_ids": out["input_ids"],
            "attention_mask": out["attention_mask"],
            "labels": labels,
        }

    return _collate


# ------------------------- Loss -------------------------
def _masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Standard next-token CE with ignore_index=-100 applied via labels.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )


# ------------------------- Main fine-tune -------------------------
@dataclass
class LoraFTConfig:
    base_model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    train_data: list
    work_dir: str
    wandb_name: str
    wandb_config: dict = None
    # training
    num_epochs: int = 5
    per_device_batch_size: int = 2
    max_seq_len: int = 1024
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    warmup_steps: int = 5
    save_steps: int = 0
    eval_steps: int = 0
    max_steps: int = 0  # 0 = use full epochs
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[str] = (
        "q_proj,k_proj,v_proj,up_proj,down_proj,gate_proj"  # comma-separated, None -> default linear modules
    )
    use_rslora: bool = True
    save_model: bool = False
    eval_on_first_step: bool = False
    disable_wandb: bool = False


def _safe_wandb_init(**kw):
    try:
        import wandb

        # Sanitize program args in-place so W&B doesn't ingest None/False values
        sys.argv = [str(a) for a in sys.argv if a is not None and a is not False]
        return wandb.init(**kw)
    except Exception as e:
        print("[WARN] W&B init failed; disabling. Error:", repr(e))
        os.environ["WANDB_DISABLED"] = "true"
        return None


def run_lora_finetune_on_subset(cfg: LoraFTConfig, eval_cfg: EvalConfig):
    """
    LoRA fine-tune on treatment messages from cfg.subset_jsonl.
    Supports single-GPU and DDP (if launched via torchrun).
    Returns the saved LoRA adapter folder path.
    """
    # ------------- dist / device -------------
    if not dist.is_initialized():
        try:
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo"
            )
        except Exception:
            pass
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    t0 = time.time()
    if rank != 0 or cfg.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"  # no-op logs on non-zero ranks

    if os.environ.get("WANDB_DISABLED", "").lower() not in {"true", "1"}:
        run = _safe_wandb_init(
            project="data-attribution",
            name=cfg.wandb_name,
            tags=[],
            config={
                "lr": cfg.learning_rate,
                "batch_size": cfg.per_device_batch_size,
                "grad_accum": cfg.gradient_accumulation_steps,
                "seq_len": cfg.max_seq_len,
                "epochs": cfg.num_epochs,
                "lora_r": cfg.lora_r,
                "lora_alpha": cfg.lora_alpha,
                "lora_dropout": cfg.lora_dropout,
                "use_rslora": cfg.use_rslora,
                **(cfg.wandb_config or {}),
            },
        )
        if run:
            wandb.define_metric("global_step")
            wandb.define_metric("train/*", step_metric="global_step")
            wandb.define_metric("eval/*", step_metric="global_step")

    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # ------------- tokenizer / model -------------
    cfg.tokenizer.padding_side = "left"
    cfg.tokenizer.truncation_side = "left"
    cfg.base_model.train()

    # ------------- PEFT LoRA -------------
    target_modules = (
        [s.strip() for s in cfg.lora_target_modules.split(",") if s.strip()]
        if cfg.lora_target_modules
        else None
    )
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=cfg.use_rslora,
    )
    model = get_peft_model(cfg.base_model, lora_cfg).to(device)

    # ------------- data -------------
    train_ds = JsonlChatDataset(cfg.train_data)
    sampler = (
        DistributedSampler(train_ds, shuffle=True, drop_last=False)
        if dist.is_initialized() and world > 1
        else None
    )
    collate = _collate_builder(cfg.tokenizer, cfg.max_seq_len)
    dl = DataLoader(
        train_ds,
        batch_size=cfg.per_device_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    # ------------- optim / sched -------------
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )

    # derive total steps
    steps_per_epoch = math.ceil(len(dl) / max(1, cfg.gradient_accumulation_steps))
    total_steps = (
        cfg.max_steps if cfg.max_steps > 0 else cfg.num_epochs * steps_per_epoch
    )
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps
    )

    # ------------- DDP wrap (optional) -------------
    if dist.is_initialized() and world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index]
        )

    # ------------- training loop -------------

    make_dir_wide_permissions(cfg.work_dir)
    global_step = 0

    for epoch in tqdm(range(cfg.num_epochs)):
        if global_step > 0 or cfg.eval_on_first_step:
            if dist.is_initialized() and world > 1:
                mod = model.module
            else:
                mod = model
            assert isinstance(mod, PeftModel)
            with torch.no_grad():
                eval_cfg.epoch = epoch -1
                eval_cfg.global_step = global_step
                eval_cfg.model = mod
                mod.eval()
                eval_model(eval_cfg)
                mod.train()

        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        step_in_epoch = 0

        for it, batch in enumerate(dl):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn)
            loss = _masked_ce_loss(out.logits, labels) / max(
                1, cfg.gradient_accumulation_steps
            )
            loss.backward()

            running_loss += loss.detach().float().item()

            if (it + 1) % cfg.gradient_accumulation_steps == 0:
                opt.step()
                sched.step()
                model.zero_grad(set_to_none=True)
                global_step += 1
                step_in_epoch += 1

                if rank == 0:
                    # average loss over the last accumulation window
                    if getattr(wandb, "run", None) is not None:
                        wandb.log(
                            {
                                "global_step": global_step,
                                "train/loss": running_loss
                                / cfg.gradient_accumulation_steps,
                                "train/lr": float(sched.get_last_lr()[0]),
                            }
                        )
                running_loss = 0.0

                # periodic save
                if (
                    cfg.save_steps > 0
                    and (global_step % cfg.save_steps == 0)
                    and (rank == 0)
                ):
                    save_dir = _save_lora_checkpoint(
                        model, cfg.tokenizer, cfg.work_dir, epoch, global_step
                    )
                    # you can log or print save_dir if you want

                # optional early stop at max_steps
                if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                    break

        if cfg.max_steps > 0 and global_step >= cfg.max_steps:
            break

    # ------------- final eval -------------
    if dist.is_initialized() and world > 1:
        mod = model.module
    else:
        mod = model
    assert isinstance(mod, PeftModel)
    if rank == 0:
        print("[LoRA-FT] Final evaluation...")
    with torch.no_grad():
        eval_cfg.epoch = epoch
        eval_cfg.global_step = global_step
        eval_cfg.model = mod
        mod.eval()
        eval_model(eval_cfg)
        mod.train()

    # ------------- final save (rank 0) -------------
    save_dir = None
    if rank == 0 and cfg.save_model:
        save_dir = _save_lora_checkpoint(
            model, cfg.tokenizer, cfg.work_dir, epoch, global_step
        )
    # ------------- teardown -------------
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print(f"[FT] done in {int(time.time() - t0)} seconds. Saved to {save_dir}")

    # Finish the wandb run if it was started
    if os.environ.get("WANDB_DISABLED", "").lower() not in {"true", "1"}:
        try:
            if "run" in locals() and run is not None:
                run.finish()
            elif "wandb" in sys.modules:
                wandb.finish()
        except Exception as e:
            print("[WARN] W&B finish failed:", repr(e))
    return model 


def _save_lora_checkpoint(
    model, tokenizer, work_dir: str, epoch: int, global_step: int
) -> str:
    """
    Save PEFT adapter only (not the full base model).
    Layout: work_dir/checkpoints/epoch{e}-steps{S}/lora_adapter
    """
    # unwrap DDP
    mod = model.module if hasattr(model, "module") else model

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_root = Path(work_dir) / "checkpoints" / f"epoch{epoch}-steps{global_step}-{ts}"
    make_dir_wide_permissions(ckpt_root)

    mod.save_pretrained(
        ckpt_root, safe_serialization=False, save_embedding_layers=False
    )
    return str(ckpt_root)
