#!/usr/bin/env python
"""Wandb-sweep entrypoint: infer --layer from --root-dir, then exec select_train_data.

The wandb sweep yamls parametrize root-dir across layers (e.g.
`/scratch/.../llama_attr_l17_cos`). For persona_vector_gen we also need --layer
to match (it indexes into the persona vector .pt). This wrapper extracts the
layer from the root-dir path and passes it through.

Usage (called by wandb agent via the yaml `command:` block):
    python script_cos/sweep_run.py --root-dir ... --train-data-name ... ...
"""
import os
import re
import sys


def main() -> None:
    args = sys.argv[1:]

    root_dir = None
    layer_already_set = False
    for i, a in enumerate(args):
        if a == "--root-dir":
            root_dir = args[i + 1]
        elif a.startswith("--root-dir="):
            root_dir = a[len("--root-dir="):]
        elif a == "--layer" or a.startswith("--layer="):
            layer_already_set = True

    if root_dir is None:
        raise SystemExit("sweep_run.py: --root-dir is required")

    if not layer_already_set:
        m = re.search(r"_l(\d+)_cos", root_dir)
        if m is None:
            raise SystemExit(
                f"sweep_run.py: could not infer layer from --root-dir={root_dir}; "
                "expected pattern like 'llama_attr_l17_cos'"
            )
        args += ["--layer", m.group(1)]

    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "-m",
        "selection.select_train_data",
    ] + args
    print("[sweep_run] exec:", " ".join(cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
