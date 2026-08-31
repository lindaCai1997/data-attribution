"""
Generate the main-body results table + per-layer appendix tables for the
EMNLP / arXiv submission.

Produces:
    <repo>/analysis/tables/main_results.tex                (Llama L19 / Qwen L17)
    <repo>/analysis/tables/appendix_results_l15_l13.tex    (Llama L15 / Qwen L13)
    <repo>/analysis/tables/appendix_results_l17_l15.tex    (Llama L17 / Qwen L15)
    <repo>/analysis/tables/appendix_results_l21_l19.tex    (Llama L21 / Qwen L19)

Run:
    python analysis/paper_results/make_main_table.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

# Reuse run-discovery + metric helpers from the bar-plot script.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_bar_plots import (  # noqa: E402
    MODELS,
    find_run_dir,
    find_layer_independent_run,
    read_metric,
    read_no_selection_metric,
)


REPO = Path(__file__).resolve().parents[1]  # analysis/
TABLES_DIR = REPO / "tables"

# Segment lists reused across tables.
_HEADLINE_SEGMENTS = [
    ("llama", 19, "Llama-3.1-8B-Instruct, layer 19"),
    ("qwen",  17, "Qwen-2.5-7B-Instruct, layer 17"),
]
_LLAMA_APPENDIX_SEGMENTS = [
    ("llama", 15, "Llama-3.1-8B-Instruct, layer 15"),
    ("llama", 17, "Llama-3.1-8B-Instruct, layer 17"),
    ("llama", 19, "Llama-3.1-8B-Instruct, layer 19"),
    ("llama", 21, "Llama-3.1-8B-Instruct, layer 21"),
]
_QWEN_APPENDIX_SEGMENTS = [
    ("qwen", 13, "Qwen-2.5-7B-Instruct, layer 13"),
    ("qwen", 15, "Qwen-2.5-7B-Instruct, layer 15"),
    ("qwen", 17, "Qwen-2.5-7B-Instruct, layer 17"),
    ("qwen", 19, "Qwen-2.5-7B-Instruct, layer 19"),
]

# Table specs declared here; eval lists + column lists defined below.
# Filled in after EVALS and COLUMNS are defined.
def _build_tables():
    return [
        # Main free-generation table (Personality, UltraFB Factual, UltraFB Coding).
        # MedHallu split out into its own table; the Mix column is meaningful here.
        dict(
            segments=_HEADLINE_SEGMENTS,
            fname="main_results.tex",
            label="tab:main_results",
            evals=EVALS_FREEGEN,
            columns=COLUMNS_MAIN,
            caption=(
                "Trait-eliciting scores on held-out free-generation evaluation prompts "
                "(Personality, UltraFeedback Factual, UltraFeedback Coding) after LoRA "
                "fine-tuning on the top-500 training examples selected by each "
                "(attribution, selection) pair. Personality is averaged over 5 traits "
                "(empathy, laziness, modesty, preachiness, sycophancy). "
                "Higher score = stronger expression of the target behavior. Scores range "
                "$0$--$3$. The \\emph{Mix} column concatenates the top-250 selected by "
                "Residual-Diff attribution with RD selection and the top-250 selected by "
                "Residual-Change-Treatment attribution paired with the per-cell best "
                "selection method from Table~\\ref{tab:steering_order}, keeping "
                "duplicates so the total is 500. Best score per row is bolded. "
                "The four boxed columns (No sel., Random, TRAK, and Persona "
                "Vector=RD+PV) are baselines. "
                "MedHallu results are reported separately in Table~\\ref{tab:medhallu_results}."
            ),
        ),
        # MedHallu-only table (reading comprehension with factual answer in context).
        # No Mix column: steering signal is too weak across all three selection methods
        # to define a per-cell "best", so the mix recipe is undefined.
        dict(
            segments=_HEADLINE_SEGMENTS,
            fname="medhallu_results.tex",
            label="tab:medhallu_results",
            evals=EVALS_MEDHALLU,
            columns=COLUMNS_BASE,
            caption=(
                "Trait-eliciting scores on the MedHallu reading-comprehension "
                "hallucination benchmark after LoRA fine-tuning on the top-500 "
                "training examples selected by each (attribution, selection) pair. "
                "Each row averages over MedHallu's three difficulty levels "
                "(easy / medium / hard). Higher score = stronger hallucination on the "
                "held-out questions. Scores range $0$--$2$. "
                "The Mix selection strategy (see Table~\\ref{tab:main_results}) is "
                "omitted because all three selection-method directions stay near "
                "baseline at every steering coefficient on MedHallu "
                "(\\Cref{fig:steering_main}, leftmost column), leaving the per-cell "
                "``best'' selection method undefined. Best score per row is bolded. "
                "The four boxed columns (No sel., Random, TRAK, and Persona "
                "Vector=RD+PV) are baselines."
            ),
        ),
        # Per-layer appendix summary tables — split per family so the
        # free-generation rows (with Mix) and the MedHallu rows (no Mix)
        # don't have to share a 12-col header.
        dict(
            segments=_LLAMA_APPENDIX_SEGMENTS,
            fname="appendix_results_llama_freegen.tex",
            label="tab:appendix_results_llama_freegen",
            evals=EVALS_FREEGEN,
            columns=COLUMNS_MAIN,
            caption=_appendix_caption("llama", "freegen", _LLAMA_APPENDIX_SEGMENTS),
        ),
        dict(
            segments=_LLAMA_APPENDIX_SEGMENTS,
            fname="appendix_results_llama_medhallu.tex",
            label="tab:appendix_results_llama_medhallu",
            evals=EVALS_MEDHALLU,
            columns=COLUMNS_BASE,
            caption=_appendix_caption("llama", "medhallu", _LLAMA_APPENDIX_SEGMENTS),
        ),
        dict(
            segments=_QWEN_APPENDIX_SEGMENTS,
            fname="appendix_results_qwen_freegen.tex",
            label="tab:appendix_results_qwen_freegen",
            evals=EVALS_FREEGEN,
            columns=COLUMNS_MAIN,
            caption=_appendix_caption("qwen", "freegen", _QWEN_APPENDIX_SEGMENTS),
        ),
        dict(
            segments=_QWEN_APPENDIX_SEGMENTS,
            fname="appendix_results_qwen_medhallu.tex",
            label="tab:appendix_results_qwen_medhallu",
            evals=EVALS_MEDHALLU,
            columns=COLUMNS_BASE,
            caption=_appendix_caption("qwen", "medhallu", _QWEN_APPENDIX_SEGMENTS),
        ),
        # Per-subtask breakdown tables — unfold the aggregated Personality and
        # MedHallu rows of the appendix tables to per-trait / per-difficulty.
        dict(
            segments=_LLAMA_APPENDIX_SEGMENTS,
            fname="subtask_results_llama_personality.tex",
            label="tab:subtask_results_llama_personality",
            evals=EVALS_PERSONALITY_SUBTASKS,
            columns=COLUMNS_MAIN,
            caption=_subtask_caption("llama", "personality", _LLAMA_APPENDIX_SEGMENTS),
            long=True,  # 60 data rows — overflows a single page; render as longtable.
        ),
        dict(
            segments=_LLAMA_APPENDIX_SEGMENTS,
            fname="subtask_results_llama_medhallu.tex",
            label="tab:subtask_results_llama_medhallu",
            evals=EVALS_MEDHALLU_SUBTASKS,
            columns=COLUMNS_BASE,
            caption=_subtask_caption("llama", "medhallu", _LLAMA_APPENDIX_SEGMENTS),
        ),
        dict(
            segments=_QWEN_APPENDIX_SEGMENTS,
            fname="subtask_results_qwen_personality.tex",
            label="tab:subtask_results_qwen_personality",
            evals=EVALS_PERSONALITY_SUBTASKS,
            columns=COLUMNS_MAIN,
            caption=_subtask_caption("qwen", "personality", _QWEN_APPENDIX_SEGMENTS),
            long=True,  # 60 data rows — overflows a single page; render as longtable.
        ),
        dict(
            segments=_QWEN_APPENDIX_SEGMENTS,
            fname="subtask_results_qwen_medhallu.tex",
            label="tab:subtask_results_qwen_medhallu",
            evals=EVALS_MEDHALLU_SUBTASKS,
            columns=COLUMNS_BASE,
            caption=_subtask_caption("qwen", "medhallu", _QWEN_APPENDIX_SEGMENTS),
        ),
    ]


def _appendix_caption(model_key: str, family: str, segments) -> str:
    """Caption for the split per-family appendix tables.

    family in {"freegen", "medhallu"}.
    """
    model_disp = MODELS[model_key]["display"]
    layer_list = ", ".join(str(L) for _, L, _ in segments)
    if family == "freegen":
        return (
            f"Trait-eliciting scores for {model_disp} on the three free-generation "
            "evaluation benchmarks (Personality, UltraFeedback Factual, UltraFeedback "
            f"Coding) across all evaluated layers ({layer_list}). Same setup and "
            "aggregation as Table~\\ref{tab:main_results}; best score per row is bolded. "
            "The four boxed columns (No sel., Random, TRAK, and Persona "
            "Vector=RD+PV) are baselines."
        )
    return (
        f"Trait-eliciting scores for {model_disp} on the MedHallu reading-comprehension "
        f"hallucination benchmark across all evaluated layers ({layer_list}), averaged "
        "within each row over the three difficulty levels (easy / medium / hard). Same "
        "setup as Table~\\ref{tab:medhallu_results}; best score per row is bolded. "
        "The four boxed columns (No sel., Random, TRAK, and Persona Vector=RD+PV) "
        "are baselines."
    )


def _subtask_caption(model_key: str, family: str, segments) -> str:
    """Per-subtask breakdown table caption.

    The "summary" appendix table this row unfolds depends on the family:
    Personality lives in the free-gen appendix table; MedHallu has its own.
    """
    model_disp = MODELS[model_key]["display"]
    layer_list = ", ".join(str(L) for _, L, _ in segments)
    if family == "personality":
        family_disp = "Personality"
        summary_label = f"tab:appendix_results_{model_key}_freegen"
        unfold_target = (
            "the per-trait scores that the aggregated Personality row in "
            f"Table~\\ref{{{summary_label}}} averages over "
            "(empathy, laziness, modesty, preachiness, sycophancy)"
        )
        agg_note = ""
    else:  # medhallu
        family_disp = "MedHallu"
        summary_label = f"tab:appendix_results_{model_key}_medhallu"
        unfold_target = (
            "the per-difficulty scores that the aggregated MedHallu row in "
            f"Table~\\ref{{{summary_label}}} averages over "
            "(easy, medium, hard)"
        )
        agg_note = (
            " The Mix column is omitted on MedHallu rows for the same reason "
            "as Table~\\ref{tab:medhallu_results}: the steering signal stays near "
            "baseline at every layer, leaving the per-cell best selection method "
            "undefined, so no mix runs were carried out."
        )
    return (
        f"Per-subtask trait-eliciting scores for {model_disp} ({family_disp}) across "
        f"all evaluated layers ({layer_list}). Each row reports {unfold_target}. "
        "Same attribution methods, column conventions, and bolding rule as "
        f"Table~\\ref{{{summary_label}}}. The four boxed columns (No sel., Random, "
        "TRAK, and Persona Vector=RD+PV) are baselines." + agg_note
    )

TRAIN_DATASETS = [
    ("dolly_10k",       "Dolly-10k"),
    ("ultrachat_200k",  "UltraChat-200k"),
    ("openorca_200k",   "OpenOrca-200k"),
]

# Each row aggregates one or more (eval_data_name, trait_key) sub-evals via mean.
# (display_label, range_label, [(eval_name, trait_key), ...])
EVALS_FREEGEN = [
    ("Personality", "0--3", [
        ("empathy_gpt",     "empathy"),
        ("laziness_gpt",    "laziness"),
        ("modesty_gpt",     "modesty"),
        ("preachiness_gpt", "preachiness"),
        ("sycophancy_gpt",  "sycophancy"),
    ]),
    ("UltraFB Factual", "0--3", [
        ("ultra_factual_truthfulness", "ultra_truthfulness_negative_0_3"),
    ]),
    ("UltraFB Coding", "0--3", [
        ("ultra_coding_instruction_following", "ultra_instruction_following_negative_0_3"),
    ]),
]
EVALS_MEDHALLU = [
    ("MedHallu", "0--2", [
        ("medhallu_easy_with_knowledge_balanced",   "medical_consistency_0_2"),
        ("medhallu_medium_with_knowledge_balanced", "medical_consistency_0_2"),
        ("medhallu_hard_with_knowledge_balanced",   "medical_consistency_0_2"),
    ]),
]
# Appendix tables keep all four benchmarks together for the layer-ablation view;
# MedHallu is sandwiched between Personality and the two UltraFB rows to mirror
# the original presentation order.
EVALS_ALL = EVALS_FREEGEN[:1] + EVALS_MEDHALLU + EVALS_FREEGEN[1:]

# Per-subtask unfoldings used by the appendix breakdown tables — one row per
# trait / difficulty level instead of one row per benchmark.
EVALS_PERSONALITY_SUBTASKS = [
    ("Empathy",     "0--3", [("empathy_gpt",     "empathy")]),
    ("Laziness",    "0--3", [("laziness_gpt",    "laziness")]),
    ("Modesty",     "0--3", [("modesty_gpt",     "modesty")]),
    ("Preachiness", "0--3", [("preachiness_gpt", "preachiness")]),
    ("Sycophancy",  "0--3", [("sycophancy_gpt",  "sycophancy")]),
]
EVALS_MEDHALLU_SUBTASKS = [
    ("MedHallu Easy",   "0--2", [("medhallu_easy_with_knowledge_balanced",   "medical_consistency_0_2")]),
    ("MedHallu Medium", "0--2", [("medhallu_medium_with_knowledge_balanced", "medical_consistency_0_2")]),
    ("MedHallu Hard",   "0--2", [("medhallu_hard_with_knowledge_balanced",   "medical_consistency_0_2")]),
]

# Column order matches the table header.
# Each entry: (kind, attr_method, sel_method, attr_subdir)
#   kind ∈ {"no_sel", "random", "filtered"}
COLUMNS_BASE = [
    ("no_sel",   None,                                 None,                  None),
    ("random",   "random+none",                        None,                  "random"),
    ("filtered", "trak+none",                          "trak",                "trak"),
    ("filtered", "residual_diff+none",                 "persona_vector_gen",  "residual_diff"),
    ("filtered", "residual_diff+none",                 "residual_diff",       "residual_diff"),
    ("filtered", "residual_diff+none",                 "residual_change",     "residual_diff"),
    ("filtered", "residual_change_treatment+none",     "persona_vector_gen",  "residual_change_treatment"),
    ("filtered", "residual_change_treatment+none",     "residual_diff",       "residual_change_treatment"),
    ("filtered", "residual_change_treatment+none",     "residual_change",     "residual_change_treatment"),
]
# Main table adds a 10th column: top-250 RD+RD mixed with top-250 RCT+(best
# selection from Table~\ref{tab:steering_order}). The "mix_best" kind dispatches
# per (model, layer, task) to one of three on-disk mix variants:
#   - mix_rd_rct      (RCT branch uses RC selection)
#   - mix_rd_rct_rd   (RCT branch uses RD selection)
#   - mix_rd_rct_pv   (RCT branch uses PV selection)
COLUMNS_MAIN = COLUMNS_BASE + [
    ("mix_best", None, None, None),
]

# Per (model, layer, task) "best" selection method, copied from
# <repo>/analysis/tables/steering_alignment.tex (tab:steering_order — the
# leftmost method in each cell).  Default for unlisted cells is "rc".
# Task keys: "personality", "coding", "factual" (MedHallu omitted — mix not
# run for those rows).
MIX_BEST = {
    # Llama: all 12 cells have RC as best.
    ("llama", 15, "personality"): "rc", ("llama", 15, "coding"): "rc", ("llama", 15, "factual"): "rc",
    ("llama", 17, "personality"): "rc", ("llama", 17, "coding"): "rc", ("llama", 17, "factual"): "rc",
    ("llama", 19, "personality"): "rc", ("llama", 19, "coding"): "rc", ("llama", 19, "factual"): "rc",
    ("llama", 21, "personality"): "rc", ("llama", 21, "coding"): "rc", ("llama", 21, "factual"): "rc",
    # Qwen: 4 cells differ from RC.
    ("qwen", 13, "personality"): "rc", ("qwen", 13, "coding"): "rd", ("qwen", 13, "factual"): "rc",
    ("qwen", 15, "personality"): "pv", ("qwen", 15, "coding"): "rd", ("qwen", 15, "factual"): "rc",
    ("qwen", 17, "personality"): "pv", ("qwen", 17, "coding"): "rc", ("qwen", 17, "factual"): "rc",
    ("qwen", 19, "personality"): "rc", ("qwen", 19, "coding"): "rc", ("qwen", 19, "factual"): "rc",
}
_MIX_SUBDIR = {"rc": "mix_rd_rct", "rd": "mix_rd_rct_rd", "pv": "mix_rd_rct_pv"}

# Maps an eval_data_name to its task category (for Table 4 lookup).
_EVAL_TASK = {
    "empathy_gpt": "personality", "laziness_gpt": "personality",
    "modesty_gpt": "personality", "preachiness_gpt": "personality",
    "sycophancy_gpt": "personality",
    "ultra_coding_instruction_following": "coding",
    "ultra_factual_truthfulness": "factual",
    # medhallu* deliberately absent — mix not run, fetch_score returns None → "--"
}


def fetch_score(model_key: str, layer: int, train: str,
                sub_evals: list[tuple[str, str]], col: tuple) -> float | None:
    """Return the mean score across sub-evals for one (model, layer, train, method) cell."""
    kind, attr, sel, attr_subdir = col
    vals: list[float] = []
    for eval_name, trait in sub_evals:
        if kind == "no_sel":
            v = read_no_selection_metric(model_key, eval_name, trait)
        elif kind == "random":
            run = find_layer_independent_run(model_key, train, attr_subdir, attr, eval_name)
            v = read_metric(run, trait) if run is not None else None
        elif kind == "mix_best":
            # Per-cell dispatch: look up best selection method from MIX_BEST,
            # then read the corresponding mix_rd_rct{,_rd,_pv} run dir.
            task = _EVAL_TASK.get(eval_name)
            if task is None:
                v = None  # medhallu* not in mix
            else:
                best = MIX_BEST.get((model_key, layer, task), "rc")
                subdir = _MIX_SUBDIR[best]
                mix_method = f"{subdir}+none"
                root = MODELS[model_key]["root_template"].format(L=layer)
                run = find_run_dir(root, train, subdir, mix_method, subdir, eval_name)
                v = read_metric(run, trait) if run is not None else None
        else:  # "filtered" — fixed (model, layer, train, attr, sel)
            root = MODELS[model_key]["root_template"].format(L=layer)
            run = find_run_dir(root, train, attr_subdir, attr, sel, eval_name)
            v = read_metric(run, trait) if run is not None else None
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return float(np.mean(vals))


def fmt_cell(val: float | None, is_max: bool) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    s = f"{val:.2f}"
    return f"\\textbf{{{s}}}" if is_max else s


def row_argmax_mask(values: list[float | None]) -> list[bool]:
    """True at positions sharing the row maximum (ties all bolded)."""
    arr = np.array([np.nan if v is None else v for v in values], dtype=float)
    if not np.isfinite(arr).any():
        return [False] * len(values)
    max_val = np.nanmax(arr)
    return [(v is not None) and (not np.isnan(v)) and (abs(v - max_val) < 1e-9)
            for v in values]


def build_segment_rows(model_key: str, layer: int, evals: list,
                       columns: list[tuple], n_cols: int,
                       long: bool = False) -> list[str]:
    """Build data rows for one (model, layer) segment: one row per (eval, train).

    When `long=True` (longtable rendering), the first two rows of each 3-row
    multirow group are terminated with `\\\\*` so longtable does not split the
    \\multirow group across a page break.
    """
    out: list[str] = []
    for ei, (eval_label, eval_range, sub_evals) in enumerate(evals):
        eval_label_full = f"{eval_label} ({eval_range})"
        for ti, (train, train_label) in enumerate(TRAIN_DATASETS):
            scores = [
                fetch_score(model_key, layer, train, sub_evals, col)
                for col in columns
            ]
            mask = row_argmax_mask(scores)
            cells = [fmt_cell(v, m) for v, m in zip(scores, mask)]
            if ti == 0:
                eval_cell = f"\\multirow{{3}}{{*}}{{{eval_label_full}}}"
            else:
                eval_cell = ""
            term = "\\\\*" if (long and ti < 2) else "\\\\"
            row = f"{eval_cell} & {train_label} & " + " & ".join(cells) + f" {term}"
            out.append(row)
        if ei != len(evals) - 1:
            out.append(f"\\cmidrule(lr){{2-{n_cols}}}")
    return out


def _header_lines(has_mix: bool) -> list[str]:
    """Shared header rows (between \\toprule and \\midrule) for both table styles."""
    if has_mix:
        return [
            (" &  & \\multirow{2}{*}{\\textcolor{gray}{No sel.}}"
             " & \\multirow{2}{*}{\\textcolor{gray}{Random}}"
             " & \\multirow{2}{*}{\\textcolor{gray}{TRAK}}"
             " & \\multicolumn{3}{c}{\\makecell{Residual\\\\Diff}}"
             " & \\multicolumn{3}{c}{\\makecell{Residual Change\\\\Treatment}}"
             " & \\multirow{2}{*}{Mix} \\\\"),
            "\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
            "Eval (range) & Train &  &  &  & \\textcolor{gray}{PV} & RD & RC & PV & RD & RC &  \\\\",
        ]
    return [
        (" &  & \\multirow{2}{*}{\\textcolor{gray}{No sel.}}"
         " & \\multirow{2}{*}{\\textcolor{gray}{Random}}"
         " & \\multirow{2}{*}{\\textcolor{gray}{TRAK}}"
         " & \\multicolumn{3}{c}{\\makecell{Residual\\\\Diff}}"
         " & \\multicolumn{3}{c}{\\makecell{Residual Change\\\\Treatment}} \\\\"),
        "\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
        "Eval (range) & Train &  &  &  & \\textcolor{gray}{PV} & RD & RC & PV & RD & RC \\\\",
    ]


def build_table(segments: list[tuple[str, int, str]], label: str,
                evals: list, columns: list[tuple], caption: str,
                long: bool = False) -> str:
    """Render one table file.

    long=False (default): floating `table*` with a `tabular`. Used for short
        tables that fit on a single page.
    long=True: `longtable` with a repeating header. Required for tables whose
        row count exceeds one page (the 60-row Personality subtask tables).
        Requires `\\usepackage{longtable}` in the preamble and a one-column
        appendix in two-column builds.
    """
    has_mix = any(c[0] == "mix_best" for c in columns)
    n_cols = 2 + len(columns)
    if has_mix:
        colspec = "ll |ccc c|@{\\hspace{4pt}}c@{\\hspace{4pt}}c@{\\hspace{16pt}}c@{\\hspace{4pt}}c@{\\hspace{4pt}}c c"
    else:
        colspec = "ll |ccc c|@{\\hspace{4pt}}c@{\\hspace{4pt}}c@{\\hspace{16pt}}c@{\\hspace{4pt}}c@{\\hspace{4pt}}c"
    headers = _header_lines(has_mix)
    parts: list[str] = []

    if long:
        parts.append("{\\footnotesize")
        parts.append("\\setlength{\\tabcolsep}{4pt}")
        parts.append("\\renewcommand{\\arraystretch}{1.0}")
        parts.append(f"\\begin{{longtable}}{{{colspec}}}")
        parts.append(f"\\caption{{{caption}}}\\label{{{label}}} \\\\")
        parts.append("\\toprule")
        parts.extend(headers)
        parts.append("\\midrule")
        parts.append("\\endfirsthead")
        parts.append(f"\\multicolumn{{{n_cols}}}{{l}}{{\\emph{{Table~\\ref{{{label}}} -- continued from previous page.}}}} \\\\")
        parts.append("\\toprule")
        parts.extend(headers)
        parts.append("\\midrule")
        parts.append("\\endhead")
        parts.append("\\midrule")
        parts.append(f"\\multicolumn{{{n_cols}}}{{r}}{{\\emph{{Continued on next page.}}}} \\\\")
        parts.append("\\endfoot")
        parts.append("\\bottomrule")
        parts.append("\\endlastfoot")
        for i, (model_key, layer, header) in enumerate(segments):
            parts.append(f"\\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{{header}}}}} \\\\*")
            parts.append("\\midrule")
            parts.extend(build_segment_rows(model_key, layer, evals, columns, n_cols, long=True))
            if i != len(segments) - 1:
                parts.append("\\midrule")
        parts.append("\\end{longtable}")
        parts.append("}")
    else:
        parts.append("\\begin{table*}[!t]")
        parts.append("\\centering")
        parts.append("\\footnotesize")
        parts.append("\\setlength{\\tabcolsep}{4pt}")
        parts.append("\\renewcommand{\\arraystretch}{1.0}")
        parts.append(f"\\begin{{tabular}}{{{colspec}}}")
        parts.append("\\toprule")
        parts.extend(headers)
        parts.append("\\midrule")
        for i, (model_key, layer, header) in enumerate(segments):
            parts.append(f"\\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{{header}}}}} \\\\")
            parts.append("\\midrule")
            parts.extend(build_segment_rows(model_key, layer, evals, columns, n_cols))
            if i != len(segments) - 1:
                parts.append("\\midrule")
        parts.append("\\bottomrule")
        parts.append("\\end{tabular}")
        parts.append(f"\\caption{{{caption}}}")
        parts.append(f"\\label{{{label}}}")
        parts.append("\\end{table*}")

    return "\n".join(parts) + "\n"


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for spec in _build_tables():
        text = build_table(
            segments=spec["segments"],
            label=spec["label"],
            evals=spec["evals"],
            columns=spec["columns"],
            caption=spec["caption"],
            long=spec.get("long", False),
        )
        out_path = TABLES_DIR / spec["fname"]
        out_path.write_text(text)
        n_data_rows = sum(1 for L in text.splitlines() if L.endswith(" \\\\") and "multicolumn" not in L)
        print(f"wrote {out_path}  ({n_data_rows} data rows incl. headers)")


if __name__ == "__main__":
    main()
