#!/usr/bin/env python3
"""Plot predicted build wall-clock as kernels migrate to NVRTC, per arch."""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt


# sm_89 progression — from BUILD_TIME_ISSUE.md's "Predicted wall-clock by phase" table.
# Tuples: (step label, wall after step, ceiling TU, NVRTC-amenable step?)
SM89_STEPS = [
    ("0: current",                           107, "ln_fwd_cuda_kernel.cu",     True),
    ("1: + norm NVRTC",                      105, "cast_transpose_fusion.cu",  True),
    ("2: + cast_transpose_fusion complete",   94, "gelu.cu",                   True),
    ("3: + activation NVRTC",                 93, "fused_attn_fp8.cu",         True),
    ("4: + fused_attn cuDNN trim",            80, "fused_topk.cu",            False),
    ("5: + NVFP4 + cutlass + fused_router",   72, "normalization/common.cpp",  True),
    ("6: + cuDNN/registry header trim",       37, "cudnn_utils.cpp",          False),
]


# sm_100a — built from ninja-log data + the amenability classification in BUILD_TIME_ISSUE.md.
# (label, elapsed_s, NVRTC-amenable?). Top-25 from /home/cgomes/te-bench-100a/build/cmake/.ninja_log.
SM100A_TUS = [
    ("activation/gelu.cu",                                   323.3, True),
    ("activation/relu.cu",                                   280.9, True),
    ("activation/swiglu.cu",                                 197.7, True),
    ("normalization/layernorm/ln_fwd_cuda_kernel.cu",        113.6, True),
    ("transpose/cast_transpose_fusion.cu",                   112.3, True),
    ("cast/cast.cu",                                         105.7, True),
    ("fused_attn/fused_attn_fp8.cu",                         102.2, False),
    ("normalization/layernorm/ln_bwd_semi_cuda_kernel.cu",    97.3, True),
    ("fused_attn/fused_attn_f16_arbitrary_seqlen.cu",         93.7, False),
    ("normalization/rmsnorm/rmsnorm_fwd_cuda_kernel.cu",      87.0, True),
    ("normalization/rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu", 84.5, True),
    ("fused_attn/utils.cu",                                   80.7, False),
    ("gemm/cutlass_grouped_gemm.cu",                          77.5, False),
    ("transpose/quantize_transpose_vector_blockwise_fp4.cu",  71.8, True),
    ("hadamard_transform/row_cast_col_hadamard_fusion.cu",    68.8, True),
    ("hadamard_transform/graph_safe_group_row_cast.cu",       63.3, True),
    ("hadamard_transform/group_row_cast_col_hadamard.cu",     61.6, True),
    ("fused_router/fused_topk_with_score_function.cu",        51.5, True),
    ("activation/glu.cu",                                     50.1, True),
    ("normalization/common.cpp",                              50.0, False),
    ("hadamard_transform/group_hadamard_transform.cu",        41.8, True),
    ("hadamard_transform/hadamard_transform_cast_fusion.cu",  38.3, True),
    ("gemm/cublaslt_gemm.cu",                                 30.1, False),
    ("permutation/permutation.cu",                            28.4, True),
    ("transpose/multi_cast_transpose.cu",                     25.2, True),
]

NVRTC_FLOOR_S = 5.0  # Per BUILD_TIME_ISSUE.md "Realistic floor" column for amenable TUs.


def simulate_sm100a():
    """Greedy: at each step migrate the longest currently-NVRTC-amenable TU to the floor.

    Wall after step k = max elapsed over all TUs (migrated ones reduced to NVRTC_FLOOR_S).
    Non-amenable TUs are never migrated; once they bound the wall, further NVRTC work
    yields no improvement.
    """
    tus = [{"label": l, "elapsed": e, "amenable": a, "migrated": False}
           for l, e, a in SM100A_TUS]

    def wall():
        return max((NVRTC_FLOOR_S if t["migrated"] else t["elapsed"]) for t in tus)

    def ceiling_label():
        worst = max(tus, key=lambda t: NVRTC_FLOOR_S if t["migrated"] else t["elapsed"])
        return worst["label"]

    steps = [("0: current", wall(), ceiling_label(), True)]
    step_idx = 1
    while True:
        candidates = [t for t in tus if t["amenable"] and not t["migrated"]
                      and t["elapsed"] > NVRTC_FLOOR_S]
        if not candidates:
            break
        # Pick the candidate whose migration would lower the wall the most;
        # ties broken by largest elapsed.
        candidates.sort(key=lambda t: t["elapsed"], reverse=True)
        target = candidates[0]
        # Only meaningful to migrate it if it's at or above current wall.
        # Otherwise migrating it doesn't change the wall — skip ahead to non-amenable bound.
        if target["elapsed"] < wall():
            break
        target["migrated"] = True
        short = target["label"].split("/")[-1].replace(".cu", "").replace(".cpp", "")
        steps.append((f"{step_idx}: + {short} NVRTC", wall(), ceiling_label(), True))
        step_idx += 1

    # One more synthetic point showing what bounds us next (a non-NVRTC TU).
    remaining = max((t for t in tus if not t["migrated"]),
                    key=lambda t: t["elapsed"])
    if not remaining["amenable"]:
        steps.append((f"{step_idx}: (NVRTC ceiling — bounded by {remaining['label'].split('/')[-1]})",
                      wall(), remaining["label"], False))
    return steps


def plot_arch(ax, title, steps, color):
    xs = list(range(len(steps)))
    ys = [s[1] for s in steps]
    amenable_flags = [s[3] for s in steps]
    labels = [s[0] for s in steps]
    ceiling = [s[2] for s in steps]

    ax.plot(xs, ys, color=color, marker="o", linewidth=2, markersize=8, zorder=2)
    # Overlay non-NVRTC steps with a different marker (X).
    for x, y, a in zip(xs, ys, amenable_flags):
        if not a:
            ax.plot([x], [y], marker="X", color="firebrick", markersize=12, zorder=3)

    for x, y, lbl, c in zip(xs, ys, labels, ceiling):
        ax.annotate(f"{y:.0f}s", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, fontweight="bold")
        ax.annotate(f"ceiling: {c.split('/')[-1]}", (x, y),
                    textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=7, color="gray")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Wall time (s)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(ys) * 1.18)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=pathlib.Path("/home/cgomes/transformerengine"))
    args = parser.parse_args()

    sm100a_steps = simulate_sm100a()

    for arch_name, steps, color, fname in [
        ("sm_89 (from BUILD_TIME_ISSUE.md predictions)",
         SM89_STEPS, "steelblue", "nvrtc_progression_sm89.png"),
        ("sm_100a + fast-math activations (measured + simulated)",
         sm100a_steps, "darkorange", "nvrtc_progression_sm100a.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        plot_arch(ax, arch_name, steps, color)
        # Red X legend hint.
        from matplotlib.lines import Line2D
        legend = [
            Line2D([0], [0], marker="o", linestyle="-", color=color,
                   label="wall after step"),
            Line2D([0], [0], marker="X", linestyle="", color="firebrick",
                   label="not NVRTC-amenable (requires C++ work)"),
        ]
        ax.legend(handles=legend, loc="upper right", fontsize=9)
        fig.tight_layout()
        path = args.out_dir / fname
        fig.savefig(path, dpi=130)
        print(f"wrote {path}")
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
