# plot_robustness_aggregated.py
#
# Usage:
#   python plot_robustness_aggregated.py robustness_results_all_models.csv plots
#
# Output:
#   plots/clean_accuracy_mean_std.png
#   plots/robustness_gaussian.png
#   plots/robustness_ch_dropout.png
#   plots/robustness_time_mask.png
#   plots/robustness_time_shift.png

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path("results/robustness_results.csv")
    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required = {"model", "member_id", "corruption", "param", "severity", "accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Ensure numeric severity (so sorting/plotting works)
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce")

    # 1) Clean accuracy: mean +/- std across members
    clean = df[df["corruption"] == "clean"].copy()
    if not clean.empty:
        g = clean.groupby("model")["accuracy"]
        stats = g.agg(["mean", "std"]).reset_index().sort_values("mean", ascending=False)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(stats["model"], stats["mean"], yerr=stats["std"])
        ax.set_title("Clean accuracy (mean ± std across members)")
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(outdir / "clean_accuracy_mean_std.png", dpi=200)
        plt.close(fig)

    # 2) Robustness curves per corruption: mean +/- std across members, per model, per severity
    corruptions = [c for c in df["corruption"].unique() if c != "clean"]
    for corr in sorted(corruptions):
        d = df[df["corruption"] == corr].copy()
        if d.empty:
            continue

        # Aggregate across members for each model & severity
        agg = (
            d.groupby(["model", "param", "severity"])["accuracy"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values(["model", "severity"])
        )

        param_name = agg["param"].iloc[0] if "param" in agg.columns and len(agg) else "severity"

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for model_name, dm in agg.groupby("model"):
            dm = dm.sort_values("severity")
            ax.plot(dm["severity"], dm["mean"], marker="o", label=model_name)
            # error bars (std across members)
            ax.fill_between(
                dm["severity"],
                (dm["mean"] - dm["std"]).fillna(dm["mean"]),
                (dm["mean"] + dm["std"]).fillna(dm["mean"]),
                alpha=0.2,
            )

        ax.set_title(f"Robustness to {corr} (mean ± std across members)")
        ax.set_xlabel(f"Severity ({param_name})")
        ax.set_ylabel("Accuracy")
        ax.legend(title="Model", loc="best")
        fig.tight_layout()

        safe_corr = corr.replace("/", "_").replace(" ", "_")
        fig.savefig(outdir / f"robustness_{safe_corr}.png", dpi=200)
        plt.close(fig)

    print(f"Saved aggregated plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
