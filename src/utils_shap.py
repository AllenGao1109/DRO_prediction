import shap
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
from typing import List, Union


def wrap_text(text: str, max_chars_per_line: int = 100) -> str:
    return "\n".join(
        textwrap.wrap(text, width=max_chars_per_line, break_long_words=False)
    )


def compute_and_plot_shap_kernel_autoquestion(
    model: torch.nn.Module,
    X_sample: Union[np.ndarray, List[List[float]]],
    feature_names: List[str],
    class_index: int = 3,
    top_n: int = 20,
    n_bootstrap: int = 1000,
    csv_path: str = "data/supplementary_table.csv",
) -> pd.DataFrame:
    # Load variable-to-question mapping
    df_q = pd.read_csv(csv_path)
    df_q = df_q.dropna(subset=["variable", "question"])
    df_q["variable"] = df_q["variable"].str.strip()
    df_q["question"] = df_q["question"].str.strip()
    variable_to_question = {
        var: wrap_text(q, max_chars_per_line=120)
        for var, q in zip(df_q["variable"], df_q["question"])
    }

    # Define model prediction function
    def predict_fn(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = model(X_tensor)
            return logits[:, class_index].numpy().reshape(-1, 1)

    # Compute SHAP values
    background = shap.sample(X_sample, 100)
    X_explain = X_sample[: top_n * 10]

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_explain, nsamples="auto")
    shap_values = shap_values[:, :, 0]

    # Organize into DataFrame
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    abs_shap_mean = shap_df.abs().mean()
    top_features = abs_shap_mean.sort_values(ascending=False).index[:top_n]

    # Bootstrap CI
    ci_lower, ci_upper = [], []
    for col in top_features:
        boots = [
            shap_df[col].abs().sample(frac=1, replace=True).mean()
            for _ in range(n_bootstrap)
        ]
        ci_lower.append(np.percentile(boots, 2.5))
        ci_upper.append(np.percentile(boots, 97.5))

    # Summary table
    summary_df = pd.DataFrame(
        {
            "Feature": top_features,
            "Abs SHAP": abs_shap_mean[top_features].values,
            "CI Lower": ci_lower,
            "CI Upper": ci_upper,
        }
    )

    summary_df["Question"] = (
        summary_df["Feature"].map(variable_to_question).fillna(summary_df["Feature"])
    )
    summary_df = summary_df.sort_values(by="Abs SHAP", ascending=False)

    # Plot with seaborn
    plt.figure(figsize=(12, 10))
    sns.pointplot(
        data=summary_df,
        x="Abs SHAP",
        y="Question",
        join=False,
        errorbar=None,
        color="steelblue",
    )

    for i, row in summary_df.iterrows():
        plt.errorbar(
            x=row["Abs SHAP"],
            y=i,
            xerr=[
                [row["Abs SHAP"] - row["CI Lower"]],
                [row["CI Upper"] - row["Abs SHAP"]],
            ],
            fmt="o",
            color="gray",
            capsize=4,
        )

    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.title(f"Top {top_n} Features by |SHAP| Value (Class {class_index}) with 95% CI")
    plt.tight_layout()
    plt.show()

    return summary_df


def compute_and_plot_shap_with_ci_classed(
    model: torch.nn.Module,
    X_sample: np.ndarray,
    feature_names: list,
    csv_path: str,
    class_index: int = 3,
    top_n: int = 15,
    n_bootstrap: int = 1000,
    wrap_width: int = 40,
    figsize: tuple = (12, 8),
    save_path: str = None,
):
    """
    Compute SHAP values for a given model and input data, summarize by dataset and respondent,
    calculate bootstrap-based confidence intervals, and generate a dot plot with error bars.

    Args:
        model: Trained PyTorch model.
        X_sample: Input data for SHAP (numpy array).
        feature_names: List of variable names matching X_sample columns.
        csv_path: Path to variable metadata CSV with columns [variable, dataset, respondent].
        class_index: Class to explain (for multi-class models).
        top_n: Number of top datasets to display.
        n_bootstrap: Number of bootstrap resamples for CI.
        wrap_width: Maximum characters per line for y-axis labels.
        figsize: Tuple defining figure size.
        save_path: If provided, saves the figure to this path.
    """

    # 1. Load metadata
    df_meta = pd.read_csv(csv_path)
    df_meta["variable"] = df_meta["variable"].str.strip()
    df_meta["dataset"] = df_meta["dataset"].str.strip()
    df_meta["respondent"] = df_meta["respondent"].str.strip()
    variable_to_dataset = df_meta.set_index("variable")["dataset"].to_dict()
    variable_to_respondent = df_meta.set_index("variable")["respondent"].to_dict()

    # 2. Set device (prefer MPS for Apple Silicon)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    def predict_fn(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            logits = model(X_tensor)
            return logits[:, class_index].cpu().numpy().reshape(-1, 1)

    # 3. Compute SHAP values
    background = shap.sample(X_sample, 100)
    X_explain = X_sample[:200]
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values_raw = explainer.shap_values(X_explain, nsamples="auto")
    shap_values = (
        shap_values_raw[class_index]
        if isinstance(shap_values_raw, list)
        else shap_values_raw
    )
    if shap_values.ndim == 3 and shap_values.shape[2] == 1:
        shap_values = shap_values.squeeze(-1)
    shap_df = pd.DataFrame(shap_values, columns=feature_names).abs()

    # 4. Match features to metadata (exclude unmapped ones)
    all_features = pd.DataFrame(
        {
            "Feature": feature_names,
            "Dataset": [variable_to_dataset.get(f) for f in feature_names],
            "Respondent": [variable_to_respondent.get(f) for f in feature_names],
        }
    )

    missing = all_features[all_features.isna().any(axis=1)]["Feature"].tolist()
    if missing:
        print(f"⚠️ Skipping {len(missing)} feature(s) not found in metadata: {missing}")

    meta_df = all_features.dropna(subset=["Dataset", "Respondent"])

    # 5. Bootstrap SHAP sums and confidence intervals
    results = []
    for _, row in meta_df.iterrows():
        values = shap_df[row["Feature"]].values
        boots = [
            np.sum(np.random.choice(values, size=len(values), replace=True))
            for _ in range(n_bootstrap)
        ]
        results.append(
            {
                "Dataset": row["Dataset"],
                "Respondent": row["Respondent"],
                "Feature": row["Feature"],
                "Sum SHAP": np.mean(boots),
                "CI Lower": np.percentile(boots, 2.5),
                "CI Upper": np.percentile(boots, 97.5),
            }
        )

    summary_df = pd.DataFrame(results)

    # 6. Aggregate by Dataset + Respondent
    grouped = (
        summary_df.groupby(["Dataset", "Respondent"])
        .agg({"Sum SHAP": "sum", "CI Lower": "sum", "CI Upper": "sum"})
        .reset_index()
    )

    top_datasets = (
        grouped.groupby("Dataset")["Sum SHAP"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    grouped = grouped[grouped["Dataset"].isin(top_datasets)]
    dataset_order = (
        grouped.groupby("Dataset")["Sum SHAP"].sum().sort_values(ascending=False).index
    )
    grouped["Dataset"] = pd.Categorical(
        grouped["Dataset"], categories=dataset_order, ordered=True
    )
    grouped = grouped.sort_values("Dataset")

    # 7. Plot: Dot plot with CI error bars
    plt.figure(figsize=figsize)
    color_map = {"Parent": "steelblue", "Youth": "coral"}
    for _, row in grouped.iterrows():
        plt.plot(
            [row["CI Lower"], row["CI Upper"]],
            [row["Dataset"], row["Dataset"]],
            color=color_map[row["Respondent"]],
            linewidth=3,  # ✅ thicker line
            zorder=1,
        )

    sns.scatterplot(
        data=grouped,
        x="Sum SHAP",
        y="Dataset",
        hue="Respondent",
        palette={"Parent": "steelblue", "Youth": "coral"},
        s=80,
        zorder=2,
    )

    # Wrap long y-axis labels
    ax = plt.gca()
    labels = [
        textwrap.fill(label.get_text(), wrap_width) for label in ax.get_yticklabels()
    ]
    ax.set_yticklabels(labels)

    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Absolute SHAP value sum")
    plt.ylabel("Predictor category")
    plt.legend(title="Respondent")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return grouped
