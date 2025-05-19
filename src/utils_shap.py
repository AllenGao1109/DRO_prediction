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
