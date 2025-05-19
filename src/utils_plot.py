import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import ttest_rel
from scipy.spatial.distance import cdist
import ot


def plot_auc_demographic_comparison(
    trad_auc,
    reg_auc,
    dro_auc,
    white_proportion,
    black_proportion,
    asian_proportion,
    hispano_proportion,
    other_proportion,
    filtered_list,
    site_level_comparison=True,
):
    def calculate_min_mean(column, n):
        return column.nsmallest(n).mean()

    n_values = list(range(1, len(filtered_list) + 1))

    # Section 1: AUC dataframe

    comparison_df = pd.DataFrame(
        [trad_auc, reg_auc, dro_auc], index=["Traditional", "Regularization", "DRO"]
    )
    hist_reg = reg_auc
    hist_dro = dro_auc

    mean_dro_plot = [calculate_min_mean(comparison_df.loc["DRO"], n) for n in n_values]
    mean_reg_plot = [
        calculate_min_mean(comparison_df.loc["Regularization"], n) for n in n_values
    ]
    mean_tra_plot = [
        calculate_min_mean(comparison_df.loc["Traditional"], n) for n in n_values
    ]

    # Section 2: Minimum-N mean plot
    plt.figure(figsize=(8, 6))
    plt.plot(n_values, mean_dro_plot, label="DRO", marker="o")
    plt.plot(n_values, mean_reg_plot, label="REG", marker="s")
    plt.plot(n_values, mean_tra_plot, label="TRA", marker="x")
    plt.xlabel("Worst N Sites", fontsize=12)
    plt.ylabel("AUC", fontsize=12)
    plt.title("Minimum-N Mean AUC Comparison", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Section 3: AUC & demographic proportions
    x_indices = np.arange(len(filtered_list))
    bar_width = 0.2
    color = ["lightblue", "purple", "teal", "orange", "lightgray"]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(
        x_indices - bar_width * 1.5,
        white_proportion,
        width=bar_width,
        color=color[0],
        label="White",
    )
    ax1.bar(
        x_indices - bar_width * 0.5,
        black_proportion,
        width=bar_width,
        color=color[1],
        label="Black",
    )
    ax1.bar(
        x_indices + bar_width * 0.5,
        asian_proportion,
        width=bar_width,
        color=color[3],
        label="Asian",
    )
    ax1.bar(
        x_indices + bar_width * 1.5,
        hispano_proportion,
        width=bar_width,
        color=color[2],
        label="Hispano",
    )
    ax1.set_ylabel("Demographic Proportion (%)", fontsize=12)
    ax1.set_ylim(0, 4)
    ax1.set_xlabel("Site Index", fontsize=12)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twinx()
    ax2.plot(x_indices, hist_dro, color="blue", label="DRO (AUC)", marker="o")
    ax2.plot(x_indices, hist_reg, color="orange", label="REG (AUC)", marker="s")
    ax2.plot(x_indices, trad_auc, color="black", label="TRA (AUC)", marker="^")
    ax2.set_ylabel("AUC Scores", fontsize=12)
    ax2.set_ylim(0.5, 0.9)
    ax2.legend(loc="upper right", fontsize=10)

    plt.xticks(ticks=x_indices, labels=filtered_list)
    plt.title("AUC and Demographic Proportions Across Sites", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Section 4: DRO - REG scatter difference
    chaju = [d - r for d, r in zip(hist_dro, hist_reg)]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(
        x_indices - bar_width * 2,
        white_proportion,
        width=bar_width,
        color=color[0],
        label="White",
    )
    ax1.bar(
        x_indices - bar_width * 1,
        black_proportion,
        width=bar_width,
        color=color[1],
        label="Black",
    )
    ax1.bar(
        x_indices, hispano_proportion, width=bar_width, color=color[2], label="Hispano"
    )
    ax1.bar(
        x_indices + bar_width * 1,
        asian_proportion,
        width=bar_width,
        color=color[3],
        label="Asian",
    )
    ax1.bar(
        x_indices + bar_width * 2,
        other_proportion,
        width=bar_width,
        color=color[4],
        label="Other",
    )
    ax1.set_ylabel("Demographic Proportion", fontsize=12)
    ax1.set_ylim(0, 2)
    ax1.set_xlabel("Site Index", fontsize=12)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twinx()
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    for i in range(len(chaju)):
        color_dot = "blue" if chaju[i] >= 0 else "red"
        ax2.scatter(
            x_indices[i],
            chaju[i],
            color=color_dot,
            marker="o",
            label="Diff" if i == 0 else "",
        )
    ax2.set_ylabel("AUC Diff (DRO - REG)", fontsize=12)
    ax2.set_ylim(-0.12, 0.12)
    ax2.legend(loc="upper right", fontsize=10)

    plt.title("AUC Difference and Demographic Proportions Across Sites", fontsize=16)
    plt.xticks(ticks=x_indices, labels=filtered_list)
    plt.tight_layout()
    plt.show()

    # Section 5: Paired t-test
    df = comparison_df
    results = []
    rows = df.index.tolist()
    for i in range(1, len(rows)):
        for j in range(i + 1, len(rows)):
            t_stat, p_value = ttest_rel(df.loc[rows[i]], df.loc[rows[j]])
            results.append((f"{rows[i]} vs {rows[j]}", t_stat, p_value))

    results_df = pd.DataFrame(results, columns=["Comparison", "t_stat", "p_value"])
    print("\nMean's Comparison Results:")
    print(results_df)

    return results_df


def plot_wasserstein_and_urban_vs_auc(
    data_train,
    y_train,
    hatdata,
    data_mapping,
    filtered_list,
    top_vars,
    trad_auc,
    reg_auc,
    dro_auc,
    site_title="AUC and Urban Distribution with Wasserstein",
    kappa=1e-5,
):
    # --- Section 1: Compute site-wise urban proportions ---
    urban_props_by_site = []
    for site in filtered_list:
        site_data = data_mapping[site]
        counts = site_data["urban"].value_counts(normalize=True).sort_index()
        props = [counts.get(k, 0.0) for k in range(4)]  # urban 0~3
        urban_props_by_site.append(props)

    # --- Section 2: Compute site-wise Wasserstein distance ---
    was_value = []
    for i in filtered_list:
        was = get_wasserstein(
            data_train[top_vars],
            data_mapping[i][top_vars],
            y_train,
            data_mapping[i]["y_{t+1}"],
            kappa,
        )
        was_value.append(was)

    # --- Compute train vs hat Wasserstein distance ---
    was_train_hat = get_wasserstein(
        data_train[top_vars], hatdata[top_vars], y_train, hatdata["y_{t+1}"], kappa
    )

    # --- Section 3: Plot urban, Wasserstein, AUC ---
    x_indices = np.arange(len(filtered_list))
    bar_width = 0.15
    colors = ["lightgray", "skyblue", "orange", "teal", "red"]
    hist_dro = dro_auc
    hist_reg = reg_auc
    hist_tra = trad_auc

    fig, ax1 = plt.subplots(figsize=(12, 6))

    for i in range(4):  # urban 0~3
        urban_vals = [p[i] * 10 for p in urban_props_by_site]
        ax1.bar(
            x_indices + (i - 2) * bar_width,
            urban_vals,
            width=bar_width,
            color=colors[i],
            label=f"Urban = {i}",
        )

    # Wasserstein bar
    ax1.bar(
        x_indices + 2 * bar_width,
        was_value,
        width=bar_width,
        color=colors[4],
        label="Wasserstein",
    )

    # Horizontal line for train vs hatdata
    ax1.axhline(
        was_train_hat,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Train vs Hat W",
    )

    ax1.set_ylabel("Urban Proportion (%) / Wasserstein", fontsize=12)
    ax1.set_xlabel("Site Index", fontsize=12)
    ax1.set_ylim(0, max(max(was_value) * 1.2, 200))
    ax1.set_title(site_title, fontsize=14)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(title="Legend", loc="upper left", fontsize=10)

    # Annotate site IDs

    # Plot AUC lines
    ax2 = ax1.twinx()
    ax2.plot(x_indices, hist_dro, color="blue", label="DRO (AUC)", marker="o")
    ax2.plot(x_indices, hist_reg, color="orange", label="REG (AUC)", marker="s")
    ax2.plot(x_indices, hist_tra, color="black", label="TRA (AUC)", marker="^")
    ax2.set_ylabel("AUC Scores", fontsize=12)
    ax2.set_ylim(0.6, 1.0)
    ax2.legend(loc="upper right", fontsize=10)

    plt.xticks(ticks=x_indices)
    plt.tight_layout()
    plt.show()

    # --- Section 4: Wasserstein vs AUC Difference (DRO - REG) ---
    chaju = [d - r for d, r in zip(hist_dro, hist_reg)]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(
        x_indices,
        was_value,
        width=bar_width * 1.5,
        color="skyblue",
        label="Wasserstein",
    )
    ax1.axhline(
        was_train_hat,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Train vs Hat W",
    )
    ax1.set_ylabel("Wasserstein Distance", fontsize=12)
    ax1.set_xlabel("Site Index", fontsize=12)
    ax1.set_ylim(0, max(was_value) * 1.2)
    ax1.set_title("Wasserstein vs AUC Difference (DRO - REG)", fontsize=14)
    ax1.legend(loc="upper left", fontsize=10)

    ax2 = ax1.twinx()
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    for i in range(len(chaju)):
        color_dot = "blue" if chaju[i] >= 0 else "red"
        ax2.scatter(
            x_indices[i],
            chaju[i],
            color=color_dot,
            marker="o",
            label="DRO - REG" if i == 0 else "",
        )

    ax2.set_ylabel("AUC Difference (DRO - REG)", fontsize=12)
    ax2.set_ylim(-0.1, 0.1)
    ax2.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.show()

    # --- Section 5: Paired t-test ---
    comparison_df = pd.DataFrame(
        [trad_auc, reg_auc, dro_auc], index=["Traditional", "Regularization", "DRO"]
    )
    results = []
    rows = comparison_df.index.tolist()
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            t_stat, p_value = ttest_rel(
                comparison_df.loc[rows[i]], comparison_df.loc[rows[j]]
            )
            results.append((f"{rows[i]} vs {rows[j]}", t_stat, p_value))

    results_df = pd.DataFrame(results, columns=["Comparison", "t_stat", "p_value"])
    print("\nT-test Comparison Results:")
    print(results_df)

    return was_value, chaju, was_train_hat, results_df


def get_wasserstein(x, x_hat, y, y_hat, kappa):
    x = x.to_numpy()
    x_hat = x_hat.to_numpy()
    p = np.ones(len(x)) / len(x)
    q = np.ones(len(x_hat)) / len(x_hat)
    M = cdist(x, x_hat, metric="euclidean")
    wasserstein_distance = ot.sinkhorn2(p, q, M, reg=0.1, numItermax=500)
    y = np.array(y).reshape(-1, 1)
    y_hat = np.array(y_hat).reshape(-1, 1)
    p_y = np.ones(len(y)) / len(y)
    q_y = np.ones(len(y_hat)) / len(y_hat)
    M_y = (y - y_hat.T) ** 2
    wasserstein_distance += ot.emd2(p_y, q_y, M_y) * kappa / 2
    return wasserstein_distance


def plot_site_auc_with_population_bars(
    trad_auc,
    reg_auc,
    dro_auc,
    people_num,
    site_labels,
):
    x_indices = np.arange(len(site_labels))
    bar_width = 0.35

    # Section 1: AUC line plot (left y-axis)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(x_indices, trad_auc, label="Traditional", marker="^", color="black")
    ax1.plot(x_indices, reg_auc, label="Regularization", marker="s", color="orange")
    ax1.plot(x_indices, dro_auc, label="DRO", marker="o", color="blue")
    ax1.set_xlabel("Site", fontsize=12)
    ax1.set_ylabel("AUC Score", fontsize=12)
    ax1.set_ylim(0.3, 0.9)
    ax1.set_xticks(x_indices, labels=site_labels)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True)

    # Section 2: People count bar chart (right y-axis)
    ax2 = ax1.twinx()
    ax2.bar(
        x_indices,
        people_num,
        width=bar_width,
        color="darkblue",
        alpha=0.4,
        label="People Count",
    )
    ax2.set_ylabel("Number of People", fontsize=12, color="darkblue")
    ax2.tick_params(axis="y", labelcolor="darkblue")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.set_ylim(0, 2.3 * max(people_num))
    plt.title("Site-wise AUC Comparison with People Count", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Section 3: Paired t-tests
    comparison_df = pd.DataFrame(
        [trad_auc, reg_auc, dro_auc], index=["Traditional", "Regularization", "DRO"]
    )
    results = []
    rows = comparison_df.index.tolist()
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            t_stat, p_value = ttest_rel(
                comparison_df.loc[rows[i]], comparison_df.loc[rows[j]]
            )
            results.append((f"{rows[i]} vs {rows[j]}", t_stat, p_value))

    results_df = pd.DataFrame(results, columns=["Comparison", "t_stat", "p_value"])
    print("\nPaired t-test Results:")
    print(results_df)

    return results_df
