import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def results_to_dataframe(results, variable_name="Layers", matlab_results=False):
    if matlab_results:
        results = convert_from_mat_to_dict(results)

    rows = []

    for model_name, times in results.items():
        parts = model_name.rsplit("_", 2)
        name = parts[0].replace("_", " ")
        var = int(parts[1])

        mean_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)

        row = [name, var, mean_time, min_time, max_time, std_time] + list(times)
        rows.append(row)

    time_names = [f"Time {n}" for n in range(1, times.shape[0] + 1)]
    df = pd.DataFrame(
        rows, columns=["Model", variable_name, "Mean", "Min", "Max", "Std"] + time_names
    )
    model_name_mapping = {
        "gru": "GRU",
        "vanila lstm": "LSTM",
        "peephole lstm": "LSTM with peepholes",
        "residual lstm": "Residual LSTM",
        "peepholeLSTM": "LSTM with peepholes",
        "residualLSTM": "Residual LSTM",
    }
    df["Model"] = df["Model"].replace(model_name_mapping)
    return df


def convert_from_mat_to_dict(mat_results):
    results = {}
    for result in mat_results:
        model_name = result[0][0][0]
        # rmse = result[0][1][0]
        time = result[0][2][0]
        results[model_name] = np.array(time)
    return results


def plot_training_time(
    df,
    variable_name="Layers",
    use_bars=False,
    show_yerr=False,
    bar_width=1.0,
    rotation=90,
    title="Tensorflow Mean Training Time vs. Number of Hidden Units for Different LSTM Cell Types",
    xlabel="Number of Units",
    ylabel="Mean Training Time [s]",
    save_path=None,
    figsize=(8, 4),
):
    width = bar_width / len(df["Model"].unique())

    fig = plt.figure(figsize=figsize)
    for i, (model, group) in enumerate(df.groupby("Model")):
        if use_bars:
            if show_yerr:
                plt.bar(
                    group[variable_name]
                    + (i - len(df["Model"].unique()) / 2 + 0.5) * width,
                    group["Mean"],
                    width=width,
                    label=model,
                    yerr=group["Std"],
                )
            else:
                plt.bar(
                    group[variable_name]
                    + (i - len(df["Model"].unique()) / 2 + 0.5) * width,
                    group["Mean"],
                    width=width,
                    label=model,
                )
        else:
            plt.plot(group[variable_name], group["Mean"], label=model)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if use_bars:
        plt.xticks(
            df[variable_name].unique(),
            rotation=rotation,
        )
    else:
        plt.xticks(df[variable_name].unique(), rotation=rotation)
    plt.show()
    if save_path:
        fig.savefig(save_path, format="eps", bbox_inches="tight")
