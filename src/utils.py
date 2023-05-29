from typing import List, Union

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


__COLORS = [
    "#8A2BE2",
    "#6495ED",
    "#6A5ACD",
    "#663399",
    "#483D8B",
    "#D2691E",
    "#FF7F50",
    "#DC143C",
    "#FF4500",
    "#FF6347",
    "#FFD700",
    "#DAA520",
    "#FFA500",
    "#FFFF00",
    "#B8860B",
]


def test_train_split(dataframe: pd.DataFrame, test_size: int):
    dataframe_ = dataframe.copy(deep=True)
    if isinstance(dataframe_.index, pd.DatetimeIndex) is False:
        return
    dataframe_.sort_index(ascending=True)

    train_df = dataframe_[:-test_size].copy(deep=True)
    test_df = dataframe_[-test_size:].copy(deep=True)

    return train_df, test_df


def seperate_target(dataframe: pd.DataFrame, target_columns: str):
    dataframe_ = dataframe.copy(deep=True)
    target_columns = (
        target_columns if isinstance(target_columns, list) else [target_columns]
    )
    target = dataframe_[target_columns].copy(deep=True)

    for col in target_columns:
        dataframe_.pop(col)

    return dataframe_, target


def show_dataset(dataframe: pd.DataFrame, columns: Union[str, List] = None):
    if isinstance(dataframe.index, pd.DatetimeIndex) is False:
        return
    columns = dataframe.columns.tolist() if columns is None else columns
    columns = [columns] if isinstance(columns, str) else columns

    row_n = len(columns)
    _, axis = plt.subplots(nrows=row_n, ncols=1, figsize=(12, row_n * 3))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    for idx, col in enumerate(columns):
        axis_t = axis[idx] if row_n > 1 else axis
        colors_t = __COLORS[(idx + 6) % len(__COLORS)]  # gROUP OF 6 cOLORS
        axis_t.plot(
            dataframe.index,
            dataframe[col],
            linewidth=2,
            alpha=0.5,
            c=colors_t,
            label=col,
        )
        axis_t.scatter(dataframe.index, dataframe[col], marker="o", s=8, c=colors_t)
    plt.legend()
    plt.show()


def show_series(dataframes: List[pd.Series], labels: Union[List[str], str]):
    for df in dataframes:
        if isinstance(df.index, pd.DatetimeIndex) is False:
            return
    _, axis = plt.subplots(nrows=1, ncols=1, figsize=(12, 3))
    plt.tight_layout()
    for idx, (df, lbs) in enumerate(zip(dataframes, labels)):
        colors_t = __COLORS[idx % len(__COLORS)]
        axis.plot(df.index, df, linewidth=2, alpha=0.5, c=colors_t, label=lbs)
        axis.scatter(df.index, df, marker="o", s=8, c=colors_t)
    plt.legend()
    plt.show()


def plot_metrics(dataframe: pd.DataFrame):
    fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))

    dataframe_ = dataframe.T.copy(deep=True)

    colormap = matplotlib.colormaps["tab20"]
    pallette = colormap(range(len(dataframe_.index)))

    dataframe_["mae"].sort_values().plot(kind="bar", color=pallette, ax=axis[0][0])
    axis[0][0].set_title("MAE Metric, lower is better")

    dataframe_["rmse"].sort_values().plot(kind="bar", color=pallette, ax=axis[0][1])
    axis[0][1].set_title("RMSE Metric, lower is better")

    dataframe_["mape"].sort_values().plot(kind="bar", color=pallette, ax=axis[1][0])
    axis[1][0].set_title("MAPE Metric, lower is better")

    dataframe_["r2"].sort_values().plot(kind="bar", color=pallette, ax=axis[1][1])
    axis[1][1].set_title("R2 Metric, higher is better")

    plt.tight_layout()
    plt.show()


def plot_tf_training_history(history, title="Training History"):
    training_loss = history.history["loss"]
    validtion_loss = history.history["val_loss"]

    training_err = history.history["mae"]
    validtion_err = history.history["val_mae"]

    epoches = range(len(training_loss))

    fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    axis[0].plot(epoches, training_loss, c="#6495ED", label="training")
    axis[0].plot(epoches, validtion_loss, c="#DC143C", label="Validation")
    axis[0].title.set_text(f"{title} loss")
    axis[0].legend()

    axis[1].plot(epoches, training_err, c="#6495ED", label="training")
    axis[1].plot(epoches, validtion_err, c="#DC143C", label="Validation")
    axis[1].title.set_text(f"{title} Error")
    axis[1].legend()

    plt.show()
