import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from joblib import dump
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def train(alpha=1.0, plot_graphs=False, feature_scales=None):
    if feature_scales is None:
        feature_scales = {}

    csv_directory = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    csv_file_path = os.path.join(csv_directory, "face_data.csv")
    data = pd.read_csv(csv_file_path)

    def extract_features(json_str):
        try:
            json_str = json_str.replace("'", '"')
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}

    data["Parsed_Data"] = data["Data"].apply(extract_features)
    data_features = data["Parsed_Data"].apply(pd.Series)

    for feature in ["yaw", "horizontal_ratio", "pitch", "vertical_ratio"]:
        scale = feature_scales.get(feature, 1.0)
        data_features[feature] = data_features[feature] * scale

    data = pd.concat([data, data_features], axis=1).drop(
        columns=["Data", "Parsed_Data"]
    )

    X_x = data[["yaw", "horizontal_ratio"]]
    X_y = data[["pitch", "vertical_ratio"]]

    y_x = data["Click X"]
    y_y = data["Click Y"]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_x_scaled = scaler_x.fit_transform(X_x)
    X_y_scaled = scaler_y.fit_transform(X_y)

    model_x = Ridge(alpha=alpha)
    model_x.fit(X_x_scaled, y_x)

    model_y = Ridge(alpha=alpha)
    model_y.fit(X_y_scaled, y_y)

    if plot_graphs:
        predictions_x = model_x.predict(X_x_scaled)
        predictions_y = model_y.predict(X_y_scaled)
        plot_results(
            X_x_scaled,
            y_x,
            predictions_x,
            X_y_scaled,
            y_y,
            predictions_y,
            scaler_x,
            scaler_y,
        )

    model_directory = os.path.join(csv_directory, "models")
    os.makedirs(model_directory, exist_ok=True)
    dump(model_x, os.path.join(model_directory, "ridge_regression_model_x.joblib"))
    dump(model_y, os.path.join(model_directory, "ridge_regression_model_y.joblib"))
    dump(scaler_x, os.path.join(model_directory, "scaler_x.joblib"))
    dump(scaler_y, os.path.join(model_directory, "scaler_y.joblib"))


def plot_results(X_x, y_x, predictions_x, X_y, y_y, predictions_y, scaler_x, scaler_y):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    X_x_inv = scaler_x.inverse_transform(X_x)
    X_y_inv = scaler_y.inverse_transform(X_y)

    axs[0, 0].scatter(X_x_inv[:, 0], y_x, color="blue", label="Actual")
    axs[0, 0].plot(
        np.sort(X_x_inv[:, 0]),
        predictions_x[np.argsort(X_x_inv[:, 0])],
        color="red",
        label="Predicted",
        linewidth=2,
    )
    axs[0, 0].set_title("Yaw vs Click X")
    axs[0, 0].legend()

    axs[0, 1].scatter(X_x_inv[:, 1], y_x, color="blue", label="Actual")
    axs[0, 1].plot(
        np.sort(X_x_inv[:, 1]),
        predictions_x[np.argsort(X_x_inv[:, 1])],
        color="red",
        label="Predicted",
        linewidth=2,
    )
    axs[0, 1].set_title("Horizontal Ratio vs Click X")
    axs[0, 1].legend()

    axs[1, 0].scatter(X_y_inv[:, 0], y_y, color="blue", label="Actual")
    axs[1, 0].plot(
        np.sort(X_y_inv[:, 0]),
        predictions_y[np.argsort(X_y_inv[:, 0])],
        color="red",
        label="Predicted",
        linewidth=2,
    )
    axs[1, 0].set_title("Pitch vs Click Y")
    axs[1, 0].legend()

    axs[1, 1].scatter(X_y_inv[:, 1], y_y, color="blue", label="Actual")
    axs[1, 1].plot(
        np.sort(X_y_inv[:, 1]),
        predictions_y[np.argsort(X_y_inv[:, 1])],
        color="red",
        label="Predicted",
        linewidth=2,
    )
    axs[1, 1].set_title("Vertical Ratio vs Click Y")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
