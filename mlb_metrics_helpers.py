from typing import Literal
import datetime

import pybaseball as pb
import statsapi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.pipeline import make_pipeline, Pipeline


def player_id(last_name: str, first_name: str, player_num: int = 0) -> int:
    """
    Finds the player ID based on the player's last name and first name.
    Uses pybaseball's playerid_lookup function.

    Args:
        last_name (str): The last name of the player.
        first_name (str): The first name of the player.
        player_num (int, optional): The player number to specify which player to look at if there are multiple players with the same name. Defaults to 0.

    Returns:
        int: The player ID.

    Raises:
        ValueError: If the player ID lookup is empty.
    """

    try:
        return pb.playerid_lookup(last_name, first_name)["key_mlbam"][player_num]
    except KeyError:
        raise ValueError(
            "Player ID lookup failed. No player found with the given name."
        )


def player_general_metrics(
    player_id: int, timeline_type: Literal["career", "season"] = "career"
) -> dict:
    """
    Retrieves the general metrics for a player based on their ID.
    Uses MLB-StatsAPI's player_stat_data function.

    Args:
        player_id (int): The ID of the player.
        timeline_type (str): The type of metrics to retrieve. Should be either "career" or "season". Defaults to "career".

    Returns:
        dict: A dictionary containing the general metrics of the player.
    """

    return statsapi.player_stat_data(player_id, type=timeline_type)


def parse_career_timeline(player_metrics: dict) -> tuple[str, str]:
    """
    Parses the career timeline of a player and returns a tuple with the start and end dates in the format "YYYY-MM-DD".
    If player is still active, the end date will be the current date.


    Args:
        player_metrics (dict): A dictionary containing the player's metrics from MLB-StatsAPI.

    Returns:
        tuple[str, str]: A tuple with the start and end dates of the player's career.
    """

    start_dt = player_metrics["mlb_debut"]
    end_dt = player_metrics["last_played"] or datetime.date.today().strftime("%Y-%m-%d")

    return start_dt, end_dt


def player_specific_metrics(
    player_id: int,
    metric_type: Literal["pitching", "batting"],
    start_dt: str,
    end_dt: str,
) -> pd.DataFrame:
    """
    Retrieves the specific metrics for a player based on their ID, metric type, and date range.
    Only works for pitcher and batter metrics.
    Uses pybaseball's statcast_pitcher and statcast_batter functions.

    Args:
        player_id (int): The ID of the player.
        metric_type (Literal["pitching", "batting"]): The type of metric to retrieve (either "pitching" or "batting").
        start_dt (str): The start date for the metrics retrieval in the format "YYYY-MM-DD".
        end_dt (str): The end date for the metrics retrieval in the format "YYYY-MM-DD".

    Returns:
        pd.DataFrame: A DataFrame containing the specific metrics of the player.
    """

    if metric_type == "pitching":
        return pb.statcast_pitcher(
            start_dt=start_dt, end_dt=end_dt, player_id=player_id
        )
    elif metric_type == "batting":
        return pb.statcast_batter(start_dt=start_dt, end_dt=end_dt, player_id=player_id)
    else:
        raise ValueError("Invalid metric_type. Must be either 'pitching' or 'batting'.")


def plate_crossing_metrics(
    player_specific_metrics: pd.DataFrame, metric_type: Literal["pitching", "batting"]
) -> pd.DataFrame:
    """
    Retrieves the plate crossing metrics for a specific player.

    Parameters:
        player_specific_metrics (pd.DataFrame): DataFrame containing player-specific metrics from pybaseball statcast API.
        metric_type (Literal["pitching", "batting"]): The type of metric to retrieve (either "pitching" or "batting").

    Returns:
        pd.DataFrame: DataFrame containing plate crossing metrics for the player.
    """
    plate_crossing_metrics = player_specific_metrics[
        ~player_specific_metrics["plate_x"].isna()
        & ~player_specific_metrics["plate_z"].isna()
    ]

    if metric_type == "pitching":
        return plate_crossing_metrics[["pitch_name", "plate_x", "plate_z"]]
    else:
        return plate_crossing_metrics[["description", "plate_x", "plate_z"]]


def plate_crossing_scatter(
    specific_stats: pd.DataFrame, metric_type: Literal["pitching", "batting"]
):
    """
    Generates a scatter plot for plate crossing metrics for either pitching or batting.

    Parameters:
        specific_stats (pd.DataFrame): DataFrame containing player-specific stats.
        metric_type (Literal["pitching", "batting"]): The type of metrics to plot (either "pitching" or "batting").

    Returns:
        matplotlib.figure.Figure: A matplotlib figure containing the generated scatter plot.
    """

    # Retrieve the relevant plate crossing metrics
    if metric_type == "pitching":
        data = plate_crossing_metrics(specific_stats, "pitching")
        hue = "pitch_name"
        title = "Scatter Plot of Types of Pitches Crossing Plate"
    else:
        data = plate_crossing_metrics(specific_stats, "batting")
        hue = "description"
        title = "Scatter Plot of Batting Events Crossing Plate"

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create scatter plot
    sns.scatterplot(
        data=data, x="plate_x", y="plate_z", hue=hue, s=100, alpha=0.5, ax=ax
    )

    # Customize the axes
    ax.axvline(x=-0.71, color="gray", linestyle="--")  # left bound of strike zone
    ax.axvline(x=0.71, color="gray", linestyle="--")  # right bound of strike zone

    # Add labels and legend
    ax.set_title(title)
    ax.set_xlabel("Horizontal Position (feet)")
    ax.set_ylabel("Vertical Position (feet)")
    ax.legend(title=hue)

    return fig


def plot_prediction_probas(prediction_probas, class_labels):
    """
    Creates a bar plot of prediction probabilities using Seaborn and returns the Matplotlib figure.

    Parameters:
        prediction_probas (list): List of prediction probabilities.
        class_labels (list): List of class labels corresponding to the probabilities.

    Returns:
        matplotlib.figure.Figure: A matplotlib figure containing the generated bar plot.
    """
    # Create a color palette with the same number of colors as there are classes
    palette = sns.color_palette("tab10")

    # Create a Matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a Seaborn barplot on the created axes
    sns.barplot(x=class_labels, y=prediction_probas, palette=palette, ax=ax)

    # Customize the plot
    ax.set_xlabel("Class")
    ax.set_ylabel("Probability")
    ax.set_title("Model Prediction Probabilities")
    ax.set_xticklabels(class_labels)

    return fig


def pitcher_model_data(player_specific_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Processes player-specific metrics for model training.

    Parameters:
        player_specific_metrics (pd.DataFrame): DataFrame containing player-specific metrics.

    Returns:
        pd.DataFrame: DataFrame prepared for pitcher model training (predict zone for throw).
    """
    # columns to use as features and classes
    col_to_keep = [
        "pitch_type",
        "release_speed",
        "release_pos_x",
        "release_pos_y",
        "release_spin_rate",
        "spin_axis",
        "p_throws",
        "vx0",
        "vy0",
        "vz0",
        "ax",
        "ay",
        "az",
        "zone",
    ]

    # Select only the relevant target columns
    pitcher_model_data = player_specific_metrics[col_to_keep]

    # Drop rows with missing values
    pitcher_model_data.dropna(inplace=True)

    return pitcher_model_data


def batter_model_data(player_specific_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Processes player-specific metrics for model training.

    Parameters:
        player_specific_metrics (pd.DataFrame): DataFrame containing player-specific metrics.

    Returns:
        pd.DataFrame: DataFrame prepared for batter model training (predict result of swing).
    """
    # columns to use as features and classes
    col_to_keep = [
        "pitch_type",
        "release_speed",
        "release_pos_x",
        "release_pos_y",
        "release_spin_rate",
        "spin_axis",
        "p_throws",
        "plate_x",
        "plate_z",
        "vx0",
        "vy0",
        "vz0",
        "ax",
        "ay",
        "az",
        "description",
    ]

    # Select only the relevant target columns
    batter_model_data = player_specific_metrics[col_to_keep]
    batter_model_data = batter_model_data[
        (batter_model_data["description"] == "hit_into_play")
        | (batter_model_data["description"] == "swinging_strike")
    ]

    # Drop rows with missing values
    batter_model_data.dropna(inplace=True)

    return batter_model_data


def model_datasets(
    model_data: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the model data into training and testing datasets.

    Parameters:
        model_data (pd.DataFrame): DataFrame containing model data.
        target (str): The target column to predict.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the training and testing datasets.
            - X_train (pd.DataFrame): Training feature dataset.
            - X_test (pd.DataFrame): Testing feature dataset.
            - y_train (pd.Series): Training class dataset.
            - y_test (pd.Series): Testing class dataset.
    """

    # Split into feature and class datasets
    X = model_data.drop(columns=[target])
    y = model_data[target]

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.1, stratify=y
    )

    return (X_train, X_test, y_train, y_test)


def trained_model(
    X_train: pd.DataFrame, y_train: pd.Series, sklearn_model: Pipeline
) -> Pipeline:
    """
    Trains a baseball model using the provided training data.

    Parameters:
        X_train (pd.DataFrame): Training feature dataset.
        y_train (pd.Series): Training class dataset.
        sklearn_model (object): Type of model to be trained.

    Returns:
        Pipeline : sklearn pipeline for processing and predicting baseball data point.
    """
    # Select numerical and categorical columns
    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)

    numerical_columns = numerical_columns_selector(X_train)
    categorical_columns = categorical_columns_selector(X_train)

    # Preprocess categorical and numerical columns
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = StandardScaler()

    preprocessor = ColumnTransformer(
        [
            ("one-hot-encoder", categorical_preprocessor, categorical_columns),
            ("standard_scaler", numerical_preprocessor, numerical_columns),
        ]
    )

    # Create and train the model
    model = make_pipeline(preprocessor, sklearn_model)
    model.fit(X_train, y_train)

    return model


def tested_model(
    model_data: pd.DataFrame,
    target: str,
    sklearn_model_type: Literal[
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "hist_gradient_boosting",
        "svc",
    ],
) -> tuple[Pipeline, float]:
    """
    Trains and evaluates a player model using the specified sklearn model type.

    Args:
        model_data (pd.DataFrame): The input data for training the model.
        target (str): The target column to predict.
        sklearn_model_type (Literal): The type of sklearn model to use. Must be one of:
            - "logistic_regression"
            - "random_forest"
            - "gradient_boosting"
            - "hist_gradient_boosting"
            - "svc"

    Returns:
        tuple[Pipeline, float]: A tuple containing the trained model pipeline and the accuracy score.
    """
    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = model_datasets(model_data, target)

    # Train the model
    if sklearn_model_type == "logistic_regression":
        model_type = LogisticRegression(random_state=0)
    elif sklearn_model_type == "random_forest":
        model_type = RandomForestClassifier(random_state=0)
    elif sklearn_model_type == "gradient_boosting":
        model_type = GradientBoostingClassifier(random_state=0)
    elif sklearn_model_type == "hist_gradient_boosting":
        model_type = HistGradientBoostingClassifier(random_state=0)
    elif sklearn_model_type == "svc":
        model_type = SVC(random_state=0, probability=True)
    else:
        raise ValueError(
            "Invalid sklearn_model_type. Please choose a valid model type."
        )

    model = trained_model(X_train, y_train, model_type)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)

    return (model, accuracy)


def model_prediction(model: Pipeline, sample_X: pd.DataFrame) -> (str, list, list):
    """
    Makes predictions using the trained model on the provided sample data.

    Parameters:
        model (Pipeline): The trained sklearn model pipeline.
        sample_X (pd.DataFrame): Sample feature dataset for prediction.

    Returns:
        Tuple containing:
        - str: Predicted class label.
        - list: Prediction probabilities for each class.
        - list: Class labels used by the model.
    """
    prediction = model.predict(sample_X)[0]
    prediction = str(prediction)

    prediction_probas = model.predict_proba(sample_X)[0].tolist()
    prediction_probas = [float(x) for x in prediction_probas]

    # Retrieve class labels
    class_labels = model.classes_.tolist()

    return prediction, prediction_probas, class_labels
