from . import metrics

import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def prepare_past_tasks_features(data_df, task):
    """
    Filters past tasks similar to the given task and computes relevant features for scoring translators.

    Parameters:
        data_df (pd.DataFrame): Full dataset of past translation tasks.
        task (object): The task object containing attributes like TASK_TYPE, SOURCE_LANG, etc.

    Returns:
        pd.DataFrame: A filtered and enriched DataFrame with computed features and filled missing values.
    """
    df = data_df[
        (data_df['TASK_TYPE'] == task.TASK_TYPE) &
        (data_df['SOURCE_LANG'] == task.SOURCE_LANG) & 
        (data_df['TARGET_LANG'] == task.TARGET_LANG)
    ].copy()  
    df = df[df['TASK_ID'] != task.TASK_ID].copy()  # Exclude the current task

    df = metrics.compute_quality_by_languages(data_df, df, source_lang=task.SOURCE_LANG, target_lang=task.TARGET_LANG)
    df = metrics.compute_quality_by_task_type(data_df, df, task_type=task.TASK_TYPE)
    df = metrics.compute_experience(df, data_df, task.TASK_TYPE, task.SOURCE_LANG, task.TARGET_LANG, task.MANUFACTURER_INDUSTRY, task.MANUFACTURER_SUBINDUSTRY)
    df = metrics.compute_experience_for_client(df, data_df, task.MANUFACTURER)
    df = metrics.compute_delay_percentage(data_df, df)

    # Rellenar NaNs generados en las nuevas columnas con valores razonables (ejemplo):
    fill_defaults = {
        'AVG_QUALITY_BY_LG': 5,
        'AVG_QUALITY_BY_TASK': 5,
        'EXPERIENCE_SCORE': 0,
        'EXPERIENCE_CLIENT': 0,
        'AVG_DELAY_PERCENTAGE': 0,
        'HOURLY_RATE': df['HOURLY_RATE'].median() if 'HOURLY_RATE' in df else 0,
    }
    for col, val in fill_defaults.items():
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(val)

    return df


def train_feature_weight_model(df_past_tasks, target='QUALITY_EVALUATION'):
    """
    Trains a linear regression model to learn the importance (weights) of different features 
    in predicting a target such as quality.

    Parameters:
        df_past_tasks (pd.DataFrame): DataFrame with past task features and quality evaluations.
        target (str): The name of the target column (default is 'QUALITY_EVALUATION').

    Returns:
        np.ndarray: A normalized weight vector (1D array) indicating the importance of each feature.
    """
    categorical_cols = ['MANUFACTURER', 'MANUFACTURER_INDUSTRY', 'MANUFACTURER_SUBINDUSTRY']

    numeric_cols = ['HOURLY_RATE', 'AVG_QUALITY_BY_LG', 'AVG_QUALITY_BY_TASK',
                'AVG_DELAY_PERCENTAGE', 'EXPERIENCE_SCORE', 'EXPERIENCE_CLIENT']

    df_past_tasks = df_past_tasks.copy()

    # Rellenar NaNs en features numéricas
    fill_defaults = {
        'AVG_QUALITY_BY_LG': 5,
        'AVG_QUALITY_BY_TASK': 5,
        'EXPERIENCE_SCORE': 0,
        'EXPERIENCE_CLIENT': 0,
        'AVG_DELAY_PERCENTAGE': 0,
        'HOURLY_RATE': df_past_tasks['HOURLY_RATE'].median() if 'HOURLY_RATE' in df_past_tasks else 0,
    }
    for col, val in fill_defaults.items():
        if col in df_past_tasks.columns:
            df_past_tasks.loc[:, col] = df_past_tasks[col].fillna(val)

    X = df_past_tasks[categorical_cols + numeric_cols]
    y = df_past_tasks[target]

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ], remainder='passthrough')

    model = Pipeline([
        ('pre', preprocessor),
        ('reg', LinearRegression())
    ])

    model.fit(X, y)

    coef = model.named_steps['reg'].coef_
    feature_names = model.named_steps['pre'].get_feature_names_out()

    feature_weights = {
        name: abs(weight)
        for name, weight in zip(feature_names, coef)
        if name in numeric_cols
    }

    weights_vector = np.array([feature_weights.get(f, 0.0) for f in numeric_cols])
    s = weights_vector.sum()
    if s == 0:
        weights_vector = np.ones_like(weights_vector) / len(weights_vector)
    else:
        weights_vector = weights_vector / s

    return weights_vector


def get_dynamic_ideal(df_past_tasks, features):
    """
    Computes the ideal feature vector for a given task using the top 10 highest quality past translations 
    for the same task type and language pair.

    Parameters:
        df_past_tasks (pd.DataFrame): DataFrame with all past tasks.
        features (list): List of feature names to include in the ideal vector.

    Returns:
        np.ndarray: Mean feature vector representing an "ideal" translator for the given task.
    """
    top_translators = df_past_tasks.sort_values(by='QUALITY_EVALUATION', ascending=False).head(10)

    ideal_vector = top_translators[features].mean().values
    return ideal_vector


def knn(df_filtered, data_df, task, metric='euclidean', need_wildcard=False):
    """
    Finds the nearest translators to the dynamically computed ideal translator using weighted KNN.

    Parameters:
        df_filtered (pd.DataFrame): Candidate translators already filtered by basic criteria.
        task (object): The new task for which we want to find the best translator.
        df_past_tasks (pd.DataFrame): All past tasks used for computing the dynamic ideal.
        weight_vector (np.ndarray): Feature importance weights learned from historical data.
        metric (str): Distance metric to use in KNN (default is 'euclidean').
        need_wildcard (bool): If True, skip wildcard weighting adjustment.

    Returns:
        tuple: (distances, indexes) from KNN representing closeness to the ideal translator.
    """
    features = ['HOURLY_RATE', 'AVG_QUALITY_BY_LG', 'AVG_QUALITY_BY_TASK',
                'AVG_DELAY_PERCENTAGE', 'EXPERIENCE_SCORE', 'EXPERIENCE_CLIENT']

    # 1. Pesos dinámicos aprendidos desde el histórico
    past_data = prepare_past_tasks_features(data_df, task)
    #weights = np.array([1, 1.5, 1.5, 0.25, 1, 0.5])  # Default weights for the features
    weights = train_feature_weight_model(past_data)

    # 3. Vector ideal dinámico
    ideal_values = get_dynamic_ideal(past_data, features)

    # 4. Preparar datos para KNN
    X = df_filtered[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_weighted = X_scaled * weights

    knn = NearestNeighbors(metric=metric)
    knn.fit(X_weighted)

    task_df = pd.DataFrame([ideal_values], columns=features)
    task_scaled = scaler.transform(task_df)
    task_weighted = task_scaled * weights

    distances, indexes = knn.kneighbors(task_weighted, n_neighbors=len(df_filtered))

    return distances, indexes

def get_best_translators(df_filtered, indexes, distances):
    """
    Get the best translators based on the KNN results.
    
    Args:
        df_filtered (pd.DataFrame): 
            Contains the filtered translators' attributes (name, language, price, quality, speed).
        indexes (np.ndarray): 
            Indices of the nearest neighbors in the df_filtered.
        distances (np.ndarray): 
            Distances of the nearest neighbors.
            
    Returns:
        df_filtered (pd.DataFrame): 
            Contains the filtered translators' attributes (name, language, price, quality, speed AND similarity_score).
    """
    
    selected_translators = df_filtered.iloc[indexes[0]].copy()
    
    # Add the similarity score
    selected_translators['Similarity Score'] = distances[0].round(2)  # Round to 2 decimal places

    # Sort by similarity score (ascending: closest match first)
    selected_translators = selected_translators.sort_values(by='Similarity Score', ascending=False) 

    return selected_translators