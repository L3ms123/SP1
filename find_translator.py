import pandas as pd
from utils import metrics
from utils.data import Task, data_df, schedules_df, clients_df, transl_cost_pairs_df
from utils import knn

def get_top_translators_for_task(task_input_dict, 
                                  data_df,  
                                  transl_cost_pairs_df, 
                                  clients_df, 
                                  schedules_df,
                                  top_k=4):
    ## It is important that the task_input_dict has ASSIGNED = timedelta(now)
    """
    Given task attributes from the front-end, returns top-k recommended translators.

    Parameters:
        task_input_dict (dict): Dictionary containing task attributes ('TASK_ID', 'TASK_TYPE', 'SOURCE_LANG', 'TARGET_LANG', 
        'MANUFACTURER', 'MANUFACTURER_INDUSTRY', 'MANUFACTURER_SUBINDUSTRY', 'SELLING_HOURLY_PRICE','MIN_QUALITY', 'WILDCARD', 'ASSIGNED').
        data_df (pd.DataFrame): Full historical dataset.
        train_df (pd.DataFrame): Training dataset of past tasks.
        transl_cost_pairs_df (pd.DataFrame): Translator data with costs and metrics.
        clients_df (pd.DataFrame): Client-specific preferences (e.g. wildcard, quality threshold).
        schedules_df (pd.DataFrame): Availability and scheduling info.
        top_k (int): Number of top translators to return (default=4).

    Returns:
        list[dict]: A list of dictionaries, each containing features of a top recommended translator.
    """
    # Compute translator metrics
    translators_df = metrics.compute_delay_percentage(data_df, transl_cost_pairs_df)
    translators_df = metrics.compute_number_tasks(data_df, translators_df)

    # Normalize task fields
    task_row = pd.Series(task_input_dict)

    # Get client preferences
    client_match = clients_df[clients_df['CLIENT_NAME'].str.strip() == task_row['MANUFACTURER'].strip()]
    if not client_match.empty:
        task_row['WILDCARD'] = client_match.iloc[0]['WILDCARD']
        task_row['HOURLY_RATE'] = client_match.iloc[0]['SELLING_HOURLY_PRICE']
        task_row['QUALITY_EVALUATION'] = client_match.iloc[0]['MIN_QUALITY']
    else:
        task_row['WILDCARD'] = 'Quality'
        task_row['HOURLY_RATE'] = task_row['SELLING_HOURLY_PRICE']
        task_row['QUALITY_EVALUATION'] = task_row['MIN_QUALITY']

    # Create Task object
    task = Task(**task_row.to_dict())

    # Filter candidate translators
    df_filtered = metrics.filter_language_price_quality_availability(data_df, schedules_df, translators_df, task)
    if df_filtered.empty:
        return []

    # Compute experience features
    df_filtered = metrics.compute_experience(df_filtered, task.TASK_TYPE, task.SOURCE_LANG, task.TARGET_LANG, task.MANUFACTURER_INDUSTRY, task.MANUFACTURER_SUBINDUSTRY)
    df_filtered= metrics.compute_experience_for_client(df_filtered, task.MANUFACTURER)

    # Prepare past task features and learn weights
    past_data = metrics.prepare_past_tasks_features(data_df, task)
    weights = metrics.train_feature_weight_model(past_data)

    # Run KNN to get similar translators
    distances, indexes = knn(df_filtered, task, data_df, weights, metric='euclidean', need_wildcard=False)

    # Select top-k best translators
    top_translators_df = metrics.get_best_translators(df_filtered, indexes, distances).head(top_k)

    return top_translators_df.to_dict(orient='records')


## MAKE FUNCTION TO UPDATE THE UNAVAILABLE TRANSLATORS


