from utils import metrics
from utils.data import Task, data_df, schedules_df, clients_df, transl_cost_pairs_df
from utils import knn
import logging

logging.basicConfig(level=logging.CRITICAL)

def drop_and_save_translator_labels(df, translator_column="TRANSLATOR"):
    """
    Extracts and removes translators from test_df, then saves them as a label dict.
    
    Args:
        df (pd.DataFrame): The dataframe containing translator data.
        translator_column (str): The column name that holds the translator labels.
        
    Returns:
        pd.DataFrame: The test_df without the translators column.
        dict: Dictionary of translator labels {index: translators}
    """
    if translator_column not in df.columns:
        raise ValueError(f"Column '{translator_column}' not found in test_df.")
    
    # Extract labels
    translator_labels = df[translator_column].to_dict()
    
    # Drop the column from the DataFrame
    df = df.drop(columns=[translator_column])
    
    return df, translator_labels

def main():
    # 1. Barajar y tomar 1000 filas como validación
    validation_df_clean = data_df.sample(n=1000, random_state=42).copy()

    # 2. Guardar los labels reales y quitar la columna TRANSLATOR
    validation_df_clean, validation_translator_labels = drop_and_save_translator_labels(validation_df_clean)

    # 3. Crear el conjunto de entrenamiento excluyendo los IDs de validación
    validation_ids = validation_df_clean['TASK_ID'].tolist()
    data_df_train = data_df[~data_df['TASK_ID'].isin(validation_ids)].copy()

    # Creates a dataframe with  the additional attributes
    translators_attributes_df = metrics.compute_delay_percentage(data_df_train, transl_cost_pairs_df)
    translators_attributes_df = metrics.compute_number_tasks(data_df_train, translators_attributes_df) 

    validation_df_clean = validation_df_clean[0:100]  # For testing purposes, limit to 100 rows
    # Define the top-k value for evaluation
    k = 5
    correct_predictions = 0
    total_predictions = 0

    # Iterate over each row in the validation set
    for idx, task_row in validation_df_clean.iterrows():
        #Convert the current row to a task
        new_task = task_row.copy()
        new_task = new_task.rename({'HOURLY_RATE': 'SELLING_HOURLY_PRICE', 'QUALITY_EVALUATION': 'MIN_QUALITY'})
        
        match = clients_df[clients_df['CLIENT_NAME'].str.strip() == new_task['MANUFACTURER'].strip()]

        if not match.empty:
            new_task['WILDCARD'] = match.iloc[0]['WILDCARD']
            new_task['HOURLY_RATE'] = match.iloc[0]['SELLING_HOURLY_PRICE']
            new_task['QUALITY_EVALUATION'] = match.iloc[0]['MIN_QUALITY']
        else:
            print("WARNING: No match found in schedules_df for the given client. Setting default values.")
            new_task['WILDCARD'] = 'Quality'
            new_task['HOURLY_RATE'] = new_task['SELLING_HOURLY_PRICE']
            new_task['QUALITY_EVALUATION'] = new_task['MIN_QUALITY']
        
        new_task = Task(**new_task.to_dict())  # Convert the task to the Task object
        
        # Filter translators based on task attributes
        df_filtered = metrics.filter_language_price_quality_availability(data_df_train, schedules_df, translators_attributes_df, new_task)

        # if df_filtered.empty:
        #     df_filtered = translators_attributes_df[
        #         (translators_attributes_df['SOURCE_LANG'] == new_task.SOURCE_LANG) & 
        #         (translators_attributes_df['TARGET_LANG'] == new_task.TARGET_LANG)]
        #     df_filtered = metrics.compute_quality_by_task_type(data_df, df_filtered, task_type=new_task.TASK_TYPE)
        #     df_filtered = metrics.compute_quality_by_languages(data_df, df_filtered, source_lang=new_task.SOURCE_LANG, target_lang=new_task.TARGET_LANG)
        # Compute experience scores for the filtered translators
        df_filtered = metrics.compute_experience(df_filtered, data_df_train, task_type=new_task.TASK_TYPE, source_lang=new_task.SOURCE_LANG, target_lang=new_task.TARGET_LANG, industry=new_task.MANUFACTURER_INDUSTRY, subindustry=new_task.MANUFACTURER_SUBINDUSTRY)
        df_filtered = metrics.compute_experience_for_client(df_filtered, data_df_train, client=new_task.MANUFACTURER)

        # Get the distances and indexes from KNN
        distances, indexes = knn.knn(df_filtered, data_df_train, new_task, metric='euclidean', need_wildcard=False)
        
        # Get the best translators
        selected_translators = knn.get_best_translators(df_filtered, indexes, distances)
        
        # Retrieve the true translator label for the current task
        true_translator = validation_translator_labels[idx]  # Assuming you have the true labels in the dictionary from earlier
        
        # Check if any of the top-k translators match the true translator
        top_k_translators = selected_translators.iloc[:k]  # Top-k translators
        if true_translator in top_k_translators['TRANSLATOR'].values:
            correct_predictions += 1
        
        total_predictions += 1

    # Calculate the top-k accuracy
    top_k_accuracy = correct_predictions / total_predictions
    print(f"Top-{k} Accuracy: {top_k_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
