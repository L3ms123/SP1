import pandas as pd
import numpy as np
from datetime import timedelta
from . import data
import logging
import os 
import redis 
from dotenv import load_dotenv


load_dotenv()
logger = logging.getLogger(__name__)

redis_client = redis.StrictRedis(host='172.18.203.199', port=6379, db=0, decode_responses=True)
UNAVAILABLE_KEY = os.getenv("REDIS_APIKEY")

# def mark_translator_unavailable(translator):
#     redis_client.sadd(UNAVAILABLE_KEY, translator)

# def mark_translator_available(translator):
#     redis_client.srem(UNAVAILABLE_KEY, translator)

# def get_unavailable_translators():
#     return redis_client.smembers(UNAVAILABLE_KEY)

def compute_delay_percentage(data_df, transl_cost_pairs_df):
    """
    Compute the delay percentage of each translator based on task completion times.
    Negative values indicate early delivery, positive values indicate late delivery.

    Args:
        data_df (pd.DataFrame): Task data with 'TRANSLATOR', 'START', 'END', 'DELIVERED'
        transl_cost_pairs_df (pd.DataFrame): Translator costs with 'TRANSLATOR', 'COST'

    Returns:
        pd.DataFrame: Merged dataframe with 'TRANSLATOR', 'COST', 'AVG_DELAY_PERCENTAGE'
    """
    try:
        # Convert date columns to datetime safely
        date_cols = ['START', 'END', 'DELIVERED']
        for col in date_cols:
            data_df[col] = pd.to_datetime(data_df[col], errors='coerce')

        # Calculate duration and filter out bad rows
        duration = data_df['END'] - data_df['START']
        invalid_rows = duration <= pd.Timedelta(0)
        if invalid_rows.any():
            logger.info(f"[INFO] Removed {invalid_rows.sum()} rows with zero or negative durations.")
            data_df = data_df[~invalid_rows]
            duration = duration[~invalid_rows]

        # Calculate delay percentage
        delay = (data_df['DELIVERED'] - data_df['END']) / duration * 100
        delay = delay.replace([np.inf, -np.inf, np.nan], 0).clip(-100, 100)
        data_df = data_df.copy()
        data_df['DELAY_PERCENTAGE'] = delay

        # Compute average delay per translator
        avg_delay = (
            data_df.groupby('TRANSLATOR')['DELAY_PERCENTAGE']
            .mean()
            .round(2)
            .reset_index()
            .rename(columns={'DELAY_PERCENTAGE': 'AVG_DELAY_PERCENTAGE'})
        )

        # Merge with cost data
        merged = transl_cost_pairs_df.merge(avg_delay, on='TRANSLATOR', how='left')
        merged['AVG_DELAY_PERCENTAGE'] = merged['AVG_DELAY_PERCENTAGE'].fillna(0)

        return merged

    except Exception as e:
        logger.warning(f"[ERROR] Failed to compute delay percentages: {e}")
        return transl_cost_pairs_df.assign(AVG_DELAY_PERCENTAGE=0)


def compute_number_tasks(data_df, translators_attributes_df):
    """
    Computes the number of tasks for each translator.
    
    Args:
        data_df (pd.DataFrame): 
            DataFrame containing the data of the tasks.
        df_filtered (pd.DataFrame): 
            DataFrame containing the filtered translators' attributes.

    Returns:
        translators_attributes_df (pd.DataFrame) with the delay_percentage.
    """
    # Count the number of tasks each translator has done
    task_counts = data_df.groupby('TRANSLATOR').size().reset_index(name='NUM_TASKS')

    # Merge the task counts into the filtered dataframe
    translators_attributes_df = translators_attributes_df.merge(task_counts, on='TRANSLATOR', how='left')

    # Fill missing values (i.e., translators with no tasks) with 0
    translators_attributes_df['NUM_TASKS'] = translators_attributes_df['NUM_TASKS'].fillna(0).astype(int)

    return translators_attributes_df


def compute_quality_by_languages(data_df, df_filtered, source_lang, target_lang):
    """
    Computes average quality for a given language pair (source_lang → target_lang).
    If the translator has no experience with that task, falls back to:
      - their overall average quality (with a penalty), or
      - a 5 if no task has been done.
    
    Args:
        df_filtered (pd.DataFrame): Filtered translators.
        source_lang (str): Source language.
        target_lang (str): Target language.
    
    Returns:
        pd.DataFrame: Same df_filtered with new 'AVG_QUALITY_BY_LG' column.
    """
    if df_filtered.empty:
        logger.warning(f"Warning: No translators found in the filtered dataframe.")
        return df_filtered
    
    df_filtered = df_filtered.copy()

    translators = df_filtered['TRANSLATOR'].unique()
    
    # Filter tasks dataframe by the language pair and translators in df_filtered
    mask_lang_pair = (
        (data_df['SOURCE_LANG'] == source_lang) &
        (data_df['TARGET_LANG'] == target_lang) &
        (data_df['TRANSLATOR'].isin(df_filtered['TRANSLATOR']))
    )


    # Compute the average quality for each translator in the filtered dataframe
    avg_quality = (
        data_df[mask_lang_pair]
        .groupby('TRANSLATOR')['QUALITY_EVALUATION']
        .mean()
        .round(2)
    )

    # Assing the average quality to the filtered df
    df_filtered['AVG_QUALITY_BY_LG'] = df_filtered['TRANSLATOR'].map(avg_quality)

    # Fallback to penalized overall average
    mask_missing = df_filtered['AVG_QUALITY_BY_LG'].isna()

    overall_avg = (
        data_df[data_df['TRANSLATOR'].isin(translators)]
        .groupby('TRANSLATOR')['QUALITY_EVALUATION']
        .mean()
        .round(2)
        .apply(lambda x: x - 1 if pd.notnull(x) else None)  # configurable penalization, for flexibility
        #For a data-driven approach, use can standard deviation or percentile-based penalization to adapt to the distribution of quality scores
    )

    df_filtered.loc[mask_missing, 'AVG_QUALITY_BY_LG'] = df_filtered.loc[mask_missing, 'TRANSLATOR'].map(overall_avg)

    # Para los traductores que no existen en data_df → asignar calidad por defecto (ej. 5)
    df_filtered['AVG_QUALITY_BY_LG'] = df_filtered['AVG_QUALITY_BY_LG'].fillna(5)
        
    return df_filtered


def compute_quality_by_task_type(data_df, df_filtered, task_type):
    """
    Computes the average quality for each translator for a given task type.
    If the translator has no experience with that task, falls back to:
      - their overall average quality (with a penalty), or
      - a 5 if no task has been done.
    
    Args:
        df_filtered (pd.DataFrame): DataFrame with filtered translators.
        task_type (str): The specific task type to evaluate.
    
    Returns:
        pd.DataFrame: df_filtered with 'AVG_QUALITY_BY_TASK'.
    """
    if df_filtered.empty:
        logger.warning(f"Warning: No translators found in the filtered dataframe.")
        return df_filtered
    
    df_filtered = df_filtered.copy()
    
    translators = df_filtered['TRANSLATOR'].unique()

    # 1. Compute average quality for given task type
    mask_task = (
        (data_df['TASK_TYPE'] == task_type) &
        (data_df['TRANSLATOR'].isin(translators))
    )

    avg_by_task = (
        data_df[mask_task]
        .groupby('TRANSLATOR')['QUALITY_EVALUATION']
        .mean()
        .round(2)
    )

    df_filtered['AVG_QUALITY_BY_TASK'] = df_filtered['TRANSLATOR'].map(avg_by_task)

    # 2. Fallback to penalized overall average
    mask_missing = df_filtered['AVG_QUALITY_BY_TASK'].isna()

    overall_avg = (
        data_df[data_df['TRANSLATOR'].isin(translators)]
        .groupby('TRANSLATOR')['QUALITY_EVALUATION']
        .mean()
        .round(2)
        .apply(lambda x: x - 1 if pd.notnull(x) else None)  # configurable penalization, for flexibility
        #For a data-driven approach, use can standard deviation or percentile-based penalization to adapt to the distribution of quality scores
    )

    df_filtered.loc[mask_missing, 'AVG_QUALITY_BY_TASK'] = df_filtered.loc[mask_missing, 'TRANSLATOR'].map(overall_avg)

    df_filtered['AVG_QUALITY_BY_TASK'] = df_filtered['AVG_QUALITY_BY_TASK'].fillna(5)

    return df_filtered


def compute_experience(df_filtered, data_df, task_type, source_lang, target_lang, industry, subindustry):
    """
    Computes a soft experience score for each translator based on how many
    dimensions match (task_type, language pair, industry, subindustry).

    Args:
        df_filtered (pd.DataFrame): Filtered translators' dataframe.

    Returns:
        pd.DataFrame: With added column 'EXPERIENCE_SCORE'.
    """
    TASK_TYPE_BONUS = {
        'LanguageLead': 0.5,
        'ProofReading': 0.5,
        'Spotcheck': 0.5
    }

    df_filtered = df_filtered.copy()

    translators = df_filtered['TRANSLATOR'].unique()

    df = data_df[data_df['TRANSLATOR'].isin(translators)].copy()

    # Base score: match on source, target, task_type
    df['score'] = 0
    df['score'] += (df['SOURCE_LANG'] == source_lang).astype(int)
    df['score'] += (df['TARGET_LANG'] == target_lang).astype(int)
    df['score'] += (df['TASK_TYPE'] == task_type).astype(int)

    # Only add 1 point if industry or subindustry match
    industry_match = (df['MANUFACTURER_INDUSTRY'] == industry)
    subindustry_match = (df['MANUFACTURER_SUBINDUSTRY'] == subindustry)
    df['score'] += ((industry_match | subindustry_match)).astype(int)

    # Advanced task bonus
    bonus_df = df[df['TASK_TYPE'].isin(TASK_TYPE_BONUS)].copy()
    bonus_df['bonus'] = bonus_df['TASK_TYPE'].map(TASK_TYPE_BONUS)
    bonus_scores = bonus_df.groupby('TRANSLATOR')['bonus'].sum()

    # Base score
    base_scores = df.groupby('TRANSLATOR')['score'].sum()

    # Total experience = base + bonus
    total_score = base_scores.add(bonus_scores, fill_value=0)

    df_filtered['EXPERIENCE_SCORE'] = df_filtered['TRANSLATOR'].map(total_score).fillna(0)

    # Normalize between 0 and 10
    min_score = df_filtered['EXPERIENCE_SCORE'].min()
    max_score = df_filtered['EXPERIENCE_SCORE'].max()

    if max_score > min_score:
        df_filtered['EXPERIENCE_SCORE'] = (
            (df_filtered['EXPERIENCE_SCORE'] - min_score) / (max_score - min_score)
        ) * 10
    else:
        df_filtered['EXPERIENCE_SCORE'] = 0

    df_filtered['EXPERIENCE_SCORE'] = df_filtered['EXPERIENCE_SCORE'].round(2)

    # Detect translators not present in data_df (no prior tasks)
    missing_translators_mask = ~df_filtered['TRANSLATOR'].isin(data_df['TRANSLATOR'])

    # if not missing_translators.empty:
    #     print("Translators with no experience data:")
    #     print(missing_translators.tolist())

    # Compute average from those with scores
    avg_experience = df_filtered.loc[~missing_translators_mask, 'EXPERIENCE_SCORE'].mean()
    # print(f"Assigning average experience score of {round(avg_experience, 2)} to missing translators.")

    # Assign average to missing
    df_filtered.loc[missing_translators_mask, 'EXPERIENCE_SCORE'] = avg_experience

    return df_filtered


def compute_experience_for_client(df_filtered, data_df, client):
    """
    Computes an experience score for each translator based on a specific client

    Args:
        df_filtered (pd.DataFrame): Filtered translators' dataframe.

    Returns:
        pd.DataFrame: With added column 'EXPERIENCE_CLIENT'.
    """
    df_filtered = df_filtered.copy()
    translators = df_filtered['TRANSLATOR'].unique()
    df = data_df[data_df['TRANSLATOR'].isin(translators)].copy()


    df['score'] = 0
    df['score'] += (df['MANUFACTURER'] == client).astype(int)

    # Total experience score = sum of weights per translator
    experience_scores = df.groupby('TRANSLATOR')['score'].sum()

    # Add to filtered dataframe
    df_filtered['EXPERIENCE_CLIENT'] = df_filtered['TRANSLATOR'].map(experience_scores).fillna(0).astype(int)

    # Normalizar entre 0 y 10
    min_score = df_filtered['EXPERIENCE_CLIENT'].min()
    max_score = df_filtered['EXPERIENCE_CLIENT'].max()

    if max_score > min_score:  # Evitar división por 0
        df_filtered['EXPERIENCE_CLIENT'] = ((df_filtered['EXPERIENCE_CLIENT'] - min_score) / (max_score - min_score)) * 10
    else:
        df_filtered['EXPERIENCE_CLIENT'] = 0  # Si todos los scores son iguales

    df_filtered['EXPERIENCE_CLIENT'] = df_filtered['EXPERIENCE_CLIENT'].round(2)

    return df_filtered


def available_translators(task, translators_attributes_df, schedules_df):
    """
    Checks if translators are available for the task based on their weekly working schedule.
    This, for now just takes into account the day of the week and the start time of the task. 
    TAKE INTO ACCOUNT: This can have problems if the translator is at the end of their weekly shedule, it also doesnt take into account multitasking.
    
    Args:
        task (Task object): The task for which we want to check availability.
        translators_attributes_df (pd.DataFrame): DataFrame containing the translators' attributes.
        schedules_df (pd.DataFrame): DataFrame containing the weekly schedules of translators.
        TRANSLATORS_UNAVAILABLE (list): List of translators who are unavailable.
        
    Returns:
        df_filtered (pd.DataFrame): Filtered DataFrame containing translators who are available.
    """
    # 1. Remove explicitly unavailable translators
    # translators_unavailable = get_unavailable_translators()
    # df_filtered = translators_attributes_df[~translators_attributes_df['TRANSLATOR'].isin(translators_unavailable)].copy()
    df_filtered = translators_attributes_df.copy()
    # 2. Extract day of week and time from task
    task_day = task.ASSIGNED.strftime('%a').upper()  #day of the week  e.g., 'MON', 'TUE'

    # 3. Merge schedule info
    df_filtered = df_filtered.merge(schedules_df, left_on='TRANSLATOR', right_on='NAME').drop(columns=['NAME'])

    def is_available(row):
        # 1. Verificar si trabaja ese día
        if row[task_day] != 1:
            return False

        # 2. Tiempos
        task_start_time = timedelta(hours=task.ASSIGNED.hour, minutes=task.ASSIGNED.minute)
        work_end_time = timedelta(hours=row['END'].hour, minutes=row['END'].minute)

        # 3. Caso 1: el turno dura al menos una hora desde el inicio de la tarea
        if work_end_time - task_start_time >= timedelta(hours=1):
            return True

        # 4. Caso 2: turno corto, pero trabaja mañana
        days = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
        next_day = days[(days.index(task_day) + 1) % 7]
        return row[next_day] == 1

    # 4. Apply availability logic
    df_filtered['IS_AVAILABLE'] = df_filtered.apply(is_available, axis=1)
    df_filtered = df_filtered[df_filtered['IS_AVAILABLE'] == True].drop(columns=['IS_AVAILABLE'])

    return df_filtered


# ----- PRINCIPAL FUNCTION TO FILTER THE TRANSLATORS' ATTRIBUTES -----
def filter_language_price_quality_availability(data_df, schedules_df, translators_attributes_df, task=data.Task, need_wildcard = False):
    """
    Filters the translators' attributes by languages, price, quality and availability.
    If need_wildcard is True, it will skip the filter corresponding to the wildcard.

    Structured fallback:
        Tries a strict filter.
        If that fails, it retries with a wildcard (skipping one constraint).
        If that also fails, it relaxes all filters except language pair. That's reasonable.
    
    Args:
        translators_attributes_df (pd.DataFrame): 
            DataFrame containing the translators' attributes (name, languages, price, speed).
        task (Task object): 
            The task for which we want to filter the translators.
        need_wildcard (bool): 
            If True, skip the filter corresponding to the wildcard.
            
    Returns:
        pd.DataFrame: 
            Filtered DataFrame containing translators who meet the criteria.
    """
    
    if not need_wildcard:
        # Filter by language, price HARD FILTER
        df_filtered = translators_attributes_df[
            (translators_attributes_df['SOURCE_LANG'] == task.SOURCE_LANG) & 
            (translators_attributes_df['TARGET_LANG'] == task.TARGET_LANG) &
            (translators_attributes_df['HOURLY_RATE'] <= task.SELLING_HOURLY_PRICE) 
        ].copy()

        # If the filtered dataframe is empty, it's because the budget is too low, therefore we have to relax the filter
        if df_filtered.empty:
            if task.WILDCARD == "Price":
                logger.warning(f" No translators found for task {task.TASK_ID}, wildcard= {task.WILDCARD} because the BUDGET is too low. Trying with wildcard...")
                return filter_language_price_quality_availability(data_df, schedules_df, translators_attributes_df, task = task, need_wildcard = True)
            else: 
                logger.warning(f" No translators found for task {task.TASK_ID}, wildcard= {task.WILDCARD} because the BUDGET is too low. Relaxing price filter...")
                # Skip the price filter and try again
                df_filtered = translators_attributes_df[
                    (translators_attributes_df['SOURCE_LANG'] == task.SOURCE_LANG) &
                    (translators_attributes_df['TARGET_LANG'] == task.TARGET_LANG)
                ].copy()

        # add the average quality column
        df_filtered = compute_quality_by_task_type(data_df, df_filtered, task_type=task.TASK_TYPE)
        df_filtered = compute_quality_by_languages(data_df, df_filtered, source_lang=task.SOURCE_LANG, target_lang=task.TARGET_LANG)

        df_filtered_quality = df_filtered[
            (df_filtered['AVG_QUALITY_BY_LG'] >= task.MIN_QUALITY) | 
            (df_filtered['AVG_QUALITY_BY_TASK'] >= task.MIN_QUALITY)]

        # If the filtered dataframe is empty, it's because the quality is too high.
        if df_filtered_quality.empty:
            if task.WILDCARD == "Quality":
                logger.warning(f" No translators found for task {task.TASK_ID}, wildcard= {task.WILDCARD} because the QUALITY is too high. Trying with wildcard...")
                return filter_language_price_quality_availability(data_df, schedules_df, translators_attributes_df, task = task, need_wildcard = True)
            else: 
                logger.warning(f" No translators found for task {task.TASK_ID}, wildcard= {task.WILDCARD} because the QUALITY is too high. Relaxing quality filter...")
                # Skip the quality filter and try again
                penalization = df_filtered['AVG_QUALITY_BY_LG'].std() * 2
                df_filtered_quality = df_filtered[
                    (df_filtered['AVG_QUALITY_BY_LG'] >= task.MIN_QUALITY-penalization) | 
                    (df_filtered['AVG_QUALITY_BY_TASK'] >= task.MIN_QUALITY-penalization)]
        # Filter by availability
        df_filtered_availability = available_translators(task, df_filtered_quality, schedules_df)

        if df_filtered_availability.empty:
            if task.WILDCARD == "Deadline":
                logger.warning(f" No translators found for task {task.TASK_ID}, wildcard= {task.WILDCARD} because TRANSLATORS ARE UNAVAILABLE. Trying with wildcard...")
                return filter_language_price_quality_availability(data_df, schedules_df, translators_attributes_df, task = task, need_wildcard = True)
            else: 
                logger.warning(f" No translators found for task {task.TASK_ID}, wildcard= {task.WILDCARD} because TRANSLATORS ARE UNAVAILABLE. Relaxing availability filter...")
                if df_filtered_quality.empty:
                    df_filtered_quality = translators_attributes_df[
                        (translators_attributes_df['SOURCE_LANG'] == task.SOURCE_LANG) & 
                        (translators_attributes_df['TARGET_LANG'] == task.TARGET_LANG) &
                        (translators_attributes_df['HOURLY_RATE'] <= task.SELLING_HOURLY_PRICE)
                    ].copy()
                    df_filtered_quality = compute_quality_by_task_type(data_df, df_filtered, task_type=task.TASK_TYPE)
                    df_filtered_quality = compute_quality_by_languages(data_df, df_filtered, source_lang=task.SOURCE_LANG, target_lang=task.TARGET_LANG)
                return df_filtered_quality

        return df_filtered_availability
    
    # same code as above but with the wildcard, it will skip the filter corresponding to the wildcard
    else:
        # if the wildcard is "Price", we don't filter by price
        price_condition = (translators_attributes_df['HOURLY_RATE'] <= task.SELLING_HOURLY_PRICE) if task.WILDCARD != "Price" else True
        # Filter by language, price 
        df_filtered = translators_attributes_df[
            (translators_attributes_df['SOURCE_LANG'] == task.SOURCE_LANG) & 
            (translators_attributes_df['TARGET_LANG'] == task.TARGET_LANG) &
            price_condition 
        ].copy()

        if df_filtered.empty:
            logger.warning(f" No translators found for task {task.TASK_ID}, wildcard= {task.WILDCARD} because the BUDGET is too low. Relaxing price filter...")
            # Skip the price filter and try again
            df_filtered = translators_attributes_df[
                (translators_attributes_df['SOURCE_LANG'] == task.SOURCE_LANG) &
                (translators_attributes_df['TARGET_LANG'] == task.TARGET_LANG)
            ].copy()

        # add the average quality column
        df_filtered = compute_quality_by_languages(data_df, df_filtered, source_lang=task.SOURCE_LANG, target_lang=task.TARGET_LANG)
        df_filtered = compute_quality_by_task_type(data_df, df_filtered, task_type=task.TASK_TYPE)

        if task.WILDCARD != "Quality":
            df_filtered_quality = df_filtered[
                (df_filtered['AVG_QUALITY_BY_LG'] >= task.MIN_QUALITY) | 
                (df_filtered['AVG_QUALITY_BY_TASK'] >= task.MIN_QUALITY)]

            if df_filtered_quality.empty:
                logger.warning(f" No translators found for task {task.TASK_ID}, wildcard= {task.WILDCARD} because the QUALITY is too high. Relaxing quality filter...")
                # Skip the quality filter and try again
                penalization = df_filtered['AVG_QUALITY_BY_LG'].std() 
                df_filtered_quality = df_filtered[
                    (df_filtered['AVG_QUALITY_BY_LG'] >= task.MIN_QUALITY-penalization) | 
                    (df_filtered['AVG_QUALITY_BY_TASK'] >= task.MIN_QUALITY-penalization)]
        else:
            df_filtered_quality = df_filtered.copy()

        if task.WILDCARD != "Deadline":  
            # Filter by availability
            df_filtered_availability = available_translators(task, df_filtered_quality, schedules_df)

            if df_filtered_availability.empty:
                logger.warning(f" No translators found for task {task.TASK_ID}, wildcard= {task.WILDCARD} because TRANSLATORS ARE UNAVAILABLE. Relaxing availability filter...")
                
                if df_filtered_quality.empty:
                    df_filtered_quality = translators_attributes_df[
                        (translators_attributes_df['SOURCE_LANG'] == task.SOURCE_LANG) & 
                        (translators_attributes_df['TARGET_LANG'] == task.TARGET_LANG) &
                        (translators_attributes_df['HOURLY_RATE'] <= task.SELLING_HOURLY_PRICE)
                    ].copy()
                    df_filtered_quality = compute_quality_by_task_type(data_df, df_filtered, task_type=task.TASK_TYPE)
                    df_filtered_quality = compute_quality_by_languages(data_df, df_filtered, source_lang=task.SOURCE_LANG, target_lang=task.TARGET_LANG)
                return df_filtered_quality
        
            
        return df_filtered
