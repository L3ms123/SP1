import os
import pandas as pd

from . import cleaning_utils as cu

RANDOM_SEED = 42

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "Data")

parquet_data = os.path.join(data_path, "cleaned_data.parquet")

if os.path.exists(parquet_data):
    data_df = pd.read_parquet(parquet_data)
    schedules_df = pd.read_parquet(os.path.join(data_path, "schedules.parquet"))
    clients_df = pd.read_parquet(os.path.join(data_path, "clients.parquet"))
    transl_cost_pairs_df = pd.read_parquet(os.path.join(data_path, "translators.parquet"))
else:
    schedules_df = pd.read_excel(os.path.join(data_path, "Schedules.xlsx"))
    data_df = pd.read_excel(os.path.join(data_path, "Data.xlsx"))
    clients_df = pd.read_excel(os.path.join(data_path, "Clients.xlsx"))
    transl_cost_pairs_df = pd.read_excel(os.path.join(data_path, "TranslatorsCost+Pairs.xlsx"))

    # Limpieza data_df
    data_df['START'] = pd.to_datetime(data_df['START'], errors='coerce')
    data_df, _ = cu.drop_invalid_dates(data_df, 'START')
    data_df, _ = cu.drop_invalid_dates(data_df, 'END')
    data_df, _ = cu.drop_invalid_dates(data_df, 'DELIVERED')
    data_df, _ = cu.drop_invalid_dates(data_df, 'ASSIGNED')
    data_df = cu.drop_invalid_rows(data_df)

    # Guardar todo en parquet para próximas veces
    data_df.to_parquet(parquet_data)
    schedules_df.to_parquet(os.path.join(data_path, "schedules.parquet"))
    clients_df.to_parquet(os.path.join(data_path, "clients.parquet"))
    transl_cost_pairs_df.to_parquet(os.path.join(data_path, "translators.parquet"))


## -------Task Class------- ##
class Task:
    def __init__(self, **kwargs):
        """
        A class used to represent a Task. It initializes the attributes dynamically 
        using the keyword arguments passed. Default values are provided for certain fields.
        """
        self.PROJECT_ID = kwargs.get('PROJECT_ID', None)
        self.TASK_ID = kwargs.get('TASK_ID', None)
        self.ASSIGNED = kwargs.get('ASSIGNED', None)
        self.END = kwargs.get('END', None)
        self.SELLING_HOURLY_PRICE = kwargs.get('SELLING_HOURLY_PRICE', None)
        self.MIN_QUALITY = kwargs.get('MIN_QUALITY', None)
        self.WILDCARD = kwargs.get('WILDCARD', None) 
        self.TASK_TYPE = kwargs.get('TASK_TYPE', None)
        self.SOURCE_LANG = kwargs.get('SOURCE_LANG', None)
        self.TARGET_LANG = kwargs.get('TARGET_LANG', None)
        self.MANUFACTURER = kwargs.get('MANUFACTURER', None)
        self.MANUFACTURER_SECTOR = kwargs.get('MANUFACTURER_SECTOR', None)
        self.MANUFACTURER_INDUSTRY_GROUP = kwargs.get('MANUFACTURER_INDUSTRY_GROUP', None)
        self.MANUFACTURER_INDUSTRY = kwargs.get('MANUFACTURER_INDUSTRY', None)
        self.MANUFACTURER_SUBINDUSTRY = kwargs.get('MANUFACTURER_SUBINDUSTRY', None)
        
        # Optional attributes with None default value
        self.START = kwargs.get('START', None)
        self.PM = kwargs.get('PM', None)
        self.TRANSLATOR = kwargs.get('TRANSLATOR', None)
        self.READY = kwargs.get('READY', None)
        self.WORKING = kwargs.get('WORKING', None)
        self.DELIVERED = kwargs.get('DELIVERED', None)
        self.RECEIVED = kwargs.get('RECEIVED', None)
        self.CLOSE = kwargs.get('CLOSE', None)
        self.FORECAST = kwargs.get('FORECAST', None)
        self.HOURLY_RATE = kwargs.get('HOURLY_RATE', None)
        self.COST = kwargs.get('COST', None)
        self.QUALITY_EVALUATION = kwargs.get('QUALITY_EVALUATION', None)
        
    
    def __str__(self):
        return (
            f"Task Details:\n"
            f"  - Task ID: {self.TASK_ID}\n"
            f"  - Type: {self.TASK_TYPE}\n"
            f"  - Client: {self.MANUFACTURER}\n"
            f"  - Sector: {self.MANUFACTURER_SECTOR}\n"
            f"  - Industry (Subsector): {self.MANUFACTURER_INDUSTRY}\n"
            f"  - Start: {self.START}\n"
            f"  - Budget: {self.SELLING_HOURLY_PRICE}\n"
            f"  - Quality: {self.MIN_QUALITY}\n"
            f"  - Wildcard: {self.WILDCARD}\n"
            f"  - Source Language: {self.SOURCE_LANG}\n"
            f"  - Target Language: {self.TARGET_LANG}"
        )
