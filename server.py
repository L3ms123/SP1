from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from find_translator import *
import os
app = FastAPI()


script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, ".", "Data")
parquet_data = os.path.join(data_path, "cleaned_data.parquet")

schedules_df = pd.read_parquet(os.path.join(data_path, "schedules.parquet"))
data_df = pd.read_parquet(parquet_data)
transl_cost_pairs_df = pd.read_parquet(os.path.join(data_path, "translators.parquet"))
clients_df = pd.read_parquet(os.path.join(data_path, "clients.parquet"))

@app.get("/")
def read_root():
    return {"message": "API funcionando"}

class Task(BaseModel):
    sourceLanguage: str 
    targetLanguage: str
    taskType: str
    manufacturer: str 
    manufacturerIndustry: str 
    manufacturerSubindustry: str 
    minQuality: int 
    wildcard: str 
    pricePerHour: int

@app.post("/get-translators")
async def getTranslators(task: Task):
    get_top_translators_for_task(task_input_dict=task,
                                 data_df=data_df,
                                 transl_cost_pairs_df=transl_cost_pairs_df,
                                 clients_df=clients_df,
                                 schedules_df=schedules_df,
                                 top_k=4)
      
