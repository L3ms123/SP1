from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from find_translator import *
from pprint import pprint
from datetime import datetime
import os
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    processed_task= {
        'TASK_TYPE': task.taskType,
        'SOURCE_LANG': task.sourceLanguage,
        'TARGET_LANG': task.targetLanguage,
        'TASK_ID': 32134,

        'MANUFACTURER': task.manufacturer,
        'MANUFACTURER_INDUSTRY': task.manufacturerIndustry,
        'MANUFACTURER_SUBINDUSTRY': task.manufacturerSubindustry,
        'SELLING_HOURLY_PRICE': task.pricePerHour,
        'MIN_QUALITY': task.minQuality,
        'ASSIGNED':  datetime.now()
    }
    '''
    print(f"source: {task.sourceLanguage}")
    print(f"target: {task.targetLanguage}")
    print(f"type: {task.taskType}")
    print(f"manufacturer: {task.manufacturer}")
    print(f"industry: {task.manufacturerIndustry}")
    print(f"subindustry: {task.manufacturerSubindustry}")
    print(f"quality> {task.minQuality}")
    print(f"wildcard: {task.wildcard}")
    print(f"price/h: {task.pricePerHour}")
    print(f"assgined: {datetime.now()}")
    '''
    try:
        top = get_top_translators_for_task(task_input_dict=processed_task,
                                 data_df=data_df,
                                 transl_cost_pairs_df=transl_cost_pairs_df,
                                 clients_df=clients_df,
                                 schedules_df=schedules_df,
                                 top_k=4)
        for t in top:
            pprint(t)
            print("\n\n")

        return top
    except Exception as e:
        print("ERROR:", e)
        return []
      
