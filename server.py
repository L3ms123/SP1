from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()


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
    print(task)
    return
