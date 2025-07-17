
from pydantic import BaseModel
from typing import List, Dict, Any

class OMOPQueryRequest(BaseModel):
    """Message from Orchestrator to OMOP Agent"""
    question: str

class OMOPQueryResponse(BaseModel):
    """Message from OMOP Agent to Orchestrator"""
    generated_sql: str
    query_result: List[Dict[str, Any]]
