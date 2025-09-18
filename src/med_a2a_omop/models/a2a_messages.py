
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class OMOPQueryRequest(BaseModel):
    """Message from Orchestrator to OMOP Agent"""
    question: str
    semantic_context: Optional[Dict[str, Any]] = None

class OMOPQueryResponse(BaseModel):
    """Message from OMOP Agent to Orchestrator"""
    generated_sql: str
    query_result: List[Dict[str, Any]]

class SemanticAnalysisRequest(BaseModel):
    """Message for Semantic Agent analysis"""
    question: str

class SemanticAnalysisResponse(BaseModel):
    """Response from Semantic Agent"""
    medical_terms: List[Dict[str, Any]]
    temporal_constraints: List[Dict[str, Any]]
    query_intent: Dict[str, Any]
    relationships: List[Dict[str, Any]]
    original_query: str
