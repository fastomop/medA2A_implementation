import json
import uuid
from typing import List, Dict, Any, Optional
from collections.abc import AsyncIterable
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

from a2a.types import Message, TextPart, Part, Role, SendMessageSuccessResponse, JSONRPCErrorResponse
from a2a_medical.base.agent import MedicalAgent, ProcessedObservation, Action, ActionResult, MentalState
from a2a_medical.integrations.ollama import OllamaReasoningMixin
from a2a_medical.integrations.mcp_official import MCPDiscoveryMixin, MCPServer
from a2a.types import AgentCard, AgentCapabilities
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.server.context import ServerCallContext

from ..models.a2a_messages import OMOPQueryRequest, OMOPQueryResponse

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class OMOPTable:
    """Comprehensive OMOP CDM table definition"""
    name: str
    domain: str  # Clinical domain (e.g., "Person", "Condition", "Drug")
    description: str
    standard_columns: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # col_name -> {type, required, fk_table}
    primary_key: str = ""
    foreign_keys: Dict[str, str] = field(default_factory=dict)  # column -> referenced_table
    indexes: List[str] = field(default_factory=list)
    business_rules: List[str] = field(default_factory=list)
    common_joins: List[str] = field(default_factory=list)
    
    # Runtime discovered info
    actual_columns: Dict[str, str] = field(default_factory=dict)  # discovered columns
    row_count: Optional[int] = None
    sample_data: List[Dict[str, Any]] = field(default_factory=list)
    last_explored: Optional[datetime] = None
    confidence: float = 0.0
    confirmed_missing_columns: set[str] = field(default_factory=set) # New field for confirmed missing columns

@dataclass
class OMOPDomain:
    """OMOP Clinical Domain with its relationships"""
    name: str
    description: str
    primary_table: str
    related_tables: List[str] = field(default_factory=list)
    concept_classes: List[str] = field(default_factory=list)
    vocabularies: List[str] = field(default_factory=list)
    common_patterns: List[str] = field(default_factory=list)

@dataclass
class OMOPQueryTemplate:
    """Reusable query templates for common OMOP patterns"""
    name: str
    description: str
    sql_template: str
    parameters: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    complexity: str = "simple"  # simple, intermediate, complex
    success_rate: float = 0.0
    usage_count: int = 0
    examples: List[Dict[str, Any]] = field(default_factory=list) # New field for examples

class ComprehensiveOMOPWorldModel:
    """
    Comprehensive OMOP CDM v5.4 World Model with complete standard knowledge
    and dynamic learning capabilities.
    """
    
    def __init__(self):
        self.schema_prefix = "base"
        self.cdm_version = "5.4"
        
        # Initialize comprehensive OMOP CDM v5.4 structure
        self.omop_tables = self._initialize_omop_cdm_structure()
        self.omop_domains = self._initialize_omop_domains()
        self.query_templates = self._initialize_query_templates()
        
        # Runtime learning
        self.successful_queries: List[Dict[str, Any]] = []
        self.failed_queries: List[Dict[str, Any]] = []
        self.learned_patterns: Dict[str, OMOPQueryTemplate] = {}
        self.vocabulary_cache: Dict[str, List[Dict]] = {}  # concept searches
        
        # Exploration tracking
        self.exploration_status: Dict[str, Dict[str, Any]] = {}
        
    def _initialize_omop_cdm_structure(self) -> Dict[str, OMOPTable]:
        """Initialize complete OMOP CDM v5.4 table structure"""
        tables = {}
        
        # PERSON - Demographics
        tables["person"] = OMOPTable(
            name="person",
            domain="Demographics",
            description="Unique patients/persons in the database",
            standard_columns={
                "person_id": {"type": "INTEGER", "required": True, "description": "Unique identifier"},
                "gender_concept_id": {"type": "INTEGER", "required": True, "fk_table": "concept"},
                "year_of_birth": {"type": "INTEGER", "required": True},
                "month_of_birth": {"type": "INTEGER", "required": False},
                "day_of_birth": {"type": "INTEGER", "required": False},
                "birth_datetime": {"type": "DATETIME", "required": False},
                "race_concept_id": {"type": "INTEGER", "required": True, "fk_table": "concept"},
                "ethnicity_concept_id": {"type": "INTEGER", "required": True, "fk_table": "concept"},
                "location_id": {"type": "INTEGER", "required": False, "fk_table": "location"},
                "provider_id": {"type": "INTEGER", "required": False, "fk_table": "provider"},
                "care_site_id": {"type": "INTEGER", "required": False, "fk_table": "care_site"},
                "person_source_value": {"type": "VARCHAR(50)", "required": False},
                "gender_source_value": {"type": "VARCHAR(50)", "required": False},
                "gender_source_concept_id": {"type": "INTEGER", "required": False, "fk_table": "concept"},
                "race_source_value": {"type": "VARCHAR(50)", "required": False},
                "race_source_concept_id": {"type": "INTEGER", "required": False, "fk_table": "concept"},
                "ethnicity_source_value": {"type": "VARCHAR(50)", "required": False},
                "ethnicity_source_concept_id": {"type": "INTEGER", "required": False, "fk_table": "concept"}
            },
            primary_key="person_id",
            business_rules=[
                "Each person must have a valid gender_concept_id",
                "Year of birth must be reasonable (e.g., > 1900)",
                "All concept_id fields must reference valid concepts"
            ]
        )
        
        # CONDITION_OCCURRENCE - Medical Conditions
        tables["condition_occurrence"] = OMOPTable(
            name="condition_occurrence",
            domain="Condition",
            description="Records of medical conditions/diagnoses",
            standard_columns={
                "condition_occurrence_id": {"type": "INTEGER", "required": True, "description": "Unique identifier"},
                "person_id": {"type": "INTEGER", "required": True, "fk_table": "person"},
                "condition_concept_id": {"type": "INTEGER", "required": True, "fk_table": "concept"},
                "condition_start_date": {"type": "DATE", "required": True},
                "condition_start_datetime": {"type": "DATETIME", "required": False},
                "condition_end_date": {"type": "DATE", "required": False},
                "condition_end_datetime": {"type": "DATETIME", "required": False},
                "condition_type_concept_id": {"type": "INTEGER", "required": True, "fk_table": "concept"},
                "condition_status_concept_id": {"type": "INTEGER", "required": False, "fk_table": "concept"},
                "stop_reason": {"type": "VARCHAR(20)", "required": False},
                "provider_id": {"type": "INTEGER", "required": False, "fk_table": "provider"},
                "visit_occurrence_id": {"type": "INTEGER", "required": False, "fk_table": "visit_occurrence"},
                "visit_detail_id": {"type": "INTEGER", "required": False, "fk_table": "visit_detail"},
                "condition_source_value": {"type": "VARCHAR(50)", "required": False},
                "condition_source_concept_id": {"type": "INTEGER", "required": False, "fk_table": "concept"},
                "condition_status_source_value": {"type": "VARCHAR(50)", "required": False}
            },
            primary_key="condition_occurrence_id",
            foreign_keys={
                "person_id": "person.person_id",
                "condition_concept_id": "concept.concept_id",
                "condition_type_concept_id": "concept.concept_id",
                "visit_occurrence_id": "visit_occurrence.visit_occurrence_id"
            },
            common_joins=["person", "concept", "visit_occurrence"],
            business_rules=[
                "condition_concept_id must be from Condition domain",
                "condition_start_date cannot be in the future",
                "condition_end_date must be >= condition_start_date if present"
            ]
        )
        
        # DRUG_EXPOSURE - Medications
        tables["drug_exposure"] = OMOPTable(
            name="drug_exposure",
            domain="Drug",
            description="Records of drug/medication exposures",
            standard_columns={
                "drug_exposure_id": {"type": "INTEGER", "required": True},
                "person_id": {"type": "INTEGER", "required": True, "fk_table": "person"},
                "drug_concept_id": {"type": "INTEGER", "required": True, "fk_table": "concept"},
                "drug_exposure_start_date": {"type": "DATE", "required": True},
                "drug_exposure_start_datetime": {"type": "DATETIME", "required": False},
                "drug_exposure_end_date": {"type": "DATE", "required": True},
                "drug_exposure_end_datetime": {"type": "DATETIME", "required": False},
                "verbatim_end_date": {"type": "DATE", "required": False},
                "drug_type_concept_id": {"type": "INTEGER", "required": True, "fk_table": "concept"},
                "stop_reason": {"type": "VARCHAR(20)", "required": False},
                "refills": {"type": "INTEGER", "required": False},
                "quantity": {"type": "FLOAT", "required": False},
                "days_supply": {"type": "INTEGER", "required": False},
                "sig": {"type": "TEXT", "required": False},
                "route_concept_id": {"type": "INTEGER", "required": False, "fk_table": "concept"},
                "lot_number": {"type": "VARCHAR(50)", "required": False},
                "provider_id": {"type": "INTEGER", "required": False, "fk_table": "provider"},
                "visit_occurrence_id": {"type": "INTEGER", "required": False, "fk_table": "visit_occurrence"},
                "visit_detail_id": {"type": "INTEGER", "required": False, "fk_table": "visit_detail"},
                "drug_source_value": {"type": "VARCHAR(50)", "required": False},
                "drug_source_concept_id": {"type": "INTEGER", "required": False, "fk_table": "concept"},
                "route_source_value": {"type": "VARCHAR(50)", "required": False},
                "dose_unit_source_value": {"type": "VARCHAR(50)", "required": False}
            },
            primary_key="drug_exposure_id",
            foreign_keys={
                "person_id": "person.person_id",
                "drug_concept_id": "concept.concept_id",
                "drug_type_concept_id": "concept.concept_id"
            },
            common_joins=["person", "concept", "visit_occurrence"],
            business_rules=[
                "drug_concept_id must be from Drug domain",
                "drug_exposure_end_date must be >= drug_exposure_start_date"
            ]
        )
        
        # MEASUREMENT - Lab Results, Vitals
        tables["measurement"] = OMOPTable(
            name="measurement",
            domain="Measurement",
            description="Laboratory tests, vital signs, and other measurements",
            standard_columns={
                "measurement_id": {"type": "INTEGER", "required": True},
                "person_id": {"type": "INTEGER", "required": True, "fk_table": "person"},
                "measurement_concept_id": {"type": "INTEGER", "required": True, "fk_table": "concept"},
                "measurement_date": {"type": "DATE", "required": True},
                "measurement_datetime": {"type": "DATETIME", "required": False},
                "measurement_time": {"type": "VARCHAR(10)", "required": False},
                "measurement_type_concept_id": {"type": "INTEGER", "required": True, "fk_table": "concept"},
                "operator_concept_id": {"type": "INTEGER", "required": False, "fk_table": "concept"},
                "value_as_number": {"type": "FLOAT", "required": False},
                "value_as_concept_id": {"type": "INTEGER", "required": False, "fk_table": "concept"},
                "unit_concept_id": {"type": "INTEGER", "required": False, "fk_table": "concept"},
                "range_low": {"type": "FLOAT", "required": False},
                "range_high": {"type": "FLOAT", "required": False},
                "provider_id": {"type": "INTEGER", "required": False, "fk_table": "provider"},
                "visit_occurrence_id": {"type": "INTEGER", "required": False, "fk_table": "visit_occurrence"},
                "visit_detail_id": {"type": "INTEGER", "required": False, "fk_table": "visit_detail"},
                "measurement_source_value": {"type": "VARCHAR(50)", "required": False},
                "measurement_source_concept_id": {"type": "INTEGER", "required": False, "fk_table": "concept"},
                "unit_source_value": {"type": "VARCHAR(50)", "required": False},
                "unit_source_concept_id": {"type": "INTEGER", "required": False, "fk_table": "concept"},
                "value_source_value": {"type": "VARCHAR(50)", "required": False},
                "measurement_event_id": {"type": "INTEGER", "required": False},
                "meas_event_field_concept_id": {"type": "INTEGER", "required": False, "fk_table": "concept"}
            },
            primary_key="measurement_id",
            foreign_keys={
                "person_id": "person.person_id",
                "measurement_concept_id": "concept.concept_id"
            },
            common_joins=["person", "concept", "visit_occurrence"]
        )
        
        # CONCEPT - Vocabulary/Terminology
        tables["concept"] = OMOPTable(
            name="concept",
            domain="Vocabulary",
            description="Standardized medical concepts and terminology",
            standard_columns={
                "concept_id": {"type": "INTEGER", "required": True},
                "concept_name": {"type": "VARCHAR(255)", "required": True},
                "domain_id": {"type": "VARCHAR(20)", "required": True},
                "vocabulary_id": {"type": "VARCHAR(20)", "required": True},
                "concept_class_id": {"type": "VARCHAR(20)", "required": True},
                "standard_concept": {"type": "VARCHAR(1)", "required": False},
                "concept_code": {"type": "VARCHAR(50)", "required": True},
                "valid_start_date": {"type": "DATE", "required": True},
                "valid_end_date": {"type": "DATE", "required": True},
                "invalid_reason": {"type": "VARCHAR(1)", "required": False}
            },
            primary_key="concept_id",
            business_rules=[
                "standard_concept = 'S' for standard concepts",
                "domain_id determines which clinical tables can use this concept",
                "valid_start_date <= valid_end_date"
            ]
        )
        
        # Add more core tables...
        tables["observation"] = OMOPTable(
            name="observation",
            domain="Observation", 
            description="Clinical facts that don't fit other domains",
            standard_columns={
                "observation_id": {"type": "INTEGER", "required": True},
                "person_id": {"type": "INTEGER", "required": True, "fk_table": "person"},
                "observation_concept_id": {"type": "INTEGER", "required": True, "fk_table": "concept"},
                "observation_date": {"type": "DATE", "required": True},
                "observation_datetime": {"type": "DATETIME", "required": False},
                "observation_type_concept_id": {"type": "INTEGER", "required": True, "fk_table": "concept"},
                "value_as_number": {"type": "FLOAT", "required": False},
                "value_as_string": {"type": "VARCHAR(60)", "required": False},
                "value_as_concept_id": {"type": "INTEGER", "required": False, "fk_table": "concept"},
                "qualifier_concept_id": {"type": "INTEGER", "required": False, "fk_table": "concept"},
                "unit_concept_id": {"type": "INTEGER", "required": False, "fk_table": "concept"},
                "provider_id": {"type": "INTEGER", "required": False, "fk_table": "provider"},
                "visit_occurrence_id": {"type": "INTEGER", "required": False, "fk_table": "visit_occurrence"},
                "visit_detail_id": {"type": "INTEGER", "required": False, "fk_table": "visit_detail"},
                "observation_source_value": {"type": "VARCHAR(50)", "required": False},
                "observation_source_concept_id": {"type": "INTEGER", "required": False, "fk_table": "concept"},
                "unit_source_value": {"type": "VARCHAR(50)", "required": False},
                "qualifier_source_value": {"type": "VARCHAR(50)", "required": False},
                "value_source_value": {"type": "VARCHAR(50)", "required": False},
                "observation_event_id": {"type": "INTEGER", "required": False},
                "obs_event_field_concept_id": {"type": "INTEGER", "required": False, "fk_table": "concept"}
            },
            primary_key="observation_id",
            foreign_keys={
                "person_id": "person.person_id",
                "observation_concept_id": "concept.concept_id"
            }
        )
        
        return tables
    
    def _initialize_omop_domains(self) -> Dict[str, OMOPDomain]:
        """Initialize OMOP clinical domains with their relationships"""
        domains = {}
        
        domains["Person"] = OMOPDomain(
            name="Person",
            description="Patient demographics, age, gender, race, ethnicity",
            primary_table="person",
            related_tables=["location", "provider", "care_site"],
            concept_classes=["Gender", "Race", "Ethnicity"],
            vocabularies=["Gender", "Race", "Ethnicity"],
            common_patterns=[
                "Count total patients",
                "Calculate average age",
                "Age distribution analysis",
                "Demographics breakdown",
                "Gender distribution"
            ]
        )
        
        domains["Condition"] = OMOPDomain(
            name="Condition",
            description="Medical conditions, diagnoses, symptoms",
            primary_table="condition_occurrence",
            related_tables=["person", "concept", "visit_occurrence"],
            concept_classes=["Clinical Finding", "Disorder", "Disease"],
            vocabularies=["SNOMED", "ICD10CM", "ICD9CM"],
            common_patterns=[
                "Count patients with condition",
                "Find comorbidities", 
                "Condition prevalence by demographics",
                "Time to diagnosis"
            ]
        )
        
        domains["Drug"] = OMOPDomain(
            name="Drug",
            description="Medications, prescriptions, drug exposures",
            primary_table="drug_exposure",
            related_tables=["person", "concept", "visit_occurrence"],
            concept_classes=["Ingredient", "Brand Name", "Clinical Drug"],
            vocabularies=["RxNorm", "NDC", "CVX"],
            common_patterns=[
                "Drug utilization patterns",
                "Polypharmacy analysis",
                "Drug-condition associations",
                "Adherence calculations"
            ]
        )
        
        domains["Measurement"] = OMOPDomain(
            name="Measurement",
            description="Laboratory tests, vital signs, measurements",
            primary_table="measurement",
            related_tables=["person", "concept", "visit_occurrence"],
            concept_classes=["Lab Test", "Vital Sign", "Clinical Measurement"],
            vocabularies=["LOINC", "SNOMED"],
            common_patterns=[
                "Lab value distributions",
                "Abnormal results identification",
                "Longitudinal trending",
                "Reference range analysis"
            ]
        )
        
        return domains
    
    def _initialize_query_templates(self) -> Dict[str, OMOPQueryTemplate]:
        """Initialize common OMOP query patterns"""
        templates = {}
        
        templates["count_patients_with_condition"] = OMOPQueryTemplate(
            name="count_patients_with_condition",
            description="Count distinct patients with a specific condition",
            sql_template="""
SELECT COUNT(DISTINCT p.person_id) as patient_count
FROM {schema}.person p
JOIN {schema}.condition_occurrence co ON p.person_id = co.person_id
JOIN {schema}.concept c ON co.condition_concept_id = c.concept_id
WHERE c.standard_concept = 'S'
  AND c.domain_id = 'Condition'
  AND LOWER(c.concept_name) LIKE LOWER('%{condition_name}%')
            """.strip(),
            parameters=["condition_name"],
            domains=["Condition"],
            complexity="simple"
        )
        
        templates["drug_exposure_by_person"] = OMOPQueryTemplate(
            name="drug_exposure_by_person",
            description="Find drug exposures for patients",
            sql_template="""
SELECT p.person_id, c.concept_name as drug_name, 
       de.drug_exposure_start_date, de.days_supply
FROM {schema}.person p
JOIN {schema}.drug_exposure de ON p.person_id = de.person_id
JOIN {schema}.concept c ON de.drug_concept_id = c.concept_id
WHERE c.standard_concept = 'S'
  AND c.domain_id = 'Drug'
  AND LOWER(c.concept_name) LIKE LOWER('%{drug_name}%')
            """.strip(),
            parameters=["drug_name"],
            domains=["Drug"],
            complexity="simple"
        )
        
        templates["average_patient_age"] = OMOPQueryTemplate(
            name="average_patient_age",
            description="Calculate average age of patients using year_of_birth",
            sql_template="""
SELECT AVG(EXTRACT(YEAR FROM CURRENT_DATE) - p.year_of_birth) as avg_age
FROM {schema}.person p
WHERE p.year_of_birth IS NOT NULL
  AND p.year_of_birth > 1900
  AND p.year_of_birth <= EXTRACT(YEAR FROM CURRENT_DATE)
            """.strip(),
            parameters=[],
            domains=["Person"],
            complexity="simple"
        )
        
        templates["patient_age_distribution"] = OMOPQueryTemplate(
            name="patient_age_distribution",
            description="Get age distribution of patients",
            sql_template="""
SELECT 
    CASE 
        WHEN (EXTRACT(YEAR FROM CURRENT_DATE) - p.year_of_birth) < 18 THEN 'Under 18'
        WHEN (EXTRACT(YEAR FROM CURRENT_DATE) - p.year_of_birth) BETWEEN 18 AND 65 THEN '18-65'
        ELSE 'Over 65'
    END as age_group,
    COUNT(*) as patient_count
FROM {schema}.person p
WHERE p.year_of_birth IS NOT NULL
  AND p.year_of_birth > 1900
GROUP BY age_group
            """.strip(),
            parameters=[],
            domains=["Person"],
            complexity="intermediate"
        )

        templates["average_age_with_demographic_filter"] = OMOPQueryTemplate(
            name="average_age_with_demographic_filter",
            description="Calculate the average age of patients filtered by a specific demographic (e.g., gender, race).",
            sql_template="""
SELECT AVG(EXTRACT(YEAR FROM CURRENT_DATE) - p.year_of_birth) as avg_age
FROM {schema}.person p
WHERE p.{demographic_column} IN (
    SELECT c.concept_id
    FROM {schema}.concept c
    WHERE c.standard_concept = 'S'
      AND c.domain_id = '{domain_id}'
      AND LOWER(c.concept_name) LIKE LOWER('%{concept_value}%')
)
  AND p.year_of_birth IS NOT NULL
  AND p.year_of_birth > 1900
            """.strip(),
            parameters=["demographic_column", "domain_id", "concept_value"],
            domains=["Person"],
            complexity="intermediate",
            examples=[
                {"question": "average age of female patients", "demographic_column": "gender_concept_id", "domain_id": "Gender", "concept_value": "female"},
                {"question": "average age of asian patients", "demographic_column": "race_concept_id", "domain_id": "Race", "concept_value": "asian"}
            ]
        )
          
        return templates
    
    async def explore_database_schema(self, mcp_manager) -> Dict[str, Any]:
        """Comprehensively explore the actual database schema"""
        exploration_results = {
            "tables_found": [],
            "tables_missing": [],
            "schema_differences": {},
            "exploration_timestamp": datetime.now()
        }
        
        for table_name, omop_table in self.omop_tables.items():
            try:
                # Test table existence and get sample data
                sample_query = f"SELECT * FROM {self.schema_prefix}.{table_name} LIMIT 3"
                result = await mcp_manager.call_tool("omop_db_server:Select_Query", {"query": sample_query})
                
                if self._is_successful_result(result):
                    exploration_results["tables_found"].append(table_name)
                    
                    # Extract actual schema from sample data
                    actual_columns = self._extract_columns_from_result(result)
                    omop_table.actual_columns = actual_columns
                    
                    # Compare with expected OMOP structure
                    schema_diff = self._compare_with_standard(omop_table, actual_columns)
                    if schema_diff:
                        exploration_results["schema_differences"][table_name] = schema_diff
                    
                    # Get row count
                    count_query = f"SELECT COUNT(*) as row_count FROM {self.schema_prefix}.{table_name}"
                    count_result = await mcp_manager.call_tool("omop_db_server:Select_Query", {"query": count_query})
                    if self._is_successful_result(count_result):
                        omop_table.row_count = self._extract_count_from_result(count_result)
                    
                    omop_table.last_explored = datetime.now()
                    omop_table.confidence = 1.0
                    
                else:
                    exploration_results["tables_missing"].append(table_name)
                    
            except Exception as e:
                logger.debug(f"Could not explore table {table_name}: {e}")
                exploration_results["tables_missing"].append(table_name)
        
        self.exploration_status["last_full_exploration"] = exploration_results
        return exploration_results
    
    def get_comprehensive_context(self, question: str, extracted_context: Dict[str, Any], failed_attempts: Optional[List[Dict]] = None) -> str:
        """
        Generates a hyper-focused context string using extracted entities
        to query the world model for relevant schemas and templates.
        """
        context_parts = []
        
        # Use domains from the structured extraction
        relevant_domains = extracted_context.get("domains", [])
        if not relevant_domains:
            # Fallback to keyword-based identification if extraction fails
            relevant_domains = self._identify_relevant_domains(question)
        
        context_parts.append("=== OMOP CDM v5.4 CONTEXT ===")
        context_parts.append(f"Question: {question}")
        context_parts.append(f"Identified Domains: {', '.join(relevant_domains)}")
        context_parts.append(f"Query Type: {extracted_context.get('query_type', 'unknown')}")
        
        # 2. Add relevant table schemas based on extracted domains
        if relevant_domains:
            context_parts.append("\n=== RELEVANT TABLES & SCHEMAS ===")
            seen_tables = set()
            for domain_name in relevant_domains:
                if domain_name in self.omop_domains:
                    domain = self.omop_domains[domain_name]
                    
                    # Add primary table for the domain
                    tables_to_describe = {domain.primary_table}
                    # Also include related tables
                    tables_to_describe.update(domain.related_tables)
                    
                    for table_name in tables_to_describe:
                        if table_name not in seen_tables and table_name in self.omop_tables:
                            table = self.omop_tables[table_name]
                            context_parts.append(f"\nTABLE: {self.schema_prefix}.{table.name} ({table.domain} Domain)")
                            context_parts.append(f"  Description: {table.description}")
                            
                            # Display actual columns if discovered, otherwise show standard columns
                            if table.actual_columns:
                                context_parts.append(f"  Actual Columns (Discovered):")
                                for col_name, col_type in table.actual_columns.items():
                                    context_parts.append(f"    - {col_name} ({col_type})")
                            else:
                                context_parts.append(f"  Standard Columns (Default):")
                                for col_name, col_info in table.standard_columns.items():
                                    fk_info = f" -> FK to {col_info['fk_table']}" if col_info.get("fk_table") else ""
                                    context_parts.append(f"    - {col_name} ({col_info.get('type', 'UNKNOWN')}){fk_info}")
                            seen_tables.add(table_name)

        # 3. Add relevant query templates based on extracted concepts/type
        relevant_templates = self._find_relevant_templates(
            question, 
            relevant_domains, 
            extracted_context.get("concepts", []),
            extracted_context.get("query_type", "")
        )
        if relevant_templates:
            context_parts.append("\n=== SUGGESTED QUERY PATTERNS ===")
            for template_name, template in relevant_templates.items():
                context_parts.append(f"\n-- Pattern: {template.name} --")
                context_parts.append(f"  Description: {template.description}")
                context_parts.append(f"  SQL Template:")
                formatted_sql = template.sql_template.format(schema=self.schema_prefix, **{p: f"{{{p}}}" for p in template.parameters})
                for line in formatted_sql.split('\n'):
                    context_parts.append(f"    {line}")

        # 4. Add lessons from failures
        if failed_attempts:
            context_parts.append("\n=== LESSONS FROM PREVIOUS FAILURES ===")
            lessons = self._extract_lessons_from_failures(failed_attempts)
            for lesson in lessons:
                context_parts.append(f"  - {lesson}")
        
        return "\n".join(context_parts)
    
    def _identify_relevant_domains(self, question: str) -> List[str]:
        """Identify relevant OMOP domains from the question"""
        question_lower = question.lower()
        relevant_domains = []
        
        # Person/Demographics keywords (including age)
        person_keywords = ["age", "birth", "year", "old", "demographics", "gender", "race", "ethnicity", "patient count", "how many patients"]
        if any(keyword in question_lower for keyword in person_keywords):
            relevant_domains.append("Person")
        
        # Medical conditions keywords
        condition_keywords = ["condition", "diagnosis", "disease", "disorder", "hypertension", "diabetes", "cancer", "infection", "syndrome"]
        if any(keyword in question_lower for keyword in condition_keywords):
            relevant_domains.append("Condition")
        
        # Drug/medication keywords  
        drug_keywords = ["drug", "medication", "prescription", "medicine", "treatment", "therapy", "pill", "dose"]
        if any(keyword in question_lower for keyword in drug_keywords):
            relevant_domains.append("Drug")
        
        # Measurement/lab keywords
        measurement_keywords = ["lab", "test", "measurement", "result", "value", "level", "blood", "vital", "pressure", "glucose"]
        if any(keyword in question_lower for keyword in measurement_keywords):
            relevant_domains.append("Measurement")
        
        # Observation keywords
        observation_keywords = ["observation", "note", "finding", "history", "family history", "social"]
        if any(keyword in question_lower for keyword in observation_keywords):
            relevant_domains.append("Observation")
        
        # Default to Person if asking about patients but no specific domain found
        if not relevant_domains and any(keyword in question_lower for keyword in ["patient", "person", "people", "individual"]):
            relevant_domains.append("Person")
        
        return relevant_domains
    
    def _find_relevant_templates(self, question: str, domains: List[str], concepts: List[str], query_type: str) -> Dict[str, OMOPQueryTemplate]:
        """Finds query templates relevant to the extracted context."""
        relevant = {}
        question_lower = question.lower()
        
        for template_name, template in self.query_templates.items():
            # Match by domain
            if any(domain in template.domains for domain in domains):
                relevant[template_name] = template
            
            # Match by query type
            if query_type in template_name:
                relevant[template_name] = template
            
            # Match by concepts
            for concept in concepts:
                if concept in template.description or concept in template_name:
                    relevant[template_name] = template
        
        return relevant
    
    def _extract_lessons_from_failures(self, failed_attempts: List[Dict]) -> List[str]:
        """Extract actionable lessons from failure patterns"""
        lessons = set()
        
        for attempt in failed_attempts:
            error = attempt.get('error', '')
            
            if 'does not have a column named' in error:
                lessons.add("Verify column names exist in target tables before joining")
                
                # Extract specific column issues
                import re
                match = re.search(r'does not have a column named "(\w+)"', error)
                if match:
                    missing_col = match.group(1)
                    lessons.add(f"Column '{missing_col}' does not exist - check OMOP standard schema")
            
            elif 'Table with name' in error and 'does not exist' in error:
                lessons.add(f"Always use '{self.schema_prefix}.' prefix for table references")
                
            elif 'domain_id' in error:
                lessons.add("Verify concept.domain_id matches the clinical domain being queried")
            
            elif 'standard_concept' in error:
                lessons.add("Use standard_concept = 'S' to filter for standard OMOP concepts")
        
        return list(lessons)
    
    def learn_from_query_execution(self, sql: str, result: Any, success: bool, error_message: Optional[str] = None):
        """Enhanced learning from query execution with pattern recognition"""
        query_info = {
            "sql": sql,
            "success": success,
            "timestamp": datetime.now(),
            "error": error_message,
            "pattern_type": self._classify_query_pattern(sql)
        }
        
        if success:
            self.successful_queries.append(query_info)
            self._learn_from_success(sql, result)
        else:
            self.failed_queries.append(query_info)
            if error_message:
                self._learn_from_failure(sql, error_message)
    
    def _classify_query_pattern(self, sql: str) -> str:
        """Classify the type of query pattern"""
        sql_upper = sql.upper()
        
        if "COUNT(" in sql_upper and "DISTINCT" in sql_upper:
            return "count_distinct_patients"
        elif "COUNT(" in sql_upper:
            return "count_records"
        elif "JOIN" in sql_upper and "CONDITION_OCCURRENCE" in sql_upper:
            return "condition_analysis"
        elif "JOIN" in sql_upper and "DRUG_EXPOSURE" in sql_upper:
            return "drug_analysis"
        elif "JOIN" in sql_upper and "MEASUREMENT" in sql_upper:
            return "measurement_analysis"
        else:
            return "general_query"
    
    def _learn_from_success(self, sql: str, result: Any):
        """Learn patterns from successful queries"""
        pattern_type = self._classify_query_pattern(sql)
        
        # Update template success rates
        for template_name, template in self.query_templates.items():
            if self._sql_matches_template(sql, template):
                template.success_rate = min(1.0, template.success_rate + 0.1)
                template.usage_count += 1
                template.examples.append({"sql": sql, "timestamp": datetime.now()})
    
    def _sql_matches_template(self, sql: str, template: OMOPQueryTemplate) -> bool:
        """Check if a SQL query matches a template pattern"""
        sql_upper = sql.upper()
        template_upper = template.sql_template.upper()
        
        # Simple pattern matching - could be enhanced with more sophisticated NLP
        key_phrases = ["COUNT(DISTINCT", "JOIN", "WHERE"]
        template_phrases = [phrase for phrase in key_phrases if phrase in template_upper]
        
        return all(phrase in sql_upper for phrase in template_phrases)
    
    def _learn_from_failure(self, sql: str, error_message: str):
        """Enhanced learning from query failures"""
        # Extract table and column information from errors
        if "does not have a column named" in error_message:
            table_match = re.search(r'Table "(\w+)" does not have a column named "(\w+)"', error_message)
            if table_match:
                table_alias, missing_column = table_match.groups()
                
                # Update our knowledge about what columns DON'T exist
                for table_name, omop_table in self.omop_tables.items():
                    if table_alias.lower() in table_name.lower():
                        # Mark this column as confirmed missing
                        if "confirmed_missing_columns" not in omop_table.__dict__:
                            omop_table.confirmed_missing_columns = set()
                        omop_table.confirmed_missing_columns.add(missing_column)
                        logger.info(f"Learned: {table_name} does NOT have column {missing_column}")
    
    def _is_successful_result(self, result: Any) -> bool:
        """Check if MCP result indicates success"""
        if isinstance(result, dict):
            return not result.get("isError", False)
        return result is not None
    
    def _extract_columns_from_result(self, result: Any) -> Dict[str, str]:
        """Extract column information from query result"""
        # This would parse the actual MCP result format
        # Implementation depends on the specific MCP result structure
        columns = {}
        # TODO: Implement based on actual MCP result format
        return columns
    
    def _compare_with_standard(self, omop_table: OMOPTable, actual_columns: Dict[str, str]) -> Dict[str, Any]:
        """Compare actual schema with OMOP standard"""
        differences = {}
        
        # Find missing expected columns
        expected_cols = set(omop_table.standard_columns.keys())
        actual_cols = set(actual_columns.keys())
        
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols
        
        if missing_cols:
            differences["missing_columns"] = list(missing_cols)
        if extra_cols:
            differences["extra_columns"] = list(extra_cols)
        
        return differences
    
    def _extract_count_from_result(self, result: Any) -> Optional[int]:
        """Extract row count from COUNT query result"""
        # TODO: Implement based on actual MCP result format
        return None

    def update_schema_from_discovery(self, discovered_columns: List[Dict[str, Any]]):
        """
        Updates the in-memory OMOP table schemas based on a single, efficient
        discovery query to the information_schema.
        """
        tables_updated = set()
        for row in discovered_columns:
            table_name = row.get('table_name')
            column_name = row.get('column_name')
            data_type = row.get('data_type')

            if table_name and column_name and table_name in self.omop_tables:
                # Update the actual_columns dictionary for the table
                if table_name not in tables_updated:
                    self.omop_tables[table_name].actual_columns = {} # Clear previous knowledge
                    tables_updated.add(table_name)
                
                if data_type:
                    self.omop_tables[table_name].actual_columns[column_name] = data_type
        
        logger.info(f"World Model schema updated for {len(tables_updated)} tables from database discovery.")

    async def perform_smart_discovery(self, mcp_manager) -> bool:
        """
        Performs a single, efficient query to discover the actual schema of key OMOP tables
        and updates the world model.
        """
        core_tables = [
            'person', 'condition_occurrence', 'drug_exposure', 
            'measurement', 'observation', 'concept', 'visit_occurrence'
        ]
        
        # DuckDB uses upper-case for information_schema
        discovery_query = f"""
        SELECT
            LOWER(table_name) as table_name,
            LOWER(column_name) as column_name,
            data_type
        FROM information_schema.columns
        WHERE table_schema = '{self.schema_prefix}'
          AND table_name IN ({', '.join([f"'{t}'" for t in core_tables])})
        ORDER BY table_name, column_name;
        """
        
        try:
            logger.info("Performing smart schema discovery...")
            result = await mcp_manager.call_tool("omop_db_server:Select_Query", {"query": discovery_query})
            
            # The result parsing logic is now in the agent, so we need a simplified check here
            if result and isinstance(result, dict) and not result.get("isError"):
                # This part is tricky as the full result parsing is in the agent.
                # We'll assume a successful query returns a result with a 'result' key
                # that can be parsed. A more robust solution might involve moving
                # result parsing to a shared utility.
                
                raw_result_str = result.get('result', '{}')
                parsed_result = json.loads(raw_result_str)
                content = parsed_result.get('content', [])
                
                if content and content[0].get('type') == 'text':
                    text_data = content[0]['text']
                    lines = text_data.strip().split('\n')
                    if len(lines) > 1:
                        headers = [h.strip() for h in lines[0].split('\t')]
                        discovered_columns = [dict(zip(headers, line.split('\t'))) for line in lines[1:]]
                        
                        self.update_schema_from_discovery(discovered_columns)
                        return True
            
            logger.warning("Smart schema discovery did not return a valid result.")
            return False

        except Exception as e:
            logger.error(f"Smart schema discovery failed: {e}", exc_info=True)
            return False

class OMOPDatabaseAgent(MCPDiscoveryMixin, OllamaReasoningMixin, MedicalAgent, RequestHandler):
    """
    An intelligent agent that performs text-to-SQL conversion and then calls an 
    external MCP tool to execute the query against the OMOP CDM.
    """
    def __init__(self, agent_id: str, mcp_servers: List[MCPServer], ollama_model: str = "llama3.1:8b"):
        super().__init__(
            agent_id=agent_id,
            agent_type="omop_database_client",
            capabilities=["omop_text_to_sql", "mcp_tool_caller"],
            world_model=None,
            mcp_servers=mcp_servers,
            model_name=ollama_model
        )
        # Now that the mixins are initialized, we can access self.mcp_manager
        self.mcp_manager = self.mcp_manager
        self.omop_world_model = ComprehensiveOMOPWorldModel()
        print("[OMOPDatabaseAgent] Initialized with Comprehensive OMOP World Model.")

    async def _async_init(self):
        """Asynchronous initialization to discover MCP servers and validate connection."""
        print("[OMOPDatabaseAgent] Starting MCP server discovery...")
        await self.mcp_manager.discover_servers()
        print("[OMOPDatabaseAgent] MCP server discovery completed.")
        
        # Perform a single, efficient discovery of the live database schema
        print("[OMOPDatabaseAgent] 🧠 Performing smart schema discovery...")
        success = await self.omop_world_model.perform_smart_discovery(self.mcp_manager)
        if success:
            print("[OMOPDatabaseAgent] ✅ World model updated with live schema.")
        else:
            print("[OMOPDatabaseAgent] ⚠️ Warning: Could not perform smart discovery. Using default OMOP schema.")
        
        print("[OMOPDatabaseAgent] 🚀 Fast initialization completed.")

    @classmethod
    async def create(cls, agent_id: str, mcp_servers: List[MCPServer], ollama_model: str = "llama3.1:8b"):
        """Factory method to create and asynchronously initialize the agent."""
        agent = cls(agent_id, mcp_servers, ollama_model)
        await agent._async_init()
        return agent

    async def perceive(self, observation: Any) -> ProcessedObservation:
        """Parses the incoming A2A Message to extract the user's question."""
        logger.debug(f"[OMOPDatabaseAgent] Perceiving observation: {observation}")
        if isinstance(observation, Message):
            try:
                part = observation.parts[0].root
                if isinstance(part, TextPart):
                    logger.debug(f"[OMOPDatabaseAgent] Received TextPart: {part.text}")
                    request_data = json.loads(part.text)
                    request = OMOPQueryRequest(**request_data)
                    return ProcessedObservation(data=request, timestamp=0, source="a2a_message")
                else:
                    logger.warning(f"[OMOPDatabaseAgent] Received non-text part: {part}")
                    return ProcessedObservation(data=None, timestamp=0, source="a2a_message")
            except (json.JSONDecodeError, IndexError) as e:
                logger.error(f"[OMOPDatabaseAgent] Error decoding message: {e}")
                return ProcessedObservation(data=None, timestamp=0, source="a2a_message")
        logger.warning(f"[OMOPDatabaseAgent] Unsupported observation type: {type(observation)}")
        return ProcessedObservation(data=None, timestamp=0, source="unsupported_observation")

    async def learn(self, state: MentalState, observation: ProcessedObservation) -> MentalState:
        """Updates the mental state with the perceived observation."""
        logger.debug(f"[OMOPDatabaseAgent] Learning from observation: {observation.data}")
        if observation.data:
            state.memory["current_nl_query"] = observation.data.question
        logger.debug(f"[OMOPDatabaseAgent] Updated mental state: {state.memory}")
        return state

    async def reason(self, state: MentalState) -> Action:
        """Generates a SQL query based on the natural language question."""
        nl_query = state.memory.get("current_nl_query")
        logger.debug(f"[OMOPDatabaseAgent] Reasoning with query: {nl_query}")
        if not nl_query:
            logger.error("[OMOPDatabaseAgent] No natural language query found in mental state.")
            return Action(action_type="error", parameters={"message": "No natural language query to process."})

        # Check for previous failed attempts
        failed_attempts = state.memory.get("failed_sql_attempts", [])
        
        # Step 1: Extract structured context from the query
        print("[OMOPDatabaseAgent] 🧠 Step 1/3: Extracting query context...")
        extracted_context = await self._extract_query_context(nl_query)
        
        # Step 2: Get hyper-focused context from the world model
        print("[OMOPDatabaseAgent] 📚 Step 2/3: Retrieving targeted world model context...")
        world_model_context = self.omop_world_model.get_comprehensive_context(
            nl_query, extracted_context, failed_attempts
        )
        
        # Step 3: Generate the SQL query
        print("[OMOPDatabaseAgent]  SQL Step 3/3: Generating SQL query with focused context...")
        if failed_attempts:
            # Build a detailed prompt for refinement
            prompt = self._build_refinement_prompt(nl_query, world_model_context, failed_attempts)
        else:
            # Build a concise, focused prompt for the first attempt
            prompt = self._build_initial_prompt(nl_query, world_model_context)

        ollama_response = await self.ollama_reason(prompt["prompt"], system_prompt=prompt["system_prompt"], include_tools=False)

        # Extract SQL from response
        sql_query = self._extract_sql_from_response(ollama_response)
        logger.debug(f"[OMOPDatabaseAgent] Generated SQL query: {sql_query}")

        action = Action(
            action_type="call_mcp_tool", 
            parameters={
                "tool_id": "omop_db_server:Select_Query",
                "tool_parameters": {"query": sql_query}
            }
        )
        logger.debug(f"[OMOPDatabaseAgent] Generated action: {action}")
        return action

    def _build_initial_prompt(self, nl_query: str, context: str) -> Dict[str, str]:
        """Builds a concise and focused prompt for the first attempt."""
        system_prompt = """
You are an expert SQL generator for OMOP CDM v5.4 using DuckDB syntax.
Your goal is to generate a single, valid, and executable SQL query.

CRITICAL RULES:
1.  **Start with SELECT only.** No WITH clauses, CTEs, or multiple statements.
2.  **Always use the `base.` schema prefix** for all tables (e.g., `base.person`).
3.  **Use `EXTRACT()` for dates**, not `date_part()` (e.g., `EXTRACT(YEAR FROM CURRENT_DATE)`).
4.  **Filter concepts** using `standard_concept = 'S'`.
5.  **Use `LOWER()` and `LIKE`** for case-insensitive text matching.
6.  **For age calculations**, use `(EXTRACT(YEAR FROM CURRENT_DATE) - year_of_birth)`.

Use the provided context to write the query. Generate ONLY the SQL query.
        """.strip()
        
        prompt = f"""
### CONTEXT
{context}

### QUESTION
"{nl_query}"

### SQL QUERY
        """.strip()
        
        return {"system_prompt": system_prompt, "prompt": prompt}

    def _build_refinement_prompt(self, nl_query: str, context: str, failed_attempts: List[Dict]) -> Dict[str, str]:
        """Builds a detailed prompt for refining a failed query."""
        system_prompt = self._build_initial_prompt(nl_query, context)["system_prompt"]
        
        failure_context = "\n\n### PREVIOUS FAILED ATTEMPTS\n"
        for i, attempt in enumerate(failed_attempts, 1):
            failure_context += f"Attempt {i} SQL: {attempt['sql']}\n"
            failure_context += f"Attempt {i} Error: {attempt['error']}\n---\n"
        
        prompt = f"""
### CONTEXT
{context}

{failure_context}

### INSTRUCTIONS
Analyze the previous errors and the context to generate a corrected SQL query for the original question. Pay close attention to table and column names, join logic, and function syntax.

### QUESTION
"{nl_query}"

### CORRECTED SQL QUERY
        """.strip()
        
        return {"system_prompt": system_prompt, "prompt": prompt}

    async def _extract_query_context(self, nl_query: str) -> Dict[str, Any]:
        """
        Extracts structured context from the query using an LLM call.
        Returns a dictionary with domains, concepts, query_type, and tables.
        """
        extraction_prompt = f"""
Analyze this medical question and extract key information in a JSON format:

Question: "{nl_query}"

Respond with a single JSON object containing these keys:
- "domains": list of relevant medical domains (e.g., "Condition", "Drug", "Person").
- "concepts": list of specific medical concepts (e.g., "Essential Hypertension", "metformin").
- "query_type": a simple category (e.g., "count", "average", "list", "distribution").
- "tables": list of likely OMOP tables needed (e.g., "condition_occurrence", "drug_exposure").

Example for "What is the most common disease?":
{{
  "domains": ["Condition"],
  "concepts": ["disease"],
  "query_type": "list",
  "tables": ["condition_occurrence", "concept"]
}}

Example for "Average age of female patients?":
{{
  "domains": ["Person"],
  "concepts": ["age", "female"],
  "query_type": "average",
  "tables": ["person", "concept"]
}}

Your JSON response:
        """
        
        system_prompt = "You are an OMOP CDM expert. Extract key information from medical questions and respond ONLY with a valid JSON object."
        
        default_context = {
            "domains": [], "concepts": [nl_query], "query_type": "unknown", "tables": []
        }

        try:
            response = await self.ollama_reason(extraction_prompt, system_prompt=system_prompt, include_tools=False)
            response_text = self._extract_sql_from_response(response)  # Reusing this to get the core text
            
            # Find the JSON object in the response text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                extracted_json = json_match.group(0)
                parsed_context = json.loads(extracted_json)
                logger.debug(f"[OMOPDatabaseAgent] Extracted structured context: {parsed_context}")
                return parsed_context
            else:
                logger.warning("[OMOPDatabaseAgent] No JSON object found in context extraction response.")
                return default_context
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"[OMOPDatabaseAgent] Context extraction failed to parse JSON: {e}")
            return default_context
        except Exception as e:
            logger.error(f"[OMOPDatabaseAgent] An unexpected error occurred during context extraction: {e}")
            return default_context

    def _extract_sql_from_response(self, ollama_response: Any) -> str:
        """Extract SQL query from Ollama response."""
        sql_query = ""
        if isinstance(ollama_response, dict):
            if 'response' in ollama_response:
                sql_query = ollama_response['response']
            elif 'message' in ollama_response and isinstance(ollama_response['message'], dict):
                sql_query = ollama_response['message'].get('content', '')
            elif 'message' in ollama_response:
                sql_query = str(ollama_response['message'])
            elif 'content' in ollama_response:
                sql_query = ollama_response['content']
            else:
                sql_query = str(ollama_response)
        else:
            sql_query = str(ollama_response)
        
        sql_query = sql_query.strip()
        
        # Extract SQL from markdown code blocks if present
        if '```sql' in sql_query:
            sql_matches = re.findall(r'```sql\s*(.*?)\s*```', sql_query, re.DOTALL | re.IGNORECASE)
            if sql_matches:
                sql_query = sql_matches[-1].strip()
        elif '```' in sql_query:
            code_matches = re.findall(r'```\s*(.*?)\s*```', sql_query, re.DOTALL)
            if code_matches:
                # Look for the one that looks most like SQL
                for match in code_matches:
                    match_upper = match.strip().upper()
                    if any(keyword in match_upper for keyword in ['SELECT', 'FROM', 'WHERE', 'JOIN']):
                        sql_query = match.strip()
                        break
        
        return sql_query

    async def execute(self, action: Action) -> ActionResult:
        """Executes the generated action, learning from MCP server feedback."""
        logger.debug(f"[OMOPDatabaseAgent] Executing action: {action.action_type}")
        
        if action.action_type == "call_mcp_tool":
            tool_id = action.parameters.get("tool_id")
            tool_params = action.parameters.get("tool_parameters")

            if not tool_id or not isinstance(tool_id, str):
                return ActionResult(success=False, error="Invalid or missing tool_id.")
            if not tool_params or not isinstance(tool_params, dict):
                return ActionResult(success=False, error="Invalid or missing tool_parameters.")

            try:
                logger.debug(f"[OMOPDatabaseAgent] Calling MCP tool '{tool_id}' with params: {tool_params}")
                result = await self.call_mcp_tool(tool_id, tool_params)
                logger.debug(f"[OMOPDatabaseAgent] MCP tool result: {result}")
                
                # Check if the result indicates an error
                if self._is_mcp_error_result(result):
                    # Extract error message and store for learning
                    error_message = self._extract_error_from_result(result)
                    logger.warning(f"[OMOPDatabaseAgent] MCP execution failed: {error_message}")
                    
                    # Feed error information to world model for learning
                    self.omop_world_model.learn_from_query_execution(tool_params.get("query", ""), result, False, error_message)
                    
                    # Store failed attempt in memory for next iteration
                    failed_sql = tool_params.get("query", "")
                    self.mental_state.memory["failed_sql_attempts"] = self.mental_state.memory.get("failed_sql_attempts", [])
                    self.mental_state.memory["failed_sql_attempts"].append({
                        "sql": failed_sql,
                        "error": error_message,
                        "attempt_number": len(self.mental_state.memory.get("failed_sql_attempts", [])) + 1
                    })
                    
                    # Check if we should try again or give up
                    max_attempts = 5 # Reduced from 10 for better interactive performance
                    if len(self.mental_state.memory.get("failed_sql_attempts", [])) >= max_attempts:
                        logger.error(f"[OMOPDatabaseAgent] Max attempts ({max_attempts}) reached")
                        return ActionResult(success=False, error=f"Failed to generate valid SQL after {max_attempts} attempts. Last error: {error_message}")
                    
                    # Try again with refinement
                    logger.info(f"[OMOPDatabaseAgent] Attempting to refine SQL (attempt {len(self.mental_state.memory.get('failed_sql_attempts', []))} of {max_attempts})")
                    refined_action = await self.reason(self.mental_state)
                    return await self.execute(refined_action)
                
                else:
                    # Success! Parse the actual result
                    actual_result = self._parse_successful_result(result)
                    logger.debug(f"[OMOPDatabaseAgent] Parsed successful result: {actual_result}")

                    response_data = OMOPQueryResponse(
                        generated_sql=tool_params.get("query", ""),
                        query_result=actual_result if isinstance(actual_result, list) else []
                    )
                    
                    # Clear failed attempts on success
                    if "failed_sql_attempts" in self.mental_state.memory:
                        del self.mental_state.memory["failed_sql_attempts"]
                    
                    # Learn from the successful query result
                    self.omop_world_model.learn_from_query_execution(tool_params.get("query", ""), actual_result, True)
                    
                    logger.info(f"[OMOPDatabaseAgent] Successfully executed SQL query")
                    return ActionResult(success=True, data=response_data.model_dump())
                    
            except Exception as e:
                logger.error(f"[OMOPDatabaseAgent] MCP Tool call failed: {e}", exc_info=True)
                return ActionResult(success=False, error=f"MCP Tool call failed: {str(e)}")
        
        logger.error(f"[OMOPDatabaseAgent] Unknown action type: {action.action_type}")
        return ActionResult(success=False, error=f"Unknown action type: {action.action_type}")

    def _is_mcp_error_result(self, result: Any) -> bool:
        """Check if MCP result indicates an error."""
        if isinstance(result, dict):
            # Check for explicit error flag
            if result.get("isError", False):
                return True
            # Check for error content in the result
            if result.get("result"):
                try:
                    import json
                    parsed_result = json.loads(result["result"])
                    if parsed_result.get("isError", False):
                        return True
                    # Check if content contains error messages
                    content = parsed_result.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text" and item.get("text"):
                                text = item["text"]
                                if any(error_indicator in text.lower() for error_indicator in 
                                      ["failed to execute", "error:", "binder error", "syntax error"]):
                                    return True
                except (json.JSONDecodeError, AttributeError):
                    pass
        return False

    def _extract_error_from_result(self, result: Any) -> str:
        """Extract error message from MCP result."""
        try:
            if isinstance(result, dict) and result.get("result"):
                import json
                parsed_result = json.loads(result["result"])
                content = parsed_result.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text" and item.get("text"):
                            return item["text"]
            return str(result)
        except:
            return str(result)

    def _parse_successful_result(self, result: Any) -> list:
        """Parse successful MCP result into structured data."""
        actual_result = []
        if result and isinstance(result, dict):
            if result.get("result"):
                import json
                try:
                    parsed_result = json.loads(result["result"])
                    if parsed_result.get("content") and isinstance(parsed_result["content"], list):
                        for content_item in parsed_result["content"]:
                            if content_item.get("type") == "text" and content_item.get("text"):
                                text_result = content_item["text"].strip()
                                lines = text_result.split('\n')
                                if len(lines) >= 2:
                                    headers = [h.strip() for h in lines[0].split('\t') if h.strip()]
                                    for line in lines[1:]:
                                        if line.strip():
                                            values = [v.strip() for v in line.split('\t')]
                                            if len(values) == len(headers):
                                                row_dict = dict(zip(headers, values))
                                                actual_result.append(row_dict)
                                            else:
                                                if len(headers) == 1 and len(values) == 1:
                                                    actual_result.append({headers[0]: values[0]})
                                                elif len(values) == 1:
                                                    actual_result.append({"result": values[0]})
                except json.JSONDecodeError as e:
                    logger.error(f"[OMOPDatabaseAgent] Failed to parse successful result: {e}")
            else:
                actual_result = result.get("result", result) if isinstance(result, dict) else result
        else:
            actual_result = result
        
        return actual_result if isinstance(actual_result, list) else []

    def build_agent_card(self) -> AgentCard:
        return AgentCard(name=self.agent_name, description=self.agent_description, version="1.0.0", url=f"http://localhost:8002/{self.agent_id}", capabilities=AgentCapabilities(streaming=False), skills=[], default_input_modes=[], default_output_modes=[])

    # Implement abstract methods from RequestHandler
    async def on_message_send(self, params: Any, context: Optional[ServerCallContext] = None) -> Message:
        # Delegate to the agent's PCE cycle
        observation = await self.perceive(params.message)
        self.mental_state = await self.learn(self.mental_state, observation)
        action = await self.reason(self.mental_state)
        result = await self.execute(action)

        if result.success:
            # If successful, return the data as a JSON string in a TextPart
            message_content = json.dumps(result.data)
            return Message(
                message_id=str(uuid.uuid4()),
                role=Role.agent,
                parts=[Part(root=TextPart(kind="text", text=message_content))]
            )
        else:
            # If there's an error, return the error message as a JSON string in a TextPart
            error_content = json.dumps({"error": result.error})
            return Message(
                message_id=str(uuid.uuid4()),
                role=Role.agent,
                parts=[Part(root=TextPart(kind="text", text=error_content))]
            )

    async def on_message_send_stream(self, params: Any, context: Optional[ServerCallContext] = None) -> AsyncIterable[Any]:
        # Implement streaming logic if needed, otherwise raise NotImplementedError
        raise NotImplementedError("Streaming not implemented for OMOPDatabaseAgent")

    async def on_get_task(self, params: Any, context: Optional[ServerCallContext] = None) -> Any:
        raise NotImplementedError("Get task not implemented for OMOPDatabaseAgent")

    async def on_cancel_task(self, params: Any, context: Optional[ServerCallContext] = None) -> Any:
        raise NotImplementedError("Cancel task not implemented for OMOPDatabaseAgent")

    async def on_resubscribe_to_task(self, params: Any, context: Optional[ServerCallContext] = None) -> AsyncIterable[Any]:
        raise NotImplementedError("Resubscribe to task not implemented for OMOPDatabaseAgent")

    async def on_get_task_push_notification_config(self, params: Any, context: Optional[ServerCallContext] = None) -> Any:
        raise NotImplementedError("Get task push notification config not implemented for OMOPDatabaseAgent")

    async def on_set_task_push_notification_config(self, params: Any, context: Optional[ServerCallContext] = None) -> Any:
        raise NotImplementedError("Set task push notification config not implemented for OMOPDatabaseAgent")

    async def on_list_task_push_notification_config(self, params: Any, context: Optional[ServerCallContext] = None) -> Any:
        raise NotImplementedError("List task push notification config not implemented for OMOPDatabaseAgent")

    async def on_delete_task_push_notification_config(self, params: Any, context: Optional[ServerCallContext] = None) -> Any:
        raise NotImplementedError("Delete task push notification config not implemented for OMOPDatabaseAgent")
