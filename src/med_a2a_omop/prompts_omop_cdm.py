"""
OMOP CDM World Model Knowledge for Medical Queries
"""

OMOP_CDM_CONTEXT = """
## OMOP CDM Query Guidelines:

### Drug Queries:
1. **Always use RxNorm codes when specific drugs are mentioned**:
   - Search for drug names in the concept table with vocabulary_id = 'RxNorm'
   - Use concept_relationship to map to standard concepts
   - Use concept_ancestor for drug hierarchies

2. **For drug interaction/co-prescription queries**:
   - "Patients taking drug A and drug B within X days" means:
     - Find patients who have exposures to BOTH drugs
     - Where the time between the two drug exposures is ≤ X days
     - This requires a self-join on drug_exposure table

3. **Standard drug query pattern**:
   ```sql
   -- First find the drug concept IDs
   WITH drug_concepts AS (
     SELECT c.concept_id, c.concept_name
     FROM concept c
     WHERE c.vocabulary_id = 'RxNorm' 
       AND c.standard_concept = 'S'
       AND LOWER(c.concept_name) LIKE '%[drug_name]%'
   )
   ```

4. **For "within X days" temporal relationships**:
   - Use date arithmetic between two events
   - Pattern: `ABS(date1 - date2) <= X` or 
   - `GREATEST(date1, date2) - LEAST(date1, date2) <= INTERVAL 'X days'`

### Common Drug RxNorm Codes:
- hydrochlorothiazide 25 MG: RxNorm 310798
- lisinopril 10 MG: RxNorm 314076
- metformin: RxNorm varies by dosage
- simvastatin: RxNorm varies by dosage

### Query Patterns:

**Pattern 1: Single Drug Count**
```sql
SELECT COUNT(DISTINCT person_id) 
FROM drug_exposure de
JOIN concept c ON de.drug_concept_id = c.concept_id
WHERE c.concept_code = '[rxnorm_code]' 
  AND c.vocabulary_id = 'RxNorm';
```

**Pattern 2: Drug Co-prescription Within X Days**
```sql
SELECT COUNT(DISTINCT a.person_id)
FROM (
  SELECT person_id, drug_exposure_start_date 
  FROM drug_exposure 
  WHERE drug_concept_id IN (SELECT concept_id FROM concept WHERE concept_code = '[rxnorm1]')
) a
JOIN (
  SELECT person_id, drug_exposure_start_date 
  FROM drug_exposure 
  WHERE drug_concept_id IN (SELECT concept_id FROM concept WHERE concept_code = '[rxnorm2]')
) b ON a.person_id = b.person_id
WHERE ABS(EXTRACT(EPOCH FROM a.drug_exposure_start_date - b.drug_exposure_start_date)/86400) <= X;
```
"""

DRUG_INTERACTION_PROMPT = """
When the user asks about "patients taking drug A and drug B within X days":
1. This is a CO-PRESCRIPTION query, not separate drug queries
2. Generate a single SQL that finds patients with BOTH drugs
3. The temporal constraint means the drugs were prescribed within X days of EACH OTHER
4. Do NOT interpret as "in the last X days from today"
5. Use proper joins to ensure both drugs are present for the same patient
"""

def enhance_drug_query_prompt(original_question: str) -> str:
    """Enhance drug-related queries with OMOP CDM context."""
    
    # Keywords that indicate drug interaction queries
    interaction_keywords = ["and", "with", "together", "both", "combination"]
    temporal_keywords = ["within", "days", "period", "timeframe"]
    
    is_interaction = any(kw in original_question.lower() for kw in interaction_keywords)
    has_temporal = any(kw in original_question.lower() for kw in temporal_keywords)
    
    if is_interaction and has_temporal:
        return f"""
{DRUG_INTERACTION_PROMPT}

Original Question: {original_question}

IMPORTANT: This is asking for patients who took BOTH medications where the prescriptions were within the specified number of days of each other, NOT within that many days from today.
"""
    
    return original_question