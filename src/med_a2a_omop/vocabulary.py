"""
OMOP Vocabulary Integration Module

This module provides comprehensive access to OMOP vocabulary data for the semantic agent.
It loads and indexes all standard concepts from the OMOP vocabulary files to enable
proper concept code lookup and medical terminology standardization.
"""

import csv
import json
import sqlite3
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

@dataclass
class OMOPConcept:
    """Represents an OMOP concept with all relevant metadata."""
    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    standard_concept: str
    concept_code: str
    valid_start_date: str
    valid_end_date: str
    invalid_reason: Optional[str] = None

@dataclass
class ConceptMatch:
    """Represents a matched concept with confidence score."""
    concept: OMOPConcept
    match_type: str  # 'exact', 'partial', 'synonym', 'fuzzy'
    confidence: float
    matched_text: str

class OMOPVocabularyIndex:
    """
    Comprehensive OMOP vocabulary index for fast concept lookup.
    
    This class loads the entire OMOP vocabulary and creates multiple indexes
    for efficient concept searching by name, code, synonyms, etc.
    """
    
    def __init__(self, vocab_path: Optional[str] = None):
        # Import here to avoid circular import
        from .config import get_config

        if vocab_path is None:
            config = get_config()
            configured_path = config.get_vocabulary_path()
            if configured_path is None:
                raise ValueError(
                    "OMOP vocabulary path not found. Please:\n"
                    "1. Set 'vocabulary_path' in .medA2A.config.json, or\n"
                    "2. Set OMOP_VOCABULARY_PATH environment variable, or\n"
                    "3. Place vocabulary files in ~/Downloads/omop_vocab_current/"
                )
            vocab_path = str(configured_path)

        self.vocab_path = Path(vocab_path)
        self.concepts: Dict[int, OMOPConcept] = {}
        self.concepts_by_code: Dict[str, List[OMOPConcept]] = defaultdict(list)
        self.concepts_by_name: Dict[str, List[OMOPConcept]] = defaultdict(list) 
        self.concepts_by_domain: Dict[str, List[OMOPConcept]] = defaultdict(list)
        self.concepts_by_vocabulary: Dict[str, List[OMOPConcept]] = defaultdict(list)
        
        # Indexes for fast searching
        self.name_tokens: Dict[str, List[int]] = defaultdict(list)  # token -> concept_ids
        self.synonyms: Dict[int, List[str]] = defaultdict(list)  # concept_id -> synonyms
        
        # Cached database path for persistence
        self.cache_db_path = Path(".cache/vocabulary_index.db")
        self.cache_db_path.parent.mkdir(exist_ok=True)
        
        logger.info(f"Initializing OMOP vocabulary index from: {vocab_path}")
    
    def load_vocabulary(self, force_reload: bool = False) -> bool:
        """Load the OMOP vocabulary from CSV files."""
        try:
            # Check if we have cached data
            if not force_reload and self._load_from_cache():
                logger.info("Loaded vocabulary from cache")
                return True
            
            logger.info("Loading vocabulary from CSV files...")
            
            # Load main concepts
            self._load_concepts()
            
            # Load synonyms
            self._load_synonyms()
            
            # Build search indexes
            self._build_search_indexes()
            
            # Cache the loaded data
            self._save_to_cache()
            
            logger.info(f"Loaded {len(self.concepts)} concepts with indexes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {e}")
            return False
    
    def _load_concepts(self):
        """Load concepts from CONCEPT.csv, focusing on standard concepts."""
        concept_file = self.vocab_path / "CONCEPT.csv"
        
        if not concept_file.exists():
            raise FileNotFoundError(f"CONCEPT.csv not found at {concept_file}")
        
        logger.info("Loading standard concepts from CONCEPT.csv...")
        count = 0
        standard_count = 0
        
        with open(concept_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for row in reader:
                count += 1
                
                # Only load standard concepts (S) and classification concepts (C)
                standard_concept = row.get('standard_concept', '').strip()
                if standard_concept not in ['S', 'C']:
                    continue
                
                try:
                    concept = OMOPConcept(
                        concept_id=int(row['concept_id']),
                        concept_name=row['concept_name'].strip(),
                        domain_id=row['domain_id'].strip(),
                        vocabulary_id=row['vocabulary_id'].strip(),
                        concept_class_id=row['concept_class_id'].strip(),
                        standard_concept=standard_concept,
                        concept_code=row['concept_code'].strip(),
                        valid_start_date=row['valid_start_date'].strip(),
                        valid_end_date=row['valid_end_date'].strip(),
                        invalid_reason=row.get('invalid_reason', '').strip() or None
                    )
                    
                    # Add to main index
                    self.concepts[concept.concept_id] = concept
                    
                    # Add to secondary indexes
                    self.concepts_by_code[concept.concept_code].append(concept)
                    self.concepts_by_name[concept.concept_name.lower()].append(concept)
                    self.concepts_by_domain[concept.domain_id].append(concept)
                    self.concepts_by_vocabulary[concept.vocabulary_id].append(concept)
                    
                    standard_count += 1
                    
                    if standard_count % 100000 == 0:
                        logger.info(f"Loaded {standard_count} standard concepts...")
                        
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid concept row {count}: {e}")
                    continue
        
        logger.info(f"Loaded {standard_count} standard concepts out of {count} total")
    
    def _load_synonyms(self):
        """Load concept synonyms from CONCEPT_SYNONYM.csv."""
        synonym_file = self.vocab_path / "CONCEPT_SYNONYM.csv"
        
        if not synonym_file.exists():
            logger.warning(f"CONCEPT_SYNONYM.csv not found at {synonym_file}")
            return
        
        logger.info("Loading concept synonyms...")
        count = 0
        
        with open(synonym_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for row in reader:
                try:
                    concept_id = int(row['concept_id'])
                    synonym_name = row['concept_synonym_name'].strip()
                    
                    # Only load synonyms for concepts we have
                    if concept_id in self.concepts:
                        self.synonyms[concept_id].append(synonym_name)
                        count += 1
                        
                        if count % 50000 == 0:
                            logger.info(f"Loaded {count} synonyms...")
                            
                except (ValueError, KeyError) as e:
                    continue
        
        logger.info(f"Loaded {count} synonyms for {len(self.synonyms)} concepts")
    
    def _build_search_indexes(self):
        """Build token-based search indexes for fast concept lookup."""
        logger.info("Building search indexes...")
        
        for concept_id, concept in self.concepts.items():
            # Tokenize concept name
            tokens = self._tokenize_text(concept.concept_name)
            for token in tokens:
                self.name_tokens[token].append(concept_id)
            
            # Tokenize synonyms
            for synonym in self.synonyms.get(concept_id, []):
                tokens = self._tokenize_text(synonym)
                for token in tokens:
                    self.name_tokens[token].append(concept_id)
        
        logger.info(f"Built search index with {len(self.name_tokens)} tokens")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for search indexing."""
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove very short tokens (< 3 characters) except numbers
        tokens = [t for t in tokens if len(t) >= 3 or t.isdigit()]
        
        return tokens
    
    def _save_to_cache(self):
        """Save vocabulary index to SQLite cache."""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            
            # Create tables
            conn.execute('''
                CREATE TABLE IF NOT EXISTS concepts (
                    concept_id INTEGER PRIMARY KEY,
                    concept_name TEXT,
                    domain_id TEXT,
                    vocabulary_id TEXT,
                    concept_class_id TEXT,
                    standard_concept TEXT,
                    concept_code TEXT,
                    valid_start_date TEXT,
                    valid_end_date TEXT,
                    invalid_reason TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS indexes (
                    name TEXT PRIMARY KEY,
                    data BLOB
                )
            ''')
            
            # Save concepts
            concept_data = [
                (c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id,
                 c.concept_class_id, c.standard_concept, c.concept_code,
                 c.valid_start_date, c.valid_end_date, c.invalid_reason)
                for c in self.concepts.values()
            ]
            
            conn.executemany('''
                INSERT OR REPLACE INTO concepts VALUES (?,?,?,?,?,?,?,?,?,?)
            ''', concept_data)
            
            # Save indexes as pickled data
            indexes_to_save = {
                'concepts_by_code': dict(self.concepts_by_code),
                'concepts_by_name': dict(self.concepts_by_name),
                'concepts_by_domain': dict(self.concepts_by_domain),
                'concepts_by_vocabulary': dict(self.concepts_by_vocabulary),
                'name_tokens': dict(self.name_tokens),
                'synonyms': dict(self.synonyms)
            }
            
            for name, data in indexes_to_save.items():
                conn.execute('INSERT OR REPLACE INTO indexes VALUES (?, ?)',
                           (name, pickle.dumps(data)))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved vocabulary cache to {self.cache_db_path}")
            
        except Exception as e:
            logger.error(f"Failed to save vocabulary cache: {e}")
    
    def _load_from_cache(self) -> bool:
        """Load vocabulary index from SQLite cache."""
        try:
            if not self.cache_db_path.exists():
                return False
            
            conn = sqlite3.connect(self.cache_db_path)
            
            # Load concepts
            cursor = conn.execute('SELECT * FROM concepts')
            for row in cursor:
                concept = OMOPConcept(*row)
                self.concepts[concept.concept_id] = concept
            
            # Load indexes
            cursor = conn.execute('SELECT name, data FROM indexes')
            for name, data_blob in cursor:
                data = pickle.loads(data_blob)
                
                if name == 'concepts_by_code':
                    self.concepts_by_code = defaultdict(list, data)
                elif name == 'concepts_by_name':
                    self.concepts_by_name = defaultdict(list, data)
                elif name == 'concepts_by_domain':
                    self.concepts_by_domain = defaultdict(list, data)
                elif name == 'concepts_by_vocabulary':
                    self.concepts_by_vocabulary = defaultdict(list, data)
                elif name == 'name_tokens':
                    self.name_tokens = defaultdict(list, data)
                elif name == 'synonyms':
                    self.synonyms = defaultdict(list, data)
            
            conn.close()
            
            if self.concepts:
                logger.info(f"Loaded {len(self.concepts)} concepts from cache")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to load vocabulary cache: {e}")
            return False
    
    def search_concepts(self, 
                       query: str, 
                       domain_filter: Optional[str] = None,
                       vocabulary_filter: Optional[str] = None,
                       limit: int = 10) -> List[ConceptMatch]:
        """
        Search for concepts matching the query string.
        
        Args:
            query: Search query string
            domain_filter: Filter by domain (e.g., 'Drug', 'Condition')
            vocabulary_filter: Filter by vocabulary (e.g., 'RxNorm', 'SNOMED')
            limit: Maximum number of results to return
            
        Returns:
            List of ConceptMatch objects ordered by relevance
        """
        if not query.strip():
            return []
        
        matches = []
        query_lower = query.lower().strip()
        query_tokens = self._tokenize_text(query)
        
        # 1. Exact name match
        exact_matches = self.concepts_by_name.get(query_lower, [])
        for concept in exact_matches:
            if self._filter_concept(concept, domain_filter, vocabulary_filter):
                matches.append(ConceptMatch(
                    concept=concept,
                    match_type='exact',
                    confidence=1.0,
                    matched_text=concept.concept_name
                ))
        
        # 2. Token-based search
        candidate_concept_ids = set()
        for token in query_tokens:
            candidate_concept_ids.update(self.name_tokens.get(token, []))
        
        # Score candidates
        for concept_id in candidate_concept_ids:
            if concept_id in [m.concept.concept_id for m in matches]:
                continue  # Skip already matched
            
            concept = self.concepts[concept_id]
            if not self._filter_concept(concept, domain_filter, vocabulary_filter):
                continue
            
            # Calculate match score
            score = self._calculate_match_score(query, concept, query_tokens)
            if score > 0.3:  # Minimum threshold
                match_type = 'partial' if score < 0.8 else 'exact'
                matches.append(ConceptMatch(
                    concept=concept,
                    match_type=match_type,
                    confidence=score,
                    matched_text=concept.concept_name
                ))
        
        # 3. Synonym search
        for concept_id, synonyms in self.synonyms.items():
            if concept_id in [m.concept.concept_id for m in matches]:
                continue
            
            concept = self.concepts[concept_id]
            if not self._filter_concept(concept, domain_filter, vocabulary_filter):
                continue
            
            for synonym in synonyms:
                if query_lower in synonym.lower():
                    score = len(query_lower) / len(synonym) if synonym else 0
                    if score > 0.3:
                        matches.append(ConceptMatch(
                            concept=concept,
                            match_type='synonym',
                            confidence=score * 0.9,  # Slight penalty for synonym
                            matched_text=synonym
                        ))
                        break
        
        # Sort by confidence and return top matches
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches[:limit]
    
    def _filter_concept(self, 
                       concept: OMOPConcept, 
                       domain_filter: Optional[str],
                       vocabulary_filter: Optional[str]) -> bool:
        """Filter concept based on domain and vocabulary."""
        if domain_filter and concept.domain_id != domain_filter:
            return False
        if vocabulary_filter and concept.vocabulary_id != vocabulary_filter:
            return False
        return True
    
    def _calculate_match_score(self, query: str, concept: OMOPConcept, query_tokens: List[str]) -> float:
        """Calculate relevance score for a concept match."""
        concept_name_lower = concept.concept_name.lower()
        concept_tokens = self._tokenize_text(concept.concept_name)
        
        # Exact substring match gets high score
        if query.lower() in concept_name_lower:
            return min(1.0, len(query) / len(concept_name_lower) + 0.5)
        
        # Token overlap score
        if not query_tokens or not concept_tokens:
            return 0.0
        
        matching_tokens = set(query_tokens) & set(concept_tokens)
        token_score = len(matching_tokens) / len(query_tokens)
        
        # Boost score for important vocabularies
        vocab_boost = 1.0
        if concept.vocabulary_id in ['RxNorm', 'SNOMED', 'LOINC']:
            vocab_boost = 1.1
        
        # Boost score for standard concepts
        standard_boost = 1.1 if concept.standard_concept == 'S' else 1.0
        
        return token_score * vocab_boost * standard_boost
    
    def get_concept_by_code(self, concept_code: str, vocabulary_id: str = None) -> Optional[OMOPConcept]:
        """Get concept by exact concept code and optionally vocabulary."""
        candidates = self.concepts_by_code.get(concept_code, [])
        
        if vocabulary_id:
            candidates = [c for c in candidates if c.vocabulary_id == vocabulary_id]
        
        # Prefer standard concepts
        standard_candidates = [c for c in candidates if c.standard_concept == 'S']
        if standard_candidates:
            return standard_candidates[0]
        
        return candidates[0] if candidates else None
    
    def get_concepts_by_domain(self, domain_id: str) -> List[OMOPConcept]:
        """Get all concepts in a specific domain."""
        return self.concepts_by_domain.get(domain_id, [])
    
    def get_drug_concepts(self, query: str, limit: int = 10) -> List[ConceptMatch]:
        """Search specifically for drug concepts."""
        return self.search_concepts(
            query=query,
            domain_filter='Drug',
            vocabulary_filter='RxNorm',
            limit=limit
        )
    
    def get_condition_concepts(self, query: str, limit: int = 10) -> List[ConceptMatch]:
        """Search specifically for condition concepts."""
        return self.search_concepts(
            query=query,
            domain_filter='Condition',
            vocabulary_filter='SNOMED',
            limit=limit
        )


# Global vocabulary index instance
_vocabulary_index: Optional[OMOPVocabularyIndex] = None

def get_vocabulary_index() -> OMOPVocabularyIndex:
    """Get the global vocabulary index, loading it if necessary."""
    global _vocabulary_index
    
    if _vocabulary_index is None:
        _vocabulary_index = OMOPVocabularyIndex()
        _vocabulary_index.load_vocabulary()
    
    return _vocabulary_index

def clear_vocabulary_cache():
    """Clear the vocabulary cache to force reload."""
    global _vocabulary_index
    _vocabulary_index = None
    
    cache_path = Path(".cache/vocabulary_index.db")
    if cache_path.exists():
        cache_path.unlink()
        logger.info("Cleared vocabulary cache")
