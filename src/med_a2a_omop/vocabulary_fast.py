"""
Fast OMOP Vocabulary Integration - Optimized for Speed

Ultra-fast vocabulary system focused on essential concepts only:
- RxNorm drugs, SNOMED conditions, CPT4 procedures
- Target: <10s loading, <5ms searches
- Memory-efficient with selective loading
"""

import csv
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

@dataclass
class FastConcept:
    """Lightweight concept for fast lookups."""
    concept_id: int
    concept_name: str
    concept_code: str
    vocabulary_id: str
    domain_id: str

@dataclass 
class FastMatch:
    """Fast concept match result."""
    concept: FastConcept
    confidence: float

class FastVocabularyIndex:
    """Ultra-fast vocabulary index for essential medical concepts."""
    
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
        
        # Core indexes
        self.concepts: Dict[int, FastConcept] = {}
        self.name_to_concepts: Dict[str, List[FastConcept]] = defaultdict(list)
        self.drug_names: Dict[str, List[FastConcept]] = defaultdict(list)
        self.condition_names: Dict[str, List[FastConcept]] = defaultdict(list)
        
        # Cache
        self.cache_path = Path(".cache/fast_vocabulary.pkl")
        self.cache_path.parent.mkdir(exist_ok=True)
        
        logger.info(f"Fast vocabulary initialized from: {vocab_path}")
    
    def load_vocabulary(self, force_reload: bool = False) -> bool:
        """Load essential vocabulary with maximum speed optimization."""
        try:
            if not force_reload and self._load_from_cache():
                logger.info(f"✅ Loaded {len(self.concepts)} concepts from cache")
                return True
            
            logger.info("🚀 Fast loading essential concepts...")
            self._load_essential_concepts()
            self._build_indexes()
            self._save_to_cache()
            
            logger.info(f"✅ Loaded {len(self.concepts)} essential concepts")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {e}")
            return False
    
    def _load_essential_concepts(self):
        """Load only RxNorm drugs + SNOMED conditions + CPT4 procedures."""
        concept_file = self.vocab_path / "CONCEPT.csv"
        
        if not concept_file.exists():
            raise FileNotFoundError(f"CONCEPT.csv not found")
        
        # Load essential medical concepts
        targets = {
            ('RxNorm', 'Drug'),           # Drugs
            ('SNOMED', 'Condition'),      # Conditions/diseases
            ('CPT4', 'Procedure'),        # Procedures
            ('LOINC', 'Measurement'),     # Lab tests/measurements
        }
        
        count = 0
        loaded = 0
        
        with open(concept_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for row in reader:
                count += 1
                
                # Only standard concepts in target vocabularies/domains
                if (row.get('standard_concept', '').strip() != 'S' or
                    (row.get('vocabulary_id', '').strip(), row.get('domain_id', '').strip()) not in targets):
                    continue
                
                try:
                    concept = FastConcept(
                        concept_id=int(row['concept_id']),
                        concept_name=row['concept_name'].strip(),
                        concept_code=row['concept_code'].strip(),
                        vocabulary_id=row['vocabulary_id'].strip(),
                        domain_id=row['domain_id'].strip()
                    )
                    
                    self.concepts[concept.concept_id] = concept
                    loaded += 1
                    
                    if loaded % 50000 == 0:
                        logger.info(f"Loaded {loaded} concepts...")
                        
                except (ValueError, KeyError):
                    continue
        
        logger.info(f"Loaded {loaded}/{count} essential concepts")
    
    def _build_indexes(self):
        """Build optimized lookup indexes."""
        for concept in self.concepts.values():
            name_lower = concept.concept_name.lower()
            self.name_to_concepts[name_lower].append(concept)
            
            if concept.domain_id == 'Drug':
                self.drug_names[name_lower].append(concept)
            elif concept.domain_id == 'Condition':
                self.condition_names[name_lower].append(concept)
    
    def _save_to_cache(self):
        """Save to pickle cache."""
        try:
            cache_data = {
                'concepts': dict(self.concepts),
                'name_to_concepts': dict(self.name_to_concepts),
                'drug_names': dict(self.drug_names),
                'condition_names': dict(self.condition_names)
            }
            
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def _load_from_cache(self) -> bool:
        """Load from pickle cache."""
        try:
            if not self.cache_path.exists():
                return False
            
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.concepts = cache_data['concepts']
            self.name_to_concepts = defaultdict(list, cache_data['name_to_concepts'])
            self.drug_names = defaultdict(list, cache_data['drug_names'])
            self.condition_names = defaultdict(list, cache_data['condition_names'])
            
            return len(self.concepts) > 0
            
        except Exception as e:
            return False
    
    def get_drug_concepts(self, query: str, limit: int = 5) -> List[FastMatch]:
        """Fast drug search."""
        query_lower = query.lower().strip()
        matches = []
        
        # Exact match
        exact_drugs = self.drug_names.get(query_lower, [])
        matches.extend([FastMatch(concept=c, confidence=1.0) for c in exact_drugs[:limit]])
        
        # Substring match (limited)
        if len(matches) < limit:
            for name, concepts in list(self.drug_names.items())[:500]:  # Limit search
                if len(matches) >= limit:
                    break
                if query_lower in name and name not in [query_lower]:
                    for concept in concepts:
                        if len(matches) >= limit:
                            break
                        confidence = len(query_lower) / len(name)
                        matches.append(FastMatch(concept=concept, confidence=confidence))
        
        return matches[:limit]
    
    def get_condition_concepts(self, query: str, limit: int = 5) -> List[FastMatch]:
        """Fast condition search."""
        query_lower = query.lower().strip()
        matches = []
        
        # Exact match
        exact_conditions = self.condition_names.get(query_lower, [])
        matches.extend([FastMatch(concept=c, confidence=1.0) for c in exact_conditions[:limit]])
        
        # Substring match (limited)
        if len(matches) < limit:
            for name, concepts in list(self.condition_names.items())[:500]:  # Limit search
                if len(matches) >= limit:
                    break
                if query_lower in name and name not in [query_lower]:
                    for concept in concepts:
                        if len(matches) >= limit:
                            break
                        confidence = len(query_lower) / len(name)
                        matches.append(FastMatch(concept=concept, confidence=confidence))
        
        return matches[:limit]
    
    def search_concepts(self, query: str, domain_filter: Optional[str] = None, limit: int = 5) -> List[FastMatch]:
        """General concept search."""
        if domain_filter == 'Drug':
            return self.get_drug_concepts(query, limit)
        elif domain_filter == 'Condition':
            return self.get_condition_concepts(query, limit)
        else:
            # General search
            query_lower = query.lower().strip()
            matches = []
            
            exact_matches = self.name_to_concepts.get(query_lower, [])
            for concept in exact_matches[:limit]:
                if not domain_filter or concept.domain_id == domain_filter:
                    matches.append(FastMatch(concept=concept, confidence=1.0))
            
            return matches[:limit]


# Global instance
_fast_vocabulary: Optional[FastVocabularyIndex] = None

def get_fast_vocabulary() -> FastVocabularyIndex:
    """Get the global fast vocabulary index."""
    global _fast_vocabulary
    
    if _fast_vocabulary is None:
        _fast_vocabulary = FastVocabularyIndex()
        _fast_vocabulary.load_vocabulary()
    
    return _fast_vocabulary

def clear_fast_vocabulary_cache():
    """Clear the fast vocabulary cache."""
    global _fast_vocabulary
    _fast_vocabulary = None
    
    cache_path = Path(".cache/fast_vocabulary.pkl")
    if cache_path.exists():
        cache_path.unlink()
