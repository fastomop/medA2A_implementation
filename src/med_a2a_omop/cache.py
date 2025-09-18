"""
Comprehensive Query Pattern Caching System for Medical A2A OMOP.

This module provides intelligent caching for:
1. Query patterns - Semantic similarity-based query matching
2. SQL templates - Reusable SQL patterns with parameters
3. Database results - Cached query results
4. Semantic analysis - Medical term parsing and concept mapping
"""

import json
import hashlib
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field, asdict
import logging
import sqlite3
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)

@dataclass
class CachedQuery:
    """Represents a cached SQL template with its metadata."""
    query_hash: str
    original_query: str
    normalized_query: str
    medical_entities: List[str] = field(default_factory=list)
    sql_query: Optional[str] = None
    # REMOVED: result field - we never cache results, only SQL templates
    semantic_context: Optional[Dict[str, Any]] = None
    success: bool = False
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0
    response_time: float = 0.0
    confidence_score: float = 0.0
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['tags'] = list(data['tags'])  # Convert set to list for JSON
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedQuery':
        """Create from dictionary."""
        if 'tags' in data:
            data['tags'] = set(data['tags'])  # Convert list back to set
        return cls(**data)

@dataclass
class QueryPattern:
    """Represents a reusable query pattern."""
    pattern_id: str
    pattern_name: str
    description: str
    normalized_template: str
    medical_domains: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    sql_template: Optional[str] = None
    success_rate: float = 0.0
    usage_count: int = 0
    examples: List[Dict[str, Any]] = field(default_factory=list)
    last_used: float = field(default_factory=time.time)

class QueryNormalizer:
    """Normalizes queries for pattern matching and caching."""
    
    def __init__(self):
        # Common medical abbreviations and their expansions
        self.medical_abbreviations = {
            'dm': 'diabetes mellitus',
            't2dm': 'type 2 diabetes mellitus', 
            't1dm': 'type 1 diabetes mellitus',
            'htn': 'hypertension',
            'cad': 'coronary artery disease',
            'chf': 'congestive heart failure',
            'copd': 'chronic obstructive pulmonary disease',
            'mi': 'myocardial infarction',
            'afib': 'atrial fibrillation',
            'ckd': 'chronic kidney disease',
            'icd': 'international classification of diseases',
            'cpt': 'current procedural terminology',
            'loinc': 'logical observation identifiers names and codes',
            'rxnorm': 'rxnorm',
        }
        
        # Medical entities patterns
        self.medical_entity_patterns = [
            r'\b(?:patient|pt|pts|person|people|individual|subject)s?\b',
            r'\b(?:condition|diagnosis|disease|disorder|illness)s?\b',
            r'\b(?:medication|drug|medicine|pharmaceutical|rx|prescription)s?\b',
            r'\b(?:procedure|treatment|intervention|therapy)s?\b',
            r'\b(?:measurement|lab|test|result|value)s?\b',
            r'\b(?:age|gender|race|ethnicity|demographic)s?\b',
            r'\b(?:count|number|total|sum|average|mean|median)s?\b',
        ]
        
    def normalize_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Normalize a query for pattern matching.
        
        Returns:
            Tuple of (normalized_query, extracted_medical_entities)
        """
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Extract medical entities before normalization
        medical_entities = self.extract_medical_entities(normalized)
        
        # Expand medical abbreviations
        for abbrev, expansion in self.medical_abbreviations.items():
            normalized = re.sub(r'\b' + re.escape(abbrev) + r'\b', expansion, normalized)
        
        # Normalize common patterns
        normalized = re.sub(r'\b(?:how many|what is the number of|count of)\b', 'COUNT', normalized)
        normalized = re.sub(r'\b(?:what is the average|average|mean)\b', 'AVERAGE', normalized)
        normalized = re.sub(r'\b(?:patients? with|people with|individuals with)\b', 'PATIENTS_WITH', normalized)
        normalized = re.sub(r'\b(?:taking|prescribed|on|using)\b', 'TAKING', normalized)
        
        # Remove extra whitespace and punctuation
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized, medical_entities
    
    def extract_medical_entities(self, query: str) -> List[str]:
        """Extract medical entities from the query."""
        entities = []
        
        # Extract specific medical terms (this could be enhanced with NLP)
        medical_terms = re.findall(r'\b(?:diabetes|hypertension|cancer|heart disease|asthma|copd|pneumonia|stroke|depression|anxiety|covid|influenza|tuberculosis|malaria|hiv|aids)\b', query, re.IGNORECASE)
        entities.extend(medical_terms)
        
        # Extract drug names (simple patterns)
        drug_patterns = re.findall(r'\b(?:metformin|insulin|aspirin|warfarin|atorvastatin|lisinopril|amlodipine|omeprazole|simvastatin|levothyroxine)\b', query, re.IGNORECASE)
        entities.extend(drug_patterns)
        
        return list(set(entities))  # Remove duplicates
    
    def calculate_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two normalized queries."""
        # Simple token-based similarity (could be enhanced with embeddings)
        tokens1 = set(query1.split())
        tokens2 = set(query2.split())
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0

class QueryPatternCache:
    """High-performance query pattern caching system."""
    
    def __init__(self, cache_dir: str = ".cache", max_memory_entries: int = 1000, 
                 similarity_threshold: float = 0.8):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_memory_entries = max_memory_entries
        self.similarity_threshold = similarity_threshold
        
        # In-memory caches (LRU-style)
        self.query_cache: OrderedDict[str, CachedQuery] = OrderedDict()
        self.pattern_cache: Dict[str, QueryPattern] = {}
        # REMOVED: result_cache - we never cache results, only SQL templates
        
        # Database connection for persistent cache
        self.db_path = self.cache_dir / "query_cache.db"
        self.db_lock = threading.Lock()
        
        # Query normalizer
        self.normalizer = QueryNormalizer()
        
        # Initialize database
        self._init_database()
        self._load_from_database()
        
        logger.info(f"Query pattern cache initialized at {self.cache_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for persistent caching."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cached_queries (
                    query_hash TEXT PRIMARY KEY,
                    original_query TEXT,
                    normalized_query TEXT,
                    medical_entities TEXT,
                    sql_query TEXT,
                    -- REMOVED: result column - never store results, only SQL templates
                    semantic_context TEXT,
                    success BOOLEAN,
                    timestamp REAL,
                    hit_count INTEGER,
                    response_time REAL,
                    confidence_score REAL,
                    tags TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS query_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_name TEXT,
                    description TEXT,
                    normalized_template TEXT,
                    medical_domains TEXT,
                    parameters TEXT,
                    sql_template TEXT,
                    success_rate REAL,
                    usage_count INTEGER,
                    examples TEXT,
                    last_used REAL
                )
            ''')
            
            # Create indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_normalized_query ON cached_queries(normalized_query)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON cached_queries(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_success ON cached_queries(success)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_medical_entities ON cached_queries(medical_entities)')
    
    def _load_from_database(self):
        """Load frequently used cache entries from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load top queries by hit count
                cursor = conn.execute('''
                    SELECT * FROM cached_queries 
                    WHERE success = 1 
                    ORDER BY hit_count DESC, timestamp DESC 
                    LIMIT ?
                ''', (self.max_memory_entries // 2,))
                
                for row in cursor:
                    try:
                        cached_query = self._row_to_cached_query(row)
                        self.query_cache[cached_query.query_hash] = cached_query
                    except Exception as e:
                        logger.warning(f"Failed to load cached query: {e}")
                
                # Load patterns
                cursor = conn.execute('SELECT * FROM query_patterns ORDER BY usage_count DESC')
                for row in cursor:
                    try:
                        pattern = self._row_to_pattern(row)
                        self.pattern_cache[pattern.pattern_id] = pattern
                    except Exception as e:
                        logger.warning(f"Failed to load pattern: {e}")
                
                logger.info(f"Loaded {len(self.query_cache)} queries and {len(self.pattern_cache)} patterns from cache")
        except Exception as e:
            logger.error(f"Failed to load cache from database: {e}")
    
    def _row_to_cached_query(self, row) -> CachedQuery:
        """Convert database row to CachedQuery object."""
        return CachedQuery(
            query_hash=row[0],
            original_query=row[1],
            normalized_query=row[2],
            medical_entities=json.loads(row[3] or '[]'),
            sql_query=row[4],
            # REMOVED: result field - indices shift down by 1
            semantic_context=json.loads(row[5]) if row[5] else None,
            success=bool(row[6]),
            timestamp=row[7],
            hit_count=row[8],
            response_time=row[9],
            confidence_score=row[10],
            tags=set(json.loads(row[11] or '[]'))
        )
    
    def _row_to_pattern(self, row) -> QueryPattern:
        """Convert database row to QueryPattern object."""
        return QueryPattern(
            pattern_id=row[0],
            pattern_name=row[1],
            description=row[2],
            normalized_template=row[3],
            medical_domains=json.loads(row[4] or '[]'),
            parameters=json.loads(row[5] or '[]'),
            sql_template=row[6],
            success_rate=row[7],
            usage_count=row[8],
            examples=json.loads(row[9] or '[]'),
            last_used=row[10]
        )
    
    def _generate_query_hash(self, query: str) -> str:
        """Generate a hash for the query."""
        normalized, _ = self.normalizer.normalize_query(query)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def find_similar_queries(self, query: str, limit: int = 5) -> List[CachedQuery]:
        """Find similar cached queries."""
        normalized, entities = self.normalizer.normalize_query(query)
        similar_queries = []
        
        # Check in-memory cache first
        for cached_query in self.query_cache.values():
            if not cached_query.success:
                continue
                
            similarity = self.normalizer.calculate_similarity(normalized, cached_query.normalized_query)
            if similarity >= self.similarity_threshold:
                similar_queries.append((similarity, cached_query))
        
        # If not enough similar queries in memory, check database
        if len(similar_queries) < limit:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # Search for queries with similar medical entities
                    entity_conditions = ' OR '.join([f"medical_entities LIKE '%{entity}%'" for entity in entities])
                    if entity_conditions:
                        cursor = conn.execute(f'''
                            SELECT * FROM cached_queries 
                            WHERE success = 1 AND ({entity_conditions})
                            ORDER BY hit_count DESC, timestamp DESC
                            LIMIT ?
                        ''', (limit * 2,))
                        
                        for row in cursor:
                            try:
                                cached_query = self._row_to_cached_query(row)
                                if cached_query.query_hash not in self.query_cache:
                                    similarity = self.normalizer.calculate_similarity(normalized, cached_query.normalized_query)
                                    if similarity >= self.similarity_threshold:
                                        similar_queries.append((similarity, cached_query))
                            except Exception as e:
                                logger.warning(f"Failed to process similar query: {e}")
            except Exception as e:
                logger.error(f"Failed to search database for similar queries: {e}")
        
        # Sort by similarity and return top results
        similar_queries.sort(key=lambda x: x[0], reverse=True)
        return [query for _, query in similar_queries[:limit]]
    
    def get_cached_result(self, query: str) -> Optional[CachedQuery]:
        """Get cached result for a query."""
        query_hash = self._generate_query_hash(query)
        
        # Check in-memory cache first
        if query_hash in self.query_cache:
            cached_query = self.query_cache[query_hash]
            cached_query.hit_count += 1
            # Move to end (LRU)
            self.query_cache.move_to_end(query_hash)
            self._update_hit_count_in_db(query_hash, cached_query.hit_count)
            return cached_query
        
        # Check database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT * FROM cached_queries WHERE query_hash = ?', (query_hash,))
                row = cursor.fetchone()
                if row:
                    cached_query = self._row_to_cached_query(row)
                    cached_query.hit_count += 1
                    
                    # Add to memory cache
                    self._add_to_memory_cache(cached_query)
                    self._update_hit_count_in_db(query_hash, cached_query.hit_count)
                    return cached_query
        except Exception as e:
            logger.error(f"Failed to retrieve from cache database: {e}")
        
        return None
    
    def cache_sql_template(self, query: str, sql_query: Optional[str] = None,
                          semantic_context: Optional[Dict[str, Any]] = None, success: bool = False,
                          response_time: float = 0.0, confidence_score: float = 0.0, 
                          tags: Optional[Set[str]] = None):
        """Cache a successful SQL template for future query generation assistance."""
        # Only cache if we have a successful SQL query
        if not success or not sql_query:
            logger.debug(f"Skipping cache - only successful SQL templates are cached")
            return
            
        query_hash = self._generate_query_hash(query)
        normalized, entities = self.normalizer.normalize_query(query)
        
        cached_query = CachedQuery(
            query_hash=query_hash,
            original_query=query,
            normalized_query=normalized,
            medical_entities=entities,
            sql_query=sql_query,
            # REMOVED: result=result - never cache results
            semantic_context=semantic_context,
            success=success,
            timestamp=time.time(),
            hit_count=1,
            response_time=response_time,
            confidence_score=confidence_score,
            tags=tags or set()
        )
        
        # Add to memory cache
        self._add_to_memory_cache(cached_query)
        
        # Persist to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO cached_queries 
                    (query_hash, original_query, normalized_query, medical_entities, 
                     sql_query, semantic_context, success, timestamp, 
                     hit_count, response_time, confidence_score, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    cached_query.query_hash,
                    cached_query.original_query,
                    cached_query.normalized_query,
                    json.dumps(cached_query.medical_entities),
                    cached_query.sql_query,
                    # REMOVED: result field - never store results
                    json.dumps(cached_query.semantic_context) if cached_query.semantic_context else None,
                    cached_query.success,
                    cached_query.timestamp,
                    cached_query.hit_count,
                    cached_query.response_time,
                    cached_query.confidence_score,
                    json.dumps(list(cached_query.tags))
                ))
        except Exception as e:
            logger.error(f"Failed to persist query to cache database: {e}")
        
        logger.debug(f"Cached query: {query[:100]}... -> Success: {success}")
    
    def _add_to_memory_cache(self, cached_query: CachedQuery):
        """Add query to in-memory cache with LRU eviction."""
        # Add to cache
        self.query_cache[cached_query.query_hash] = cached_query
        
        # Evict old entries if cache is full
        while len(self.query_cache) > self.max_memory_entries:
            oldest_hash, _ = self.query_cache.popitem(last=False)
            logger.debug(f"Evicted query from memory cache: {oldest_hash}")
    
    def _update_hit_count_in_db(self, query_hash: str, hit_count: int):
        """Update hit count in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'UPDATE cached_queries SET hit_count = ? WHERE query_hash = ?',
                    (hit_count, query_hash)
                )
        except Exception as e:
            logger.error(f"Failed to update hit count: {e}")
    
    def add_query_pattern(self, pattern: QueryPattern):
        """Add a reusable query pattern."""
        self.pattern_cache[pattern.pattern_id] = pattern
        
        # Persist to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO query_patterns
                    (pattern_id, pattern_name, description, normalized_template,
                     medical_domains, parameters, sql_template, success_rate,
                     usage_count, examples, last_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern.pattern_id,
                    pattern.pattern_name,
                    pattern.description,
                    pattern.normalized_template,
                    json.dumps(pattern.medical_domains),
                    json.dumps(pattern.parameters),
                    pattern.sql_template,
                    pattern.success_rate,
                    pattern.usage_count,
                    json.dumps(pattern.examples),
                    pattern.last_used
                ))
        except Exception as e:
            logger.error(f"Failed to persist pattern to database: {e}")
    
    def find_matching_patterns(self, query: str) -> List[QueryPattern]:
        """Find query patterns that match the given query."""
        normalized, entities = self.normalizer.normalize_query(query)
        matching_patterns = []
        
        for pattern in self.pattern_cache.values():
            similarity = self.normalizer.calculate_similarity(normalized, pattern.normalized_template)
            if similarity >= self.similarity_threshold:
                matching_patterns.append((similarity, pattern))
        
        # Sort by similarity and success rate
        matching_patterns.sort(key=lambda x: (x[0], x[1].success_rate), reverse=True)
        return [pattern for _, pattern in matching_patterns]
    
    def clear_expired_entries(self, max_age_days: int = 30):
        """Clear expired cache entries."""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        # Clear from memory cache
        expired_hashes = [h for h, q in self.query_cache.items() if q.timestamp < cutoff_time]
        for hash_key in expired_hashes:
            del self.query_cache[hash_key]
        
        # Clear from database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('DELETE FROM cached_queries WHERE timestamp < ?', (cutoff_time,))
                deleted_count = cursor.rowcount
                logger.info(f"Cleared {deleted_count} expired cache entries")
        except Exception as e:
            logger.error(f"Failed to clear expired entries: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'memory_cache_size': len(self.query_cache),
            'pattern_cache_size': len(self.pattern_cache),
            'successful_queries': sum(1 for q in self.query_cache.values() if q.success),
            'total_hits': sum(q.hit_count for q in self.query_cache.values()),
            'average_response_time': 0.0
        }
        
        if self.query_cache:
            stats['average_response_time'] = sum(q.response_time for q in self.query_cache.values()) / len(self.query_cache)
        
        # Get database stats
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM cached_queries')
                stats['database_cache_size'] = cursor.fetchone()[0]
                
                cursor = conn.execute('SELECT COUNT(*) FROM cached_queries WHERE success = 1')
                stats['database_successful_queries'] = cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            stats['database_cache_size'] = 0
            stats['database_successful_queries'] = 0
        
        return stats

# Global cache instance
_cache_instance: Optional[QueryPatternCache] = None

def get_query_cache() -> QueryPatternCache:
    """Get the global query cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = QueryPatternCache()
    return _cache_instance

def clear_query_cache():
    """Clear the global cache instance (useful for testing)."""
    global _cache_instance
    if _cache_instance is not None:
        # Clear in-memory caches
        _cache_instance.query_cache.clear()
        _cache_instance.pattern_cache.clear()
        # REMOVED: result_cache.clear() - no result cache anymore
        
        # Clear database cache
        try:
            import os
            if _cache_instance.db_path.exists():
                os.remove(_cache_instance.db_path)
                logger.info(f"Cleared cache database: {_cache_instance.db_path}")
        except Exception as e:
            logger.error(f"Failed to clear cache database: {e}")
        
        # Reset instance
        _cache_instance = None
        logger.info("Query cache cleared completely")

def clear_corrupted_cache():
    """Clear corrupted cache data - for system recovery."""
    global _cache_instance
    cache_dir = Path(".cache")
    db_path = cache_dir / "query_cache.db"
    
    try:
        if db_path.exists():
            import os
            os.remove(db_path)
            logger.info(f"Removed corrupted cache database: {db_path}")
        
        # Reset instance to force reinitialization
        _cache_instance = None
        logger.info("Corrupted cache cleared - system will start fresh")
        return True
    except Exception as e:
        logger.error(f"Failed to clear corrupted cache: {e}")
        return False