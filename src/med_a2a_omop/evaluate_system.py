#!/usr/bin/env python3uv
"""
Comprehensive Evaluation Script for Medical A2A OMOP System
Evaluates the system against ground truth dataset with SQL queries and expected results.
"""

import json
import asyncio
import argparse
import time
import signal
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import re
import numpy as np
from dataclasses import dataclass, asdict
import sys
import os

# Initialize logger
logger = logging.getLogger(__name__)

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from .runner import MedA2AInterface

@dataclass
class EvaluationMetrics:
    """Metrics for evaluation results."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    exact_matches: int = 0
    partial_matches: int = 0
    no_matches: int = 0
    avg_response_time: float = 0.0
    median_response_time: float = 0.0
    min_response_time: float = 0.0
    max_response_time: float = 0.0
    accuracy: float = 0.0
    success_rate: float = 0.0
    exact_match_rate: float = 0.0
    partial_match_rate: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class QueryResult:
    """Individual query evaluation result."""
    id: int
    question: str
    ground_truth: Any
    system_response: Any
    sql_generated: Optional[str] = None
    is_success: bool = False
    is_exact_match: bool = False
    is_partial_match: bool = False
    error_message: Optional[str] = None
    response_time: float = 0.0
    match_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)

class SystemEvaluator:
    """Evaluates the Medical A2A OMOP system against ground truth data."""
    
    def __init__(self, dataset_path: str, output_dir: str = "evaluation_results"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load ground truth dataset
        with open(self.dataset_path, 'r') as f:
            self.dataset = json.load(f)
        
        self.interface: Optional[MedA2AInterface] = None
        self.results: List[QueryResult] = []
        self.metrics = EvaluationMetrics()
        
    async def initialize(self):
        """Initialize the MedA2A system."""
        print("🚀 Initializing Medical A2A OMOP System for evaluation...")
        self.interface = MedA2AInterface()
        await self.interface.initialize()
        print("✅ System initialized successfully!")
        
    async def cleanup(self):
        """Cleanup the system."""
        if self.interface:
            self.interface.cleanup_all()
        
    async def evaluate_single_query(self, query_data: Dict) -> QueryResult:
        """Evaluate a single query against ground truth."""
        query_id = query_data['id']
        question = query_data['text']
        ground_truth = query_data['result']
        expected_sql = query_data.get('sql', '')
        
        print(f"\n📊 Evaluating Query {query_id}: {question[:100]}...")
        
        result = QueryResult(
            id=query_id,
            question=question,
            ground_truth=ground_truth,
            system_response=None  # Will be set later
        )
        
        start_time = time.time()
        
        try:
            # Ask the system
            response = await self.interface.ask_single_question(question)
            result.response_time = time.time() - start_time
            
            if response.get('success'):
                result.is_success = True
                result.system_response = response.get('answer', response.get('data'))
                
                # Extract SQL from orchestrator's executed steps (the correct approach)
                if hasattr(self.interface, 'orchestrator') and self.interface.orchestrator:
                    executed_steps = self.interface.orchestrator.world_model.executed_steps
                    
                    # Find the last SQL query generated (should be the correct one)
                    for step in reversed(executed_steps):  # Start from last step
                        step_result = step.get('result', {})
                        if isinstance(step_result, dict) and 'generated_sql' in step_result:
                            result.sql_generated = step_result['generated_sql']
                            break
                
                # Fallback: try to extract from response structure
                if not result.sql_generated:
                    if 'sql_query' in response:
                        result.sql_generated = response['sql_query']
                    elif 'generated_sql' in response:
                        result.sql_generated = response['generated_sql']
                    elif 'data' in response and isinstance(response['data'], dict):
                        result.sql_generated = response['data'].get('sql_query') or response['data'].get('generated_sql')
                
                # Last resort: try to extract SQL from the answer text
                if not result.sql_generated and 'answer' in response:
                    result.sql_generated = self._extract_sql_from_text(response['answer'])
                
                # Evaluate the response
                result.is_exact_match, result.is_partial_match, result.match_score = \
                    self.compare_results(ground_truth, result.system_response)
                
                if result.is_exact_match:
                    print(f"  ✅ Exact match! (Score: {result.match_score:.2f})")
                elif result.is_partial_match:
                    print(f"  ⚠️ Partial match (Score: {result.match_score:.2f})")
                else:
                    print(f"  ❌ No match (Score: {result.match_score:.2f})")
                    
            else:
                result.is_success = False
                result.error_message = response.get('error', 'Unknown error')
                print(f"  ❌ Query failed: {result.error_message}")
                
        except Exception as e:
            result.response_time = time.time() - start_time
            result.is_success = False
            result.error_message = str(e)
            print(f"  ❌ Exception: {e}")
            
        return result
    
    def compare_results(self, ground_truth: Any, system_response: Any) -> Tuple[bool, bool, float]:
        """
        Compare system response with ground truth.
        For multi-step queries, extracts the actual result from the final answer step.
        Returns: (is_exact_match, is_partial_match, match_score)
        """
        # Extract ground truth value
        gt_value = self.extract_numeric_value(ground_truth)
        
        # Try to get the actual numeric result from executed steps
        sys_value = self._extract_from_final_step()
        
        # Fall back to extracting from the system response text if needed
        if sys_value is None:
            sys_value = self.extract_numeric_value(system_response)
        
        if gt_value is None or sys_value is None:
            if sys_value is None:
                print(f"  ⚠️ Could not extract numeric value from system response")
            return False, False, 0.0
        
        # Check for exact match
        if gt_value == sys_value:
            return True, True, 1.0
        
        # Check for partial match (within 10% tolerance)
        if gt_value != 0:
            relative_diff = abs(gt_value - sys_value) / abs(gt_value)
            if relative_diff <= 0.1:  # 10% tolerance
                score = 1.0 - relative_diff
                return False, True, score
        
        # Calculate similarity score
        if gt_value == 0 and sys_value == 0:
            score = 1.0
        elif gt_value == 0 or sys_value == 0:
            score = 0.0
        else:
            score = 1.0 - min(abs(gt_value - sys_value) / max(abs(gt_value), abs(sys_value)), 1.0)
        
        return False, False, score
    
    def _extract_from_final_step(self) -> Optional[float]:
        """
        Extract numeric result from the step marked as final answer.
        Falls back to the last step if no step is explicitly marked.
        """
        if not hasattr(self.interface, 'orchestrator') or not self.interface.orchestrator:
            return None
        
        executed_steps = self.interface.orchestrator.world_model.executed_steps
        if not executed_steps:
            return None
        
        # First, look for a step explicitly marked as final answer
        final_step = None
        for step in executed_steps:
            if step.get('is_final_answer', False):
                final_step = step
                print(f"  🎯 Found marked final step: {step.get('sub_question', '')[:100]}")
                break
        
        # If no step is marked as final, use the last step
        if final_step is None:
            final_step = executed_steps[-1]
            print(f"  🔍 Using last step as final: {final_step.get('sub_question', '')[:100]}")
        
        # Extract the numeric value from the final step
        value = self._extract_value_from_step(final_step)
        
        # Debug output for multi-step queries
        if len(executed_steps) > 1:
            print(f"  📊 Multi-step query with {len(executed_steps)} steps")
            print(f"  🔢 Extracted value: {value}")
            # Show all step results for debugging
            for i, step in enumerate(executed_steps, 1):
                step_val = self._extract_value_from_step(step)
                is_final = " [🎯 FINAL]" if step.get('is_final_answer', False) else ""
                print(f"     Step {i}: {step.get('sub_question', '')[:50]}... = {step_val}{is_final}")
        
        return value
    
    def _extract_value_from_step(self, step: Dict) -> Optional[float]:
        """
        Extract numeric value from a single executed step result.
        """
        result = step.get('result', {})
        
        if isinstance(result, dict):
            # Look for query_result which contains the actual DB query results
            query_result = result.get('query_result', [])
            if query_result and isinstance(query_result, list) and len(query_result) > 0:
                first_row = query_result[0]
                if isinstance(first_row, dict):
                    # Strategy 1: Try common standardized column names
                    for key in ['patient_count', 'distinct_patient_count', 'distinct_person_count', 'distinct_patients', 'num_patients', 'num_persons', 'person_count', 'count', 'total', 'value', 'result']:
                        if key in first_row:
                            val = first_row[key]
                            try:
                                # Handle both string and numeric values
                                return float(val) if isinstance(val, str) else float(val)
                            except (ValueError, TypeError):
                                continue
                    
                    # Strategy 2: Pattern match for SQL-generated column names
                    for key, val in first_row.items():
                        key_lower = key.lower()
                        # Match patterns like "count(distinct person_id)", "count(*)", etc.
                        if any(pattern in key_lower for pattern in [
                            'count(distinct', 'count(*)', 'count(', 
                            'patient_count', 'person_count', 'distinct_person', 'distinct_patient'
                        ]):
                            try:
                                return float(val) if isinstance(val, str) else float(val)
                            except (ValueError, TypeError):
                                continue
                    
                    # Strategy 3: Any column with count-like patterns
                    for key, val in first_row.items():
                        key_lower = key.lower()
                        if any(pattern in key_lower for pattern in ['count', 'patient', 'person', 'distinct']):
                            try:
                                return float(val) if isinstance(val, str) else float(val)
                            except (ValueError, TypeError):
                                continue
                    
                    # Strategy 4: Last resort - first numeric value found
                    for key, val in first_row.items():
                        try:
                            return float(val) if isinstance(val, str) else float(val)
                        except (ValueError, TypeError):
                            continue
            
            # Also check if the result dict itself has count fields
            for key in ['count', 'total', 'value', 'result', 'patient_count', 'distinct_patient_count', 'distinct_person_count', 'distinct_patients', 'num_patients', 'num_persons', 'person_count']:
                if key in result:
                    try:
                        return float(result[key])
                    except (ValueError, TypeError):
                        continue
        
        # Log debug info for failed extractions
        logger.debug(f"Failed to extract numeric value from step result: {result}")
        if isinstance(result, dict) and 'query_result' in result:
            query_result = result['query_result']
            if query_result and isinstance(query_result, list) and len(query_result) > 0 and isinstance(query_result[0], dict):
                first_row = query_result[0]
                logger.debug(f"Available columns: {list(first_row.keys())}")
                logger.debug(f"Column values: {first_row}")
            else:
                logger.debug(f"Query result structure: {query_result}")
        
        return None
    
    def extract_numeric_value(self, response: Any) -> Optional[float]:
        """Extract numeric value from various response formats using intelligent detection."""
        if isinstance(response, (int, float)):
            return float(response)
        
        if isinstance(response, list):
            return self._extract_from_list(response)
        
        if isinstance(response, str):
            # Extract numbers from string using regex
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if numbers:
                return float(numbers[0])
        
        if isinstance(response, dict):
            # Try common keys
            for key in ['result', 'count', 'value', 'answer', 'total']:
                if key in response:
                    return self.extract_numeric_value(response[key])
        
        return None
    
    def _extract_from_list(self, data: list) -> Optional[float]:
        """Intelligently extract numeric values from list structures."""
        if not data:
            return None
        
        # Case 1: Single value list [4] or [4.5]
        if len(data) == 1 and isinstance(data[0], (int, float)):
            return float(data[0])
        
        # Case 2: Nested single value [[4]] or [[4.5]]
        if (len(data) == 1 and isinstance(data[0], list) and 
            len(data[0]) == 1 and isinstance(data[0][0], (int, float))):
            return float(data[0][0])
        
        # Case 3: Multiple values [1, 2, 3] - sum them
        if all(isinstance(item, (int, float)) for item in data):
            return sum(float(item) for item in data)
        
        # Case 4: Nested lists (distribution tables) - intelligent analysis
        if all(isinstance(item, list) for item in data):
            return self._extract_from_distribution_table(data)
        
        # Case 5: Mixed content - try to extract numbers
        numeric_values = []
        for item in data:
            if isinstance(item, (int, float)):
                numeric_values.append(float(item))
            elif isinstance(item, str) and item.isdigit():
                numeric_values.append(float(item))
            elif isinstance(item, list):
                nested_value = self._extract_from_list(item)
                if nested_value is not None:
                    numeric_values.append(nested_value)
        
        return sum(numeric_values) if numeric_values else None
    
    def _extract_from_distribution_table(self, table: list) -> Optional[float]:
        """Intelligently extract counts from distribution tables."""
        if not table or not all(isinstance(row, list) for row in table):
            return None
        
        # Analyze the structure to determine the best extraction strategy
        row_lengths = [len(row) for row in table]
        unique_lengths = set(row_lengths)
        
        # Strategy 1: All rows have same length (structured table)
        if len(unique_lengths) == 1:
            row_length = row_lengths[0]
            
            if row_length == 1:
                # Single column: sum all values
                return self._sum_numeric_values([row[0] for row in table])
            
            elif row_length == 2:
                # Two columns: assume second is count
                return self._sum_numeric_values([row[1] for row in table])
            
            else:
                # Multiple columns: intelligently detect count column
                return self._detect_and_sum_count_column(table)
        
        # Strategy 2: Mixed row lengths - analyze each row
        else:
            return self._extract_from_mixed_structure(table)
    
    def _detect_and_sum_count_column(self, table: list) -> Optional[float]:
        """Detect which column contains counts and sum them."""
        if not table or not table[0]:
            return None
        
        num_cols = len(table[0])
        column_scores = []
        
        # Score each column based on how many numeric values it contains
        for col_idx in range(num_cols):
            numeric_count = 0
            total_count = 0
            
            for row in table:
                if col_idx < len(row):
                    total_count += 1
                    item = row[col_idx]
                    if isinstance(item, (int, float)) or (isinstance(item, str) and item.isdigit()):
                        numeric_count += 1
            
            # Score: ratio of numeric values in this column
            score = numeric_count / total_count if total_count > 0 else 0
            column_scores.append((col_idx, score, numeric_count))
        
        # Find the column with highest score (most numeric values)
        best_col = max(column_scores, key=lambda x: (x[1], x[2]))
        
        if best_col[1] > 0:  # If we found a column with numeric values
            return self._sum_numeric_values([row[best_col[0]] for row in table if best_col[0] < len(row)])
        
        return None
    
    def _extract_from_mixed_structure(self, table: list) -> Optional[float]:
        """Extract from tables with varying row structures."""
        total = 0
        for row in table:
            if isinstance(row, list):
                # For each row, try to find numeric values
                row_values = []
                for item in row:
                    if isinstance(item, (int, float)):
                        row_values.append(float(item))
                    elif isinstance(item, str) and item.isdigit():
                        row_values.append(float(item))
                
                # If row has multiple values, take the last one (usually count)
                if row_values:
                    total += row_values[-1]
        
        return total if total > 0 else None
    
    def _sum_numeric_values(self, values: list) -> Optional[float]:
        """Sum numeric values from a list, handling mixed types."""
        total = 0
        for value in values:
            if isinstance(value, (int, float)):
                total += float(value)
            elif isinstance(value, str) and value.isdigit():
                total += float(value)
        
        return total if total > 0 else None
    
    def _extract_sql_from_text(self, text: str) -> Optional[str]:
        """Extract SQL query from text content (for medA2A responses)."""
        if not text:
            return None
        
        # Look for SQL patterns in the text
        sql_patterns = [
            r'```sql\s*(.*?)\s*```',           # Markdown SQL blocks
            r'SQL:\s*(SELECT.*?);',            # SQL: SELECT... format
            r'(SELECT\s+.*?FROM\s+.*?;)',      # Direct SELECT statements
            r'(WITH\s+.*?SELECT\s+.*?;)',      # CTE queries
        ]
        
        for pattern in sql_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                sql = match.group(1).strip()
                # Clean up common issues
                sql = re.sub(r'\s+', ' ', sql)  # Normalize whitespace
                return sql
        
        return None
    
    async def evaluate_all(self, limit: Optional[int] = None):
        """Evaluate all queries in the dataset."""
        queries_to_eval = self.dataset[:limit] if limit else self.dataset
        total = len(queries_to_eval)
        
        print(f"\n🔍 Starting evaluation of {total} queries...")
        print("=" * 60)
        
        response_times = []
        
        for i, query_data in enumerate(queries_to_eval, 1):
            print(f"\n[{i}/{total}] Processing...")
            result = await self.evaluate_single_query(query_data)
            self.results.append(result)
            response_times.append(result.response_time)
            
            # Update running metrics
            if result.is_success:
                self.metrics.successful_queries += 1
            else:
                self.metrics.failed_queries += 1
            
            if result.is_exact_match:
                self.metrics.exact_matches += 1
            elif result.is_partial_match:
                self.metrics.partial_matches += 1
            else:
                self.metrics.no_matches += 1
            
            # Show progress
            if i % 10 == 0:
                print(f"\n📈 Progress: {i}/{total} ({i*100/total:.1f}%)")
                print(f"   Success rate: {self.metrics.successful_queries}/{i} ({self.metrics.successful_queries*100/i:.1f}%)")
                print(f"   Exact matches: {self.metrics.exact_matches}/{self.metrics.successful_queries if self.metrics.successful_queries > 0 else 1}")
        
        # Calculate final metrics
        self.metrics.total_queries = total
        self.metrics.success_rate = self.metrics.successful_queries / total if total > 0 else 0
        self.metrics.accuracy = self.metrics.exact_matches / total if total > 0 else 0
        self.metrics.exact_match_rate = self.metrics.exact_matches / self.metrics.successful_queries if self.metrics.successful_queries > 0 else 0
        self.metrics.partial_match_rate = self.metrics.partial_matches / self.metrics.successful_queries if self.metrics.successful_queries > 0 else 0
        
        if response_times:
            self.metrics.avg_response_time = np.mean(response_times)
            self.metrics.median_response_time = np.median(response_times)
            self.metrics.min_response_time = np.min(response_times)
            self.metrics.max_response_time = np.max(response_times)
    
    def generate_report(self):
        """Generate comprehensive evaluation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'metadata': {
                    'timestamp': timestamp,
                    'dataset': str(self.dataset_path),
                    'total_queries': self.metrics.total_queries
                },
                'metrics': self.metrics.to_dict(),
                'results': [r.to_dict() for r in self.results]
            }, f, indent=2)
        
        # Generate human-readable report
        report_file = self.output_dir / f"evaluation_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MEDICAL A2A OMOP SYSTEM EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Dataset: {self.dataset_path}\n\n")
            
            f.write("OVERALL METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Queries: {self.metrics.total_queries}\n")
            f.write(f"Successful Queries: {self.metrics.successful_queries} ({self.metrics.success_rate*100:.1f}%)\n")
            f.write(f"Failed Queries: {self.metrics.failed_queries}\n\n")
            
            f.write("ACCURACY METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Exact Matches: {self.metrics.exact_matches} ({self.metrics.exact_match_rate*100:.1f}% of successful)\n")
            f.write(f"Partial Matches: {self.metrics.partial_matches} ({self.metrics.partial_match_rate*100:.1f}% of successful)\n")
            f.write(f"No Matches: {self.metrics.no_matches}\n")
            f.write(f"Overall Accuracy: {self.metrics.accuracy*100:.1f}%\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Response Time: {self.metrics.avg_response_time:.2f}s\n")
            f.write(f"Median Response Time: {self.metrics.median_response_time:.2f}s\n")
            f.write(f"Min Response Time: {self.metrics.min_response_time:.2f}s\n")
            f.write(f"Max Response Time: {self.metrics.max_response_time:.2f}s\n\n")
            
            # Failed queries analysis
            if self.metrics.failed_queries > 0:
                f.write("FAILED QUERIES ANALYSIS\n")
                f.write("-" * 40 + "\n")
                for result in self.results:
                    if not result.is_success:
                        f.write(f"Query {result.id}: {result.question[:100]}\n")
                        f.write(f"  Error: {result.error_message}\n\n")
            
            # Mismatched queries analysis
            mismatched = [r for r in self.results if r.is_success and not r.is_exact_match]
            if mismatched:
                f.write("\nMISMATCHED QUERIES ANALYSIS\n")
                f.write("-" * 40 + "\n")
                for result in mismatched[:10]:  # Show first 10
                    f.write(f"Query {result.id}: {result.question[:100]}\n")
                    f.write(f"  Expected: {result.ground_truth}\n")
                    f.write(f"  Got: {result.system_response}\n")
                    f.write(f"  Match Score: {result.match_score:.2f}\n\n")
        
        # Generate CSV for easy analysis
        csv_file = self.output_dir / f"evaluation_summary_{timestamp}.csv"
        with open(csv_file, 'w') as f:
            f.write("ID,Success,ExactMatch,PartialMatch,MatchScore,ResponseTime,Question\n")
            for r in self.results:
                question_clean = r.question.replace('"', '""')[:100]
                f.write(f'{r.id},{r.is_success},{r.is_exact_match},{r.is_partial_match},'
                       f'{r.match_score:.3f},{r.response_time:.2f},"{question_clean}"\n')
        
        print(f"\n✅ Reports saved to {self.output_dir}/")
        print(f"  - Detailed JSON: {results_file.name}")
        print(f"  - Text Report: {report_file.name}")
        print(f"  - CSV Summary: {csv_file.name}")
        
        return results_file, report_file, csv_file
    
    def print_summary(self):
        """Print evaluation summary to console."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 Overall Performance:")
        print(f"  • Total Queries: {self.metrics.total_queries}")
        print(f"  • Success Rate: {self.metrics.success_rate*100:.1f}%")
        print(f"  • Accuracy (Exact Match): {self.metrics.accuracy*100:.1f}%")
        
        print(f"\n🎯 Match Statistics:")
        print(f"  • Exact Matches: {self.metrics.exact_matches}")
        print(f"  • Partial Matches: {self.metrics.partial_matches}")
        print(f"  • No Matches: {self.metrics.no_matches}")
        
        print(f"\n⏱️ Response Times:")
        print(f"  • Average: {self.metrics.avg_response_time:.2f}s")
        print(f"  • Median: {self.metrics.median_response_time:.2f}s")
        print(f"  • Range: {self.metrics.min_response_time:.2f}s - {self.metrics.max_response_time:.2f}s")
        
        if self.metrics.failed_queries > 0:
            print(f"\n⚠️ Failed Queries: {self.metrics.failed_queries}")
            print("  (See detailed report for error analysis)")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.interface:
            if hasattr(self.interface, 'cleanup'):
                await self.interface.cleanup()
            elif hasattr(self.interface, 'cleanup_all'):
                self.interface.cleanup_all()
            # If no cleanup method exists, that's okay

async def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate Medical A2A OMOP System against ground truth dataset"
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='Path to ground truth dataset JSON file'
    )
    parser.add_argument(
        '--output',
        default='evaluation_results',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of queries to evaluate (for testing)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show verbose output'
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = SystemEvaluator(args.dataset, args.output)
    
    try:
        # Initialize system
        await evaluator.initialize()
        
        # Run evaluation
        await evaluator.evaluate_all(limit=args.limit)
        
        # Generate reports
        evaluator.generate_report()
        
        # Print summary
        evaluator.print_summary()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluation interrupted by user")
        if evaluator.results:
            print("Generating partial results...")
            evaluator.generate_report()
            evaluator.print_summary()
    
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await evaluator.cleanup()
        print("\n✅ Evaluation complete!")

def main_sync():
    """Synchronous wrapper for the async main function."""
    asyncio.run(main())

if __name__ == "__main__":
    asyncio.run(main())