#!/usr/bin/env python3
"""
Example script demonstrating programmatic usage of the Medical A2A OMOP system.
"""

import asyncio
import json
from src.med_a2a_omop.runner import MedA2AAPI

async def main():
    """Demonstrate different ways to use the programmatic API."""
    
    # Method 1: Using async context manager (recommended)
    print("=== Method 1: Async Context Manager ===")
    async with MedA2AAPI() as api:
        # Single question
        result = await api.ask("How many patients have hypertension?")
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Success: {result['success']}")
        
        if result['success'] and 'generated_sql' in result:
            print(f"Generated SQL: {result['generated_sql']}")
        
        print("\n" + "="*50 + "\n")
        
        # Multiple questions
        questions = [
            "How many patients have diabetes?",
            "What is the average age of patients?",
            "How many patients are taking metformin?"
        ]
        
        results = await api.ask_multiple(questions)
        
        print("Multiple questions results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['question']}")
            print(f"   Answer: {result['answer']}")
            print(f"   Success: {result['success']}")
            print()
    
    # Method 2: Manual initialization and cleanup
    print("=== Method 2: Manual Management ===")
    api = MedA2AAPI()
    try:
        await api.initialize()
        
        result = await api.ask("How many patients have heart disease?")
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        
        # Save result to file
        with open('api_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("Result saved to api_result.json")
        
    finally:
        await api.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 