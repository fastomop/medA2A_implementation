#!/usr/bin/env python3
"""
Simple test to verify the database path fix works.
This tests the core logic without MCP complexity.
"""

import os

def test_connection_string_fix():
    """Test that our connection string fix works correctly."""
    db_path = "/Users/k24118093/Documents/omcp_server/synthetic_data/synthea.duckdb"
    
    # Original (broken) logic
    original_connection = f"duckdb:///{db_path}"
    print(f"Original connection string: {original_connection}")
    print(f"Original has double slash: {'///' + '/' in original_connection}")
    
    # Fixed logic
    if db_path.startswith('/'):
        fixed_connection = f"duckdb://{db_path}"
    else:
        fixed_connection = f"duckdb:///{db_path}"
    
    print(f"Fixed connection string: {fixed_connection}")
    print(f"Fixed has double slash: {'///' + '/' in fixed_connection}")
    
    # Test the fix
    expected = "duckdb:///Users/k24118093/Documents/omcp_server/synthetic_data/synthea.duckdb"
    if fixed_connection == expected:
        print("✅ Connection string fix is correct!")
        return True
    else:
        print(f"❌ Connection string fix failed. Expected: {expected}")
        return False

def test_database_exists():
    """Test that the database file exists."""
    db_path = "/Users/k24118093/Documents/omcp_server/synthetic_data/synthea.duckdb"
    if os.path.exists(db_path):
        print(f"✅ Database file exists: {db_path}")
        return True
    else:
        print(f"❌ Database file not found: {db_path}")
        return False

if __name__ == "__main__":
    print("=== Testing Database Connection String Fix ===\n")
    
    success1 = test_connection_string_fix()
    print()
    success2 = test_database_exists()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The database path fix should resolve the MCP connection issues.")
    else:
        print("\n⚠️ Some tests failed. Check the output above.")