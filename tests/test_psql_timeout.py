"""
Test timeout functionality for PostgreSQL query execution.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import pytest
from dotenv import load_dotenv
load_dotenv()
# Try to import psycopg2, skip tests if not available
try:
    import psycopg2 as psql
    from InfiniQuantumSim.db_backends import contraction_eval_psql
    PSQL_AVAILABLE = True
except ImportError:
    PSQL_AVAILABLE = False
    pytest.skip("PostgreSQL not available", allow_module_level=True)


def get_psql_connection():
    """Get a PostgreSQL connection for testing."""
    try:
        con = psql.connect(
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'password'),
            database=os.getenv('POSTGRES_DB', 'postgres'),
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', '5432')
        )
        con.set_isolation_level(psql.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        return con
    except psql.OperationalError as e:
        pytest.skip(f"Cannot connect to PostgreSQL: {e}")


def test_psql_no_timeout():
    """Test that a simple query completes without timeout."""
    print("Testing PostgreSQL query without timeout...")
    
    con = get_psql_connection()
    cur = con.cursor()
    
    # Simple query that should complete quickly
    query = "SELECT 1 as result"
    result = contraction_eval_psql(query, cur, con, timeout=5)
    
    assert result is not None, "Query should complete successfully"
    assert result == [(1,)], f"Expected [(1,)], got {result}"
    
    cur.close()
    con.close()
    print("✓ Query completed successfully without timeout")


def test_psql_with_timeout():
    """Test that a long-running query times out."""
    print("\nTesting PostgreSQL query with timeout...")
    
    con = get_psql_connection()
    cur = con.cursor()
    
    # Create a query that takes a long time (generate large series and cross join)
    query = """
    SELECT count(*) 
    FROM generate_series(1, 50000) AS t1, 
         generate_series(1, 50000) AS t2
    """
    
    start_time = time.time()
    result = contraction_eval_psql(query, cur, con, timeout=2)
    elapsed = time.time() - start_time
    
    cur.close()
    con.close()
    
    assert result is None, "Query should timeout and return None"
    assert elapsed < 4, f"Timeout should occur around 2 seconds, took {elapsed:.2f}s"
    print(f"✓ Query timed out as expected after {elapsed:.2f}s")


def test_psql_timeout_none():
    """Test that timeout=None allows query to complete normally."""
    print("\nTesting PostgreSQL query with timeout=None...")
    
    con = get_psql_connection()
    cur = con.cursor()
    
    query = "SELECT 42 as answer"
    result = contraction_eval_psql(query, cur, con, timeout=None)
    
    assert result is not None, "Query should complete successfully"
    assert result == [(42,)], f"Expected [(42,)], got {result}"
    
    cur.close()
    con.close()
    print("✓ Query with timeout=None completed successfully")


def test_psql_quick_query_with_timeout():
    """Test that a quick query completes even with a timeout set."""
    print("\nTesting quick PostgreSQL query with timeout set...")
    
    con = get_psql_connection()
    cur = con.cursor()
    
    query = "SELECT sum(x) FROM generate_series(1, 1000) AS t(x)"
    result = contraction_eval_psql(query, cur, con, timeout=10)
    
    assert result is not None, "Quick query should complete before timeout"
    assert result == [(500500,)], f"Expected [(500500,)], got {result}"
    
    cur.close()
    con.close()
    print("✓ Quick query completed before timeout")


def test_psql_with_temp_table():
    """Test timeout with a more realistic query using CTEs."""
    print("\nTesting PostgreSQL query with CTE and timeout...")
    
    con = get_psql_connection()
    cur = con.cursor()
    
    # Query that creates data similar to the tensor queries
    query = """
    WITH tensor_a(i, j, re, im) AS (
        VALUES 
            (0, 0, 1.0, 0.0),
            (0, 1, 0.0, 0.0),
            (1, 0, 0.0, 0.0),
            (1, 1, 1.0, 0.0)
    ),
    tensor_b(j, k, re, im) AS (
        VALUES 
            (0, 0, 1.0, 0.0),
            (0, 1, 0.0, 1.0),
            (1, 0, 0.0, -1.0),
            (1, 1, 1.0, 0.0)
    )
    SELECT 
        a.i, b.k,
        sum(a.re * b.re - a.im * b.im) as re,
        sum(a.re * b.im + a.im * b.re) as im
    FROM tensor_a a
    JOIN tensor_b b ON a.j = b.j
    GROUP BY a.i, b.k
    ORDER BY a.i, b.k
    """
    
    result = contraction_eval_psql(query, cur, con, timeout=5)
    
    assert result is not None, "Tensor-like query should complete"
    assert len(result) == 4, f"Expected 4 results, got {len(result)}"
    
    cur.close()
    con.close()
    print(f"✓ Tensor-like query completed successfully with {len(result)} results")


def test_psql_connection_reuse():
    """Test that connection can be reused after timeout."""
    print("\nTesting PostgreSQL connection reuse after timeout...")
    
    con = get_psql_connection()
    cur = con.cursor()
    
    # First query - should timeout
    long_query = """
    SELECT count(*) 
    FROM generate_series(1, 40000) AS t1, 
         generate_series(1, 40000) AS t2
    """
    
    result1 = contraction_eval_psql(long_query, cur, con, timeout=1)
    assert result1 is None, "Long query should timeout"
    
    # Connection should still be usable for a quick query
    # Need to recreate cursor after timeout
    cur.close()
    cur = con.cursor()
    
    quick_query = "SELECT 123"
    result2 = contraction_eval_psql(quick_query, cur, con, timeout=5)
    
    assert result2 is not None, "Connection should be reusable after timeout"
    assert result2 == [(123,)], f"Expected [(123,)], got {result2}"
    
    cur.close()
    con.close()
    print("✓ Connection successfully reused after timeout")


if __name__ == "__main__":
    if not PSQL_AVAILABLE:
        print("PostgreSQL (psycopg2) not available - skipping tests")
        exit(0)
    
    print("=" * 60)
    print("PostgreSQL Timeout Tests")
    print("=" * 60)
    
    try:
        test_psql_no_timeout()
        test_psql_timeout_none()
        test_psql_quick_query_with_timeout()
        test_psql_with_temp_table()
        test_psql_connection_reuse()
        test_psql_with_timeout()  # This one last since it's slow
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise
