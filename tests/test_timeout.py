"""
Test timeout functionality for DuckDB query execution.
"""
import time
import duckdb
from InfiniQuantumSim.sqlEinSum import contraction_eval_duckdb


def test_duckdb_no_timeout():
    """Test that a simple query completes without timeout."""
    print("Testing DuckDB query without timeout...")
    
    # Simple query that should complete quickly
    query = "SELECT 1 as result"
    result = contraction_eval_duckdb(query, timeout=5)
    
    assert result is not None, "Query should complete successfully"
    assert result == [(1,)], f"Expected [(1,)], got {result}"
    print("✓ Query completed successfully without timeout")


def test_duckdb_with_timeout():
    """Test that a long-running query times out."""
    print("\nTesting DuckDB query with timeout...")
    
    # Create a query that takes a long time (large cross join)
    query = """
    WITH RECURSIVE numbers(n) AS (
        SELECT 1
        UNION ALL
        SELECT n + 1 FROM numbers WHERE n < 100000
    )
    SELECT count(*) FROM numbers n1, numbers n2
    """
    
    start_time = time.time()
    result = contraction_eval_duckdb(query, timeout=2)
    elapsed = time.time() - start_time
    
    assert result is None, "Query should timeout and return None"
    assert elapsed < 4, f"Timeout should occur around 2 seconds, took {elapsed:.2f}s"
    print(f"✓ Query timed out as expected after {elapsed:.2f}s")


def test_duckdb_timeout_none():
    """Test that timeout=None allows query to complete normally."""
    print("\nTesting DuckDB query with timeout=None...")
    
    query = "SELECT 42 as answer"
    result = contraction_eval_duckdb(query, timeout=None)
    
    assert result is not None, "Query should complete successfully"
    assert result == [(42,)], f"Expected [(42,)], got {result}"
    print("✓ Query with timeout=None completed successfully")


def test_duckdb_quick_query_with_timeout():
    """Test that a quick query completes even with a timeout set."""
    print("\nTesting quick DuckDB query with timeout set...")
    
    query = "SELECT sum(x) FROM (SELECT unnest(range(1, 1001)) as x)"
    result = contraction_eval_duckdb(query, timeout=10)
    
    assert result is not None, "Quick query should complete before timeout"
    assert result == [(500500,)], f"Expected [(500500,)], got {result}"
    print("✓ Quick query completed before timeout")


def test_duckdb_with_temp_table():
    """Test timeout with a more realistic query using temp tables."""
    print("\nTesting DuckDB query with temp table and timeout...")
    
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
    
    result = contraction_eval_duckdb(query, timeout=5)
    
    assert result is not None, "Tensor-like query should complete"
    assert len(result) == 4, f"Expected 4 results, got {len(result)}"
    print(f"✓ Tensor-like query completed successfully with {len(result)} results")


if __name__ == "__main__":
    print("=" * 60)
    print("DuckDB Timeout Tests")
    print("=" * 60)
    
    try:
        test_duckdb_no_timeout()
        test_duckdb_timeout_none()
        test_duckdb_quick_query_with_timeout()
        test_duckdb_with_temp_table()
        test_duckdb_with_timeout()  # This one last since it's slow
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise
