"""
Test timeout functionality for SQLite query execution.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import sqlite3
from InfiniQuantumSim.db_backends import contraction_eval_sqlite


def test_sqlite_no_timeout():
    """Test that a simple query completes without timeout."""
    print("Testing SQLite query without timeout...")
    
    con = sqlite3.connect(':memory:', check_same_thread=False)
    cur = con.cursor()
    
    # Simple query that should complete quickly
    query = "SELECT 1 as result"
    result = contraction_eval_sqlite(query, con, cur, timeout=5)
    
    assert result is not None, "Query should complete successfully"
    assert result == [(1,)], f"Expected [(1,)], got {result}"
    
    con.close()
    print("✓ Query completed successfully without timeout")


def test_sqlite_with_timeout():
    """Test that a long-running query times out."""
    print("\nTesting SQLite query with timeout...")
    
    con = sqlite3.connect(':memory:', check_same_thread=False)
    cur = con.cursor()
    
    # Create a query that takes a long time (large cross join)
    query = """
    WITH RECURSIVE numbers(n) AS (
        SELECT 1
        UNION ALL
        SELECT n + 1 FROM numbers WHERE n < 50000
    )
    SELECT count(*) FROM numbers n1, numbers n2
    """
    
    start_time = time.time()
    result = contraction_eval_sqlite(query, con, cur, timeout=2)
    elapsed = time.time() - start_time
    
    con.close()
    
    assert result is None, "Query should timeout and return None"
    assert elapsed < 4, f"Timeout should occur around 2 seconds, took {elapsed:.2f}s"
    print(f"✓ Query timed out as expected after {elapsed:.2f}s")


def test_sqlite_timeout_none():
    """Test that timeout=None allows query to complete normally."""
    print("\nTesting SQLite query with timeout=None...")
    
    con = sqlite3.connect(':memory:', check_same_thread=False)
    cur = con.cursor()
    
    query = "SELECT 42 as answer"
    result = contraction_eval_sqlite(query, con, cur, timeout=None)
    
    assert result is not None, "Query should complete successfully"
    assert result == [(42,)], f"Expected [(42,)], got {result}"
    
    con.close()
    print("✓ Query with timeout=None completed successfully")


def test_sqlite_quick_query_with_timeout():
    """Test that a quick query completes even with a timeout set."""
    print("\nTesting quick SQLite query with timeout set...")
    
    con = sqlite3.connect(':memory:', check_same_thread=False)
    cur = con.cursor()
    
    # Create a table and insert data
    cur.execute("CREATE TABLE test_nums (x INTEGER)")
    for i in range(1, 1001):
        cur.execute("INSERT INTO test_nums VALUES (?)", (i,))
    con.commit()
    
    query = "SELECT sum(x) FROM test_nums"
    result = contraction_eval_sqlite(query, con, cur, timeout=10)
    
    assert result is not None, "Quick query should complete before timeout"
    assert result == [(500500,)], f"Expected [(500500,)], got {result}"
    
    con.close()
    print("✓ Quick query completed before timeout")


def test_sqlite_with_temp_table():
    """Test timeout with a more realistic query using temp tables."""
    print("\nTesting SQLite query with temp table and timeout...")
    
    con = sqlite3.connect(':memory:', check_same_thread=False)
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
    
    result = contraction_eval_sqlite(query, con, cur, timeout=5)
    
    assert result is not None, "Tensor-like query should complete"
    assert len(result) == 4, f"Expected 4 results, got {len(result)}"
    
    con.close()
    print(f"✓ Tensor-like query completed successfully with {len(result)} results")


if __name__ == "__main__":
    print("=" * 60)
    print("SQLite Timeout Tests")
    print("=" * 60)
    
    try:
        test_sqlite_no_timeout()
        test_sqlite_timeout_none()
        test_sqlite_quick_query_with_timeout()
        test_sqlite_with_temp_table()
        test_sqlite_with_timeout()  # This one last since it's slow
        
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
