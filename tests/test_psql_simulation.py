"""
Integration test for PostgreSQL timeout in quantum circuit simulation.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import pytest
import InfiniQuantumSim.TLtensor as tlt
from dotenv import load_dotenv
load_dotenv()
# Try to import psycopg2, skip tests if not available
try:
    import psycopg2 as psql
    PSQL_AVAILABLE = True
except ImportError:
    PSQL_AVAILABLE = False
    pytest.skip("PostgreSQL not available", allow_module_level=True)


def check_psql_connection():
    """Check if PostgreSQL is available and connectable."""
    try:
        con = psql.connect(
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'password'),
            database=os.getenv('POSTGRES_DB', 'postgres'),
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', '5432')
        )
        con.close()
        return True
    except:
        return False


def test_small_circuit_psql():
    """Test that a small circuit completes successfully with PostgreSQL."""
    if not check_psql_connection():
        pytest.skip("PostgreSQL not available")
        return
    
    print("Testing small circuit with PostgreSQL (no timeout)...")
    
    circuit_dict = tlt.generate_ghz_circuit(3)
    qc = tlt.QuantumCircuit(circuit_dict=circuit_dict)
    
    results = qc.benchmark_ciruit_performance(
        n_runs=2, 
        oom=['sqlite', 'np-one-shot', 'np-mps', 'ducksql'], 
        timeout_seconds=30
    )
    
    print(f"PostgreSQL results: {results['psql']}")
    
    assert results['psql']['time'] is not None, "PostgreSQL should have completed"
    assert len(results['psql']['time']) > 0, "Should have timing results"
    print(f"✓ Small circuit completed: {len(results['psql']['time'])} runs")


def test_large_circuit_psql_timeout():
    """Test that a large circuit respects timeout with PostgreSQL."""
    if not check_psql_connection():
        pytest.skip("PostgreSQL not available")
        return
    
    print("\nTesting large circuit with PostgreSQL timeout...")
    
    circuit_dict = tlt.generate_qft_circuit(12)  # 12-qubit QFT
    qc = tlt.QuantumCircuit(circuit_dict=circuit_dict)
    
    start_time = time.time()
    
    results = qc.benchmark_ciruit_performance(
        n_runs=5, 
        oom=['sqlite', 'np-one-shot', 'np-mps', 'ducksql'], 
        timeout_seconds=1  # 1 second timeout
    )
    
    elapsed = time.time() - start_time
    
    print(f"PostgreSQL results: {results['psql']}")
    print(f"Total elapsed time: {elapsed:.2f}s")
    
    # Check if timeout occurred
    if results['psql'].get('timeout'):
        print("✓ Timeout was properly detected")
    
    assert elapsed < 30, f"Should timeout quickly, took {elapsed:.2f}s"


def test_all_databases():
    """Test SQLite, DuckDB, and PostgreSQL together."""
    if not check_psql_connection():
        pytest.skip("PostgreSQL not available")
        return
    
    print("\nTesting all databases (SQLite, DuckDB, PostgreSQL) with timeout...")
    
    circuit_dict = tlt.generate_ghz_circuit(7)
    qc = tlt.QuantumCircuit(circuit_dict=circuit_dict)
    
    results = qc.benchmark_ciruit_performance(
        n_runs=3, 
        oom=['np-one-shot', 'np-mps'], 
        timeout_seconds=5
    )
    
    print(f"SQLite results: {results['sqlite']}")
    print(f"DuckDB results: {results['ducksql']}")
    print(f"PostgreSQL results: {results['psql']}")
    
    # All should complete for this moderate-sized circuit
    assert results['sqlite']['time'] is not None, "SQLite should complete"
    assert results['ducksql']['time'] is not None, "DuckDB should complete"
    assert results['psql']['time'] is not None, "PostgreSQL should complete"
    
    print(f"✓ SQLite: {len(results['sqlite']['time'])} runs")
    print(f"✓ DuckDB: {len(results['ducksql']['time'])} runs")
    print(f"✓ PostgreSQL: {len(results['psql']['time'])} runs")


if __name__ == "__main__":
    if not PSQL_AVAILABLE:
        print("PostgreSQL (psycopg2) not available - skipping tests")
        exit(0)
    
    if not check_psql_connection():
        print("PostgreSQL server not running or not accessible - skipping tests")
        exit(0)
    
    print("=" * 70)
    print("PostgreSQL Timeout Integration Tests (Quantum Circuit Simulation)")
    print("=" * 70)
    
    try:
        test_small_circuit_psql()
        test_all_databases()
        test_large_circuit_psql_timeout()
        
        print("\n" + "=" * 70)
        print("✓ All PostgreSQL integration tests completed!")
        print("=" * 70)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise
