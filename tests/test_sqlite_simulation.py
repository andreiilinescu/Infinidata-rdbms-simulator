"""
Integration test for SQLite timeout in quantum circuit simulation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import InfiniQuantumSim.TLtensor as tlt


def test_small_circuit_sqlite():
    """Test that a small circuit completes successfully with SQLite."""
    print("Testing small circuit with SQLite (no timeout)...")
    
    circuit_dict = tlt.generate_ghz_circuit(3)
    qc = tlt.QuantumCircuit(circuit_dict=circuit_dict)
    
    results = qc.benchmark_ciruit_performance(
        n_runs=2, 
        oom=['psql', 'np-one-shot', 'np-mps', 'ducksql'], 
        timeout_seconds=30
    )
    
    print(f"SQLite results: {results['sqlite']}")
    
    assert results['sqlite']['time'] is not None, "SQLite should have completed"
    assert len(results['sqlite']['time']) > 0, "Should have timing results"
    print(f"✓ Small circuit completed: {len(results['sqlite']['time'])} runs")


def test_large_circuit_sqlite_timeout():
    """Test that a large circuit respects timeout with SQLite."""
    print("\nTesting large circuit with SQLite timeout...")
    
    circuit_dict = tlt.generate_qft_circuit(12)  # 12-qubit QFT
    qc = tlt.QuantumCircuit(circuit_dict=circuit_dict)
    
    start_time = time.time()
    
    results = qc.benchmark_ciruit_performance(
        n_runs=5, 
        oom=['psql', 'np-one-shot', 'np-mps', 'ducksql'], 
        timeout_seconds=1  # 1 second timeout
    )
    
    elapsed = time.time() - start_time
    
    print(f"SQLite results: {results['sqlite']}")
    print(f"Total elapsed time: {elapsed:.2f}s")
    
    # Check if timeout occurred
    if results['sqlite'].get('timeout'):
        print("✓ Timeout was properly detected")
    
    assert elapsed < 30, f"Should timeout quickly, took {elapsed:.2f}s"


def test_both_databases():
    """Test both DuckDB and SQLite together."""
    print("\nTesting both SQLite and DuckDB with timeout...")
    
    circuit_dict = tlt.generate_ghz_circuit(8)
    qc = tlt.QuantumCircuit(circuit_dict=circuit_dict)
    
    results = qc.benchmark_ciruit_performance(
        n_runs=3, 
        oom=['psql', 'np-one-shot', 'np-mps'], 
        timeout_seconds=5
    )
    
    print(f"SQLite results: {results['sqlite']}")
    print(f"DuckDB results: {results['ducksql']}")
    
    # Both should complete for this moderate-sized circuit
    assert results['sqlite']['time'] is not None, "SQLite should complete"
    assert results['ducksql']['time'] is not None, "DuckDB should complete"
    
    print(f"✓ SQLite: {len(results['sqlite']['time'])} runs")
    print(f"✓ DuckDB: {len(results['ducksql']['time'])} runs")


if __name__ == "__main__":
    print("=" * 70)
    print("SQLite Timeout Integration Tests (Quantum Circuit Simulation)")
    print("=" * 70)
    
    try:
        test_small_circuit_sqlite()
        test_both_databases()
        test_large_circuit_sqlite_timeout()
        
        print("\n" + "=" * 70)
        print("✓ All SQLite integration tests completed!")
        print("=" * 70)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise
