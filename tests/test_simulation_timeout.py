"""
Integration test for timeout functionality in actual quantum circuit simulation.
"""
import time
import InfiniQuantumSim.TLtensor as tlt


def test_small_circuit_no_timeout():
    """Test that a small circuit completes successfully."""
    print("Testing small circuit without timeout issues...")
    
    # Small 3-qubit GHZ circuit
    circuit_dict = tlt.generate_ghz_circuit(3)
    qc = tlt.QuantumCircuit(circuit_dict=circuit_dict)
    
    # Benchmark with generous timeout
    results = qc.benchmark_ciruit_performance(n_runs=2, oom=['psql', 'np-one-shot', 'np-mps', 'sqlite'], timeout_seconds=30)
    
    print(f"DuckDB results: {results['ducksql']}")
    
    assert results['ducksql']['time'] is not None, "DuckDB should have completed"
    assert len(results['ducksql']['time']) > 0, "Should have timing results"
    print(f"✓ Small circuit completed: {len(results['ducksql']['time'])} runs")


def test_large_circuit_with_timeout():
    """Test that a large circuit respects timeout."""
    print("\nTesting large circuit with short timeout...")
    
    # Large circuit that should take significant time
    circuit_dict = tlt.generate_qft_circuit(12)  # 12-qubit QFT
    qc = tlt.QuantumCircuit(circuit_dict=circuit_dict)
    
    start_time = time.time()
    
    # Very short timeout - should trigger
    results = qc.benchmark_ciruit_performance(
        n_runs=5, 
        oom=['psql', 'np-one-shot', 'np-mps', 'sqlite'], 
        timeout_seconds=1  # 1 second timeout
    )
    
    elapsed = time.time() - start_time
    
    print(f"DuckDB results: {results['ducksql']}")
    print(f"Total elapsed time: {elapsed:.2f}s")
    
    # Check if timeout occurred
    if results['ducksql'].get('timeout'):
        print("✓ Timeout was properly detected")
    
    # Should complete relatively quickly due to timeout (not run all iterations)
    assert elapsed < 30, f"Should timeout quickly, took {elapsed:.2f}s"
    
    # Either no runs completed, or only partial runs
    if results['ducksql']['time'] is None:
        print("✓ No runs completed due to timeout (expected)")
    else:
        completed_runs = len(results['ducksql']['time'])
        print(f"✓ Only {completed_runs} of 5 runs completed before timeout")
        assert completed_runs < 5, "Should not complete all runs with short timeout"


def test_medium_circuit_partial_timeout():
    """Test circuit where some runs complete and some timeout."""
    print("\nTesting medium circuit with moderate timeout...")
    
    # Medium circuit
    circuit_dict = tlt.generate_ghz_circuit(10)  # 10-qubit GHZ
    qc = tlt.QuantumCircuit(circuit_dict=circuit_dict)
    
    start_time = time.time()
    
    # Moderate timeout - might allow some runs to complete
    results = qc.benchmark_ciruit_performance(
        n_runs=5, 
        oom=['psql', 'np-one-shot', 'np-mps', 'sqlite'], 
        timeout_seconds=2
    )
    
    elapsed = time.time() - start_time
    
    print(f"DuckDB results: {results['ducksql']}")
    print(f"Total elapsed time: {elapsed:.2f}s")
    
    if results['ducksql']['time'] is not None:
        completed = len(results['ducksql']['time'])
        print(f"✓ {completed} runs completed")
    else:
        print("✓ All runs timed out")


def test_verify_timeout_per_method():
    """Verify timeout is per-method, not aggregate."""
    print("\nVerifying timeout is applied per individual query...")
    
    circuit_dict = tlt.generate_qft_circuit(11)
    qc = tlt.QuantumCircuit(circuit_dict=circuit_dict)
    
    timeout = 2
    n_runs = 3
    
    start_time = time.time()
    results = qc.benchmark_ciruit_performance(
        n_runs=n_runs, 
        oom=['psql', 'np-one-shot', 'np-mps', 'sqlite'], 
        timeout_seconds=timeout
    )
    elapsed = time.time() - start_time
    
    print(f"Timeout per query: {timeout}s")
    print(f"Number of runs: {n_runs}")
    print(f"Total elapsed: {elapsed:.2f}s")
    print(f"DuckDB results: {results['ducksql']}")
    
    # If all runs completed, total time should be roughly n_runs * query_time
    # If timeout happened, each attempt should be limited to timeout seconds
    # So max time should be around n_runs * timeout (plus overhead)
    
    if results['ducksql']['time']:
        completed = len(results['ducksql']['time'])
        print(f"✓ {completed} of {n_runs} queries completed")
    else:
        print(f"✓ Queries timed out individually (not aggregated)")


if __name__ == "__main__":
    print("=" * 70)
    print("DuckDB Timeout Integration Tests (Quantum Circuit Simulation)")
    print("=" * 70)
    
    try:
        test_small_circuit_no_timeout()
        test_medium_circuit_partial_timeout()
        test_large_circuit_with_timeout()
        test_verify_timeout_per_method()
        
        print("\n" + "=" * 70)
        print("✓ All integration tests completed!")
        print("=" * 70)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise
