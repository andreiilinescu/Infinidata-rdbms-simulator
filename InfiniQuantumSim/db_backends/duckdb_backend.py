"""
DuckDB backend for quantum circuit simulation queries.
"""
import threading
import duckdb


def contraction_eval_duckdb(query, timeout=None):
    """Execute DuckDB query with optional timeout.
    
    Args:
        query: SQL query to execute
        timeout: Maximum execution time in seconds. If None, no timeout.
        
    Returns:
        Query results or None if timeout occurred
    """
    if timeout is None:
        return duckdb.sql(query).fetchall()
    
    # Create a persistent connection for timeout handling
    conn = duckdb.connect()
    result = [None]
    exception = [None]
    
    def run_query():
        try:
            result[0] = conn.sql(query).fetchall()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=run_query, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        # Timeout occurred - interrupt the query
        try:
            conn.interrupt()
        except:
            pass
        
        # Give thread time to clean up, but don't wait forever
        thread.join(timeout=0.5)
        
        # Close connection regardless of thread state
        try:
            conn.close()
        except:
            pass
        
        return None  # Indicate timeout
    
    # Query completed normally
    try:
        conn.close()
    except:
        pass
    
    if exception[0] is not None:
        raise exception[0]
    
    return result[0]
