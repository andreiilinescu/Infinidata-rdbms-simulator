"""
PostgreSQL backend for quantum circuit simulation queries.
"""
import threading


def contraction_eval_psql(query, db_cur, timeout=None):
    """Execute PostgreSQL query with optional timeout.
    
    Args:
        query: SQL query to execute
        db_cur: Database cursor
        timeout: Maximum execution time in seconds. If None, no timeout.
        
    Returns:
        Query results or None if timeout occurred
    """
    if timeout is None:
        db_cur.execute(query)
        return db_cur.fetchall()
    
    # PostgreSQL with timeout support
    result_data = [None]
    exception = [None]
    
    def run_query():
        try:
            db_cur.execute(query)
            result_data[0] = db_cur.fetchall()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=run_query, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        # Timeout occurred - cancel the query
        try:
            # PostgreSQL cancel requires connection.cancel()
            # This needs to be called from connection, not cursor
            # For now, we'll let the thread timeout
            pass
        except:
            pass
        
        # Give thread time to clean up
        thread.join(timeout=0.5)
        
        return None  # Indicate timeout
    
    if exception[0] is not None:
        raise exception[0]
    
    return result_data[0]
