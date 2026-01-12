"""
SQLite backend for quantum circuit simulation queries.
"""
import threading


def contraction_eval_sqlite(query, db_con, db_cur, timeout=None):
    """Execute SQLite query with optional timeout.
    
    Args:
        query: SQL query to execute
        db_con: Database connection
        db_cur: Database cursor
        timeout: Maximum execution time in seconds. If None, no timeout.
        
    Returns:
        Query results or None if timeout occurred
    """
    if timeout is None:
        result = db_cur.execute(query)
        db_con.commit()
        return result.fetchall()
    
    # SQLite requires connection to be used in the same thread
    # We need to set a timeout on the connection itself
    result_data = [None]
    exception = [None]
    
    def run_query():
        try:
            # Set a progress handler that checks if we should abort
            db_con.set_progress_handler(None, 0)  # Clear any existing handler
            result = db_cur.execute(query)
            db_con.commit()
            result_data[0] = result.fetchall()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=run_query, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        # Timeout occurred - interrupt the query
        try:
            db_con.interrupt()
        except:
            pass
        
        # Give thread time to clean up
        thread.join(timeout=0.5)
        
        return None  # Indicate timeout
    
    if exception[0] is not None:
        raise exception[0]
    
    return result_data[0]
