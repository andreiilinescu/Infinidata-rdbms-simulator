import tracemalloc
import gc
from timeit import default_timer as timer
import opt_einsum as oe
from numpy import array
import numpy as np
import sqlite3
import duckdb
import threading
import time

import InfiniQuantumSim.sql_commands as sqlC
import InfiniQuantumSim.TLtensor as tlt

def connect_and_setup_db(db: str = "sqlite"):
    match db:
        case "sqlite":
            con = sqlite3.connect(":memory:")
            cur = con.cursor()
        case "psql":
            import psycopg2 as psql
            con = psql.connect(user='postgres', password='password', database='postgres', host='localhost')
            con.set_isolation_level(psql.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            cur = con.cursor()

    return con, cur

def disconnect_db(con):
    con.close()

def change_page_size_sqlite(con, cur, size: int):
    if size < 512 or size > 65536:
        raise ValueError(f"sqlite page size has to be between 512 and 65536 bytes, not {size}!")
    if (size & (size-1) != 0):
        raise ValueError("size has to be a power of 2")
    
    cur.execute(f'PRAGMA page_size = {size};')
    cur.execute('VACUUM;')
    con.commit()

def contraction_eval_sqlite(query, db_con, db_cur):
    result = db_cur.execute(query)
    db_con.commit()
    return result.fetchall()

def contraction_eval_psql(query, db_cur):
    db_cur.execute(query)
    return db_cur.fetchall()

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
    completed = [False]
    
    def run_query():
        try:
            result[0] = conn.sql(query).fetchall()
            completed[0] = True
        except Exception as e:
            if not completed[0]:  # Only store exception if query didn't complete
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

def db_time_contraction_eval(einstein, parameters, tensors, path_info, n_runs: int, skip_db: list = [], p_size = None, timeout_seconds=None):
    eqc_tensor_mems = []
    eqc_tensor_times = []
    eqc_contr_mems = []
    eqc_contr_times = []

    tracemalloc.start()
    for _ in range(n_runs):
        tic = timer()
        tracemalloc.clear_traces()
        mem_tic, _ = tracemalloc.get_traced_memory()
        tensor_definitions = sqlC.var_list_to_sql(tensors)
        _, mem_toc = tracemalloc.get_traced_memory()
        toc = timer()
        eqc_tensor_mems.append(mem_toc - mem_tic)
        eqc_tensor_times.append(toc - tic)
        tracemalloc.clear_traces()
        
        tic = timer()
        tracemalloc.clear_traces()
        mem_tic, _ = tracemalloc.get_traced_memory()
        contraction = sqlC._einsum_notation_to_opt_sql(einstein, parameters, tensors, complex=True, path_info=path_info)
        _, mem_toc = tracemalloc.get_traced_memory()
        toc = timer()
        eqc_contr_mems.append(mem_toc - mem_tic)
        eqc_contr_times.append(toc - tic)
        tracemalloc.clear_traces()

        query = tensor_definitions + ")" + contraction.replace("WITH", ",")

    tracemalloc.stop()

    result = {"sqlite": {}, "psql": {}, "ducksql": {}, "eqc": {"tensor": {"memory": eqc_tensor_mems, "time": eqc_tensor_times}, "contraction": {"memory": eqc_contr_mems, "time": eqc_contr_times}}}
    if "sqlite" not in skip_db:
        ## one-shot SQLite performance
        con = sqlite3.connect(':memory:')
        cur = con.cursor()

        if p_size is not None:
            change_page_size_sqlite(con, cur, size=p_size)

        sqlite_mems = []
        sqlite_times = []
        contr_result = contraction_eval_sqlite(query, con, cur)
        tracemalloc.start()
        for _ in range(n_runs):
            tic = timer()
            mem_tic, _ = tracemalloc.get_traced_memory()
            contr_result = contraction_eval_sqlite(query, con, cur)
            _, mem_toc = tracemalloc.get_traced_memory()
            toc = timer()
            sqlite_mems.append(mem_toc - mem_tic)
            sqlite_times.append(toc - tic)
            tracemalloc.clear_traces()
        
        con.commit()
        con.close()

        result["sqlite"] = {"runs": n_runs, "time": sqlite_times, "memory": sqlite_mems, "non-zero": len(contr_result)}
        del con, cur, tic, toc, mem_tic, mem_toc
        gc.collect()
    else:
        result['sqlite'] = {"time": None, "memory": None, "iterations": None}

    if "psql" not in skip_db:
        ## one-shot PostgreSQL performance
        con, cur = connect_and_setup_db("psql")

        psql_mems = []
        psql_times = []
        contr_result = contraction_eval_psql(query, cur)
        tracemalloc.start()
        for _ in range(n_runs):
            tic = timer()
            mem_tic, _ = tracemalloc.get_traced_memory()
            contr_result = contraction_eval_psql(query, cur)
            _, mem_toc = tracemalloc.get_traced_memory()
            toc = timer()
            psql_mems.append(mem_toc - mem_tic)
            psql_times.append(toc - tic)
            tracemalloc.clear_traces()

        tracemalloc.stop()

        cur.close()
        con.close()

        result["psql"] = {"runs": n_runs, "time": psql_times, "memory": psql_mems, "non-zero": len(contr_result)}
        del con, cur, tic, toc, mem_tic, mem_toc
        gc.collect()
    else:
        result["psql"] = {"time": None, "memory": None, "non-zero": None}

    if "ducksql" not in skip_db:
        duck_mems = []
        duck_times = []
        
        # Initial test run to check if query completes
        contr_result = contraction_eval_duckdb(query, timeout=timeout_seconds)
        
        if contr_result is None:
            # Timeout occurred on first run
            result["ducksql"] = {"runs": 0, "time": None, "memory": None, "non-zero": None, "timeout": True}
        else:
            tracemalloc.start()
            timeout_count = 0
            
            for run_idx in range(n_runs):
                tic = timer()
                mem_tic, _ = tracemalloc.get_traced_memory()
                
                contr_result = contraction_eval_duckdb(query, timeout=timeout_seconds)
                
                if contr_result is None:
                    # Timeout on this run
                    timeout_count += 1
                    if timeout_count >= 3:  # Skip after 3 consecutive timeouts
                        break
                    continue
                
                _, mem_toc = tracemalloc.get_traced_memory()
                toc = timer()
                duck_mems.append(mem_toc - mem_tic)
                duck_times.append(toc - tic)
                tracemalloc.clear_traces()

            tracemalloc.stop()
            
            if duck_times:
                result["ducksql"] = {
                    "runs": len(duck_times), 
                    "time": duck_times, 
                    "memory": duck_mems, 
                    "non-zero": len(contr_result),
                    "timeout": timeout_count > 0
                }
            else:
                result["ducksql"] = {"runs": 0, "time": None, "memory": None, "non-zero": None, "timeout": True}
            
            gc.collect()
    else:
        result["ducksql"] = {"time": None, "memory": None, "non-zero": None}

    return result


def np_time_contraction_eval(einstein, parameters, tensors, path, circuit, n_runs, skip_method = []):
    evidence = [tensors[name] for name in parameters]

    if 'np-one-shot' not in skip_method:
        one_shot_mems = []
        one_shot_times = []
        # one-shot benchmark
        tracemalloc.start()
        for _ in range(n_runs):
            tic = timer()
            mem_tic, _ = tracemalloc.get_traced_memory()
            result = oe.contract(einstein, *evidence, optimize=path)
            _, mem_toc = tracemalloc.get_traced_memory()
            one_shot_mems.append(mem_toc - mem_tic)
            toc = timer()
            one_shot_times.append(toc - tic)
            tracemalloc.clear_traces()

        tracemalloc.stop()
        os_dict = {"runs": n_runs, "time": one_shot_times, "memory": one_shot_mems, "iterations": [1 / one_shot_time for one_shot_time in one_shot_times], "non-zero": np.count_nonzero(result)}

        del mem_tic, mem_toc, tic, toc, one_shot_mems, one_shot_times
        gc.collect()
    else:
        os_dict = {'time' : None, 'memory': None, 'iterations': None, "non-zero": None}

    if 'np-mps' not in skip_method:
        # MPS approach with sequential gate application
        mps_times = []
        mps_mems = []
        tracemalloc.start()
        for _ in range(n_runs):
            tracemalloc.clear_traces()
            mps_nonzero = 0
            mps_n_entries = 0
            sub_tic = timer()
            mem_tic, _ = tracemalloc.get_traced_memory()
            mps = tlt.MPS(circuit.num_qubits, contr_method="np")
            for gate in circuit.gates:
                mps.apply_gate(gate, max_bond_dim=10)
            _, mem_toc = tracemalloc.get_traced_memory()
            sub_toc = timer()
            mps_times.append(sub_toc - sub_tic)
            mps_mems.append(mem_toc - mem_tic)
            for tensor in mps.tensors:
                mps_nonzero += np.count_nonzero(tensor)
                mps_n_entries += tensor.size
            mps.__del__()
            del sub_tic, sub_toc, mem_toc, mem_tic
        tracemalloc.stop()
        mps_dict = {"runs": n_runs, "time": mps_times, "memory": mps_mems, "iterations": [1 / mps_time for mps_time in mps_times], "non-zero": mps_nonzero, "n_elem": mps_n_entries}
        del mps_mems, mps_times
    else:
        mps_dict = {'time' : None, 'memory': None, 'iterations': None, "non-zero": None, "n_elem": None}
    

    return {"one-shot": os_dict, "mps": mps_dict}