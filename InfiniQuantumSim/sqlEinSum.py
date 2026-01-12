import tracemalloc
import gc
from timeit import default_timer as timer
import opt_einsum as oe
from numpy import array
import numpy as np
import sqlite3

import InfiniQuantumSim.sql_commands as sqlC
import InfiniQuantumSim.TLtensor as tlt
from InfiniQuantumSim.db_backends import (
    contraction_eval_duckdb,
    contraction_eval_sqlite,
    contraction_eval_psql
)

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
        con = sqlite3.connect(':memory:', check_same_thread=False)
        cur = con.cursor()

        if p_size is not None:
            change_page_size_sqlite(con, cur, size=p_size)

        sqlite_mems = []
        sqlite_times = []
        
        # Initial test run to check if query completes
        contr_result = contraction_eval_sqlite(query, con, cur, timeout=timeout_seconds)
        
        if contr_result is None:
            # Timeout occurred on first run
            con.close()
            result["sqlite"] = {"runs": 0, "time": None, "memory": None, "non-zero": None, "timeout": True}
        else:
            tracemalloc.start()
            timeout_count = 0
            
            for run_idx in range(n_runs):
                tic = timer()
                mem_tic, _ = tracemalloc.get_traced_memory()
                
                contr_result = contraction_eval_sqlite(query, con, cur, timeout=timeout_seconds)
                
                if contr_result is None:
                    # Timeout on this run
                    timeout_count += 1
                    if timeout_count >= 3:  # Skip after 3 consecutive timeouts
                        break
                    continue
                
                _, mem_toc = tracemalloc.get_traced_memory()
                toc = timer()
                sqlite_mems.append(mem_toc - mem_tic)
                sqlite_times.append(toc - tic)
                tracemalloc.clear_traces()
            
            tracemalloc.stop()
            con.commit()
            con.close()

            if sqlite_times:
                result["sqlite"] = {
                    "runs": len(sqlite_times), 
                    "time": sqlite_times, 
                    "memory": sqlite_mems, 
                    "non-zero": len(contr_result),
                    "timeout": timeout_count > 0
                }
            else:
                result["sqlite"] = {"runs": 0, "time": None, "memory": None, "non-zero": None, "timeout": True}
            
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