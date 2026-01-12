import os, sys, json, datetime

import InfiniQuantumSim.TLtensor as tlt

def init_simulation_params():
    circuit_n_qubits = {
        'W': range(5,35),
        'QFT': range(5,25),
        'GHZ': range(5,35),
        'QPE': range(5,19)
    } 

    circuits = {
        'GHZ': tlt.generate_ghz_circuit, 
        'W': tlt.generate_w_circuit,
        'QFT': tlt.generate_qft_circuit,
        'QPE': tlt.generate_qpe_circuit,
    }

    return circuit_n_qubits, circuits


def init_test_params():
    circuit_n_qubits = {
        'W': range(5,10),
        'GHZ': range(5,10),
    } 

    circuits = {
        'GHZ': tlt.generate_ghz_circuit, 
        'W': tlt.generate_w_circuit,
    }

    return circuit_n_qubits, circuits


def save_results_to_json(results, fname = None):
    if fname is None:
        fname = datetime.datetime.now().strftime("results/%m%d%y")

    if fname[-5:] == ".json":
         fname = fname[:-5]

    if os.path.exists(fname + ".json"):
        i = 1
        while os.path.exists(fname + f"_{i}" + ".json"):
            i += 1

        fname += f"_{i}"

    fname += ".json"
    with open(fname, "w") as file:
        json.dump(results, file)


def simulation_benchmark(n_runs, circuit_n_qubits, circuits, mem_limit_bytes=2**34, time_limit_seconds=2**4, mem_db_only=False, timeout_seconds=15):
    results = {}
    skip_db = []

    if mem_db_only:
         skip_db = ["psql", "ducksql"]

    for cname, circuit in circuits.items():
        oom = skip_db
        results[cname] = {}
        progress = 0
        l_qbits = len(circuit_n_qubits[cname])-1
        for n_qubits in circuit_n_qubits[cname]:
            sys.stdout.write('\r')
            circuit_dict = circuit(n_qubits)
            qc = tlt.QuantumCircuit(circuit_dict=circuit_dict)

            results[cname][n_qubits] = qc.benchmark_ciruit_performance(n_runs, oom = oom,timeout_seconds=timeout_seconds)
            for method in results[cname][n_qubits].keys():
                if method in oom or method == "eqc":
                        continue
                mem_avg = sum(results[cname][n_qubits][method]["memory"])/n_runs
                tim_avg = sum(results[cname][n_qubits][method]["time"])/n_runs
                if "sql" in method:
                        mem_avg += sum(results[cname][n_qubits]["eqc"]["tensor"]["memory"])/n_runs
                        tim_avg += sum(results[cname][n_qubits]["eqc"]["tensor"]["time"])/n_runs
                        mem_avg += sum(results[cname][n_qubits]["eqc"]["contraction"]["memory"])/n_runs
                        tim_avg += sum(results[cname][n_qubits]["eqc"]["contraction"]["time"])/n_runs

                if  mem_avg >= mem_limit_bytes:
                        print(f'{method} kicked out due to memory limitations!\n')
                        oom.append(method)
                elif tim_avg >= time_limit_seconds:
                        print(f'{method} kicked out due to time limitations!\n')
                        oom.append(method)

            sys.stdout.write("[%-20s] %d%%" % ('='*int((20/l_qbits)*progress), progress*(100/l_qbits)))
            sys.stdout.flush()
            progress += 1

        print("\n" + cname + " done!")
    
    return results



if __name__ == "__main__":

    mem_db_only = False
    if len(sys.argv) > 1:
        if sys.argv[1] in ["sqlite", "test"]:
             mem_db_only = True
             
    
    n_runs = 20
    
    if sys.argv[1] != "test":
        circuit_n_qubits, circuits = init_simulation_params()
    else:
        circuit_n_qubits, circuits = init_test_params()
        n_runs = 2

    results = simulation_benchmark(n_runs, circuit_n_qubits, circuits, mem_db_only=mem_db_only)

    save_results_to_json(results)