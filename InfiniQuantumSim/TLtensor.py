import json, random

import numpy as np
np.set_printoptions(legacy='1.25')
import opt_einsum as oe
import InfiniQuantumSim.sqlEinSum as ses

from InfiniQuantumSim.sql_commands import sql_einsum_query
from InfiniQuantumSim.mps import MPS
from InfiniQuantumSim.utils import INDICES

def sql_to_np(db_result, expected_shape, complex_flag = False):
    # Initialize an empty NumPy array with the expected shape
    tensor = np.zeros(expected_shape, dtype=complex) if complex_flag else np.zeros(expected_shape)
    for row in db_result:
        # Extract indices and real and imaginary parts
        if complex_flag:
            *indices, re_part, im_part = row
            tensor[tuple(indices)] = re_part + 1j * im_part
        else:
            *indices, value = row
            tensor[tuple(indices)] = value
        
    return tensor


# Gate classes as provided
class Gate:
    def __init__(self, qubits, tensor, name = None, two_qubit_gate = False):
        self.qubits = qubits  # List of qubits the gate acts on
        self.tensor = tensor  # Gate matrix
        self.two_qubit_gate = two_qubit_gate
        self.gate_name = name

class HadamardGate(Gate):
    def __init__(self, qubit):
        matrix = (1/np.sqrt(2)) * np.array([[1.+0.j, 1.+0.j], [1.+0.j, -1.+0.j]])
        super().__init__([qubit], matrix, "H")

class CNOTGate(Gate):
    def __init__(self, control_qubit, target_qubit):
        # CNOT gate matrix
        matrix = np.array([
            [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],  # |00> -> |00>
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],  # |01> -> |01>
            [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],  # |10> -> |11>
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]   # |11> -> |10>
        ])
        super().__init__([control_qubit, target_qubit], matrix.reshape(2,2,2,2), "CX", True)

class SWAPGate(Gate):
    """Swap gate acting on two qubits."""
    def __init__(self, qubit1, qubit2):
        # SWAP gate matrix
        matrix = np.array([
            [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]
        ])
        super().__init__([qubit1, qubit2], matrix.reshape(2,2,2,2), "SW", True)

# Define the CZ gate
class CZGate(Gate):
    """Controlled-Z gate acting on two qubits."""
    def __init__(self, control_qubit, target_qubit):
        # CZ gate matrix
        matrix = np.array([
            [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],  # |00> -> |00>
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],  # |01> -> |01>
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],  # |10> -> |10>
            [0.+0.j, 0.+0.j, 0.+0.j, -1.+0.j]  # |11> -> -|11>
        ])
        super().__init__([control_qubit, target_qubit], matrix.reshape(2,2,2,2), "CZ", True)

class CYGate(Gate):
    def __init__(self, control_qubit, target_qubit):
        # CY gate matrix
        matrix = np.array([
            [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.-1.j],
            [0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j] 
        ])
        super().__init__([control_qubit, target_qubit], matrix.reshape(2,2,2,2), "CY", True)

class GGate(Gate):
    def __init__(self, qubit, p):
        super().__init__([qubit], np.array([[np.sqrt(1/p)+0.j, -np.sqrt(1-(1/p))+0.j], [np.sqrt(1-(1/p))+0.j, np.sqrt(1/p)+0.j]]), "G"+str(p),)

class XGate(Gate):
    def __init__(self, qubit):
        super().__init__([qubit], np.array([[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]]), "X",)

class YGate(Gate):
    def __init__(self, qubit):
        super().__init__([qubit], np.array([[0.+0.j, 0.-1j], [0.+1j, 0.+0.j]]), "Y")

class ZGate(Gate):
    def __init__(self, qubit):
        super().__init__([qubit], np.array([[1.+0.j, 0.+0.j], [0.+0.j, -1.+0.j]]), "Z")

class SGate(Gate):
    def __init__(self, qubit):
        matrix = np.array([[1.+0.j, 0.+0.j],
                           [0.+0.j, 0.+1j]], dtype=complex)
        super().__init__([qubit], matrix, "S")

class TGate(Gate):
    def __init__(self, qubit):
        matrix = np.array([[1.+0.j, 0.+0.j],
                           [0.+0.j, np.exp(1j * np.pi / 4)]], dtype=complex)
        super().__init__([qubit], matrix, "T")

class RYGate(Gate):
    def __init__(self, qubit, theta):
        matrix = np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
        super().__init__([qubit], matrix, "RY")

class RGate(Gate):
    def __init__(self, qubit, k):
        matrix = np.array([[1.+0.j, 0.+0.j], [0.+0.j, np.exp(2j*np.pi/2**k)]], dtype=complex)
        super().__init__([qubit], matrix, "R"+str(k))

class CRGate(Gate):
    def __init__(self, control_qubit, target_qubit, k):
        matrix = np.array([
            [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, np.exp(2j * np.pi / 2**k)]
        ], dtype=complex)
        
        super().__init__([control_qubit, target_qubit], matrix.reshape(2,2,2,2), "CR"+str(k), True)

class CUGate(Gate):
    def __init__(self, control_qubit, target_qubit, U, exponent, gate_name = None):
        # Compute U^e using matrix exponentiation
        U_powered = np.linalg.matrix_power(U, exponent)

        if gate_name is None:
            gate_name = "CU" + str(exponent)
        
        # Construct the 4x4 controlled-U matrix
        matrix = np.array([
            [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],               # |00⟩ stays |00⟩
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],               # |01⟩ stays |01⟩
            [0.+0.j, 0.+0.j, U_powered[0,0], U_powered[0,1]], # Controlled part on |10⟩
            [0.+0.j, 0.+0.j, U_powered[1,0], U_powered[1,1]]  # Controlled part on |11⟩
        ], dtype=complex)

        # Initialize the base Gate class with the control and target qubits, the matrix, and the name
        super().__init__([control_qubit, target_qubit], matrix.reshape(2, 2, 2, 2), gate_name, True)


# Define the quantum circuit
class QuantumCircuit:
    def __init__(self, num_qubits = None, circuit_dict = None):
        if circuit_dict is None:
            self.num_qubits = num_qubits
        else:
            n_qubit_key = [key for key in circuit_dict.keys() if "bit" in key and "n" in key]
            if len(n_qubit_key) == 0:
                raise ValueError("No key for number of qubits found in json!")
            else:
                n_qubit_key = n_qubit_key[0]

            self.num_qubits = circuit_dict[n_qubit_key]
        self.gates = []

        self.tensor_uniques ={"T0": np.array([1.+0.j, 0.+0.j])}
        self.einsum = ""
        self.mps = None
        self.con = None
        self.cur = None

        self.dispatcher = {
            'H': self.H,
            'CNOT': self.CNOT,
            'SWAP': self.SWAP,
            'CZ': self.CZ,
            'CY': self.CY,
            'X': self.X,
            'Y': self.Y,
            'Z': self.Z,
            'R': self.R,
            'G': self.G,
            'CR': self.CR,
            'CU': self.CU
        }

        if circuit_dict is not None:
            self.build_circuit_from_dict(circuit_dict['gates'])

    def add_gate(self, gate):
        self.gates.append(gate)
        if gate.gate_name not in self.tensor_uniques.keys():
            self.tensor_uniques[gate.gate_name] = gate.tensor

    def H(self, qubit):
        self.add_gate(HadamardGate(qubit))

    def CNOT(self, control_qubit, target_qubit):
        self.add_gate(CNOTGate(control_qubit, target_qubit))

    def SWAP(self, qubit1, qubit2):
        self.add_gate(SWAPGate(qubit1, qubit2))

    def CZ(self, control_qubit, target_qubit):
        self.add_gate(CZGate(control_qubit, target_qubit))

    def CY(self, control_qubit, target_qubit):
        self.add_gate(CYGate(control_qubit, target_qubit))

    def X(self, qubit):
        self.add_gate(XGate(qubit))

    def Y(self, qubit):
        self.add_gate(YGate(qubit))

    def Z(self, qubit):
        self.add_gate(ZGate(qubit))

    def R(self, qubit, k):
        self.add_gate(RGate(qubit, k))

    def RY(self, qubit, theta):
        self.add_gate(RYGate(qubit, theta))

    def S(self, qubit):
        self.add_gate(SGate(qubit))

    def T(self, qubit):
        self.add_gate(TGate(qubit))

    def G(self, qubit, p):
        self.add_gate(GGate(qubit, p))

    def CR(self, control_qubit, target_qubit, k):
        self.add_gate(CRGate(control_qubit, target_qubit, k))

    def CU(self, control_qubit, target_qubit, U, exponent, name = None):
        self.add_gate(CUGate(control_qubit, target_qubit, U, exponent, name))

    def build_circuit_from_dict(self, gates):
        for gate_info in gates:
            try:
                gate = self.dispatcher[gate_info['gate']]
            except:
                raise ValueError('invalid gate: {}'.format(gate_info['gate']))
            
            match len(gate_info['qubits']):
                case 1:
                    qubit = gate_info['qubits'][0]
                    params = gate_info['params'] if 'params' in gate_info.keys() else {}
                    gate(qubit, **params)
                case 2:
                    control_qubit = gate_info['qubits'][0]
                    target_qubit = gate_info['qubits'][1]
                    params = gate_info['params'] if 'params' in gate_info.keys() else {}
                    gate(control_qubit, target_qubit, **params)

    def to_dict(self):
        for gate_info in self.gates:
            ...

    def to_query(self, complex=True):
        einstein, index_sizes, parameters = self.convert_to_einsum()
        opt_rg = oe.RandomGreedy(max_repeats=256, parallel=True)
        views = oe.helpers.build_views(einstein, index_sizes)
        _, path_info = oe.contract_path(einstein, *views, optimize=opt_rg)

        if not complex:
            for key, tensor in self.tensor_uniques.items():
                self.tensor_uniques[key] = tensor.real

        return sql_einsum_query(einstein, parameters, self.tensor_uniques, path_info=path_info, complex=complex)


    def convert_to_einsum(self):
        einsum_notation = ""
        current_out_index = [""]*self.num_qubits
        index_sizes = {}
        parameters = []

        # start by giving each qubit a single letter
        for last_index in range(self.num_qubits):
            einsum_notation += INDICES[last_index] + ","
            current_out_index[last_index] = INDICES[last_index]
            index_sizes[INDICES[last_index]] = 2
            parameters.append("T0")

        # now the gates are applied
        for gate in self.gates:
            last_index += 1
            match gate.two_qubit_gate:
                case True:
                    einsum_notation += INDICES[last_index:last_index+2] + current_out_index[gate.qubits[0]] + current_out_index[gate.qubits[1]] + ","
                    current_out_index[gate.qubits[0]] = INDICES[last_index]
                    current_out_index[gate.qubits[1]] = INDICES[last_index+1]
                    index_sizes[INDICES[last_index]] = 2
                    last_index += 1
                case False:
                    einsum_notation += INDICES[last_index] + current_out_index[gate.qubits[0]] + ","
                    current_out_index[gate.qubits[0]] = INDICES[last_index]

            parameters.append(gate.gate_name)
            index_sizes[INDICES[last_index]] = 2

        einsum_notation = einsum_notation[:-1]
        
        last_index += 1
        einsum_notation += "->" +  "".join(current_out_index)

        self.einsum = einsum_notation
        return einsum_notation, index_sizes, parameters
    
    def run(self, contr_method="np", one_shot=True, max_bond_dim: int = 8):       
        if one_shot:
            einstein, index_sizes, parameters = self.convert_to_einsum()
            opt_rg = oe.RandomGreedy(max_repeats=256, parallel=True)
            views = oe.helpers.build_views(einstein, index_sizes)
            path, path_info = oe.contract_path(einstein, *views, optimize=opt_rg)

            if "sql" in contr_method:
                query = sql_einsum_query(einstein, parameters, self.tensor_uniques, path_info=path_info, complex=True)
                match contr_method:
                    case "psql":
                        self.con, self.cur = ses.connect_and_setup_db("psql")
                    case "sqlite":
                        self.con, self.cur = ses.connect_and_setup_db("sqlite")
                    case "ducksql":
                        ...
                    case _:
                        raise ValueError(f"{contr_method} not implemented!")
                    
            match contr_method:
                case "np":
                    evidence = [self.tensor_uniques[name] for name in parameters]
                    result = oe.contract(einstein, *evidence, optimize=path)
                case "sqlite":
                    result = ses.contraction_eval_sqlite(query, self.con, self.cur)
                    # result = sql_to_np(result, (2,)*self.num_qubits, complex_flag=True)
                case "psql":
                    result = ses.contraction_eval_psql(query, self.con, self.cur)
                    result = sql_to_np(result, (2,)*self.num_qubits, complex_flag=True)
                case "ducksql":
                    result = ses.contraction_eval_duckdb(query)
                    result = sql_to_np(result, (2,)*self.num_qubits, complex_flag=True)
                case _:
                    raise ValueError("contraction method not implemented!")
        else:
            self.mps = MPS(self.num_qubits, contr_method=contr_method)
            for gate in self.gates:
                self.mps.apply_gate(gate, max_bond_dim)
            result = self.mps
            self.mps.__del__()

        return result

    def benchmark_ciruit_performance(self, n_runs, oom = [], p_size = None, timeout_seconds=15):
        # Create dict to store all benchmarking results
        performance = {
            "psql": {},
            "sqlite": {},
            "ducksql": {},
            "np-one-shot": {},
            "np-mps": {},
            "eqc": {}
        }

        einstein, index_sizes, parameters = self.convert_to_einsum()
        opt_rg = oe.RandomGreedy(max_repeats=256, parallel=True)
        views = oe.helpers.build_views(einstein, index_sizes)
        path, path_info = oe.contract_path(einstein, *views, optimize=opt_rg)

        # opt_einsum benchmark
        perf_dict = ses.np_time_contraction_eval(einstein, parameters, self.tensor_uniques, path, self, n_runs, skip_method = oom)
        performance["np-one-shot"] = perf_dict["one-shot"]
        performance["np-mps"] = perf_dict["mps"]
        
        # sql benchmark
        ## in-db one-shot benchmark
        perf_dict = ses.db_time_contraction_eval(einstein, parameters, self.tensor_uniques, path_info, n_runs, skip_db = oom, p_size=p_size, timeout_seconds=timeout_seconds)
        performance["sqlite"] = perf_dict["sqlite"]
        performance["psql"] = perf_dict["psql"]
        performance["ducksql"] = perf_dict["ducksql"]
        performance["eqc"] = perf_dict["eqc"]

        return performance
    
    def export_circuit_query(self):
        einstein, index_sizes, parameters = self.convert_to_einsum()
        opt_rg = oe.RandomGreedy(max_repeats=256, parallel=True)
        views = oe.helpers.build_views(einstein, index_sizes)
        _, path_info = oe.contract_path(einstein, *views, optimize=opt_rg)
        query = sql_einsum_query(einstein, parameters, self.tensor_uniques, path_info=path_info, complex=True)
        return {'query': query, 'n-qubits': self.num_qubits, 'einsum': self.einsum}
        
    def qft(self, qubits = [], reverse = False):
        """if qubits is empty, apply qft circuit to the whole circuit"""
        if not reverse:
            if len(qubits) == 0:
                for i in range(self.num_qubits):
                    self.H(i)
                    for j in range(i+1, self.num_qubits):
                        self.CR(j, i, j - i + 1)
            else:
                prev_qubit = qubits[0] - 1
                for i, q in enumerate(qubits):
                    if q - prev_qubit > 1:
                        raise ValueError("Provided qubit list should be provided as list of neighbouring qubits!")
                    
                    self.H(i)
                    for j in qubits[i+1:]:
                        self.CR(j, q, j - q + 1)
                    
                    prev_qubit = q
        else:
            if len(qubits) == 0:
                for i in range(self.num_qubits):
                    for j in range(i+1, self.num_qubits):
                        self.CR(j, i, j - i + 1)
                    self.H(i)
            else:
                prev_qubit = qubits[0] - 1
                for i, q in enumerate(qubits):
                    if q - prev_qubit > 1:
                        raise ValueError("Provided qubit list should be provided as list of neighbouring qubits!")
                    
                    for j in qubits[i+1:]:
                        self.CR(j, q, j - q + 1)
                    self.H(i)

                    prev_qubit = q
                

def generate_ghz_circuit(num_qubits, reverse=False):
    gates = []
    # Apply Hadamard to the first qubit
    gates.append({"qubits": [0], "gate": "H"})
    # Apply CNOT gates from qubit 0 to all other qubits
    for q in range(1, num_qubits):
        gates.append({"qubits": [q-1, q], "gate": "CNOT"})
    if reverse:
        gates.reverse()
    return {"number_of_qubits": num_qubits, "gates": gates}


def generate_qft_circuit(num_qubits, reverse=False):
    gates = []
    for j in range(num_qubits):
        # Apply Hadamard to qubit j
        gates.append({"qubits": [j], "gate": "H"})
        # Apply controlled R_k gates
        for k in range(j + 1, num_qubits):
            angle_exponent = k - j + 1
            gates.append({"qubits": [k, j], "gate": "CR", "params": {"k": angle_exponent}})
    if reverse:
        gates.reverse()
    # Note: Swap gates are omitted
    return {"number_of_qubits": num_qubits, "gates": gates}


def generate_qpe_circuit(num_qubits):
    control_gates = []
    
    # Define the unitary matrix U with eigenvalues 1 and -1 (Pauli-Z matrix)
    U = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Initialize the control qubits with Hadamard gates
    for j in range(num_qubits):
        control_gates.append({"qubits": [j], "gate": "H"})
    
    # Apply controlled-U^(2^j) gates from each control qubit to the target qubit
    for j in range(num_qubits):
        # Use the CUGate with Pauli-Z as U and 2^j as the exponent
        exponent = 2 ** j
        control_gates.append({'qubits': [], 'gate': 'CU', 'params': {'U': U, 'exponent': exponent}})
    
    # Apply the inverse Quantum Fourier Transform on the control qubits
    for j in range(num_qubits):
        # Apply the Hadamard gate after inverse QFT rotation gates (omit swaps for simplicity)
        for k in range(j):
            angle_exponent = j - k + 1
            control_gates.append({"qubits": [k, j], "gate": "CR", "params": {"k": angle_exponent}})
        control_gates.append({"qubits": [j], "gate": "H"})
    
    # Construct the full circuit dictionary
    return {
        "number_of_qubits": num_qubits + 1,  # Control register + 1 target qubit
        "gates": control_gates
    }


def generate_w_circuit(n_qubits, reverse=False):
    gates = []
    gates.append({"qubits": [0], "gate": "X"})
    gates.append({"qubits": [1], "gate": "G", "params": {"p": n_qubits}})
    gates.append({"qubits": [1, 0], "gate": "CNOT"})

    for i in range(n_qubits-2):
        U = GGate(0, n_qubits-1-i).tensor
        gates.append({"qubits": [i+1, i+2], "gate": "CU", "params": {"U": U, "exponent": 1, 'name': 'CG' + str(n_qubits-1-i)}})
        gates.append({"qubits": [i+2, i+1], "gate": "CNOT"})
    if reverse:
        gates.reverse()

    return {"number_of_qubits": n_qubits, "gates": gates}


def generate_hadamard_wall(n_qubits):
    gates = []

    for i in range(n_qubits):
        gates.append({"qubits": [i], "gate": "H"})

    return {"number_of_qubits": n_qubits, "gates": gates}

def generate_w_qft(n_qubits):
    circuit_dict = generate_w_circuit(n_qubits)
    qft_circuit_dict = generate_qft_circuit(n_qubits)

    circuit_dict["gates"].extend(qft_circuit_dict["gates"])

    return circuit_dict


def generate_ghz_qft(n_qubits):
    circuit_dict = generate_ghz_circuit(n_qubits)
    qft_circuit_dict = generate_qft_circuit(n_qubits)

    circuit_dict["gates"].extend(qft_circuit_dict["gates"])

    return circuit_dict


def generate_w_qft(n_qubits):
    circuit_dict = generate_w_circuit(n_qubits)
    qft_circuit_dict = generate_qft_circuit(n_qubits)

    circuit_dict["gates"].extend(qft_circuit_dict["gates"])

    return circuit_dict


def generate_ghz_proned(n_qubits, depth):
    gates = []
    reversed = False
    while len(gates) < depth:
        circuit_dict = generate_ghz_circuit(n_qubits, reversed)
        gates.extend(circuit_dict["gates"])
        reversed = not reversed

    gates = gates[:depth]

    return {"number_of_qubits": n_qubits, "gates": gates}


def create_query_json():
    circuits = {
        'ghz': generate_ghz_circuit,
        'qft': generate_qft_circuit
    }

    qubit_list = [5, 10]

    json_dict = []

    for name, circ in circuits.items():
        for n_qubits in qubit_list:
            circuit_dict = circ(n_qubits)
            qc = QuantumCircuit(circuit_dict=circuit_dict)
            query_obj = qc.export_circuit_query()
            query_obj['name'] = name
            json_dict.append(query_obj)

    fname = "query_collection.json"
    with open(fname, "w") as file:
        json.dump(json_dict, file)
