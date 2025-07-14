#!/usr/bin/env python3
# Written by Andrea Giani
"""
quantum_dna_simulator.py
Quantum Genetics Accelerator - Quantum DNA Simulator
Simulates true quantum behavior for DNA sequences to validate FPGA results
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import cmath
from scipy.linalg import kron
from scipy.stats import entropy
import warnings

warnings.filterwarnings('ignore', category=np.ComplexWarning)

class DNABase(Enum):
    """DNA base enumeration"""
    A = 0  # 00
    T = 1  # 01
    G = 2  # 10
    C = 3  # 11

@dataclass
class QuantumDNAState:
    """Represents a quantum state of DNA sequence"""
    amplitudes: np.ndarray  # Complex amplitudes
    n_bases: int           # Number of DNA bases
    measurement_basis: str = "computational"  # "computational" or "biological"
    
    def __post_init__(self):
        """Normalize the quantum state"""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes = self.amplitudes / norm
        else:
            raise ValueError("Zero state vector")
    
    @property
    def probabilities(self) -> np.ndarray:
        """Get measurement probabilities"""
        return np.abs(self.amplitudes) ** 2
    
    @property
    def entropy(self) -> float:
        """Calculate von Neumann entropy of the quantum state"""
        probs = self.probabilities
        probs = probs[probs > 1e-12]  # Remove near-zero probabilities
        return entropy(probs, base=2)

class QuantumDNAGate:
    """Quantum gates for DNA processing"""
    
    @staticmethod
    def hadamard_base() -> np.ndarray:
        """Single-base Hadamard gate"""
        return np.array([[1, 1, 1, 1],
                        [1, -1, 1, -1],
                        [1, 1, -1, -1],
                        [1, -1, -1, 1]]) / 2
    
    @staticmethod
    def pauli_x_dna() -> np.ndarray:
        """DNA-specific Pauli-X (base flip)"""
        # A<->T, G<->C flips
        return np.array([[0, 1, 0, 0],  # A -> T
                        [1, 0, 0, 0],  # T -> A
                        [0, 0, 0, 1],  # G -> C
                        [0, 0, 1, 0]]) # C -> G
    
    @staticmethod
    def pauli_y_dna() -> np.ndarray:
        """DNA-specific Pauli-Y (complex base rotation)"""
        return np.array([[0, -1j, 0, 0],
                        [1j, 0, 0, 0],
                        [0, 0, 0, -1j],
                        [0, 0, 1j, 0]])
    
    @staticmethod
    def pauli_z_dna() -> np.ndarray:
        """DNA-specific Pauli-Z (phase flip)"""
        return np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, -1]])
    
    @staticmethod
    def cnot_dna() -> np.ndarray:
        """DNA CNOT gate for two bases"""
        # 16x16 matrix for 2-base system
        cnot = np.eye(16, dtype=complex)
        
        # Apply CNOT logic: if control base is A or G, flip target
        for i in range(16):
            control_base = (i >> 2) & 3  # Extract control base
            target_base = i & 3          # Extract target base
            
            if control_base in [0, 2]:  # A or G
                new_target = target_base ^ 1  # Flip T<->A, C<->G
                new_state = (control_base << 2) | new_target
                cnot[i, i] = 0
                cnot[new_state, i] = 1
        
        return cnot
    
    @staticmethod
    def aging_gate(aging_factor: float) -> np.ndarray:
        """Aging-induced decoherence gate"""
        # Decoherence increases with aging
        decoherence = aging_factor * 0.1
        
        # Dephasing matrix
        gate = np.eye(4, dtype=complex)
        for i in range(4):
            gate[i, i] *= np.exp(-decoherence * i)
        
        return gate
    
    @staticmethod
    def toffoli_dna() -> np.ndarray:
        """DNA Toffoli gate for three bases"""
        # 64x64 matrix for 3-base system
        toffoli = np.eye(64, dtype=complex)
        
        for i in range(64):
            control1 = (i >> 4) & 3  # First control base
            control2 = (i >> 2) & 3  # Second control base
            target = i & 3           # Target base
            
            # Apply Toffoli logic: if both controls are G and C, flip target
            if control1 == 2 and control2 == 3:  # G and C
                new_target = target ^ 3  # Complex flip
                new_state = (control1 << 4) | (control2 << 2) | new_target
                toffoli[i, i] = 0
                toffoli[new_state, i] = 1
        
        return toffoli

class QuantumDNASimulator:
    """
    Quantum simulator for DNA sequences with biological constraints
    """
    
    def __init__(self, n_bases: int = 8):
        self.n_bases = n_bases
        self.state_size = 4 ** n_bases
        self.gates = QuantumDNAGate()
        self.measurement_history = []
        self.fidelity_threshold = 0.95
        
    def create_initial_state(self, dna_sequence: List[DNABase]) -> QuantumDNAState:
        """Create initial quantum state from DNA sequence"""
        if len(dna_sequence) != self.n_bases:
            raise ValueError(f"DNA sequence length must be {self.n_bases}")
        
        # Create computational basis state
        state_index = 0
        for i, base in enumerate(dna_sequence):
            state_index += base.value * (4 ** (self.n_bases - 1 - i))
        
        amplitudes = np.zeros(self.state_size, dtype=complex)
        amplitudes[state_index] = 1.0
        
        return QuantumDNAState(amplitudes, self.n_bases)
    
    def create_superposition_state(self, base_states: List[List[DNABase]], 
                                 amplitudes: Optional[List[complex]] = None) -> QuantumDNAState:
        """Create superposition of multiple DNA states"""
        if amplitudes is None:
            amplitudes = [1.0] * len(base_states)
        
        if len(base_states) != len(amplitudes):
            raise ValueError("Number of states must match number of amplitudes")
        
        state_vector = np.zeros(self.state_size, dtype=complex)
        
        for dna_seq, amp in zip(base_states, amplitudes):
            state_index = 0
            for i, base in enumerate(dna_seq):
                state_index += base.value * (4 ** (self.n_bases - 1 - i))
            state_vector[state_index] = amp
        
        return QuantumDNAState(state_vector, self.n_bases)
    
    def apply_single_base_gate(self, state: QuantumDNAState, gate: np.ndarray, 
                              base_position: int) -> QuantumDNAState:
        """Apply single-base gate to specific position"""
        if base_position >= self.n_bases:
            raise ValueError(f"Base position {base_position} out of range")
        
        # Create full gate matrix using tensor products
        gate_matrices = []
        for i in range(self.n_bases):
            if i == base_position:
                gate_matrices.append(gate)
            else:
                gate_matrices.append(np.eye(4))
        
        # Build full gate through tensor products
        full_gate = gate_matrices[0]
        for i in range(1, len(gate_matrices)):
            full_gate = kron(full_gate, gate_matrices[i])
        
        new_amplitudes = full_gate @ state.amplitudes
        return QuantumDNAState(new_amplitudes, self.n_bases)
    
    def apply_two_base_gate(self, state: QuantumDNAState, gate: np.ndarray,
                           control_pos: int, target_pos: int) -> QuantumDNAState:
        """Apply two-base gate between control and target positions"""
        if control_pos >= self.n_bases or target_pos >= self.n_bases:
            raise ValueError("Base positions out of range")
        if control_pos == target_pos:
            raise ValueError("Control and target positions must be different")
        
        # For simplicity, apply gate to adjacent positions first
        # Full implementation would require complex tensor product arrangement
        if abs(control_pos - target_pos) == 1:
            # Adjacent bases - can use the gate directly
            gate_matrices = []
            for i in range(self.n_bases):
                if i == min(control_pos, target_pos):
                    gate_matrices.append(gate)
                    gate_matrices.append(np.eye(1))  # Skip next position
                elif i == max(control_pos, target_pos):
                    continue  # Already handled
                else:
                    gate_matrices.append(np.eye(4))
            
            # Build reduced gate matrix
            if len(gate_matrices) > 1:
                full_gate = gate_matrices[0]
                for i in range(1, len(gate_matrices)):
                    if gate_matrices[i].shape[0] > 1:
                        full_gate = kron(full_gate, gate_matrices[i])
            else:
                full_gate = gate_matrices[0]
            
            # Apply to reduced state space
            new_amplitudes = self._apply_reduced_gate(state.amplitudes, full_gate, 
                                                    control_pos, target_pos)
            return QuantumDNAState(new_amplitudes, self.n_bases)
        else:
            # Non-adjacent bases - use swap gates (simplified)
            return self._apply_non_adjacent_gate(state, gate, control_pos, target_pos)
    
    def _apply_reduced_gate(self, amplitudes: np.ndarray, gate: np.ndarray,
                           pos1: int, pos2: int) -> np.ndarray:
        """Apply gate to reduced state space"""
        # Simplified implementation for adjacent bases
        new_amplitudes = amplitudes.copy()
        
        # Group states by the two-base configuration
        base_pairs = {}
        for i in range(len(amplitudes)):
            if abs(amplitudes[i]) > 1e-12:
                # Extract the two-base state
                state_decomp = self._decompose_state_index(i)
                pair_key = (state_decomp[pos1], state_decomp[pos2])
                if pair_key not in base_pairs:
                    base_pairs[pair_key] = []
                base_pairs[pair_key].append(i)
        
        # Apply gate to each pair group
        for pair_indices in base_pairs.values():
            if len(pair_indices) > 1:
                # Extract substate
                substate = np.array([amplitudes[i] for i in pair_indices])
                # Apply gate (simplified)
                new_substate = gate[:len(substate), :len(substate)] @ substate
                # Put back
                for i, idx in enumerate(pair_indices):
                    if i < len(new_substate):
                        new_amplitudes[idx] = new_substate[i]
        
        return new_amplitudes
    
    def _decompose_state_index(self, index: int) -> List[int]:
        """Decompose state index into individual base values"""
        bases = []
        for i in range(self.n_bases):
            base_val = (index >> (2 * (self.n_bases - 1 - i))) & 3
            bases.append(base_val)
        return bases
    
    def _apply_non_adjacent_gate(self, state: QuantumDNAState, gate: np.ndarray,
                                control_pos: int, target_pos: int) -> QuantumDNAState:
        """Apply gate to non-adjacent bases using swap operations"""
        # Simplified: just apply to adjacent approximation
        return self.apply_two_base_gate(state, gate, 
                                      min(control_pos, target_pos),
                                      min(control_pos, target_pos) + 1)
    
    def apply_hadamard_transform(self, state: QuantumDNAState, 
                               base_positions: List[int]) -> QuantumDNAState:
        """Apply Hadamard transform to multiple bases"""
        current_state = state
        hadamard_gate = self.gates.hadamard_base()
        
        for pos in base_positions:
            current_state = self.apply_single_base_gate(current_state, hadamard_gate, pos)
        
        return current_state
    
    def apply_aging_decoherence(self, state: QuantumDNAState, 
                               aging_factor: float) -> QuantumDNAState:
        """Apply aging-induced decoherence across all bases"""
        aging_gate = self.gates.aging_gate(aging_factor)
        current_state = state
        
        for pos in range(self.n_bases):
            current_state = self.apply_single_base_gate(current_state, aging_gate, pos)
        
        return current_state
    
    def measure_state(self, state: QuantumDNAState, 
                     measurement_basis: str = "computational") -> Tuple[List[DNABase], float]:
        """Measure quantum state and collapse to classical DNA sequence"""
        probabilities = state.probabilities
        
        # Sample according to quantum probabilities
        measured_index = np.random.choice(len(probabilities), p=probabilities)
        measurement_probability = probabilities[measured_index]
        
        # Decompose to DNA sequence
        dna_sequence = []
        for i in range(self.n_bases):
            base_val = (measured_index >> (2 * (self.n_bases - 1 - i))) & 3
            dna_sequence.append(DNABase(base_val))
        
        # Store measurement history
        self.measurement_history.append({
            'timestamp': datetime.now().isoformat(),
            'sequence': [base.name for base in dna_sequence],
            'probability': measurement_probability,
            'entropy': state.entropy
        })
        
        return dna_sequence, measurement_probability
    
    def simulate_quantum_circuit(self, initial_sequence: List[DNABase],
                                 circuit_ops: List[Dict],
                                 aging_factor: float = 0.0) -> Dict:
        """Simulate complete quantum circuit on DNA sequence"""
        # Create initial state
        state = self.create_initial_state(initial_sequence)
        initial_entropy = state.entropy
        
        # Apply circuit operations
        for op in circuit_ops:
            op_type = op['type']
            
            if op_type == 'hadamard':
                positions = op.get('positions', [0])
                state = self.apply_hadamard_transform(state, positions)
            
            elif op_type == 'cnot':
                control = op['control']
                target = op['target']
                cnot_gate = self.gates.cnot_dna()
                state = self.apply_two_base_gate(state, cnot_gate, control, target)
            
            elif op_type == 'pauli_x':
                position = op['position']
                pauli_x = self.gates.pauli_x_dna()
                state = self.apply_single_base_gate(state, pauli_x, position)
            
            elif op_type == 'aging':
                factor = op.get('factor', aging_factor)
                state = self.apply_aging_decoherence(state, factor)
            
            elif op_type == 'toffoli':
                control1 = op['control1']
                control2 = op['control2']
                target = op['target']
                # Simplified toffoli for demonstration
                cnot_gate = self.gates.cnot_dna()
                state = self.apply_two_base_gate(state, cnot_gate, control1, target)
        
        # Apply aging decoherence
        if aging_factor > 0:
            state = self.apply_aging_decoherence(state, aging_factor)
        
        # Measure final state
        final_sequence, measurement_prob = self.measure_state(state)
        
        # Calculate fidelity with initial state
        initial_state_recreated = self.create_initial_state(initial_sequence)
        fidelity = self._calculate_fidelity(initial_state_recreated, state)
        
        return {
            'initial_sequence': [base.name for base in initial_sequence],
            'final_sequence': [base.name for base in final_sequence],
            'initial_entropy': initial_entropy,
            'final_entropy': state.entropy,
            'measurement_probability': measurement_prob,
            'fidelity': fidelity,
            'aging_factor': aging_factor,
            'circuit_operations': circuit_ops,
            'quantum_state_final': state
        }
    
    def _calculate_fidelity(self, state1: QuantumDNAState, 
                           state2: QuantumDNAState) -> float:
        """Calculate fidelity between two quantum states"""
        overlap = np.abs(np.vdot(state1.amplitudes, state2.amplitudes))
        return overlap ** 2
    
    def compare_with_fpga(self, fpga_result: Dict, 
                         quantum_result: Dict) -> Dict:
        """Compare quantum simulation with FPGA results"""
        # Compare sequences
        fpga_seq = fpga_result.get('processed_sequence', [])
        quantum_seq = quantum_result['final_sequence']
        
        sequence_match = fpga_seq == quantum_seq
        
        # Compare entropies
        fpga_entropy = fpga_result.get('entropy', 0)
        quantum_entropy = quantum_result['final_entropy']
        entropy_diff = abs(fpga_entropy - quantum_entropy)
        
        # Compare mutation counts
        fpga_mutations = fpga_result.get('mutation_count', 0)
        quantum_mutations = sum(1 for i, j in zip(
            quantum_result['initial_sequence'], 
            quantum_result['final_sequence']) if i != j)
        
        mutation_diff = abs(fpga_mutations - quantum_mutations)
        
        return {
            'sequence_match': sequence_match,
            'entropy_difference': entropy_diff,
            'mutation_difference': mutation_diff,
            'fidelity_score': quantum_result['fidelity'],
            'fpga_sequence': fpga_seq,
            'quantum_sequence': quantum_seq,
            'comparison_timestamp': datetime.now().isoformat()
        }
    
    def visualize_quantum_evolution(self, evolution_data: List[Dict], 
                                  save_plot: bool = False):
        """Visualize quantum state evolution"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data
        entropies = [data['final_entropy'] for data in evolution_data]
        fidelities = [data['fidelity'] for data in evolution_data]
        aging_factors = [data['aging_factor'] for data in evolution_data]
        
        # Plot 1: Entropy evolution
        axes[0, 0].plot(aging_factors, entropies, 'b-o', alpha=0.7)
        axes[0, 0].set_xlabel('Aging Factor')
        axes[0, 0].set_ylabel('Quantum Entropy')
        axes[0, 0].set_title('Quantum Entropy vs Aging')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Fidelity evolution
        axes[0, 1].plot(aging_factors, fidelities, 'r-s', alpha=0.7)
        axes[0, 1].set_xlabel('Aging Factor')
        axes[0, 1].set_ylabel('Fidelity')
        axes[0, 1].set_title('Quantum Fidelity vs Aging')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Entropy vs Fidelity
        axes[1, 0].scatter(entropies, fidelities, alpha=0.6, c=aging_factors, 
                          cmap='viridis')
        axes[1, 0].set_xlabel('Entropy')
        axes[1, 0].set_ylabel('Fidelity')
        axes[1, 0].set_title('Entropy vs Fidelity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Measurement probability distribution
        meas_probs = [data['measurement_probability'] for data in evolution_data]
        axes[1, 1].hist(meas_probs, bins=20, alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Measurement Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Measurement Probability Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('quantum_dna_evolution.png', dpi=300, bbox_inches='tight')
            print("Saved quantum evolution plot")
        
        plt.show()

def main():
    """Demonstration of quantum DNA simulation"""
    print("🧬 Quantum DNA Simulator - Demonstration")
    print("="*50)
    
    # Create simulator
    simulator = QuantumDNASimulator(n_bases=4)  # 4 bases for demo
    
    # Initial DNA sequence
    initial_dna = [DNABase.A, DNABase.T, DNABase.G, DNABase.C]
    
    # Define quantum circuit
    circuit_operations = [
        {'type': 'hadamard', 'positions': [0, 1]},
        {'type': 'cnot', 'control': 0, 'target': 1},
        {'type': 'aging', 'factor': 0.1},
        {'type': 'cnot', 'control': 2, 'target': 3}
    ]
    
    # Simulate evolution with different aging factors
    evolution_results = []
    aging_factors = np.linspace(0.0, 0.5, 10)
    
    for aging in aging_factors:
        result = simulator.simulate_quantum_circuit(
            initial_dna, circuit_operations, aging
        )
        evolution_results.append(result)
        
        print(f"Aging {aging:.2f}: {result['initial_sequence']} -> {result['final_sequence']}")
        print(f"  Entropy: {result['initial_entropy']:.3f} -> {result['final_entropy']:.3f}")
        print(f"  Fidelity: {result['fidelity']:.3f}")
        print()
    
    # Visualize results
    simulator.visualize_quantum_evolution(evolution_results, save_plot=True)
    
    # Export results
    export_data = {
        'simulation_timestamp': datetime.now().isoformat(),
        'simulator_config': {
            'n_bases': simulator.n_bases,
            'circuit_operations': circuit_operations
        },
        'evolution_results': evolution_results
    }
    
    with open('quantum_dna_simulation.json', 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print("Results exported to quantum_dna_simulation.json")

if __name__ == "__main__":
    main()