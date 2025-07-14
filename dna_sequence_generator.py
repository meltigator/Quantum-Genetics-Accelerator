#!/usr/bin/env python3
# Written by Andrea Giani

"""
dna_sequence_generator.py
Quantum Genetics Accelerator - DNA Sequence Generator
Generates synthetic DNA sequences with controlled aging and mutation patterns
"""

import numpy as np
import random
import json
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from datetime import datetime

class DNABase(Enum):
    """DNA base enumeration matching FPGA encoding"""
    A = 0  # 00 in binary
    T = 1  # 01 in binary
    G = 2  # 10 in binary
    C = 3  # 11 in binary

@dataclass
class DNASequenceProfile:
    """Profile for generating DNA sequences with specific characteristics"""
    sequence_length: int = 16  # Number of bases (32 bits total)
    aging_factor: float = 0.1  # 0.0 to 1.0, higher = more mutations
    gc_content: float = 0.5    # GC content bias
    mutation_rate: float = 0.01  # Base mutation probability
    aging_bias: str = "transitions"  # "transitions", "transversions", "random"
    repeat_regions: bool = False  # Include tandem repeats
    palindromic_regions: bool = False  # Include palindromic sequences
    
class DNASequenceGenerator:
    """
    Generates synthetic DNA sequences with aging-inspired mutations
    and biological patterns for quantum processing simulation
    """
    
    def __init__(self, profile: DNASequenceProfile):
        self.profile = profile
        self.bases = list(DNABase)
        self.mutation_matrix = self._initialize_mutation_matrix()
        self.generated_sequences = []
        self.metadata = []
        
    def _initialize_mutation_matrix(self) -> np.ndarray:
        """Initialize mutation transition matrix based on aging bias"""
        matrix = np.zeros((4, 4))
        
        if self.profile.aging_bias == "transitions":
            # Transitions: A<->G, T<->C (more common in aging)
            matrix[DNABase.A.value, DNABase.G.value] = 0.7
            matrix[DNABase.G.value, DNABase.A.value] = 0.7
            matrix[DNABase.T.value, DNABase.C.value] = 0.7
            matrix[DNABase.C.value, DNABase.T.value] = 0.7
            # Lower probability transversions
            matrix[DNABase.A.value, DNABase.T.value] = 0.1
            matrix[DNABase.A.value, DNABase.C.value] = 0.1
            matrix[DNABase.T.value, DNABase.A.value] = 0.1
            matrix[DNABase.T.value, DNABase.G.value] = 0.1
            matrix[DNABase.G.value, DNABase.T.value] = 0.1
            matrix[DNABase.G.value, DNABase.C.value] = 0.1
            matrix[DNABase.C.value, DNABase.A.value] = 0.1
            matrix[DNABase.C.value, DNABase.G.value] = 0.1
            
        elif self.profile.aging_bias == "transversions":
            # Transversions: A<->T, A<->C, G<->T, G<->C
            matrix[DNABase.A.value, DNABase.T.value] = 0.6
            matrix[DNABase.A.value, DNABase.C.value] = 0.6
            matrix[DNABase.T.value, DNABase.A.value] = 0.6
            matrix[DNABase.T.value, DNABase.G.value] = 0.6
            matrix[DNABase.G.value, DNABase.T.value] = 0.6
            matrix[DNABase.G.value, DNABase.C.value] = 0.6
            matrix[DNABase.C.value, DNABase.A.value] = 0.6
            matrix[DNABase.C.value, DNABase.G.value] = 0.6
            # Lower probability transitions
            matrix[DNABase.A.value, DNABase.G.value] = 0.2
            matrix[DNABase.G.value, DNABase.A.value] = 0.2
            matrix[DNABase.T.value, DNABase.C.value] = 0.2
            matrix[DNABase.C.value, DNABase.T.value] = 0.2
            
        else:  # random
            # Equal probability for all mutations
            for i in range(4):
                for j in range(4):
                    if i != j:
                        matrix[i, j] = 0.33
        
        # Normalize rows
        for i in range(4):
            row_sum = np.sum(matrix[i, :])
            if row_sum > 0:
                matrix[i, :] /= row_sum
                
        return matrix
    
    def generate_base_sequence(self, seed: Optional[int] = None) -> List[DNABase]:
        """Generate a base DNA sequence with specified GC content"""
        if seed is not None:
            random.seed(seed)
            
        sequence = []
        gc_target = int(self.profile.sequence_length * self.profile.gc_content)
        at_target = self.profile.sequence_length - gc_target
        
        # Create base pool
        base_pool = [DNABase.G, DNABase.C] * (gc_target // 2)
        base_pool += [DNABase.A, DNABase.T] * (at_target // 2)
        
        # Add remainder bases
        if gc_target % 2:
            base_pool.append(random.choice([DNABase.G, DNABase.C]))
        if at_target % 2:
            base_pool.append(random.choice([DNABase.A, DNABase.T]))
            
        # Shuffle and trim to exact length
        random.shuffle(base_pool)
        sequence = base_pool[:self.profile.sequence_length]
        
        # Add special patterns if requested
        if self.profile.repeat_regions:
            sequence = self._add_repeat_regions(sequence)
        if self.profile.palindromic_regions:
            sequence = self._add_palindromic_regions(sequence)
            
        return sequence
    
    def _add_repeat_regions(self, sequence: List[DNABase]) -> List[DNABase]:
        """Add tandem repeat regions to sequence"""
        if len(sequence) < 4:
            return sequence
            
        # Create a simple dinucleotide repeat
        repeat_start = random.randint(0, len(sequence) - 4)
        repeat_unit = sequence[repeat_start:repeat_start + 2]
        
        # Replace next 2 bases with repeat
        sequence[repeat_start + 2:repeat_start + 4] = repeat_unit
        
        return sequence
    
    def _add_palindromic_regions(self, sequence: List[DNABase]) -> List[DNABase]:
        """Add palindromic regions to sequence"""
        if len(sequence) < 6:
            return sequence
            
        # Create palindromic region
        palindrome_start = random.randint(0, len(sequence) - 6)
        palindrome_unit = sequence[palindrome_start:palindrome_start + 3]
        
        # Complement mapping
        complement = {
            DNABase.A: DNABase.T,
            DNABase.T: DNABase.A,
            DNABase.G: DNABase.C,
            DNABase.C: DNABase.G
        }
        
        # Create reverse complement
        reverse_complement = [complement[base] for base in reversed(palindrome_unit)]
        sequence[palindrome_start + 3:palindrome_start + 6] = reverse_complement
        
        return sequence
    
    def apply_aging_mutations(self, sequence: List[DNABase]) -> Tuple[List[DNABase], List[int]]:
        """Apply aging-induced mutations to sequence"""
        mutated_sequence = sequence.copy()
        mutation_positions = []
        
        for i, base in enumerate(sequence):
            # Higher mutation probability with aging
            mutation_prob = self.profile.mutation_rate * (1 + self.profile.aging_factor * 10)
            
            if random.random() < mutation_prob:
                # Apply mutation based on transition matrix
                probabilities = self.mutation_matrix[base.value, :]
                new_base_idx = np.random.choice(4, p=probabilities)
                new_base = DNABase(new_base_idx)
                
                if new_base != base:
                    mutated_sequence[i] = new_base
                    mutation_positions.append(i)
        
        return mutated_sequence, mutation_positions
    
    def calculate_sequence_entropy(self, sequence: List[DNABase]) -> float:
        """Calculate Shannon entropy of DNA sequence"""
        base_counts = {base: 0 for base in DNABase}
        
        for base in sequence:
            base_counts[base] += 1
            
        sequence_length = len(sequence)
        entropy = 0.0
        
        for count in base_counts.values():
            if count > 0:
                frequency = count / sequence_length
                entropy -= frequency * np.log2(frequency)
                
        return entropy
    
    def sequence_to_binary(self, sequence: List[DNABase]) -> str:
        """Convert DNA sequence to binary string for FPGA"""
        binary_str = ""
        for base in sequence:
            binary_str += f"{base.value:02b}"
        return binary_str
    
    def sequence_to_hex(self, sequence: List[DNABase]) -> str:
        """Convert DNA sequence to hexadecimal for FPGA"""
        binary_str = self.sequence_to_binary(sequence)
        # Pad to multiple of 4 bits
        while len(binary_str) % 4 != 0:
            binary_str += "0"
        
        hex_str = ""
        for i in range(0, len(binary_str), 4):
            nibble = binary_str[i:i+4]
            hex_str += f"{int(nibble, 2):X}"
        
        return hex_str
    
    def generate_sequence_batch(self, batch_size: int, 
                              aging_progression: bool = False) -> List[Dict]:
        """Generate a batch of DNA sequences with metadata"""
        sequences = []
        
        for i in range(batch_size):
            # Progressive aging if requested
            if aging_progression:
                aging_factor = (i / batch_size) * 0.8  # Scale from 0 to 0.8
                self.profile.aging_factor = aging_factor
            
            # Generate base sequence
            base_seq = self.generate_base_sequence(seed=i)
            
            # Apply mutations
            mutated_seq, mutations = self.apply_aging_mutations(base_seq)
            
            # Calculate metrics
            entropy = self.calculate_sequence_entropy(mutated_seq)
            gc_content = sum(1 for base in mutated_seq if base in [DNABase.G, DNABase.C]) / len(mutated_seq)
            
            # Create sequence data
            sequence_data = {
                'id': f"seq_{i:04d}",
                'timestamp': datetime.now().isoformat(),
                'base_sequence': [base.name for base in base_seq],
                'mutated_sequence': [base.name for base in mutated_seq],
                'binary_representation': self.sequence_to_binary(mutated_seq),
                'hex_representation': self.sequence_to_hex(mutated_seq),
                'mutation_positions': mutations,
                'mutation_count': len(mutations),
                'aging_factor': self.profile.aging_factor,
                'entropy': entropy,
                'gc_content': gc_content,
                'sequence_length': len(mutated_seq)
            }
            
            sequences.append(sequence_data)
            self.generated_sequences.append(sequence_data)
        
        return sequences
    
    def export_sequences(self, sequences: List[Dict], filename: str):
        """Export sequences to JSON file"""
        export_data = {
            'generation_timestamp': datetime.now().isoformat(),
            'profile': {
                'sequence_length': self.profile.sequence_length,
                'aging_factor': self.profile.aging_factor,
                'gc_content': self.profile.gc_content,
                'mutation_rate': self.profile.mutation_rate,
                'aging_bias': self.profile.aging_bias
            },
            'sequences': sequences
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Exported {len(sequences)} sequences to {filename}")
    
    def export_fpga_format(self, sequences: List[Dict], filename: str):
        """Export sequences in FPGA-compatible format"""
        with open(filename, 'w') as f:
            f.write("// DNA Sequences for FPGA Processing\n")
            f.write("// Generated by Quantum Genetics Accelerator\n\n")
            
            for seq in sequences:
                f.write(f"// Sequence {seq['id']}\n")
                f.write(f"// Aging Factor: {seq['aging_factor']:.3f}\n")
                f.write(f"// Mutations: {seq['mutation_count']}\n")
                f.write(f"// Entropy: {seq['entropy']:.3f}\n")
                f.write(f"32'h{seq['hex_representation']}\n\n")
        
        print(f"Exported {len(sequences)} sequences in FPGA format to {filename}")
    
    def visualize_sequences(self, sequences: List[Dict], save_plot: bool = False):
        """Visualize sequence characteristics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract metrics
        aging_factors = [seq['aging_factor'] for seq in sequences]
        mutation_counts = [seq['mutation_count'] for seq in sequences]
        entropies = [seq['entropy'] for seq in sequences]
        gc_contents = [seq['gc_content'] for seq in sequences]
        
        # Plot 1: Aging vs Mutations
        axes[0, 0].scatter(aging_factors, mutation_counts, alpha=0.6)
        axes[0, 0].set_xlabel('Aging Factor')
        axes[0, 0].set_ylabel('Mutation Count')
        axes[0, 0].set_title('Aging vs Mutations')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Entropy Distribution
        axes[0, 1].hist(entropies, bins=20, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Sequence Entropy')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Entropy Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: GC Content vs Entropy
        axes[1, 0].scatter(gc_contents, entropies, alpha=0.6, color='red')
        axes[1, 0].set_xlabel('GC Content')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].set_title('GC Content vs Entropy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Mutation Count Distribution
        axes[1, 1].hist(mutation_counts, bins=max(mutation_counts)+1, alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Mutation Count')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Mutation Count Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('dna_sequences_analysis.png', dpi=300, bbox_inches='tight')
            print("Saved plot to dna_sequences_analysis.png")
        
        plt.show()

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Generate synthetic DNA sequences for quantum processing')
    parser.add_argument('--length', type=int, default=16, help='Sequence length in bases')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of sequences to generate')
    parser.add_argument('--aging-factor', type=float, default=0.1, help='Aging factor (0.0-1.0)')
    parser.add_argument('--gc-content', type=float, default=0.5, help='GC content (0.0-1.0)')
    parser.add_argument('--mutation-rate', type=float, default=0.01, help='Base mutation rate')
    parser.add_argument('--aging-bias', choices=['transitions', 'transversions', 'random'], 
                       default='transitions', help='Aging mutation bias')
    parser.add_argument('--progression', action='store_true', help='Use progressive aging')
    parser.add_argument('--output', default='dna_sequences.json', help='Output filename')
    parser.add_argument('--fpga-format', action='store_true', help='Export FPGA format')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    
    args = parser.parse_args()
    
    # Create profile
    profile = DNASequenceProfile(
        sequence_length=args.length,
        aging_factor=args.aging_factor,
        gc_content=args.gc_content,
        mutation_rate=args.mutation_rate,
        aging_bias=args.aging_bias
    )
    
    # Generate sequences
    generator = DNASequenceGenerator(profile)
    sequences = generator.generate_sequence_batch(args.batch_size, args.progression)
    
    # Export results
    generator.export_sequences(sequences, args.output)
    
    if args.fpga_format:
        fpga_filename = args.output.replace('.json', '_fpga.v')
        generator.export_fpga_format(sequences, fpga_filename)
    
    if args.visualize:
        generator.visualize_sequences(sequences, save_plot=True)
    
    # Print summary
    print(f"\nGeneration Summary:")
    print(f"- Generated {len(sequences)} sequences")
    print(f"- Sequence length: {args.length} bases")
    print(f"- Average mutations: {np.mean([seq['mutation_count'] for seq in sequences]):.2f}")
    print(f"- Average entropy: {np.mean([seq['entropy'] for seq in sequences]):.3f}")
    print(f"- Average GC content: {np.mean([seq['gc_content'] for seq in sequences]):.3f}")

if __name__ == "__main__":
    main()