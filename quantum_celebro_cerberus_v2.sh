#!/bin/bash
# Generate data
# Written by Andrea Giani

python dna_sequence_generator.py --batch-size 100 --progression

# Run FPGA simulation
vvp dna_processor_sim

# AI Analysis
python dna_analysis_ai.py fpga_output.json

# Cross-validation
python quantum_dna_simulator.py validate fpga_output.json