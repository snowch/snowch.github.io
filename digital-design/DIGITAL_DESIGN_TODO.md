# Digital Design - Missing Prerequisites for Computer Architecture

Based on the ENGR85A course prerequisites, here are the topics needed before Computer Architecture:

## ✅ Current Coverage (Sequential Logic)
- D Flip-Flops ✓
- Latches ✓
- Timing Diagrams ✓
- Finite State Machines (FSMs) ✓

## ❌ Missing Topics

### 1. **Combinational Logic Fundamentals**
   - Boolean algebra and logic gates (AND, OR, NOT, NAND, NOR, XOR, XNOR)
   - Truth tables
   - Boolean simplification
   - Karnaugh maps (K-maps)
   - Sum of Products (SOP) and Product of Sums (POS)
   - Don't care conditions

### 2. **Combinational Building Blocks**
   - **Multiplexers (MUX)** - Critical for datapaths
     - 2:1, 4:1, 8:1 muxes
     - Using muxes to implement logic functions
   - **Demultiplexers**
   - **Decoders** - Used for address decoding
     - 2:4, 3:8 decoders
   - **Encoders**
     - Priority encoders

### 3. **Arithmetic Circuits**
   - **Adders** - Essential for ALU
     - Half adder
     - Full adder
     - Ripple-carry adder
     - Carry-lookahead adder (for performance)
   - **Subtractors**
   - **Comparators** - For conditional operations
   - **Arithmetic Logic Unit (ALU)** - Core of processor datapath
     - Multi-function ALU design
     - Operation selection

### 4. **Memory Elements**
   - **Registers** - Data storage
     - Parallel load registers
     - Shift registers
   - **Register Files** - Central to processor design
   - **RAM (Random Access Memory)**
     - SRAM basics
     - Memory arrays
     - Address decoding
   - **ROM (Read-Only Memory)**

### 5. **Hardware Description Language (Verilog/SystemVerilog)**
   - Structural vs behavioral modeling
   - Combinational logic in Verilog
   - Sequential logic in Verilog
   - Testbenches and simulation
   - Building larger systems from modules

### 6. **Additional Sequential Topics**
   - **Counters**
     - Binary counters
     - Up/down counters
     - Modulo-N counters
   - Different flip-flop types (SR, JK, T) - Currently only have D

## Priority Order for Implementation

### High Priority (Essential for Computer Architecture):
1. **ALU Design** - Core building block of processors
2. **Multiplexers** - Used everywhere in datapaths
3. **Adders** - Foundation of arithmetic operations
4. **Register Files** - Central to processor architecture
5. **Memory Basics (RAM/ROM)** - Essential for understanding memory hierarchy

### Medium Priority (Important fundamentals):
1. **Combinational Logic Basics** - Foundation for everything
2. **Decoders** - Address decoding in memory systems
3. **Comparators** - Conditional operations
4. **Counters** - Program counter in processors

### Lower Priority (Nice to have):
1. **Verilog** - More advanced, but helpful for HDL perspective
2. **K-maps** - Useful but can be supplemental
3. **Encoders** - Less commonly used in basic architecture

## Suggested Tutorial Structure

Each topic should follow the pattern of existing tutorials:
- Interactive Jupyter notebooks
- Visual diagrams (wavedrom for timing)
- Step-by-step examples
- Practice problems/exercises
- Connection to real-world applications

## Cross-references

Link from Digital Design topics to Computer Architecture applications:
- "ALUs are used in the processor datapath (see Computer Architecture)"
- "Register files form the heart of the processor (see Computer Architecture)"
- "Multiplexers control data routing in the datapath (see Computer Architecture)"
