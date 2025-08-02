# Real-Time Machine Learning - Learning Feature Trees

## Overview

This codebase implements a novel machine learning approach called **Learning Feature Trees (LF-Trees)** using **WOLF (Weighted Overlapping Linear Functions)** approximation. The system is designed for real-time pattern recognition and achieves 96% accuracy on MNIST handwritten digit classification.

Key innovations:
- **Automatic feature design** - no manual feature engineering required
- **Parallel scalability** - designed for hardware acceleration (GPU/CUDA potential)
- **Forward propagation credit assignment** - uses "engrams" to track actions for later reinforcement
- **Self-growing architecture** - trees automatically grow and partition data
- **Real-time performance** - optimized for fast inference

## Architecture

### Core Components

1. **wolf.h** - Main implementation (784 lines)
   - `cLF_tree` class: The core Learning Feature Tree implementation
   - `SB` (Sample Block) struct: Nodes in the tree that partition data
   - `SF` (Sensor Factor) struct: Feature representation
   - Tree growing, splitting, and evaluation algorithms
   - Convolution operations on features (not images)

2. **test_wolf.cpp** - Main application
   - Command-line interface for training and testing
   - MNIST data loading and preprocessing
   - Performance timing and evaluation
   - Parallel tree construction framework

3. **manipulation.cpp** - MNIST data handling
   - Binary file reading with endianness conversion
   - Memory-efficient batch processing

4. **mnist_file.h** - MNIST data structures
   - Standard MNIST format definitions
   - Image and label structures

## Key Concepts

### Sample Blocks (SB)
- Fundamental building blocks that partition the feature space
- Two types:
  1. **Feature nodes**: Split data using linear functions
  2. **Leaf nodes**: Compute class probabilities using linear functions
- Each block maintains:
  - Centroid vector `C` of all samples in the block
  - Weight vector `W` for active sensors
  - List of active (non-constant) sensors
  - Image indices belonging to this block

### Learning Process
1. Start with single root block containing all samples
2. Compute centroids and identify non-constant features
3. Create linear features to split blocks
4. Recursively partition until stopping criteria met
5. Multiple trees vote for final classification

### Convolution on Features
- Applied to feature vectors, not raw images
- Uses configurable kernel radius
- Improves generalization without speed penalty

## Usage

### Command Line
```bash
./RTML <project> <sensors> <trees> <min_split> <constancy> <conv_radius>
```

Parameters:
- `project`: Project name (e.g., "MNIST")
- `sensors`: Number of input sensors/pixels (784 for MNIST)
- `trees`: Number of LF-trees to create (multiple of 10)
- `min_split`: Minimum samples required to split a block
- `constancy`: Threshold for deeming sensors constant (std dev)
- `conv_radius`: Convolution kernel radius for features

### Example Commands
Quick test:
```bash
./RTML MNIST 784 10 10 15.0 2.0
```

Full accuracy run:
```bash
./RTML MNIST 784 200 10 15.0 2.0
```

## Performance Characteristics

### Training
- Sequential: ~40 minutes for 200 trees on 60,000 MNIST samples
- Parallel potential: ~12 seconds with GPU acceleration
- Trees grow independently - perfect for parallel processing

### Inference
- Real-time capable
- Single pass through tree for classification
- Multiple trees vote for robustness

### Memory
- Efficient storage of only active sensors per block
- Scales with tree complexity, not data size

## Implementation Details

### Tree Growing Algorithm
1. Initialize root block with all training samples
2. For each non-final block:
   - Compute centroid and variance for active sensors
   - Remove constant sensors (low variance)
   - Fit linear function using least squares
   - Create splitting feature if enough samples
   - Partition samples into child blocks
3. Mark blocks as final when:
   - Too few samples to split
   - No informative features remain

### Reinforcement Function
Located in test_wolf.cpp:
- Returns +1 for correct classification
- Returns -1 for incorrect classification
- Used for credit assignment during training

### Feature Creation
- Linear combinations of active sensors
- Convolution applied to enhance features
- Automatic selection of discriminative features

## File Structure

```
Real-Time-Machine-Learning/
├── wolf.h                 # Core LF-tree implementation
├── test_wolf.cpp         # Main training/testing application
├── manipulation.cpp      # MNIST file I/O
├── mnist_file.h         # MNIST data structures
├── README.md            # Project overview
├── LICENSE              # MIT License
├── Data/                # MNIST dataset files
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
└── RTML/               # Visual Studio project files

```

## Technical Papers

- **"A Component for Real-Time Machine Learning.pdf"** - Detailed methodology
- **"Hardware Implementation.pdf"** - Patent specifications for hardware acceleration

## Development Notes

### Code Style
- C++ with minimal dependencies
- Emphasis on performance and hardware compatibility
- Designed for parallelization (GPU/FPGA potential)

### Key Algorithms
- **findBlock()**: Traverse tree to find appropriate block for sample
- **splitSB()**: Partition block using computed feature
- **evalBoundedSB()**: Evaluate sample probability in block
- **growTree()**: Main tree construction loop

### Optimization Opportunities
- CUDA implementation for parallel tree growth
- SIMD operations for feature computation
- Hardware-specific optimizations for real-time systems

## Future Enhancements

The author suggests:
- CUDA/GPU implementation for 200-20,000x speedup
- Extension to other datasets and problem domains
- Hardware implementations for embedded systems
- Additional tree architectures and splitting criteria

## License

MIT License - Copyright (c) 2024 William Ward Armstrong