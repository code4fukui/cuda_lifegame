# cuda_lifegame

> 日本語のREADMEはこちらです: [README.ja.md](README.ja.md)

A simple CUDA-based implementation of Conway's Game of Life. This project accelerates the cellular automaton simulation by running it on an NVIDIA GPU.

The simulation starts with a pre-defined R-pentomino pattern and prints the state of the grid to the console for each iteration.

### Example Output
```
Iteration 0 (Initial R-pentomino):
00000000000000000000
00000000000000000000
00100000000000000000
00010000000000000000
01110000000000000000
00000000000000000000
...

Iteration 1:
00000000000000000000
00000000000000000000
00000000000000000000
01010000000000000000
00110000000000000000
00100000000000000000
...
```

## Features
- **GPU Acceleration:** Utilizes CUDA to perform the simulation on the GPU.
- **Fixed Grid:** Simulates a 20x20 grid with toroidal (wrapping) boundaries.
- **Pre-defined Pattern:** Initializes with a hardcoded R-pentomino pattern.
- **Fixed Iterations:** Runs for 100 iterations and prints the grid state at each step.

## Requirements
- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit installed

## Usage

### 1. Compile
Compile the source code using the NVIDIA CUDA Compiler (`nvcc`).

```sh
# The -Xcompiler flag suppresses a specific warning on some Windows toolchains
nvcc -Xcompiler "/wd 4819" lifegame.cu
```
On Windows, you can also use the provided batch script:
```sh
c.bat
```

### 2. Run
Execute the compiled program. The output file will be named `a.exe` on Windows or `a.out` on Linux/macOS.

**On Windows:**
```sh
a.exe
```

**On Linux / macOS:**
```sh
./a.out
```

The program will output the state of the 20x20 grid to your console for 100 iterations.

## License
MIT License — see [LICENSE](LICENSE).