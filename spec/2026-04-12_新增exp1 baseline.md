\# Task: Add a separate contiguous baseline benchmark for Exp1  
  
\## Goal  
Create a new standalone benchmark program for the contiguous baseline of Experiment 1.  
  
Do NOT merge this baseline into the existing \`bench\_byteplane\_scan.cu\`.  
The baseline must be implemented as a separate source file and separate binary.  
  
\## New file  
Add a new file:  
  
\`benchmarks/experiment1/bench\_contiguous\_baseline.cu\`  
  
\## Design requirements  
This new program should follow the style and structure of the existing byte-plane benchmark, but only implement the contiguous 8-byte baseline.  
  
It should:  
\- allocate one contiguous device array of \`n\` 8-byte elements  
\- run a CUDA kernel that performs:  
  - grid-stride traversal  
  - one 8-byte load per element  
  - register accumulation  
  - block reduction into \`d\_out\`  
\- measure time using the same CUDA event timing model as the original benchmark  
\- output CSV in a format suitable for later plotting against the byte-plane benchmark  
  
\## Must not do  
\- do not modify \`bench\_byteplane\_scan.cu\`  
\- do not add a new \`strategy\` into the existing program  
\- do not add \`k\` or \`plane\_bytes\` logic into the baseline program  
\- do not introduce shared memory staging, packed loads, or alternative strategies  
\- do not change the existing byte-only benchmark behavior  
  
\## CLI for the new baseline binary  
Support these options only:  
\- \`--device\`  
\- \`--n\`  
\- \`--block\`  
\- \`--grid\_mul\`  
\- \`--warmup\`  
\- \`--iters\`  
\- \`--csv\`  
  
\## Kernel requirements  
Implement a kernel like:  
  
\`\`\`cpp  
\_\_global\_\_ void scan\_contiguous\_u64(  
    const uint64\_t\* \_\_restrict\_\_ data,  
    uint64\_t n,  
    unsigned long long\* \_\_restrict\_\_ per\_block\_out);

The kernel must:

- use grid-stride traversal
- accumulate into a register
- use the same block reduction pattern as the original benchmark

## Throughput definition

For every run:

- `logical_bytes = n * 8`
- `logical_GBps = logical_bytes / seconds / 1e9`

## CSV output

The baseline program should output a simple CSV with fields like:

- benchmark
- n
- logical\_bytes
- block
- grid
- warmup
- iters
- ms\_per\_iter
- logical\_GBps
- device
- sm
- cc\_major
- cc\_minor

Set:

- `benchmark=contiguous64`

## Occupancy

Use the same occupancy-based grid sizing style as the original benchmark.

## Deliverables

Return:

1. summary
2. new file added
3. example build command
4. example run command
5. CSV schema
6. caveats

  
\---  
  
\## 你之後的命令會變成兩條線  
  
\### 1. 跑 byte-plane  
\`\`\`bash  
./build/exp1/bench\_byteplane\_scan \\  
  --strategy byte \\  
  --plane\_bytes 1 \\  
  --k\_min 1 \\  
  --k\_max 8 \\  
  --n 100000000 \\  
  --block 256 \\  
  --grid\_mul 1 \\  
  --warmup 10 \\  
  --iters 1000 \\  
  --csv results/exp1/byte\_scan.csv

### 2\. 跑 contiguous baseline

Bash

./build/exp1/bench\_contiguous\_baseline \\  
  \--n 100000000 \\  
  \--block 256 \\  
  \--grid\_mul 1 \\  
  \--warmup 10 \\  
  \--iters 1000 \\  
  \--csv results/exp1/contiguous\_baseline.csvNo. 700. So. Lesbian. None. 