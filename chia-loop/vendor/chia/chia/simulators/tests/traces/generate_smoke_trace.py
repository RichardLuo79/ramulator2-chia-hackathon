#!/usr/bin/env python3
"""Generate a minimal ChampSim trace for build-time smoke testing.

Produces a gzip-compressed trace with enough instructions to satisfy
a short warmup + simulation run. The trace contains a mix of:
  - Sequential ALU instructions (no memory)
  - Load instructions (with source_memory addresses)
  - Occasional branches (taken and not-taken)

This exercises basic prefetcher logic without requiring a real workload.
"""
from __future__ import annotations

import argparse
import gzip
import struct

# input_instr layout from inc/trace_instruction.h
# ip(Q) is_branch(B) branch_taken(B) dest_regs[2](2B) src_regs[4](4B)
# dest_mem[2](2Q) src_mem[4](4Q)
INSTR_FMT = '<QBB2B4B2Q4Q'
assert struct.calcsize(INSTR_FMT) == 64


def make_instr(ip, is_branch=0, branch_taken=0,
               dest_regs=(0, 0), src_regs=(0, 0, 0, 0),
               dest_mem=(0, 0), src_mem=(0, 0, 0, 0)):
    """Pack a single 64-byte ``input_instr`` struct."""
    return struct.pack(INSTR_FMT, ip, is_branch, branch_taken,
                       *dest_regs, *src_regs, *dest_mem, *src_mem)


def generate(output_path, num_instructions=10000):
    """Write *num_instructions* instructions to a gzip trace file.

    Instruction mix:
    - Every 3rd instruction is a load (``src_mem[0]`` set to a cache-line
      strided address).
    - Every 10th instruction is a branch (``is_branch=1``).
    - Every 20th instruction is a taken branch (``branch_taken=1``).
    - IPs are sequential: ``0x400000 + (i * 4)``.
    """
    with gzip.open(output_path, 'wb') as f:
        for i in range(num_instructions):
            ip = 0x400000 + (i * 4)
            # Every 3rd instruction is a load
            if i % 3 == 0:
                src_mem = (0x7FFF0000 + (i * 64), 0, 0, 0)
            else:
                src_mem = (0, 0, 0, 0)
            is_branch = 1 if i % 10 == 0 else 0
            branch_taken = 1 if i % 20 == 0 else 0
            f.write(make_instr(ip, is_branch, branch_taken,
                               src_mem=src_mem))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a synthetic ChampSim trace for smoke testing.')
    parser.add_argument('-o', '--output', default='smoke.champsimtrace.gz',
                        help='Output trace file path (default: smoke.champsimtrace.gz)')
    parser.add_argument('-n', '--num-instructions', type=int, default=10000,
                        help='Number of instructions to generate (default: 10000)')
    args = parser.parse_args()
    generate(args.output, args.num_instructions)
    print(f"Generated {args.num_instructions} instructions -> {args.output}")
