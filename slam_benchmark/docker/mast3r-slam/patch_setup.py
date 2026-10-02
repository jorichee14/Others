#!/usr/bin/env python3
"""Two build-time edits to the pinned MASt3R-SLAM setup.py, applied by the
Dockerfile and refused loudly if the pinned text is not what this expects.

1. `has_cuda = torch.cuda.is_available()` asks for a GPU at BUILD time. There
   is never one inside `docker build`; the compiler (nvcc) is there. So the
   backend was skipped and setup() then crashed on the undefined ext_modules.
2. The nvcc gencode list is hard-coded and stops at sm_86 (Ampere). An Ada
   (sm_89) or Hopper (sm_90) card would load the module and fail on first use
   with "no kernel image is available". sm_89, sm_90 and compute_90 PTX are
   appended, matching the TORCH_CUDA_ARCH_LIST the curope extension is built
   with in the same image.
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()
old = "has_cuda = torch.cuda.is_available()"
new = "has_cuda = True  # slambench: nvcc is present at build time; no GPU is (docker build)"
if old not in s:
    sys.exit(f"{p}: expected {old!r}; the pinned source is not what this patch was written for")
s = s.replace(old, new, 1)
anchor = '        "-gencode=arch=compute_86,code=sm_86",\n'
if anchor not in s:
    sys.exit(f"{p}: expected the sm_86 gencode line; the pinned source is not what this patch was written for")
s = s.replace(anchor, anchor
              + '        "-gencode=arch=compute_89,code=sm_89",\n'
              + '        "-gencode=arch=compute_90,code=sm_90",\n'
              + '        "-gencode=arch=compute_90,code=compute_90",\n', 1)
p.write_text(s)
print(f"patched {p}: has_cuda forced, sm_89/sm_90/compute_90 added")
