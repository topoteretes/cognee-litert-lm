#!/usr/bin/env python3
"""Patch libLiteRtTopKOpenClSampler.so to add liblitert_lm_c.so as NEEDED dep.

Strategy: repurpose the DT_SONAME entry (tag 14) → DT_NEEDED (tag 1).
SONAME string "libLiteRtTopKOpenClSampler.so" (30 bytes) is overwritten with
"liblitert_lm_c.so\0" (19 bytes), null-padded to fill original space.
SONAME is not needed for dlopen-by-name usage.
"""
import os, struct, shutil, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO_IN  = os.path.join(_REPO_ROOT, 'vendor/LiteRT-LM/prebuilt/android_arm64/libLiteRtTopKOpenClSampler.so')
SO_OUT = SO_IN  # patch in-place (original is in git)
NEW_DEP = b'liblitert_lm_c.so\x00'

shutil.copy2(SO_IN, SO_OUT)
data = bytearray(open(SO_OUT, 'rb').read())

# Idempotency check: skip if already patched
if NEW_DEP in data:
    print(f"{SO_OUT}: already patched ('{NEW_DEP.rstrip(b'\\x00').decode()}' found). Nothing to do.")
    sys.exit(0)
def ru64(off): return struct.unpack_from('<Q', data, off)[0]
def ri64(off): return struct.unpack_from('<q', data, off)[0]
def wu64(off, v): struct.pack_into('<Q', data, off, v)
def wi64(off, v): struct.pack_into('<q', data, off, v)

e_phoff = ru64(32)
e_phnum, = struct.unpack_from('<H', data, 56)

# Find PT_DYNAMIC
for i in range(e_phnum):
    ph = e_phoff + i * 56
    if struct.unpack_from('<I', data, ph)[0] == 2:  # PT_DYNAMIC
        dyn_foff   = ru64(ph + 8)
        dyn_filesz = ru64(ph + 32)
        break

# Parse .dynamic: find DT_STRTAB (file offset = vaddr for this SO), DT_SONAME
DT_NEEDED, DT_SONAME, DT_STRTAB = 1, 14, 5
strtab_vaddr = None
soname_entry_off = None
soname_val = None

for i in range(dyn_filesz // 16):
    off = dyn_foff + i * 16
    tag = ri64(off)
    val = ru64(off + 8)
    if tag == DT_STRTAB:
        strtab_vaddr = val  # equals file offset for this SO (PIE offset 0)
    elif tag == DT_SONAME:
        soname_entry_off = off
        soname_val = val
    if tag == 0:
        break

assert strtab_vaddr is not None, "DT_STRTAB not found"
assert soname_entry_off is not None, "DT_SONAME not found"
print(f"DT_STRTAB vaddr/foff={strtab_vaddr:#x}")
print(f"DT_SONAME entry at file={soname_entry_off:#x}, strtab_offset={soname_val:#x}")

# Read and verify the SONAME string
soname_foff = strtab_vaddr + soname_val
soname_bytes = data[soname_foff:]
soname_end = soname_bytes.index(0)
print(f"SONAME = '{soname_bytes[:soname_end].decode()}' ({soname_end+1} bytes with null)")

if soname_end + 1 < len(NEW_DEP):
    print(f"ERROR: SONAME too short ({soname_end+1}) to hold new dep ({len(NEW_DEP)} bytes)!")
    sys.exit(1)

# 1. Overwrite SONAME string with new dep name, null-pad the rest
data[soname_foff:soname_foff + soname_end + 1] = b'\x00' * (soname_end + 1)  # zero it first
data[soname_foff:soname_foff + len(NEW_DEP)]   = NEW_DEP
print(f"Wrote '{NEW_DEP!r}' at file offset {soname_foff:#x}")

# 2. Change DT_SONAME entry tag to DT_NEEDED (keep same strtab offset)
wi64(soname_entry_off, DT_NEEDED)
print(f"Changed DT_SONAME → DT_NEEDED at file {soname_entry_off:#x}, strtab_offset={soname_val:#x}")

# Write
open(SO_OUT, 'wb').write(bytes(data))
print(f"Written patched SO to {SO_OUT}")

# Verify round-trip
verify = open(SO_OUT, 'rb').read()
assert NEW_DEP in verify, "String not found in patched file!"
print("Verification: 'liblitert_lm_c.so' found in patched file ✓")
