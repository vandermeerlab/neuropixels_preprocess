import os
from pathlib import Path
import shutil
import glob

# === Configuration: Modify these 3 items based on your actual case ===
# 1) Source directory: point to the upper-level directory that contains CatGT exported files (the script will recursively search subdirectories)
SRC_ROOT   = Path(r"E:\SE_odor_test_all_g0")  # Example: can point to preprocessed\catgt_* or one level above the original catgt_*
# 2) Destination directory: where the organized files will be placed
DEST_DIR   = Path(r"E:\odor-pixels\Cohort5\M539\Training\SE_odor_test_all")
# 3) Required channels/bits
ANALOG_CHS = list(range(0, 7))     # Use only XA0..XA6
DIGITAL_BITS = [1, 4]              # XD1 / XD4

# Other options
SKIP_EMPTY = True                  # Skip empty files (0 bytes or unreadable)
DRY_RUN    = False                 # When True, only print actions without copying

DEST_DIR.mkdir(parents=True, exist_ok=True)

def find_one(patterns):
    """Recursively search in SRC_ROOT with multiple patterns; return the most recently modified one."""
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(str(SRC_ROOT / "**" / pat), recursive=True))
    hits = [Path(p) for p in hits]
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # Most recently modified first
    return hits[0]

def non_empty_txt(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return False

def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if DRY_RUN:
        print(f"[DRY] copy {src} -> {dst}")
    else:
        shutil.copy2(src, dst)
        print(f"[OK ] {src.name:40s} -> {dst}")

def main():
    # 1) XA (analog) — find ON/OFF for each channel
    for i in ANALOG_CHS:
        # Support two naming conventions: ...tcat.nidq.xa_i_*.txt / times_ni_xa_i_*.txt
        on_src  = find_one([f"*tcat.nidq.xa_{i}_*.txt",  f"times_ni_xa_{i}_*.txt"])
        off_src = find_one([f"*tcat.nidq.xia_{i}_*.txt", f"times_ni_xia_{i}_*.txt"])
        # Print for debugging
        print(f"[XA{i}] on={on_src.name if on_src else None}, off={off_src.name if off_src else None}")
        # Validate & copy
        if on_src and (non_empty_txt(on_src) or not SKIP_EMPTY):
            copy_file(on_src, DEST_DIR / f"XA{i}_ON.txt")
        if off_src and (non_empty_txt(off_src) or not SKIP_EMPTY):
            copy_file(off_src, DEST_DIR / f"XA{i}_OFF.txt")

    # 2) XD (digital) — find ON/OFF for each bit
    for bit in DIGITAL_BITS:
        on_src  = find_one([f"*tcat.nidq.xd_8_{bit}_*.txt",  f"times_ni_xd_8_{bit}_*.txt"])
        off_src = find_one([f"*tcat.nidq.xid_8_{bit}_*.txt", f"times_ni_xid_8_{bit}_*.txt"])
        print(f"[XD{bit}] on={on_src.name if on_src else None}, off={off_src.name if off_src else None}")
        if on_src and (non_empty_txt(on_src) or not SKIP_EMPTY):
            copy_file(on_src, DEST_DIR / f"XD{bit}_ON.txt")
        if off_src and (non_empty_txt(off_src) or not SKIP_EMPTY):
            copy_file(off_src, DEST_DIR / f"XD{bit}_OFF.txt")

if __name__ == "__main__":
    main()
