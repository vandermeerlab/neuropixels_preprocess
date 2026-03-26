# -*- coding: utf-8 -*-
"""
make_pid_peth_by_odor.py  (mV version, with TTL grouping alignment)
- Read continuous analog channel from NI *.nidq.bin -> Volt -> millivolt (mV)
- Low-pass + downsample to 100 Hz
- For each odor valve channel (XA0..XA6), compute PETH and export metrics:
  n_trials / peak_amp_mV / peak_latency_s / AUC_mV_s
"""

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, decimate

# -------------------- Paths --------------------
TRAIN_DIR = Path(r"E:\odor-pixels\Cohort5\M539\Training\SE_odor_test_all")  # directory with XA#_ON/OFF.txt, XD#_ON/OFF.txt
NI_DIR    = Path(r"E:\SE_odor_test_all_g0")                                  # NI continuous data (t0)
BIN_PATH  = NI_DIR / "SE_odor_test_all_g0_t0.nidq.bin"

# automatically find meta file
_meta_cand = [NI_DIR / "SE_odor_test_all_g0_t0.nidq", NI_DIR / "SE_odor_test_all_g0_t0.nidq.meta"]
META_PATH  = next((p for p in _meta_cand if p.exists()), None)
if META_PATH is None:
    hits = sorted(NI_DIR.glob("SE_odor_test_all_g0_t0.nidq*"), key=lambda p: len(p.name))
    META_PATH = hits[0] if hits else None

OUT_DIR   = TRAIN_DIR / "analysis_pid_by_odor_new"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Parameters --------------------
# === Odor mapping table: XA channel -> (display name, concentration string) ===
# According to your mapping: A,B,C,D,E,F correspond to XA0, XA1, XA3, XA4, XA5, XA6;
# XA2 is the neutral blank.
ODOR_MAP = {
    0: ("A: Benzaldehyde",       "16 µL"),
    1: ("B: Ethyl butyrate",     "32 µL"),
    3: ("C: Cinnamaldehyde",     "16 µL"),
    4: ("D: Isobutyric acid",    "16 µL"),
    5: ("E: 1-octanol",          "40 µL"),
    6: ("F: (+)-alpha-pinene",   "40 µL"),
    2: ("Neutral (blank/air)",   "0 µL"),
}

ODOR_XA_LIST   = list(range(0, 7))  # XA0..XA6 only
# Continuous PID channel: strongly recommended to use XA7 (even if event threshold is empty, waveform exists in bin)
AUTO_PICK_PID  = False
MANUAL_PID_XA  = 7

# Alignment mode: 'ttl' uses XD1_OFF matched to nearest XA#_ON; 'xa' uses XA#_ON directly
ALIGN_MODE     = "ttl"     # can change to "xa"
TTL_BIT_NAME   = "XD1"     # TTL bit (e.g. XD1, XD4, etc.)
TTL_EDGE_NAME  = "OFF"     # low-active TTL → use falling edge OFF

# Filtering and windows
LP_CUTOFF_HZ   = 20.0      # PID low-pass cutoff (Hz); adjust 10–30 by looking at PSD
TARGET_FS      = 100       # downsampled sampling rate
WIN            = (-2.0, 8.0)   # PETH window [s]
BASELINE       = (-2.0, -0.5)  # baseline window [s] (relative to 0)
RESP_WIN       = (0.0, 3.0)    # response window [s] for metrics

# Units: everything in mV
UNIT           = "mV"
UNIT_FACTOR    = 1e3       # V -> mV

# -------------------- Helper functions --------------------
def ensure_exists(p: Path, what: str):
    if not p or not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")

def read_meta_key(txt: str, key: str, cast=float, default=None):
    m = re.search(rf"^{re.escape(key)}=(.*)$", txt, flags=re.M)
    return cast(m.group(1)) if m else default

def load_meta(path: Path):
    t = path.read_text(encoding="utf-8", errors="ignore")
    Fs      = float(read_meta_key(t, "niSampRate"))
    n_saved = int(read_meta_key(t, "nSavedChans", int))
    rng_max = float(read_meta_key(t, "niAiRangeMax"))
    rng_min = float(read_meta_key(t, "niAiRangeMin"))
    return Fs, n_saved, rng_min, rng_max

def load_times_1d(path: Path) -> np.ndarray:
    if (not path.exists()) or path.stat().st_size == 0:
        return np.array([], dtype=float)
    arr = np.loadtxt(path, dtype=float)
    return np.atleast_1d(arr).ravel()

def dedup_events(t: np.ndarray, min_isi: float = 0.5) -> np.ndarray:
    """Remove events closer than min_isi (keep the first)."""
    if t.size == 0:
        return t
    t = np.sort(t)
    keep = [t[0]]
    for x in t[1:]:
        if x - keep[-1] >= min_isi:
            keep.append(x)
    return np.array(keep)

def assign_ttl_to_odors(ttl_times, odor_on_times_dict, max_dt=0.5):
    """
    Assign each TTL event to the nearest odor valve XA#_ON (if |dt| <= max_dt).
    Returns {odor_idx: np.array(times)}
    """
    out = {k: [] for k in odor_on_times_dict.keys()}
    sorted_on = {k: np.sort(v) for k, v in odor_on_times_dict.items()}
    for t in ttl_times:
        best_k, best_dt = None, float("inf")
        for k, arr in sorted_on.items():
            if arr.size == 0:
                continue
            j = np.searchsorted(arr, t)
            cands = []
            if j < arr.size: cands.append(arr[j])
            if j > 0:        cands.append(arr[j-1])
            for a in cands:
                dt = abs(a - t)
                if dt < best_dt:
                    best_dt, best_k = dt, k
        if best_k is not None and best_dt <= max_dt:
            out[best_k].append(t)
    return {k: np.array(v) for k, v in out.items()}

def peth_from_events(signal_ds, Fs_ds, event_times, win, baseline):
    """Return (tvec, matrix[trials, L]); subtract baseline median in baseline window."""
    pre  = int(abs(win[0]) * Fs_ds)
    post = int(win[1]       * Fs_ds)
    L    = pre + post
    tvec = (np.arange(-pre, post) / Fs_ds)

    b0 = int((baseline[0] - win[0]) * Fs_ds)
    b1 = int((baseline[1] - win[0]) * Fs_ds)

    segs = []
    for t in event_times:
        c = int(round(t * Fs_ds))
        if c - pre < 0 or c + post > len(signal_ds):
            continue
        seg = signal_ds[c - pre : c + post].copy()
        if 0 <= b0 < b1 <= len(seg):
            seg -= np.median(seg[b0:b1])
        segs.append(seg)
    if len(segs) == 0:
        return tvec, np.empty((0, L))
    return tvec, np.vstack(segs)

def metrics_from_peth(tvec, mat, resp_win):
    """Return (peak_amp_mV, peak_latency_s, auc_mV_s, n_trials)."""
    if mat.size == 0:
        return np.nan, np.nan, np.nan, 0
    m = mat.mean(0)
    i0 = np.searchsorted(tvec, resp_win[0], side="left")
    i1 = np.searchsorted(tvec, resp_win[1], side="right")
    if i1 <= i0:
        return np.nan, np.nan, np.nan, mat.shape[0]
    seg = m[i0:i1]
    pk_idx = int(np.argmax(seg))
    pk_amp = float(seg[pk_idx])
    pk_lat = float(tvec[i0 + pk_idx])
    auc    = float(np.trapz(seg, tvec[i0:i1]))
    return pk_amp, pk_lat, auc, mat.shape[0]

def safe_save_txt(path: Path, arr):
    """Avoid PermissionError on Windows: delete file if it exists first."""
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass
    np.savetxt(path, arr, delimiter=",")

# -------------------- Main --------------------
def main():
    # 0) Check paths
    ensure_exists(TRAIN_DIR, "Training dir")
    ensure_exists(BIN_PATH,  "NI bin")
    ensure_exists(META_PATH, "NI meta")

    # 1) Read meta
    Fs, n_saved, rng_min, rng_max = load_meta(META_PATH)
    print(f"[meta] Fs={Fs} Hz, nSavedChans={n_saved}, range=({rng_min},{rng_max}), meta='{META_PATH.name}'")

    # 2) Read continuous data (int16 -> Volt -> mV)
    fsize  = BIN_PATH.stat().st_size
    n_samp = fsize // (2 * n_saved)  # int16 * chans
    raw = np.memmap(BIN_PATH, dtype=np.int16, mode="r", shape=(n_samp, n_saved))
    scale_V = (rng_max - rng_min) / (2*32768)  # ADU -> Volt
    # Select PID channel
    if AUTO_PICK_PID:
        pid_xa = None
        for i in ODOR_XA_LIST:
            if (TRAIN_DIR / f"XA{i}_ON.txt").exists() and (TRAIN_DIR / f"XA{i}_ON.txt").stat().st_size > 0:
                pid_xa = i; break
        if pid_xa is None:
            pid_xa = MANUAL_PID_XA
            print(f"[pid] Auto not found, fallback XA{pid_xa}")
        else:
            print(f"[pid] Auto chose XA{pid_xa} as PID channel")
    else:
        pid_xa = MANUAL_PID_XA
        print(f"[pid] Using manually specified XA{pid_xa} as PID channel")

    pid_v_V  = np.array(raw[:, pid_xa], dtype=np.float64) * scale_V  # Volt
    pid_v_mV = pid_v_V * UNIT_FACTOR                                # mV

    # 3) PSD (in mV)
    f, Pxx = welch(pid_v_mV, fs=Fs, nperseg=int(Fs*4))
    plt.figure()
    plt.semilogy(f, Pxx)
    plt.xlim(0, 100)
    plt.xlabel("Hz")
    plt.ylabel("PSD (mV^2/Hz)")
    plt.title(f"PID raw PSD (XA{pid_xa})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pid_psd_mV.png", dpi=150)
    plt.close()
    print(f"[out] PSD -> {OUT_DIR/'pid_psd_mV.png'}")

    # 4) Low-pass + downsample to 100 Hz (still mV)
    b, a = butter(3, LP_CUTOFF_HZ/(Fs/2), btype="low")
    pid_lp = filtfilt(b, a, pid_v_mV)
    decim  = int(round(Fs / TARGET_FS))
    assert abs(Fs/decim - TARGET_FS) < 1e-6, "TARGET_FS must divide original Fs"
    pid_ds = decimate(pid_lp, decim, ftype="fir")
    Fs_ds  = Fs / decim

    # quick absolute check on one odor
    tvec, mat_abs = peth_from_events(pid_ds, Fs_ds, load_times_1d(TRAIN_DIR/'XA6_ON.txt'),
                                 WIN, baseline=(0,0))  # baseline empty → no subtraction
    m_abs = mat_abs.mean(0)
    print("XA6 absolute voltage: min/mean/max (mV) =", m_abs.min(), m_abs.mean(), m_abs.max())

    safe_save_txt(OUT_DIR / "pid_100Hz_mV.csv", pid_ds)
    print(f"[out] Downsampled PID -> {OUT_DIR/'pid_100Hz_mV.csv'}  (Fs={Fs_ds} Hz, n={len(pid_ds)})")

    # 5) Alignment events
    if ALIGN_MODE.lower() == "ttl":
        ttl_file  = TRAIN_DIR / f"{TTL_BIT_NAME}_{TTL_EDGE_NAME}.txt"
        ttl_times = dedup_events(load_times_1d(ttl_file), min_isi=0.5)
        odor_on_times = {i: load_times_1d(TRAIN_DIR / f"XA{i}_ON.txt") for i in ODOR_XA_LIST}
        ttl_groups = assign_ttl_to_odors(ttl_times, odor_on_times, max_dt=0.5)
        print(f"[align] Using {TTL_BIT_NAME}_{TTL_EDGE_NAME}, grouped to nearest XA#_ON")
    else:
        ttl_groups = None
        print("[align] Directly using XA#_ON for alignment")

    # 6) PETH for each odor
    summary_rows = []
    curves = []  # for overview plot
    for i in ODOR_XA_LIST:
        odor_name, odor_conc = ODOR_MAP.get(i, (f"XA{i}", ""))
        if ttl_groups is not None:
            align_times = ttl_groups.get(i, np.array([]))
            if align_times.size == 0:
                align_times = load_times_1d(TRAIN_DIR / f"XA{i}_ON.txt")
                if align_times.size == 0:
                    print(f"[odor XA{i}] no align events (TTL empty and XA{i}_ON empty), skipped")
                    continue
                align_label = f"XA{i}_ON"
            else:
                align_label = "TTL→odor"
        else:
            align_times = load_times_1d(TRAIN_DIR / f"XA{i}_ON.txt")
            if align_times.size == 0:
                print(f"[odor XA{i}] no on events, skipped")
                continue
            align_label = f"XA{i}_ON"

        tvec, mat = peth_from_events(pid_ds, Fs_ds, align_times, WIN, BASELINE)
        pk_amp, pk_lat, auc, ntr = metrics_from_peth(tvec, mat, RESP_WIN)
        summary_rows.append([f"XA{i}", odor_name, odor_conc, ntr, pk_amp, pk_lat, auc])

        # Save matrix and figure
        safe_save_txt(OUT_DIR / f"peth_matrix_odor_XA{i}.csv", mat)
        plt.figure()
        if mat.size > 0:
            m = mat.mean(0)
            s = mat.std(0) / np.sqrt(mat.shape[0])
            plt.plot(tvec, m, lw=1.5)
            plt.fill_between(tvec, m - s, m + s, alpha=0.3)
            curves.append((f"{odor_name} [{odor_conc}]", tvec.copy(), m.copy()))
        plt.axvline(0, ls="--", lw=1, color="k")
        plt.xlabel("Time (s)")
        plt.ylabel(f"PID ({UNIT}, baseline-subtracted)")
        plt.title(f"{odor_name} [{odor_conc}]  —  PID PETH (align={align_label}, PID=XA{pid_xa}, n={mat.shape[0]})")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"peth_odor_XA{i}.png", dpi=150)
        plt.close()
        print(f"[odor XA{i}] n={ntr}  peak={pk_amp:.3g} {UNIT} at {pk_lat:.3g}s, AUC={auc:.3g} {UNIT}.s")

    # 7) Summary CSV
    if summary_rows:
        header = "odor_channel,odor_name,odor_conc,n_trials,peak_amp_mV,peak_latency_s,AUC_mV_s"
        out_csv = OUT_DIR / "summary_by_odor.csv"
        try:
            if out_csv.exists():
                out_csv.unlink()
        except Exception:
            pass
        np.savetxt(out_csv, np.array(summary_rows, dtype=object), fmt="%s", delimiter=",", header=header, comments="")
        print(f"[out] Summary metrics -> {out_csv}")
    else:
        print("[warn] No odor channels produced PETH (no events or out of range)")

    # 8) Overview plot
    if curves:
        plt.figure()
        for label, tt, mm in curves:
            plt.plot(tt, mm, lw=1.6, label=label)
        plt.axvline(0, ls="--", lw=1, color="k")
        plt.xlabel("Time (s)")
        plt.ylabel(f"PID ({UNIT}, baseline-subtracted)")
        plt.title("PID PETH — all odors (names & concentrations)")
        plt.legend(ncols=2, frameon=False)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "peth_all_odors.png", dpi=150)
        plt.close()

if __name__ == "__main__":
    main()
