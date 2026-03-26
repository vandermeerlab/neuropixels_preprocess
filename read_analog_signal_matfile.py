"""
read_analog_signal_matfile.py
------------------------------
Utility for loading raw analog signal .mat files produced by the odor-pixels
preprocessing pipeline.

Two file variants are handled transparently:
  - MATLAB 5 format  : written by extract_raw_analog_signals.py (scipy.io.savemat)
  - MATLAB v7.3 / HDF5 : written by syncAnalogToProbe.m (MATLAB's own save)

Fields in every file
--------------------
  channel_id  : str    - SpikeInterface channel ID, e.g. 'nidq#XA7'
  tvec        : (N,)   - time vector in seconds (probe-aligned after syncAnalogToProbe)
  data        : (N,)   - signal in microvolts
  fs          : float  - sampling frequency in Hz

Usage
-----
    import read_analog_signal_matfile as ram

    # Load the unsynced file (written by extract_raw_analog_signals.py)
    channel_id, tvec, data, fs = ram.load_raw_analog_mat('XA7_unsynced.mat')

    # Load the synced file (written by syncAnalogToProbe.m)
    channel_id, tvec, data, fs = ram.load_raw_analog_mat('XA7_synced.mat')

    # Both calls return the same four variables regardless of format:
    #   channel_id  ->  'nidq#XA7'
    #   tvec        ->  array([0.000, 0.001, 0.002, ...])   (seconds)
    #   data        ->  array([...])                         (microvolts)
    #   fs          ->  1000.0

Dependencies
------------
    numpy, scipy          (always required)
    h5py                  (only needed for v7.3 files, i.e. syncAnalogToProbe output)
"""

import numpy as np
import scipy.io as scio


def _decode_mat_string(h5_dataset):
    """
    Decode a MATLAB char array stored in HDF5 (v7.3) back into a Python string.
    Uses the same uint32 -> view('U1') pattern as the NWB loading code.
    """
    temp = np.asarray(h5_dataset[:], dtype='uint32')
    temp = temp.T.view('U1')
    return ''.join(temp.flatten()).strip()


def _load_hdf5(mat_path):
    import h5py
    with h5py.File(mat_path, 'r') as f:
        data       = f['data'][:].flatten()
        tvec       = f['tvec'][:].flatten()
        fs         = float(np.asarray(f['fs']).flatten()[0])
        channel_id = _decode_mat_string(f['channel_id'])
    return channel_id, tvec, data, fs


def _load_scipy(mat_path):
    mat        = scio.loadmat(mat_path)
    data       = mat['data'].flatten()
    tvec       = mat['tvec'].flatten()
    fs         = float(np.asarray(mat['fs']).flatten()[0])
    ch         = mat['channel_id']
    channel_id = str(ch.item() if hasattr(ch, 'item') else ch.flat[0]).strip()
    return channel_id, tvec, data, fs


def load_raw_analog_mat(mat_path):
    """
    Load a {channel}_raw*.mat file regardless of whether it was written by
    scipy.io.savemat (MATLAB 5) or by MATLAB itself (v7.3 / HDF5).

    Returns
    -------
    channel_id : str
    tvec       : np.ndarray  (N,)
    data       : np.ndarray  (N,)
    fs         : float
    """
    try:
        channel_id, tvec, data, fs = _load_scipy(mat_path)
    except NotImplementedError:
        # scipy raises NotImplementedError for v7.3 HDF5 files
        channel_id, tvec, data, fs = _load_hdf5(mat_path)

    assert tvec.shape == data.shape, (
        f"tvec {tvec.shape} and data {data.shape} size mismatch in {mat_path}"
    )

    return channel_id, tvec, data, fs
