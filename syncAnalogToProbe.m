% syncAnalogToProbe.m
%
% Aligns raw analog channel data (extracted by extract_raw_analog_signals.py)
% to the reference probe timebase using TPrime-adjusted time vectors.
%
% For each channel:
%   1. Loads  {OUTPUT_DIR}/{channel}_raw.mat         – original data + tvec
%   2. Loads  {CATGT_OUTPUT_DIR}/{channel}_adj_raw_times.npy  – TPrime output
%   3. Uses interp1 to resample the data onto a regular grid in probe time
%   4. Saves  {OUTPUT_DIR}/{channel}_raw_aligned.mat
%
%
% ── User-configurable parameters ─────────────────────────────────────────────

addpath(genpath('D:\npy-matlab'))   % path to npy-matlab

% Channels to process (must match names used in the Python extraction scripts)
channels = {'XA7'};

% Directory containing the {channel}_raw.mat files (OUTPUT_DIR from params)
output_dir = 'E:\Test_Fiber_session\preprocessed';

% Directory containing the {channel}_adj_raw_times.npy files (CATGT_OUTPUT_DIR)
catgt_output_dir = 'E:\Test_Fiber_session\catgt_M653_2026_02_05_g0';

% ── Processing ────────────────────────────────────────────────────────────────

for iCh = 1:numel(channels)
    ch = channels{iCh};

    mat_path       = fullfile(output_dir,      [ch '_unsynced.mat']);
    adj_times_path = fullfile(catgt_output_dir, [ch '_adj_raw_times.npy']);

    % ── Load original data ────────────────────────────────────────────────────
    if ~isfile(mat_path)
        warning('syncAnalogToProbe: .mat file not found, skipping %s\n  %s', ch, mat_path);
        continue
    end
    d = load(mat_path);   % fields: data, tvec, fs, channel_id, session_id

    original_data = double(d.data(:));   % ensure column vector, double precision
    original_tvec = d.tvec(:);
    fs            = d.fs;

    % ── Load TPrime-adjusted time vector ──────────────────────────────────────
    if ~isfile(adj_times_path)
        warning('syncAnalogToProbe: adj_raw_times.npy not found, skipping %s\n  %s', ch, adj_times_path);
        continue
    end
    adj_tvec = double(readNPY(adj_times_path));
    adj_tvec = adj_tvec(:);

    % Sanity check: lengths must match
    if length(original_data) ~= length(adj_tvec)
        warning(['syncAnalogToProbe: data length (%d) != adj_tvec length (%d) for %s. ' ...
                 'Truncating to shorter.'], length(original_data), length(adj_tvec), ch);
        n = min(length(original_data), length(adj_tvec));
        original_data = original_data(1:n);
        adj_tvec      = adj_tvec(1:n);
    end

    % ── Build regular time grid in probe timebase ─────────────────────────────
    % adj_tvec contains where each sample lands in the probe's timebase.
    % We interpolate the data onto a regular grid at the same nominal rate.
    new_tvec = (adj_tvec(1) : 1/fs : adj_tvec(end))';

    % ── Interpolate ───────────────────────────────────────────────────────────
    aligned_data = interp1(adj_tvec, original_data, new_tvec, 'linear');

    % ── Save ──────────────────────────────────────────────────────────────────
    aligned_mat_path = fullfile(output_dir, [ch '_synced.mat']);

    % Preserve metadata from original file where available
    aligned.data       = aligned_data;
    aligned.tvec       = new_tvec;
    aligned.fs         = fs;
    aligned.channel_id = d.channel_id;

    save(aligned_mat_path, '-struct', 'aligned', '-v7.3');
    fprintf('Saved aligned data for %s → %s\n', ch, aligned_mat_path);
end

fprintf('\nsyncAnalogToProbe complete.\n');
