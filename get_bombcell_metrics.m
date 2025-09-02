%%
addpath(genpath('C:\bombcell')); % Download from https://github.com/Julie-Fabre/bombcell
addpath(genpath('C:\npy-matlab')); % Download from https://github.com/kwikteam/npy-matlab
addpath(genpath('C:\prettify_matlab')); % Download from https://github.com/Julie-Fabre/prettify_matlab
clear;

% Change this accordingly
probe = 'imec0';

% Path to where the Kilosort output lives
ephysKilosortPath = sprintf('F:\\KS4_outputs\\M591-2025-03-31_%s\\sorter_output\\', probe);

% Path to the correct CatGT-ed file
ephysRawFile = sprintf("E:\\Task2-SWR\\M591\\catgt_M591-2025-03-31_g0\\M591-2025-03-31_g0_tcat.%s.ap.bin", probe);

ephysMetaDir = dir(['E:\Task2-SWR\M591\catgt_M591-2025-03-31_g0\*' probe '.ap.meta']);

savePath = 'E:\Task2-SWR\M591\preprocessed\M591-2025-03-31\imec0_bombcell';
if ~isfolder(savePath)
    mkdir(savePath);
end
savePath = [savePath '\qMetrics']; % where you want to save the quality metrics

kilosortVersion = 4; % if using kilosort4, you need to have this value kilosertVersion=4. Otherwise it does not matter. 
gain_to_uV = NaN; % use this if you are not using spikeGLX or openEphys to record your data. this value, 
% when mulitplied by your raw data should convert it to  microvolts. 
[spikeTimes_samples, spikeClusters, templateWaveforms, templateAmplitudes, pcFeatures, ...
    pcFeatureIdx, channelPositions] = bc.load.loadEphysData(ephysKilosortPath, savePath);

param = bc.qm.qualityParamValues(ephysMetaDir, ephysRawFile, ephysKilosortPath, gain_to_uV, kilosortVersion);

% if using SpikeGLX, you can use this function: 
if ~isempty(ephysMetaDir)
    if endsWith(ephysMetaDir.name, '.ap.meta') %spikeGLX file-naming convention
        meta = bc.dependencies.SGLX_readMeta.ReadMeta(ephysMetaDir.name, ephysMetaDir.folder);
        [AP, ~, SY] = bc.dependencies.SGLX_readMeta.ChannelCountsIM(meta);
        param.nChannels = AP + SY;
        param.nSyncChannels = SY;
    end
end

[qMetric, unitType] = bc.qm.runAllQualityMetrics(param, spikeTimes_samples, spikeClusters, ...
        templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions, savePath);
close all;

%% Uncomment and run if you want to look at each cell
% https://github.com/Julie-Fabre/bombcell/wiki/Guide-to-bombcell's-GUI

% loadRawTraces = 1; % default: don't load in raw data (this makes the GUI significantly faster)
% bc.load.loadMetricsForGUI;
% 
% unitQualityGuiHandle = bc.viz.unitQualityGUI_synced(memMapData, ephysData, qMetric, forGUI, rawWaveforms, ...
%     param, probeLocation, unitType, loadRawTraces);