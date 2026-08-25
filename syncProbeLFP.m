addpath(genpath('C:\npy-matlab')) % Download from https://github.com/kwikteam/npy-matlab if not present
cd('D:\vCA1\M593\preprocessed\M593_2025_12_26'); %paste path to preprocessed folder

imec0 = load('imec0_clean_lfp.mat');
if isfile('imec1_clean_lfp.mat')
    imec1_old = load('imec1_clean_lfp.mat');
    imec1_ts = readNPY('E:\catgt_tempFiles\catgt_M588-2025-06-27_g0\imec1_adj_lfp_times.npy');% Path to the adj_lfp_times.npy file
    imec1_old.lfp_tvec = imec1_ts';
    %%
    % Sometimes the length of tvecs can  be different, if this is the case,
    % check the filesizes and meta files to confirm

    if length(imec0.lfp_tvec) ~= length(imec1_old.lfp_tvec)
        warning('The Tvec sizes are not the same, check meta files and file sizes to confirm!!')
        smaller = min([length(imec0.lfp_tvec), length(imec1_old.lfp_tvec)]);
        imec0.lfp_traces = imec0.lfp_traces(1:smaller,:);
        imec0.lfp_tvec = imec0.lfp_tvec(1:smaller);
        imec1_old.lfp_traces = imec1_old.lfp_traces(1:smaller,:);
        imec1_old.lfp_tvec = imec1_old.lfp_tvec(1:smaller);
    end
    imec1_old.adj_traces = zeros(size(imec1_old.lfp_traces));
    for iCh = 1:size(imec1_old.lfp_traces,2)
        imec1_old.adj_traces(:,iCh) = interp1(imec1_old.lfp_tvec', imec1_old.lfp_traces(:,iCh), imec0.lfp_tvec');
    end
    % imec1.adj_traces and imec0.lfp_traces are now on the unified timebase imec0.lfp_tvec
    imec1 = [];
    imec1.lfp_tvec = imec0.lfp_tvec;
    imec1.lfp_traces = imec1_old.adj_traces;
    imec1.channel_ids = imec1_old.channel_ids;
    imec1.lfp_fs = imec1_old.lfp_fs;
    imec1.depths = imec1_old.depths;
    imec1.shank_ids = imec1_old.shank_ids;
    save('imec1_clean_lfp_new.mat',  'imec1','-v7.3'); % rewriting imec1_clean_lfp.mat
end
imec0.lfp_traces = double(imec0.lfp_traces);
save('imec0_clean_lfp_new.mat',  'imec0','-v7.3'); % rewriting imec0_clean_lfp.mat