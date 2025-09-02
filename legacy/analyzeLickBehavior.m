% Script to plot lick events
cd('E:\odor-pixels\Cohort5\M539\Training\M539-2024-08-19')
load("all_events.mat");
% M524 SE: odor 4, odor 5 (rewarded), odor 2; UE_odors: odor 1, odor 6(rewarded), odor 3
% M508 SE: odor 6, odor 1 (rewarded), odor 3; UE_odors: odor 5, odor 4(rewarded), odor 2
%% Plot distribution lick durations to see if everything is sensible

lick_on = evt.t{strcmp(evt.label, 'lick on')};
lick_off = evt.t{strcmp(evt.label, 'lick off')};

if length(lick_on) == length(lick_off)
    lick_dur = lick_off - lick_on;
else
    % Implement a fix
end
histogram(lick_dur*1000, 0:1:500)
ylabel('Count')
xlabel("Lick duration (milliseconds)") 
% Decide if lick_on can be used as licks
%%
lick_on = lick_on(lick_dur < 0.1);
%% Make Raster plots centred around reward duration to see how quickly does the mouse lick rewards
reward_on = evt.t{strcmp(evt.label, 'reward on')};

lick_ts = ts(evt.t(strcmp(evt.label, 'lick on')), {'licks'});

fake_S = lick_ts;
for iR = 1:length(reward_on)
    temp_S = restrict(lick_ts, iv(reward_on(iR)-6, reward_on(iR)+6));
    fake_S.t{iR} = temp_S.t{1} - reward_on(iR);
    fake_S.label{iR} = temp_S.label{1};
end
%
figure('WindowState','maximized')
cfg = [];
cfg.openNewFig = 0;
MultiRaster(cfg, fake_S);
xline(0, '--red')
xlabel('Time from reward (s)')
ylabel('All reward delivery')
title(extractAfter(pwd, 'Training\'));


%% Make raster plots of licks with respect to various odor onsets
SE_odors = {'Odor 4', 'Odor 5', 'Odor 2', };
clist = {[1 0 0], [0 1 0], [0 1 1]};
spk_colors = [];
SE_ts = lick_ts;
for iOdor = 1:length(SE_odors)
    this_onset = evt.t{strcmp(SE_odors(iOdor), evt.label)};
    for iEvt = 1:length(this_onset)
        temp_ts = restrict(lick_ts, iv(this_onset(iEvt)-2,this_onset(iEvt)+5));
        SE_ts.t{(iOdor-1)*length(this_onset)+iEvt} = temp_ts.t{1} - this_onset(iEvt);
        SE_ts.label{(iOdor-1)*length(this_onset)+iEvt} = SE_odors{iOdor};
        spk_colors = [spk_colors; clist{iOdor}];
    end
end

figure('WindowState','maximized')
cfg = [];
cfg.openNewFig = 0;
cfg.spkColor = spk_colors;
MultiRaster(cfg, SE_ts);
xline(0, '--red')
xline(1.5, '--red')
xline(2, '--red')
xlabel('Time from odor onset (s)')
ylabel('Trials grouped by odor')
title('SE Block')

%%
UE_odors = {'Odor 1', 'Odor 6', 'Odor 3'};
UE_ts = lick_ts;
for iOdor = 1:length(UE_odors)
    this_onset = evt.t{strcmp(UE_odors(iOdor), evt.label)};
    for iEvt = 1:length(this_onset)
        temp_ts = restrict(lick_ts, iv(this_onset(iEvt)-2,this_onset(iEvt)+5));
        UE_ts.t{(iOdor-1)*length(this_onset)+iEvt} = temp_ts.t{1} - this_onset(iEvt);
        UE_ts.label{(iOdor-1)*length(this_onset)+iEvt} = UE_odors{iOdor};
    end
end

figure('WindowState','maximized')
cfg = [];
cfg.openNewFig = 0;
MultiRaster(cfg, UE_ts);
xline(0, '--red')
xline(1.5, '--red')
xline(2, '--red')
xlabel('Time from odor onset (s)')
ylabel('Trials grouped by odor')
title('UE Block')
%% Sanity check to see if the rewarded odor is indeed the correct one

ue_delay = reward_on(1:81) - evt.t{strcmp(UE_odors(2), evt.label)};
se_delay = reward_on(82:end) - evt.t{strcmp(SE_odors(2), evt.label)};
%%

%% Make raster plots of licks with respect to various odor onsets

fig = figure('WindowState','maximized');

SE_odors = {'Odor 4', 'Odor 5', 'Odor 2', };
clist = {[1 0 0], [0 1 0], [0 1 1]};
spk_colors = [];
SE_ts = lick_ts;
for iOdor = 1:length(SE_odors)
    this_onset = evt.t{strcmp(SE_odors(iOdor), evt.label)};
    for iEvt = 1:length(this_onset)
        temp_ts = restrict(lick_ts, iv(this_onset(iEvt)-2,this_onset(iEvt)+4));
        SE_ts.t{(iOdor-1)*length(this_onset)+iEvt} = temp_ts.t{1} - this_onset(iEvt);
        SE_ts.label{(iOdor-1)*length(this_onset)+iEvt} = SE_odors{iOdor};
        spk_colors = [spk_colors; clist{iOdor}];
    end
end

ax = subplot(1,2,1);
cfg = [];
cfg.openNewFig = 0;
cfg.spkColor = spk_colors;
MultiRaster(cfg, SE_ts);
xline(0, '--black')
xline(1.5, '--black')
xline(2, '--black')
xlabel('Time from odor onset (s)')
ylabel('Trials grouped by odor')
yticks([])
xticks([-2 -1 0 1 2 3 4])
xlim([-2 4])
ax.XAxis.FontSize = 36;
ax.YAxis.FontSize = 36;
ax.TickDir = 'out';
box off;
title('SE', 'FontSize', 48)


ax = subplot(1,2,2);
spk_colors = [];
UE_odors = {'Odor 1', 'Odor 6', 'Odor 3'};
UE_ts = lick_ts;
for iOdor = 1:length(UE_odors)
    this_onset = evt.t{strcmp(UE_odors(iOdor), evt.label)};
    for iEvt = 1:length(this_onset)
        temp_ts = restrict(lick_ts, iv(this_onset(iEvt)-2,this_onset(iEvt)+4));
        UE_ts.t{(iOdor-1)*length(this_onset)+iEvt} = temp_ts.t{1} - this_onset(iEvt);
        UE_ts.label{(iOdor-1)*length(this_onset)+iEvt} = UE_odors{iOdor};
        spk_colors = [spk_colors; clist{iOdor}];
    end
end

cfg = [];
cfg.openNewFig = 0;
cfg.spkColor = spk_colors;
MultiRaster(cfg, UE_ts);
xline(0, '--black')
xline(1.5, '--black')
xline(2, '--black')
xlabel('Time from odor onset (s)')
ylabel('Trials grouped by odor')
yticks([])
xticks([-2 -1 0 1 2 3 4])
xlim([-2 4])
ax.XAxis.FontSize = 36;
ax.YAxis.FontSize = 36;
ax.TickDir = 'out';
box off;
title('UE', 'FontSize', 48)
set(gcf, 'Renderer','painters')

