% Script to plot early vs late licking
figure('WindowState', 'maximized');

% First plot early behavior
cd('E:\odor-pixels\Cohort5\M539\Training\M539-2024-08-19')
load("all_events.mat");
lick_on = evt.t{strcmp(evt.label, 'lick on')};
lick_off = evt.t{strcmp(evt.label, 'lick off')};

if length(lick_on) == length(lick_off)
    lick_dur = lick_off - lick_on;
else
    % Implement a fix
end
reward_on = evt.t{strcmp(evt.label, 'reward on')};
lick_latency = zeros(size(reward_on));
for iR = 1:length(reward_on)
    post_licks = lick_on(lick_on > reward_on (iR));
    lick_latency(iR) = post_licks(1) - reward_on(iR);
end


lick_ts = ts(evt.t(strcmp(evt.label, 'lick on')), {'licks'});
fake_S = lick_ts;
for iR = 1:length(reward_on)
    temp_S = restrict(lick_ts, iv(reward_on(iR)-6, reward_on(iR)+6));
    fake_S.t{iR} = temp_S.t{1} - reward_on(iR);
    fake_S.label{iR} = temp_S.label{1};
end
ax  = subplot(2,2,1);
cfg = [];
cfg.openNewFig = 0;
MultiRaster(cfg, fake_S);
xline(0, '--red')
xlabel('Time from reward (s)')
ylabel('All reward delivery')
title('Early Lick Training', 'FontSize', 24);
box off;
ax.XAxis.FontSize = 24;
ax.YAxis.FontSize = 24;

ax  = subplot(2,2,3);
histogram(lick_latency, 0:1:10, 'Normalization', 'probability');
xlabel('First lick latency (sec)')
ylabel('Proportion of licks')
yticks([0:0.25:1])
xticks(0:1:10)
ylim([0 1])
ax.TickDir = 'out';
ax.XAxis.FontSize = 24;
ax.YAxis.FontSize = 24;
box off; 
% title(extractAfter(pwd, 'Training\'));
%%
% Next plot expert session
cd('E:\odor-pixels\Cohort4\M524\Training\M524-2024-08-08')
load("all_events.mat");
lick_on = evt.t{strcmp(evt.label, 'lick on')};
lick_off = evt.t{strcmp(evt.label, 'lick off')};

if length(lick_on) == length(lick_off)
    lick_dur = lick_off - lick_on;
else
    % Implement a fix
end
reward_on = evt.t{strcmp(evt.label, 'reward on')};
lick_latency = zeros(size(reward_on));
for iR = 1:length(reward_on)
    post_licks = lick_on(lick_on > reward_on (iR));
    lick_latency(iR) = post_licks(1) - reward_on(iR);
end

lick_ts = ts(evt.t(strcmp(evt.label, 'lick on')), {'licks'});
fake_S = lick_ts;
for iR = 1:length(reward_on)
    temp_S = restrict(lick_ts, iv(reward_on(iR)-6, reward_on(iR)+6));
    fake_S.t{iR} = temp_S.t{1} - reward_on(iR);
    fake_S.label{iR} = temp_S.label{1};
end
ax  = subplot(2,2,2);
cfg = [];
cfg.openNewFig = 0;
MultiRaster(cfg, fake_S);
xline(0, '--red')
xlabel('Time from reward (s)')
ylabel('All reward delivery')
title('Late Lick Training', 'FontSize', 24);
box off;
ax.XAxis.FontSize = 24;
ax.YAxis.FontSize = 24;


ax  = subplot(2,2,4);
histogram(lick_latency, 0:1:10, 'Normalization', 'probability');
xlabel('First lick latency (sec)')
xticks(0:1:10)
yticks([0:0.25:1])
ylabel('Proportion of licks')
ax.TickDir = 'out';
ax.XAxis.FontSize = 24;
ax.YAxis.FontSize = 24;
box off; 

