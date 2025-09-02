%%extremely ugly code to convert tprime event txt files into event object

folder_list = {'E:\Task2-SWR\M590\preprocessed\M590-2025-03-31', ...
                'E:\Task2-SWR\M591\preprocessed\M591-2025-04-02', ...
                'E:\Task2-SWR\M536\preprocessed\M536-2025-04-14'};

for iFolder = 1:length(folder_list)

    cd(folder_list{iFolder});
    % save odor on & off
    XA0_ON = readmatrix('XA0_ON.txt');
%     XA0_OFF = readmatrix('XA0_OFF.txt');
    
    XA1_ON = readmatrix('XA1_ON.txt');
%     XA1_OFF = readmatrix('XA1_OFF.txt');
    
    XA2_ON = readmatrix('XA2_ON.txt');
%     XA2_OFF = readmatrix('XA2_OFF.txt');
    
    XA3_ON = readmatrix('XA3_ON.txt');
%     XA3_OFF = readmatrix('XA3_OFF.txt');

    XA4_ON = readmatrix('XA4_ON.txt');
%     XA4_OFF = readmatrix('XA4_OFF.txt');
    
    XA5_ON = readmatrix('XA5_ON.txt');
%     XA5_OFF = readmatrix('XA5_OFF.txt');

    XA6_ON = readmatrix('XA6_ON.txt');
%     XA6_OFF = readmatrix('XA6_OFF.txt');

    XA7_ON = readmatrix('XA7_ON.txt');
%     XA7_OFF = readmatrix('XA7_OFF.txt');

%     XD1_ON = readmatrix('XD1_ON.txt');
%     XD1_OFF = readmatrix('XD1_OFF.txt');

    XD2_ON = readmatrix('XD2_ON.txt');
    XD2_OFF = readmatrix('XD2_OFF.txt');

    XD3_ON = readmatrix('XD3_ON.txt');
    XD3_OFF = readmatrix('XD3_OFF.txt');

%     XD4_ON = readmatrix('XD4_ON.txt');
%     XD4_OFF = readmatrix('XD4_OFF.txt');

    evt = ts({XA0_ON, XA1_ON, XA2_ON, XA3_ON, XA4_ON, XA5_ON, XA6_ON, XA7_ON, ...
        XD2_ON, XD2_OFF, XD3_ON, XD3_OFF}, {'Odor A','Odor B', 'Neutral', ...
        'Odor C', 'Odor D', 'Odor E', 'Odor F', 'Odor G', 'WE1 ON', 'WE1 OFF', ...
        'WE2 ON', 'WE2 OFF'});

    save("all_events.mat","evt");
end