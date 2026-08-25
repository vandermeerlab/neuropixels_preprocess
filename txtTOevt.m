%%extremely ugly code to convert tprime event txt files into event object

folder_list = {'E:\npx_quadprobe\LC_Pilot\preprocessed'};

for iFolder = 1:length(folder_list)

    cd(folder_list{iFolder});
    % save odor on & off
    % XA0_ON = readmatrix('XA1_ON.txt');
%     XA0_OFF = readmatrix('XA0_OFF.txt');
    
    XA1_ON = readmatrix('XA1_ON.txt');
%     XA1_OFF = readmatrix('XA1_OFF.txt');
    
    XA2_ON = readmatrix('XA2_ON.txt');
%     XA2_OFF = readmatrix('XA2_OFF.txt');
    
    % XA3_ON = readmatrix('XA3_ON.txt');
%     XA3_OFF = readmatrix('XA3_OFF.txt');

    % XA4_ON = readmatrix('XA4_ON.txt');
%     XA4_OFF = readmatrix('XA4_OFF.txt');
    
    % XA5_ON = readmatrix('XA5_ON.txt');
%     XA5_OFF = readmatrix('XA5_OFF.txt');

    XA6_ON = readmatrix('XA6_ON.txt');
%     XA6_OFF = readmatrix('XA6_OFF.txt');

    % XA7_ON = readmatrix('XA7_ON.txt');
%     XA7_OFF = readmatrix('XA7_OFF.txt');

%     XD1_ON = readmatrix('XD1_ON.txt');
%     XD1_OFF = readmatrix('XD1_OFF.txt');

    XD5_ON = readmatrix('XD5_ON.txt');
    XD5_OFF = readmatrix('XD5_OFF.txt');

    XD6_ON = readmatrix('XD6_ON.txt');
    XD6_OFF = readmatrix('XD6_OFF.txt');

%     XD4_ON = readmatrix('XD4_ON.txt');
%     XD4_OFF = readmatrix('XD4_OFF.txt');

    evt = ts({XA1_ON, XA2_ON, XA6_ON, ...
        XD5_ON, XD5_OFF, XD6_ON, XD6_OFF}, {'Odor B','Neutral', ...
        'Odor F','AirPuff ON', 'AirPuff OFF', ...
        'CamFrame ON', 'CamFrame OFF'});

    save("all_events.mat","evt");
end