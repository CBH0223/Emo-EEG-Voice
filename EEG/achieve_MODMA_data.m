load('chan_info_egi_128.mat')
EEG.chanlocs=chanlocs
EEG.chaninfo=chaninfo
eeglab redraw
ch = [22,9,11,24,123,33,122,12,112,28,117,26,104,37,87,47,98,62,52,92,67,77,75,70,83]
EEG1 = pop_select(EEG, 'channel', ch);
eeglab redraw
% 生成MAT文件名
mat_file_name = fullfile(output_folder_path, '02030004.mat');

% 保存为MAT文件
save(mat_file_name, 'EEG1');
clear;
clc;