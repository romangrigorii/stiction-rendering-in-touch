%% Psychophysics protocol for stiction experiment
% start
clear
subject_n = 1;
subject_initials = 'TT';
data = start_collect(20*60,1000);
save(strcat('C:\Users\atrox\Desktop\paper\code\subject results\',subject_initials,'\daqdata.mat'),'data');
