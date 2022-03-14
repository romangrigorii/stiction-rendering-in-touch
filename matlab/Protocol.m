%% Psychophysics protocol for stiction experiment
% start
clear
subject_n = 1;
subject_initials = 'ZZ';
new_trial = 1;
mkdir(strcat('C:\Users\atrox\Desktop\paper\code\subject results\',subject_initials));
if new_trial
    current_vals = 4095*[0,2,3,4,5,6,7,8]/8;
    trials_per_current = 5;
    results = -1*ones(length(current_vals)*trials_per_current,1);
    stimuli = results;
    for i = 1:length(current_vals)
        stimuli((i-1)*trials_per_current+1:i*trials_per_current) = current_vals(i)*ones(1,trials_per_current);
    end
    randloc = randperm(length(stimuli))';
    exp_times = zeros(6,trials_per_current*length(current_vals));
    save(strcat('C:\Users\atrox\Desktop\paper\code\subject results\',subject_initials,'\results.mat'),'current_vals','trials_per_current','results','stimuli','randloc','exp_times');
else
    load(strcat('C:\Users\atrox\Desktop\paper\code\subject results\',subject_initials,'\results.mat'));
end

port = start_send();
exp1 = app3;
while ~exp1.finished
    while ~exp1.finished && ~exp1.flag
        pause(.05);
    end
    exp1.flag = 0;
    exp1.command.Text = 'Remove the finger';
    data_send('z',port);
    data_send(exp1.knob_value*4095,port);  
    pause(.5);
    exp1.command.Text = 'Start swiping!';
end   
close all force
x = input('Let the experimenter know you are ready to start\n');
exp = app2;
for t = 1:length(stimuli)
    exp.finalvalue = 0;
    data_send('z',port);
    data_send(stimuli(randloc(t)),port);
    exp_times(:,t) = clock;
    pause(1);
    exp.command.Text = 'Rate how sticky the texture feels';
    while t == exp.trial
        pause(.1);
    end
    exp.command.Text = 'Remove finger from screen';
    fprintf(strcat(num2str(exp.finalvalue),'\n'));
    results(t) = exp.finalvalue;
    save(strcat('C:\Users\atrox\Desktop\paper\code\subject results\',subject_initials,'\results.mat'),'current_vals','trials_per_current','results','stimuli','randloc','exp_times');
    t = t + 1
end
exp.command.Text = 'All done!';
