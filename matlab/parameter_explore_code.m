%% This program allows free exploration of stiction paramters while being able to adjust them in real time
exp = parameter_explore;
out = [0 0 0 0 0];
maxI = 7;
exp.IstuckmAKnob.Limits = [0 maxI];
exp.IstuckmAKnob.MajorTicks = 0:maxI;
exp.IslipmAKnob.Limits = [0 maxI];
exp.IslipmAKnob.MajorTicks = 0:maxI;
exp.IslidemAKnob.Limits = [0 maxI];
exp.IslidemAKnob.MajorTicks = 0:maxI;

if isempty(instrfind)
    fclose(instrfind);
end
port = serial('COM5', 'BaudRate', 230400, 'FlowControl', 'hardware');
fopen(port);
fprintf(port,'\n');
v = fscanf(port); 

while 1
    if exp.updated == 1
        fprintf(port,'%s','m\n');
        v = fscanf(port); 
        out = [exp.Istuckval*4095/maxI,exp.Islipval*4095/maxI,exp.Islideval*4095/maxI,exp.sigma,exp.delta];
        for i = 1:length(out)
            fprintf(port,'%s\n',num2str(out(i)));
        end
    elseif exp.updated == 2
        fprintf(port,'%s\n','z');
        v = fscanf(port);
    end
    exp.updated = 0;
    pause(.01);
end