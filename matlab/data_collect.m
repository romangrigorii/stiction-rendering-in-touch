duration = 30;
sr = 10000;
st = daq.createSession('ni');
st.Rate = sr;
st.DurationInSeconds = duration;
ch1 = addAnalogInputChannel(st,'Dev2','ai1','Voltage');
ch1.InputType = 'Differential'; %% modulation values
ch2 = addAnalogInputChannel(st,'Dev2','ai2','Voltage');
ch2.InputType = 'Differential'; %% modulation values
ch3 = addAnalogInputChannel(st,'Dev2','ai3','Voltage');
ch3.InputType = 'Differential'; %% modulation values

if ~isempty(instrfind)
    fclose(instrfind);
end
port = serial('COM5', 'BaudRate', 230400, 'FlowControl', 'hardware');
fopen(port);
fprintf(port,'\n');
v = fscanf(port); 
fprintf(port,'%s\n','z');
v = fscanf(port);

Istick = [0,2047,4095];
Islip = [0,1023,2047,3071,4095];
DATA = zeros(length(Istick),length(Islip),duration*sr,3);
for i = 1:length(Istick)
    for ii = 1:length(Islip)
        fprintf(port,'%s','m\n');
        v = fscanf(port); 
        out = [Istick(i),Islip(ii),0,.1,.8];
        for v = 1:length(out)
            fprintf(port,'%s\n',num2str(out(v)));
        end
        x = input('');
        data = startForeground(st);
        DATA(i,ii,:,:) = data;
    end
end

%% processing data that explores variation in mus
out = zeros(4,20,5000,3,4);

[b,a] = butter(3,2*1000/10000);
con = 10.5/5.9;

Nc = .105; %N/volt;save
Lc = .59; %N/volt;

for i=1:4    
    lat = squeeze(DATA(i,1,1,1,:,1));
    nor = squeeze(DATA(i,1,1,1,:,3));
    cur = squeeze(DATA(i,1,1,1,:,2));
    lat = filtfilt(b,a,lat(40001:end) - mean(lat(1:3000)));
    nor = filtfilt(b,a,nor(40001:end) - mean(nor(1:3000)));
    cur = cur(40001:end);
    mu = lat./nor*Nc/Lc;
    l = 1;
    g = 1;
    while g<length(cur)
        if cur(g)>-5
            start = g;
            start2 = g;
            l = l + 1;
            flag = 1;
            while cur(g)>-5 && g<length(cur)
                if cur(g)>5
                    endd2 = g;
                end               
                g = g + 1;
            end            
            endd = g;           
            if endd-start<1000
                out(i,l,:,1,1) = zpad(mu(start:endd),5000,1);
                out(i,l,:,2,1) = zpad(lat(start:endd),5000,1);
                out(i,l,:,3,1) = zpad(nor(start:endd),5000,1);
                out(i,l,:,1,2) = zpad(mu(start:endd),5000,1);
                out(i,l,:,2,2) = zpad(lat(start:endd),5000,1);
                out(i,l,:,3,2) = zpad(nor(start:endd),5000,1); 
                out(i,l,:,1,3) = zpad(mu(start:endd2),5000,1);
                out(i,l,:,2,3) = zpad(lat(start:endd2),5000,1);
                out(i,l,:,3,3) = zpad(nor(start:endd2),5000,1);                 
                out(i,l,:,1,4) = zpad(mu(start:endd+1000),5000,1);
                out(i,l,:,2,4) = zpad(lat(start:endd+1000),5000,1);
                out(i,l,:,3,4) = zpad(nor(start:endd+1000),5000,1);
            end
        end
        g = g + 1;
    end
end

            