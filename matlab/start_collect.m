function out = start_collect(duration,sr)
st = daq.createSession('ni');
st.Rate = sr;
st.DurationInSeconds = duration;
ch1 = addAnalogInputChannel(st,'Dev2','ai2','Voltage');
ch1.InputType = 'Differential'; %% lateral force
ch2 = addAnalogInputChannel(st,'Dev2','ai3','Voltage');
ch2.InputType = 'Differential'; %% normal force
ch3 = addAnalogInputChannel(st,'Dev2','ai1','Voltage');
ch3.InputType = 'Differential'; %% normal force
out = startForeground(st);
end