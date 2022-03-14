function out = start_send()
!!!!!!!!!!!
ACCIDENTALLY REMOVED LINES FROM THIS CODE, NEED TO REWRITE IN ORDER TO READ 'z' FROM THE PIC 
!!!!!!!!!!!

if ~isempty(instrfind)
    fclose(instrfind);
end
port = serial('COM6', 'BaudRate', 230400, 'FlowControl', 'hardware');
fopen(port);
out = port;
end