function out = data_send(val,port)
if val == 'z'
    fprintf(port,'%s\n','z');
    v = fscanf(port);
    v = fscanf(port); 
else
    fprintf(port,'%s','m\n');
    v = fscanf(port);
    out = [val,val,0,.1,.8];
    for v = 1:length(out)
        fprintf(port,'%s\n',num2str(out(v)));
    end
    v = fscanf(port); 
end
end