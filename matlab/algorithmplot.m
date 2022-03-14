%% plot an example of the stiction algorithm in action

clear
run('C:\Users\atrox\Desktop\general code\genhead.m');
directory = 'C:\Users\atrox\Desktop\projects\rendering stiction\code\subject results';
dire = dir(directory);
[b,a] = butter(6,2*50/1000);
su = 8;
Nc = 1.05; %N/volt;save
Lc = .59; %N/volt;
load(strcat(directory,'\',dire(su+3).name,'\results.mat'));
load(strcat(directory,'\',dire(su+3).name,'\daqdata.mat'));
lat = filtfilt(b,a,data(:,1));
lat = lat - mean(lat(1:2000,1));
nor = filtfilt(b,a,data(:,2));
nor = nor - mean(nor(1:2000,1));
cur = data(:,3);
lat = lat/Lc;
nor = nor/Nc;
mu = lat./nor;
nor = nor.*(nor>.03);

T = {};
T{1,1} = [100200,102200];
T{2,1} = [359000,361400];
T{3,1} = [125700,128000];
T{4,1} = [81000,84000];
T{5,1} = [29000,30600];
T{6,1} = [165200,167900];
T{7,1} = [327500,332600];
T{8,1} = [243100,249800];

%% plot type 1
hold on
t = linspace(0,2,2001);
for i = 1:8
    a = plot(t,mu(T{i,1}(1):T{i,1}(1)+2000) + (i*3) - 13,'-');
    a.Color = [.8 .8 .8];
    a.LineWidth = 1.2;
    curhi = mu(T{i,1}(1):T{i,1}(1)+2000).*(cur(T{i,1}(1):T{i,1}(1)+2000)>-4.8);
    curhi(curhi==0) = NaN';
    a = plot(t,curhi+ (i*3) - 13);
    a.MarkerSize = .5;
    a.Color = [1 0 0];
    a.LineWidth = 1.2;
end
axis([0 2 -15 15]);

%% plot type 2
hold on
t = linspace(0,2,2001);
pieces = {};
lp = 1;
for i = 1:8
    ii = 1;
    mup = mu(T{i,1}(1):T{i,1}(2));
    curp = cur(T{i,1}(1):T{i,1}(2));
    while ~((mup(ii)*mup(ii+1))<0 & (mup(ii+1)>0))
        ii = ii + 1;
    end
    s = ii;
    ii = ii + 1;
    while ~((mup(ii)*mup(ii+1))<0 & (mup(ii+1)>0))
        ii = ii + 1;
    end  
    e = ii;
    pieces{i,1} = mup(s:e);
    pieces{i,2} = curp(s:e);
end


hold on
L = 100;
for i = 1:8
    a = plot(L:(length(pieces{i,1})+L-1),pieces{i,1});
    a.Color = [.8 .8 .8];
    a.LineWidth = 1.2;   
    curp = pieces{i,1}.*(pieces{i,2}>-4.8);
    curp(curp==0) = NaN;    
    a = plot(L:(length(pieces{i,1})+L-1),curp);
    a.Color = [1 0 0];
    a.LineWidth = 1.2;      
    L = L  +  1000;
end
set(gca, 'YGrid', 'on', 'XGrid', 'off')
a = plot([500 1500],[1 1]);
a.Color = [0 0 0];
a.LineWidth = 2;
axis([0 9000 -1.75 1.75]);

ylabel('$\eta$','Interpreter','latex','FontSize',11);