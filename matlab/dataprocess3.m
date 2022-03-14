%% processes experiment data
%clear
addpath('C:\Users\atrox\Desktop\Work\Research\Code\General code\MATLAB code');
directory = 'C:\Users\atrox\Desktop\Work\Research\projects\z Finished\Stiction rendering in touch\code\subject results';
L = length(dir(directory)) - 3;
dire = dir(directory);
Nc = 1.05; %N/volt;save
Lc = .59; %N/volt;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% defining variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
et = [];
ns = [];
[bl1,al1] = butter(1,2*150/1000); % optimal filtering of lateral force data
%run('C:\Users\atrox\Desktop\Work\Research\projects\z Finished\Stiction rendering in touch\code\filterformulation.m');
%bl1 = bd;
%al1 = ad;

[bl2,al2] = butter(6,2*50/1000); % extreme filtering of lateral force data
[bn,an] = butter(1,2*15/1000); % filtering normal force data

times = {};
swipes = {};
time_data = {};
mu_data = {};
dir_data = {};
nor_data = {};
lat_data = {};
trials = [];
subjects = [1,2,3,4,5,6,7,9,10,11];
currents = [0,1,2,3,4,5,6,7,8];
swipe_n = {};
res = {};
GG = {};
sti = {};
time_fix = {};
for i = 1:length(subjects)
    time_fix{i} = zeros(1,41);
end
time_fix{10}(2) = 3000;


for s = 1:length(subjects)
    su = subjects(s);
    load(strcat(directory,'\',dire(su+3).name,'\results.mat'));
    load(strcat(directory,'\',dire(su+3).name,'\daqdata.mat'));
    
    % extracting lateral, normal, and current information
    lato = data(:,1);
    
    lat = filtfilt(bl1,al1,data(:,1));
    lat = lat - mean(lat(1:2000,1));
    lato = lato - mean(lato(1:2000,1));
    lat2 = filtfilt(bl2,al2,data(:,1));
    lat2 = lat2 - mean(lat2(1:2000,1));
    nor = filtfilt(bn,an,data(:,2));
    nor = nor - mean(nor(1:2000,1));
    cur = data(:,3);
    lat = lat/Lc;
    nor = nor/Nc;
    muo = real(lat.*(nor.^(v2(s))));
    %muo = lat./nor;
    mu = lat./nor;
    mu2 = lat2./nor;
    
    dmuo = filtfilt(bl1,al1,derivR(muo,1,1000));
    dmuo(isnan(dmuo)) = 0;
    dmuo(isinf(abs(dmuo))) = 0;
    dmuo = filtfilt(bl1,al1,dmuo);
    
    dmu = filtfilt(bl1,al1,derivR(mu,1,1000));
    dmu(isnan(dmu)) = 0;
    dmu(isinf(abs(dmu))) = 0;
    dmu = filtfilt(bl1,al1,dmu);
    
    ddmu = derivR(mu,2,1000);
    ddmu(isnan(ddmu)) = 0;
    ddmu(isinf(abs(ddmu))) = 0;
    ddmu = filtfilt(bl1,al1,ddmu);
    
    dlat = derivR(lat,1,1000);
    dlat(isnan(dlat)) = 0;
    dlat(isinf(abs(dlat))) = 0;
    dlat = filtfilt(bl1,al1,dlat);
    
    ddlat = derivR(lat,2,1000);
    ddlat(isnan(ddlat)) = 0;
    ddlat(isinf(abs(ddlat))) = 0;
    ddlat = filtfilt(bl1,al1,ddlat);
    
    nor = nor.*(nor>.03);
    % setting up an array that can seperate trials
    
    time_vector = [0];
    for t = 2:40
        time_vector(t) = (exp_times(4,t)*3600+exp_times(5,t)*60 + exp_times(6,t)) - (exp_times(4,1)*3600+exp_times(5,1)*60 + exp_times(6,1));
    end
    time_vector(41) = time_vector(40)+max(diff(time_vector))*3;
    
    start = 1;
    while start<length(mu) && (nor(start)==0)
        start = start + 1;
    end
    endd = start+1;
    while endd<length(mu) && (nor(endd)~=0)
        endd = endd + 1;
    end
    start = endd + 1;
    while endd<length(mu) && (nor(endd)==0)
        endd = endd + 1;
    end
    start = round((start + endd)/2);
    time_vector = time_vector - time_vector(2) + start/1000;
    time_vector = round(time_vector(time_vector>0)*1000);
    if length(time_vector) == 40
        time_vector = [1,time_vector];
    end
    
    out = zeros(1,length(nor))';
    out(time_vector+1) = 1;
    tt_d = []; tt = 1;
    while (time_vector(end) + 51)<length(out) && sum(tt_d==0)==0
        time_vector = time_vector + 50;
        out = zeros(1,length(nor))';
        out(time_vector+1) = 1;
        tt_d(tt) = sum(out.*(nor>0));
        tt = tt + 1;
    end
    if sum(tt_d==0)==0
        [~,tt] = min(tt_d);
        time_vector = time_vector - (tt-1)*50;
    end
    out = zeros(1,length(nor))';
    out(time_vector+1) = 1;
    time_vector = time_vector + 50;
    hh = (stimuli(randloc)/4095*10)-5;
    hh2 = zeros(size(hh));
    vec = -5.625:1.25:5.625;
    for j = [1,3,4,5,6,7,8,9]
        hh2 = hh2 + (hh>vec(j)).*(hh<vec(j+1))*j;
    end
    
    % solving for the issue of pressing the submit button too early
    time_vector_mod = [time_vector(1)]; g = 2; gg = [1];
    for i = 2:length(time_vector)
        if sum(nor(time_vector(i-1):time_vector(i)))>100
            time_vector_mod(g) = time_vector(i);
            gg(g) = i;
            g = g + 1;
        end
    end
    time_vector = time_vector_mod;
    time_vector(2:end) = time_vector(2:end) + time_fix{s}(gg(2:end)-1);
    % extracting force information trial1 by trial1
    
    trial1 = 1;
    while trial1 < length(time_vector)
        start = round(time_vector(trial1)) + time_fix{s}(trial1);
        endd = round(time_vector(trial1+1)) - time_fix{s}(trial1+1);
        endds = endd;
        
        % dividng up the trial1 into sections where the subject was in contact
        % with the device
        nor_d = (nor(start:endd)>0);
        nor_d = (abs(diff(nor_d))>0);
        nor_d = nor_d.*(1:1:(length(nor_d)))';
        nor_d = nor_d(nor_d~=0);
        nor_d = nor_d(nor_d>100);
        if length(nor_d) == 1
            nor_d = [1,nor_d];
        end
        nor_ds = nor_d(1==(mod(1:1:length(nor_d),2)));
        nor_de = nor_d(0==(mod(1:1:length(nor_d),2)));
        nor_k = (nor_de-nor_ds)>100;
        nor_ds = nor_ds(nor_k)+start;
        nor_de = nor_de(nor_k)+start;
        nor_k = [];
        
        for i = 1:length(nor_ds)
            % fixing indeces to start and end with static finger contact
            while lat2(nor_ds(i))*lat2(nor_ds(i)+1)>0
                nor_ds(i) = nor_ds(i) + 1;
            end
            while lat2(nor_de(i))*lat2(nor_de(i)-1)>0
                nor_de(i) = nor_de(i) - 1;
            end
            
            % removing sporatic contact points            
            start = nor_ds(i);
            pp = mu(nor_ds(i):nor_de(i));
            pp = abs(diff(pp>0)'.*(1:1:length(pp)-1));
            pp = pp(pp~=0);
            pp1 = diff(pp);
            pp1 = pp1>50;
            
            if ~isempty(pp1)
                nor_ds(i) = min(pp)+start-1;
                nor_de(i) = max(pp)+start;
                nor_k(i) = 1;
            else
                nor_k(i) = 0;
            end
        end
        
        nor_k = nor_k == 1;
        nor_ds = nor_ds(nor_k);
        nor_de = nor_de(nor_k);

        for i = 1:length(nor_ds)
            times{s}(trial1,i,1:2) = [nor_ds(i),nor_de(i)];
        end
        swipe_n{s}(trial1) = length(nor_ds);
        trial1 = trial1 + 1;
        start = endds + 1;
    end
    
    % extracting wipe information
    for t = 1:trial1-1
        mu_data{s,t,1} = [];
        mu_data{s,t,2} = [];
        mu_data{s,t,3} = [];
        mu_data{s,t,4} = [];
        
        for n = 1:swipe_n{s}(t)
            start = times{s}(t,n,1) - 5;
              endd = times{s}(t,n,2);
            mur = (abs(mu(start:endd))>5).*(1:1:(endd-start+1))';
            mur = mur(1:round(length(mur)/2));
            mmur = max(mur(mur~=0));
            if ~isempty(mmur)
                start = mmur + start + 1;
            end          
            
            % finding current start and end times

            mulo = muo(start:endd);
            mul = mu(start:endd);
            mul2 = mu2(start:endd);
            latt = lat(start:endd);
            latt2 = lat2(start:endd);
            dlatt = dlat(start:endd);
            ddlatt = ddlat(start:endd);
            dmul = dmu(start:endd);
            norr = nor(start:endd);
            
                        % extracting transition swipe direction switch times
            time_vec = abs(diff(mul2>0));
            time_vec = [time_vec;time_vec(end)]';
            time_vec = time_vec.*(1:1:length(mul2));
            time_vec = time_vec(time_vec~=0);
            
            % removing data that is suspect
            time_vect = time_vec;
            for tv = 2:length(time_vec)
                if abs(mean(mul2(time_vec(tv-1):time_vec(tv))))<.1
                   time_vect(tv) = 0;
                end
            end
            time_vec = time_vec(time_vect~=0);
            
            time_vec = [time_vec(diff(time_vec)>50),time_vec(end)];            
            time_vec1 = time_vec;
            time_vec = [1,time_vec(1:end-1)+round(diff(time_vec)/2)];
            time_vec2 = diff(time_vec1);
            
           
            % extracting perceptual1 info
            % positive swipe info contains information about swiping from
            % right to left, so negative to positive force
           
            for ll = 1:length(time_vec1)-1
                qq = time_vec1(ll);
                if length(latt)>qq+100
                                        
                    ss = sign(mean(mul(qq:qq+100)));
                    ww = qq;
                    while (dmul(ww)*dmul(ww+1))>0
                        ww = ww + 1;
                    end
                    mu_data{s,t,1}(ll) = latt(ww);
                    ee = ww + 1;
                    while ee<length(latt) && (dmul(ee)*dmul(ee+1)>0)
                        ee = ee + 1;
                    end
                    mu_data{s,t,2}(ll) = latt(ww) - latt(ee);
                    mu_data{s,t,3}(ll) = (ee-ww+1)/1000;
                    
                    rr = ww;
                    while (rr>1)&&(ss*dmul(rr))<(ss*dmul(rr-1))
                        rr = rr - 1;
                    end
                    mu_data{s,t,4}(ll) = (ww - rr)/1000;
                    mu_data{s,t,5}(ll) = norr(ww);
                    mu_data{s,t,6}(ll) = mulo(ww);
                    mu_data{s,t,7}(ll) = max(dmul(ww:ee));
                else
                    break;
                end
            end
                
            
            time_data{s,t,n,1} = sum(squeeze(times{s}(t,n,2))-squeeze(times{s}(t,n,1)))/1000;
            cc = cur(round(time_vector_mod(t)):round(time_vector_mod(t+1)));
            if max(cc) > - 4.8
                cc = (cc>-4.8).*cc;
                cc = cc(cc~=0);
            end
            cc = histcounts(cc,vec);
            [~,cc] = max(cc);
            
            time_data{s,t,n,2} = currents(cc);
            time_data{s,t,n,3} = time_vec2(1:end-1);
        end
    end
    trial1s(s) = trial1-1;
    res{s} = results(gg(2:end)-1);
    sti{s} = stimuli((randloc(gg(2:end)-1)));
    sti{s} = sti{s}/4095*8;
    GG{s} = gg(2:end)-1;
end

%% reformatting
mus = {}; MUS = [];
lat = {}; LAT = [];
dlat = {}; DLAT = [];
dlatdt = {}; DLATDT = [];
tps = {}; TPS = [];
tts = {}; TTS = [];
jud = {}; JUD = [];
cur = {}; CUR = [];
nor = {}; NOR = [];

for s = 1:10
    y = 1;
   	for t = 1:40
        if ~isempty(mu_data{s,t,1}) && median(mu_data{s,t,5})~=0
            lat{s}(y) = median(abs(mu_data{s,t,1}));
            dlat{s}(y) = median(abs(mu_data{s,t,2}));
            %dlatdt{s}(y) = median(abs(mu_data{s,t,2})./mu_data{s,t,3});
            dlatdt{s}(y) = mean(abs(mu_data{s,t,7})); 
            tps{s}(y) = median(mu_data{s,t,4});
            tts{s}(y) = median(mu_data{s,t,3});
            jud{s}(y) = res{s}(t);
            cur{s}(y) = sti{s}(t).^2;
            nor{s}(y) = median(mu_data{s,t,5});
            mus{s}(y) = median(abs(mu_data{s,t,1})).*(median(abs(mu_data{s,t,5})).^v1(s));
            y = y + 1;
        end
    end
    
    lat{s} = (lat{s} - mean(lat{s}))/std(lat{s});
    dlat{s} = (dlat{s} - mean(dlat{s}))/std(dlat{s});
    dlatdt{s} = (dlatdt{s} - mean(dlatdt{s}))/std(dlatdt{s});
    tps{s} = (tps{s} - mean(tps{s}))/std(tps{s});
    tts{s} = (tts{s} - mean(tts{s}))/std(tts{s});
    jud{s} = (jud{s} - mean(jud{s}))/std(jud{s});
    cur{s} = (cur{s} - mean(cur{s}))/std(cur{s});
    nor{s} = (nor{s} - mean(nor{s}))/std(nor{s});
    mus{s} = (mus{s} - mean(mus{s}))/std(mus{s});
    
%     LAT = [LAT,(lat{s} - mean(lat{s}))/std(lat{s})];
%     DLAT = [DLAT,(dlat{s} - mean(dlat{s}))/std(dlat{s})];
%     DLATDT = [DLATDT,(dlatdt{s} - mean(dlatdt{s}))/std(dlatdt{s})];
%     TPS = [TPS,tps{s}];
%     TTS = [TTS, (tts{s} - mean(tts{s}))/std(tts{s})];
%     JUD = [JUD,(jud{s} - mean(jud{s}))/std(jud{s})];
%     CUR = [CUR,(cur{s} - mean(cur{s}))/std(cur{s})];
%     MUS = [MUS,(mus{s} - mean(mus{s}))/std(mus{s})];
%     NOR = [NOR,(nor{s} - mean(nor{s}))/std(nor{s})];
end

JUDt = [];
LATt = [];
DLATt = [];
DLATDTt = [];
TPSt = [];
TTSt = [];
CURt = [];
MUSt = [];
NORt = [];

for c = [0,2,3,4,5,6,7,8]
    out = ~isoutlier(JUD(CUR==c),'quartiles','ThresholdFactor',1.5);
    JUDt = [JUDt,JUD(out)];
    LATt = [LATt,LAT(out)];
    DLATt = [DLATt,DLAT(out)];
    DLATDTt = [DLATDTt,DLATDT(out)];
    TPSt = [TPSt,TPS(out)];
    TTSt = [TTSt,TTS(out)];
    CURt = [CURt,CUR(out)];
    MUSt = [MUSt,MUS(out)];
    NORt = [NORt,NOR(out)];
end

    
%% plotting
B = [];
for s = 1:10
B(s,1,:) = mvregress([(lat{s}.'-mean(lat{s}.'))/std(lat{s}.'),(dlat{s}.' - mean(dlat{s}.'))/std(dlat{s}.'),(dlatdt{s}.' - mean(dlatdt{s}.'))/std(dlatdt{s}.')],(cur{s}.^2.' - mean(cur{s}.^2.'))/std(cur{s}.^2.'));
B(s,2,:) = mvregress([(lat{s}.'-mean(lat{s}.'))/std(lat{s}.'),(dlat{s}.' - mean(dlat{s}.'))/std(dlat{s}.'),(dlatdt{s}.' - mean(dlatdt{s}.'))/std(dlatdt{s}.')],(jud{s}.' - mean(jud{s}.'))/std(jud{s}.'));
end

hold on
b = mvregress([LATt;DLATDTt].',JUDt.');
a = plot(b,'.');
a.Color = [1 0 0];
a.MarkerSize = 20;
a = plot(b,'-');
a.Color = [1 0 0];
b = mvregress([LATt;DLATDTt].',(CURt.'.^2 - mean(CURt.^2))/std(CURt.^2));
a = plot(b,'.');
a.Color = [0 0 0];
a.MarkerSize = 20;
a = plot(b,'-');
a.Color = [0 0 0];
set(gcf,'color','w');
xticklabels({'F','dF/dt'});
xticks([1 2 3]);
ylabel('coefficient val1ue');


%% perfmorming pca
kv = 1:5;
sv = 1:10;
pv = 1:2;

MAT = [];

for s = sv
    for k1 = kv
        for k2 = kv
            DM = [jud{s};lat{s};nor{s};mus{s};dlat{s};tps{s}];
            DM = DM(kv(((kv~=k1).*(kv~=k2))==1),:);
            [v,e,j] = pca(DM);
            for p = pv
                if p == 1
                    MAT(s,k1,k2,p) = corr(v(:,p),jud{s}.').^2;
                else
                    MAT(s,k1,k2,p) = MAT(s,k1,k2,p-1) + corr(v(:,p),jud{s}.').^2;
                end
            end
        end
    end
end

%% finding best componenets
MAT = [];
J = zeros(10,3);
for s = 1:10
    DM = [jud{s};lat{s};nor{s};mus{s};dlat{s};tps{s}].';  
    [ve,tra,va] = pca(DM); 
    J(s,1:3) = va(1:3)/sum(va);
    for v = 1:6
        for d = 1:4
            MAT(s,v,d) = corr(DM(:,v),tra(:,d)).^2;
        end
    end
end

%% plotting
figure
a = subplot (3,1,1);
set(gca,'TickLabelInterpreter','latex','FontSize',10);
hold on
a.XTick = [];
a.YTick = [0 1];
bar(MAT(:,:,1))
axis([.5 10.5 0 1]);
for s = 1:10
    text(s-.2,1.05,num2str(J(s,1),2),'Interpreter','latex','FontSize',10);
    a = plot([s+.5,s+.5],[0,1],':');
    a.Color = [0 0 0];
end

a = subplot(3,1,2);
set(gca,'TickLabelInterpreter','latex','FontSize',10);
hold on
a.XTick = [];
a.YTick = [0 1];
bar(MAT(:,:,2))
axis([.5 10.5 0 1]);
for s = 1:10
    text(s-.33,1.05,strcat(num2str(J(s,2),2),'(',num2str(sum(J(s,1:2)),2),')'),'Interpreter','latex','FontSize',10);
    a = plot([s+.5,s+.5],[0,1],':');
    a.Color = [0 0 0];
end
ylabel('$R^2$','Interpreter','latex','FontSize',10);


a = subplot(3,1,3);
set(gca,'TickLabelInterpreter','latex','FontSize',10);
hold on
a.XTick = [];
a.YTick = [0 1];
bar(MAT(:,:,3))
axis([.5 10.5 0 1]);
for s = 1:10
    text(s-.33,1.05,strcat(num2str(J(s,3),2),'(',num2str(sum(J(s,1:3)),2),')'),'Interpreter','latex','FontSize',10);
    a = plot([s+.5,s+.5],[0,1],':');
    a.Color = [0 0 0];
end

set(gcf,'color','w');