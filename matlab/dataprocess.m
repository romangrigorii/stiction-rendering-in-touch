%% processes experiment data
clear
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
[bn,an] = butter(1,2*25/1000); % filtering normal force data

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
    lat = filtfilt(bl1,al1,data(:,1));
    lat = lat - mean(lat(1:2000,1));
    lat2 = filtfilt(bl2,al2,data(:,1));
    lat2 = lat2 - mean(lat2(1:2000,1));
    nor = filtfilt(bn,an,data(:,2));
    nor = nor - mean(nor(1:2000,1));
    cur = data(:,3);
    lat = lat/Lc;
    nor = nor/Nc;
    mu = lat./nor;
    mu2 = lat2./nor;
    
    dmu = derivR(mu,1,1000);
    dmu(isnan(dmu)) = 0;
    dmu(isinf(abs(dmu))) = 0;
    dmu = filtfilt(bl2,al2,dmu);
    
    ddmu = derivR(mu,2,1000);
    ddmu(isnan(ddmu)) = 0;
    ddmu(isinf(abs(ddmu))) = 0;
    ddmu = filtfilt(bl2,al2,ddmu);
    
    dlat = derivR(lat,1,1000);
    dlat(isnan(dlat)) = 0;
    dlat(isinf(abs(dlat))) = 0;
    dlat = filtfilt(bl2,al2,dlat);
    
    ddlat = derivR(lat,2,1000);
    ddlat(isnan(ddlat)) = 0;
    ddlat(isinf(abs(ddlat))) = 0;
    ddlat = filtfilt(bl2,al2,ddlat);
    
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
    % extracting force information trial by trial
    
    trial = 1;
    while trial < length(time_vector)
        start = round(time_vector(trial)) + time_fix{s}(trial);
        endd = round(time_vector(trial+1)) - time_fix{s}(trial+1);
        endds = endd;
        
        % dividng up the trial into sections where the subject was in contact
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
            times{s}(trial,i,1:2) = [nor_ds(i),nor_de(i)];
        end
        swipe_n{s}(trial) = length(nor_ds);
        trial = trial + 1;
        start = endds + 1;
    end
    
    % extracting wipe information
    curr = cur>-4.8;
    for t = 1:trial-1
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
            g = start;
            while curr(g) == 1
                g = g + 1;
            end
            cur_s = [start];
            cur_e = [g];
            while g<endd
                while g<=endd && curr(g) == 0
                    g = g + 1;
                end
                e = g;
                while e<=endd && curr(e) == 1
                    e = e + 1;
                end
                cur_s = [cur_s,g];
                cur_e = [cur_e,e];
                g = e;
            end
            
            % working with force profile
            mul = mu(start:endd);
            mul2 = mu2(start:endd);
            latt = lat(start:endd);
            latt2 = lat2(start:endd);
            dlatt = dlat(start:endd);
            dmul = dmu(start:endd);
            ddmul = ddmu(start:endd);
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
            
            % exacting first derivative values of mu and lat
            dmp = zeros(length(time_vec)-1,2); dmn = dmp; dln = dmp; dlp = dmp;
            v = [];
            
            sig = dmul(time_vec(1):time_vec(2));
            if max(abs(sig))>100
                time_vec = time_vec(2:end);
            end
                        
            for i = 1:(length(time_vec)-1)
                sig = dmul(time_vec(i):time_vec(i+1));
                
                if sum(sig)>0
                    [~,sl] = max(sig.*hann(length(sig)));
                    v(i) = 1;
                else
                    [~,sl] = min(sig.*hann(length(sig)));
                    v(i) = 0;
                end
                
                slp = sl; sln = sl;
                while (sig(slp)*sig(slp+1))>0 && slp<length(sig)-1
                    slp = slp+1;
                end
                while (sig(sln)*sig(sln-1))>0 && sln>2
                    sln = sln-1;
                end
                
                subsig = sig(sln:slp);
                asubsig = abs(subsig);
                
                if sum(subsig)>0
                    dmp(i,1) = max(asubsig);
                    dmp(i,2) = mean(asubsig(asubsig>(.9*dmp(i,1))));
                else
                    dmn(i,1) = max(asubsig);
                    dmn(i,2) = mean(asubsig(asubsig>(.9*dmn(i,1))));
                end
                
                sig = dlatt(time_vec(i):time_vec(i+1));
                subsig = sig(sln:slp);
                asubsig = abs(subsig);
                
                if sum(subsig)>0
                    dlp(i,1) = max(asubsig);
                    dlp(i,2) = mean(asubsig(asubsig>(.9*dlp(i,1))));
                else
                    dln(i,1) = max(asubsig);
                    dln(i,2) = mean(asubsig(asubsig>(.9*dln(i,1))));
                end
            end
            
            dmp = dmp(dmp(:,1)~=0,:);
            dmn = dmn(dmn(:,1)~=0,:);
            dlp = dlp(dlp(:,1)~=0,:);
            dln = dln(dln(:,1)~=0,:);
            
            [sl,~] = size(dmp);
            for i = 1:sl
                mu_data{s,t,n,i,7} = dmp(i,:);
            end
            [sl,~] = size(dmn);
            for i = 1:sl
                mu_data{s,t,n,i,9} = dmn(i,:);
            end
            [sl,~] = size(dln);
            for i = 1:sl
                mu_data{s,t,n,i,10} = dln(i,:);
            end
            [sl,~] = size(dlp);
            for i = 1:sl
                mu_data{s,t,n,i,8} = dlp(i,:);
            end
            
            ps = 0;
            ns = 0;
            
            for sw = 1:length(v)
                st = time_vec1(sw);
                e = time_vec1(sw+1);
                di = round((e - st)/4);
                
                m = 1;
                while ~((dmul(m+st)*dmul(m+st+1))<0 && abs(mul(m+st+1))>.1)
                    m = m + 1;
                end
                m = m + 1;
                if (m+st-10)<=0
                    m =10 - st + 1;
                end
                kk = 0;                
                if mul(st+m) > 0
                    if kk == 0
                        dir_data{s,t} = 1;
                        kk = 1;
                    end
                    ps = ps + 1;
                    [~,ma] = max(mul(st+m-10:st+m+25));
                    qq = st+m-10+ma-1 + 20;
%                     while qq<length(latt) && abs(latt(qq))>abs(latt(qq+1))
%                         qq = qq + 1;
%                     end
%                     if qq >= length(latt)
%                         qq = st+m-10+ma-1 + 20;
%                     end
                    mu_data{s,t,n,ps,1} = mul(st+m-10+ma-1) - mul(qq);
                    mu_data{s,t,n,ps,2} = latt(st+m-10+ma-1);
                    mu_data{s,t,n,ps,3} = norr(st+m-10+ma-1);
                else
                    if kk == 0
                        dir_data{s,t} = 0;
                        kk = 1;
                    end
                    ns = ns + 1;
                    [~,ma] = min(mul(st+m-10:st+m+25));
                    qq = st+m-10+ma-1 + 20;
%                     while qq<length(latt) && abs(latt(qq))>abs(latt(qq+1))
%                         qq = qq + 1;
%                     end
%                     if qq >= length(latt)
%                         qq = st+m-10+ma-1 + 20;
%                     end
                    mu_data{s,t,n,ns,4} = mul(st+m-10+ma-1)-mul(qq);
                    mu_data{s,t,n,ns,5} = latt(st+m-10+ma-1);
                    mu_data{s,t,n,ns,6} = norr(st+m-10+ma-1);
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
            time_data{s,t,n,3} = [ps,ns];
            time_data{s,t,n,4} = time_vec2(1:end-1);
        end
    end
    trials(s) = trial-1;
    res{s} = results(gg(2:end)-1);
    sti{s} = stimuli((randloc(gg(2:end)-1)));
    sti{s} = sti{s}/4095*8;
    GG{s} = gg(2:end)-1;
end

%% reformatting
mus = {};
lat = {};
muk = {};
nor = {};
cur = {};
swipe_times = {};

for s = 1:10
    swipe_times{s} = [];
    for t = 1:trials(s)
        T = GG{s}(t);
        must = [];
        mukt = [];
        latt = [];
        nort = [];
        curt = [];
        st = 0;
        swipen = zeros(1,3);
        for n = 1:swipe_n{s}(t)
            ii = 1;
            for i = 1:time_data{s,t,n,3}(1)
                if length(mu_data{s,t,n,i,2})>0
                    must(n,ii,1) = mu_data{s,t,n,i,1};
                    latt(n,ii,1) = mu_data{s,t,n,i,2};
                    nort(n,ii,1) = mu_data{s,t,n,i,3};
                    ii = ii + 1;
                end
            end
            ii = 1;            
            for i = 1:time_data{s,t,n,3}(2)
                if length(mu_data{s,t,n,i,5})>0
                    must(n,ii,2) = mu_data{s,t,n,i,4};
                    latt(n,ii,2) = mu_data{s,t,n,i,5};
                    nort(n,ii,2) = mu_data{s,t,n,i,6};
                    ii = ii + 1;
                end
            end
            swipen = swipen + [time_data{s,t,n,3},time_data{s,t,n,1}];
        end
        for d = 1:2
            if time_data{s,t,1,3}(d)>0
                temp = squeeze(abs(must(1,:,d)));
                temp = temp(temp>=.01);
                mus{s,t,d} = temp;
                temp = squeeze(abs(latt(1,:,d)));
                temp = temp(temp>=.01);
                lat{s,t,d} = temp;
                temp = squeeze(abs(nort(1,:,d))); %% *** temp = squeeze(abs(nort(:,:,d)));
                temp = temp(temp>=.01);
                nor{s,t,d} = temp;
            end
        end
        swipe_times{s,t} = swipen;
    end
end

%clearvars -except mus lat nor cur swipe_times sti res trials time_data

%% STATS
% normalizing judgment
for i = 1:10
    res{i} = res{i}/mean(res{i});
end
% findings the outliers in judgment

stims = [0,2,3,4,5,6,7,8]; % stimuli numbers
allres = {};
mapping = {};
outliers = {};
P = ones(8,1);
for i = 1:10
    for s = 1:8
        if i == 1
            allres{s} = [];
        end
        r = res{i}(sti{i}==stims(s));
        allres{s} = [allres{s},r'];
        for p = 1:length(r)
            mapping{s,P(s)} = [i,p];
            P(s) = P(s) + 1;
        end
    end
end
P = P - 1;

for s = 1:8
    outliers{s} = ~isoutlier(allres{s},'quartiles','ThresholdFactor',1.5);
end

% removing outliers from data

fres = {};
fsti = {};
fmus = {};
flat = {};
fnor = {};
ftim = {};
fnum = {};
fswipe_times = {};
for i = 1:10
    fres{i} = [];
    fsti{i} = [];
    fmus{i} = [];
    flat{i} = [];
    fnor{i} = [];   
    ftim{i} = [];
    fnum{i} = [];
    fswipe_times{i} = [];
    for t = 1:trials(i)
        fmus{i,t,1} = [];
        fmus{i,t,2} = [];
    end
end

counter = zeros(1,10);

for s = 1:8
    for ii = 1:P(s)
        ip = mapping{s,ii};
        i = ip(1); p = ip(2);
        loc = sti{i}==stims(s);
        loc = loc.*(1:1:length(loc))';
        loc = loc(loc~=0);
        rr = res{i}(sti{i}==stims(s));
        ss = sti{i}(sti{i}==stims(s));
        if outliers{s}(ii)
            counter(i) = counter(i) + 1;
            fmus{i,counter(i),1} = mus{i,loc(p),1};
            fmus{i,counter(i),2} = mus{i,loc(p),2};
            flat{i,counter(i),1} = lat{i,loc(p),1};
            flat{i,counter(i),2} = lat{i,loc(p),2};
            fnor{i,counter(i),1} = nor{i,loc(p),1};
            fnor{i,counter(i),2} = nor{i,loc(p),2};
            fswipe_times{i,counter(i)} = swipe_times{i,loc(p)};
            fres{i}(counter(i)) = res{i}(loc(p));
            fsti{i}(counter(i)) = sti{i}(loc(p));
        end
    end
end

% finding the mean judgment that excludes outliers
RES = [];
STI = [];
for i = 1:10
    RES = [RES,fres{i}];
    STI = [STI,fsti{i}];
end
mm = [];
st = [];

for i = 1:8
    mm(i) = mean(RES(STI==stims(i)));
end


%% plot of stiction judgment as function of applied current (version 1)
j = figure;
j.Position = [j.Position(1:2),[500,400]];
col = jet;
shaps = {'^','s','v','p','x','<','>','d','o','*'};
hold on
a = plot(stims,mm,'--');
a.Color = [0 0 0];
a.LineWidth = 1;
s = 1;
for i = 1:10
    a = plot(mean(fsti{i}(fsti{i}==stims(s)))+(i - 5.5)/15,mean(fres{i}(fsti{i}==stims(s))),shaps{i});
    a.Color = [col(round(63/11*(i-1)+1),:),0];
    a.LineWidth = 1.1;
    a.MarkerSize = 5;
    if ii>3
        a.MarkerSize = 6;
    end
end

[ha,hb,hc,hd] = legend('mean','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10');
hb(12).XData = [hb(12).XData(1) + .2,hb(12).XData(2) - .2];
ha.FontSize = 7;
ha.Location = 'northwest';

for i = 1:10
    a = plot(fsti{i}+(i - 5.5)/15,fres{i},'.');
    a.Color = [.8 .8 .8];
    a.MarkerSize = 6;
end

hold on
temp = [];
for i = 1:10
    for s = 1:8
        a = plot(mean(fsti{i}(fsti{i}==stims(s)))+(i - 5.5)/15,mean(fres{i}(fsti{i}==stims(s))),shaps{i});
        a.Color = [col(round(63/11*(i-1)+1),:),0];
        a.LineWidth = 1.1;
        a.MarkerSize = 5;
        if ii>3
            a.MarkerSize = 6;
        end
        temp(i,s,1:2) = [mean(fsti{i}(fsti{i}==stims(s)))+(i - 5.5)/15,mean(fres{i}(fsti{i}==stims(s)))];
    end
end

set(gcf,'color','w');
xlabel('applied current (mApp)','Interpreter','latex','FontSize',11)
ylabel('normalized stickiness judgment','Interpreter','latex','FontSize',11)
axis([-1 9 -.25 3.5])
text(1.9,2.45,'r = .79','Interpreter','latex','FontSize',11)
text(1.9,2.15,'$\rho$ = .83','Interpreter','latex','FontSize',11)
%text(2.5,3.15,'p $\ll$ $10^{-9}$','Interpreter','latex','FontSize',11)
xticks([0,2,3,4,5,6,7,8])
ha.EdgeColor = [0 0 0];
ha.Position(3) = ha.Position(3) + .02;
ha.Position(1) = ha.Position(1) + .01;
ha.Position(4) = ha.Position(4) + .02;

%% plot of judgment and current as function of stiction values
%% finding R^2 vs n exponent

%% reformatting mus and sti results
co = 1:10;
co2 = 1:10;
hh = 1;
resnew = {};
mnew = {};
M = [];
T = [];
N = [];
mus2 = {};
lat2 = {};
nor2 = {};
fsti2 = {};
fres2 = {};
for i = 1:10
    M = [];
    for ii = 1:2
        for t = 1:counter(i)            
            m = fmus{i,t,ii};
            m = m(~isnan(m));  
            M(ii,t) = median(m);            
        end        
    end
    cord = ~boolean(sum(isnan(M)));
    M = M(:,cord);
    for ii = 1:2
        mus2{i,ii} = M(ii,:);
    end
    fsti2{i} = fsti{i}(cord);
    fres2{i} = fres{i}(cord);
    co(i) = corr(mean(M)',fsti2{i}');
    co2(i) = corr(mean(M)',fres2{i}');
end
[~,cor] = sort(co);

%% mus vs judgment vs current
a = figure;
a.Position = [a.Position(1:2),[1500,300]];
col = jet;
hold on
fun = @(x,xdata)x(1)*xdata + x(2);
shaps = {'s','o'};
P = zeros(2,2,10);
tit = {};
Mm = [];
Ss = [];
for i = 1:10
    M = [];
    for ii = 1:2
        M(ii,1:length(mus2{cor(i),ii})) = mus2{cor(i),ii};
        subplot(2,10,i);
        hold on
        S = fsti2{cor(i)};
        a = plot(M(ii,:),S-.3 + ii*.3,'.');
        if ii == 1
            a.Color = [0 0 1];
        else
            a.Color = [1 0 0];
        end
        a.MarkerSize = 8;
        for s = 1:8
            Mm(s) = mean(M(ii,S==stims(s)));
            Ss(s) = std(M(ii,S==stims(s)));
        end
        a = plot(Mm,stims-.3+ii*.3);
        if ii == 1
            a.Color = [0 0 1 .4];
        else
            a.Color = [1 0 0 .4];
        end
        a.LineWidth = 1.1;
        if cor(i) == 4
            axis([0 4.3 -.5 9]);
            xticks([0,1.0,2.0,3.0,4.0]);
        else
            axis([0 3 -.5 9]);
            xticks([0,1,2]);
        end
        yticks([0 2 4 6 8]);
        set(gca,'FontSize',9);
    end
    [hh,jj,~] = corrcoef(mean(M),S);
    P(1,i) = jj(1,2);
    P(2,i) = hh(1,2);
    if P(1,ii,i)<.001
        tit = strcat('r =',{' '},num2str(hh(1,2),' %0.2f'));%,' p $<$ .001');
    else
        tit = strcat('r =',{' '},num2str(hh(1,2),' %0.2f'));%,'  p =',{' '},num2str(jj(1,2),'%0.3f'));
    end
    if cor(i) == 4
        text(2.5,.5,tit{1},'Interpreter','latex','FontSize',9)
    else
        text(1.5,.5,tit{1},'Interpreter','latex','FontSize',9)
    end
end


set(gcf,'color','w');
subplot(2,10,1);
ylabel('applied current (mApp)','Interpreter','latex','FontSize',9);

fun = @(x,xdata)x(1)*xdata + x(2);
for i = 1:10
    M = [];
    for ii = 1:2
        M(ii,1:length(mus2{cor(i),ii})) = mus2{cor(i),ii};
        R = fres2{cor(i)}/max(fres2{cor(i)});        
        subplot(2,10,10+i);
        hold on
        a = plot(M(ii,:),R - .05 + ii*.05,'.');
        if ii == 1
            a.Color = [0 0 1];
        else
            a.Color = [1 0 0];
        end
        a.MarkerSize = 8;        
        v = polyfit(M(ii,:),R- .05 + ii*.05,1);
        
        a = plot([.1,5],[.1,5]*v(1)+v(2));
        
        if ii == 1
            a.Color = [0 0 1 .4];
        else
            a.Color = [1 0 0 .4];
        end
        
        a.LineWidth = 1.1;
        if cor(i) == 4
            axis([0 4.3 -.1 1.1]);
            xticks([0,1.0,2.0,3.0,4.0]);
        else
            axis([0 3 -.1 1.1]);
            xticks([0,1,2]);
        end
        yticks([0.0 1]);
        set(gca,'FontSize',9);
    end
    [hh,jj,~] = corrcoef(mean(M),R);
    P(1,i) = jj(1,2);
    P(2,i) = hh(1,2);
    if P(1,i)<.001
        tit = strcat('r =',{' '},num2str(hh(1,2),' %0.2f'));%,' p $<$ .001');
    else
        tit = strcat('r =',{' '},num2str(hh(1,2),' %0.2f'));%,'  p =',{' '},num2str(jj(1,2),'%0.3f'));
    end
    if cor(i) == 4
        text(2.5,0,tit{1},'Interpreter','latex','FontSize',9)
    else
        text(1.5,0,tit{1},'Interpreter','latex','FontSize',9)
    end
end

subplot(2,10,11);
ylabel('normalized judgement','Interpreter','latex','FontSize',9);
subplot(2,10,16);
xlabel('static coefficient of friction ($\mu_s$) ','Interpreter','latex','FontSize',9);


%% finding time spent per trial information
hertz = [];
nu = [];
ti = [];
for i = 1:10
    temp = [];
    for t = 1:counter(i)
        temp = [temp;fswipe_times{i,t}];
    end
    ti = [ti, temp(:,3)'];
    nu = [nu,mean(temp(:,1:2)')];
end
hertz = nu./ti;

%% checking if judgement performance changed over time

tp = [];
ii = 1;
data = {};
fun = @(x,xdata)x(1)*xdata + x(2);
for i = 1:10
    s = fsti2{i};
    r = fres2{i};
    o = lsqcurvefit(fun,[0,0],s,r);
    y = s*o(1) + o(2);
    e = (r - y).^2;
    l = round(length(e)/2);
    data{1} = e(1:l);
    data{2} = e(l+1:end);
    tp(i,1:2) = unpairedttest(data);
end

%% finding best N and F exponents
R2 = [];
LN = [];
for i = 1:10
    for L = 1:301
        for N = 1:301
            LN = [];
            for ii = 1:2
                for t = 1:counter(i)
                    l = flat{i,t,ii};
                    n = fnor{i,t,ii};
                    ln = (l.^((L-201)/100)).*(n.^((N-201)/100));
                    LN(ii,t) = mean(ln);
                end
            end
            cord = isnan(LN(1,:)).*(1:length(LN));
            cord = cord(cord~=0);
            LN(1,cord) = LN(2,cord);
            cord = isnan(LN(2,:)).*(1:length(LN));
            cord = cord(cord~=0);
            LN(2,cord) = LN(1,cord);        
            R2(i,L,N,1) = corr(mean(LN)',fres{i}','Type','Pearson');
            R2(i,L,N,2) = corr(mean(LN)',fsti{i}','Type','Pearson');  
        end
    end
    i
end

% plotting
x = linspace(-2,1,301);
[~,v] = max(squeeze((mean(squeeze(R2(:,:,:,1)),1)))');
v = v(210:end);
m = polyfit(1:92,v,1);

imagesc(x,x,squeeze((mean(squeeze(R2(:,:,:,1)),1))))
hold on
set(gca,'YDir','normal')
set(gcf,'color','w');
title('current vs $F^{d}N^{k}$','Interpreter','latex','FontSize',10);
title('current vs $F^{d}N^{k}$','Interpreter','latex','FontSize',15);
title('current vs. $F^{d}N^{k}$','Interpreter','latex','FontSize',15);
xlabel('k')
ylabel('d')
title('judgment vs. $F^{d}N^{k}$','Interpreter','latex','FontSize',15);
hold on
colorbar
text(-1,0,'y = -1.39x','Interpreter','latex','FontSize',15)
plot(x,(1/m(1))*x)

%% building conditional matrix from all data
CAn = {};
CAl = {};
CAnn = {};
dirdata = {};

CAnl = {};
CAll = {};
CAnr = {};
CAlr = {};
CAnnr = {};
CAnnl = {};

for i = 1:10
    CAn{i} = [];
    CAl{i} = [];
    CAnn{i} = [];
    dirdata{i} = [];
    
    CAnl{i} = [];
    CAll{i} = [];
    CAnr{i} = [];
    CAlr{i} = [];
    CAnnr{i} = [];
    CAnnl{i} = [];
    for t = 1:counter(i)
        if dir_data{i,t}==1 % if the swipe is rightward first
            q = 1;
            while (length(fnor{i,t,1})>=(q+1))&&(length(fnor{i,t,2})>=(q+1))   
                dirdata{i} = [dirdata{i},1];
                dirdata{i} = [dirdata{i},-1];
                CAl{i} = [CAl{i},flat{i,t,1}(q)];
                CAl{i} = [CAl{i},flat{i,t,2}(q)];
                CAnn{i} = [CAnn{i},1 - fnor{i,t,1}(q)/(fnor{i,t,2}(q)+fnor{i,t,1}(q))];
                CAnn{i} = [CAnn{i},1 - fnor{i,t,2}(q)/(fnor{i,t,1}(q+1)+fnor{i,t,2}(q))];
                CAn{i} = [CAn{i},fnor{i,t,1}(q)];
                CAn{i} = [CAn{i},fnor{i,t,2}(q)];
                
                CAll{i} = [CAll{i},flat{i,t,1}(q)];
                CAlr{i} = [CAlr{i},flat{i,t,2}(q)];
                CAnl{i} = [CAnl{i},fnor{i,t,1}(q)];
                CAnr{i} = [CAnr{i},fnor{i,t,2}(q)];
                CAnnl{i} = [CAnnl{i},fnor{i,t,1}(q+1) - fnor{i,t,1}(q)];
                CAnnr{i} = [CAnnr{i},fnor{i,t,2}(q+1) - fnor{i,t,2}(q)];
                q = q + 1;
            end
        else
            q = 1;
            while (length(fnor{i,t,1})>=(q+1))&&(length(fnor{i,t,2})>=(q+1))
                dirdata{i} = [dirdata{i},-1];
                dirdata{i} = [dirdata{i},1];
                CAl{i} = [CAl{i},flat{i,t,2}(q)];
                CAl{i} = [CAl{i},flat{i,t,1}(q)];
                CAnn{i} = [CAnn{i},1 - fnor{i,t,2}(q)/(fnor{i,t,1}(q)+fnor{i,t,1}(q))];
                CAnn{i} = [CAnn{i},1 - fnor{i,t,1}(q)/(fnor{i,t,2}(q+1)+fnor{i,t,1}(q))];
                CAn{i} = [CAn{i},fnor{i,t,2}(q)];
                CAn{i} = [CAn{i},fnor{i,t,1}(q)];
                
                CAll{i} = [CAll{i},flat{i,t,1}(q)];
                CAlr{i} = [CAlr{i},flat{i,t,2}(q)];
                CAnl{i} = [CAnl{i},fnor{i,t,1}(q)];
                CAnr{i} = [CAnr{i},fnor{i,t,2}(q)];
                CAnnl{i} = [CAnnl{i},fnor{i,t,1}(q+1) - fnor{i,t,1}(q)];   
                CAnnr{i} = [CAnnr{i},fnor{i,t,2}(q+1) - fnor{i,t,2}(q)];
                q = q + 1;
            end
        end
    end
end

%% making across subject correlation 
CAna = [];
CAnna = [];
CAla = [];
dird = [];

CAnla = [];
CAnra = [];
CAlla = [];
CAlra = [];
CAnnla = [];
CAnnra = [];

for i = 1:10
    CAna = [CAna,(CAn{i} - mean(CAn{i}))/std(CAn{i})];
    CAla = [CAla,(CAl{i} - mean(CAl{i}))/std(CAl{i})];
    CAnna = [CAnna,(CAnn{i} - mean(CAnn{i}))/std(CAnn{i})];
    CAnla = [CAnla,(CAnl{i} - mean(CAnl{i}))/std(CAnl{i})];
    CAnra = [CAnra,(CAnr{i} - mean(CAnr{i}))/std(CAnr{i})];
    CAlla = [CAlla,(CAll{i} - mean(CAll{i}))/std(CAll{i})];
    CAlra = [CAlra,(CAlr{i} - mean(CAlr{i}))/std(CAlr{i})];
    CAnnra = [CAnnra,(CAnnr{i} - mean(CAnnr{i}))/std(CAnnr{i})];
    CAnnla = [CAnnla,(CAnnl{i} - mean(CAnnl{i}))/std(CAnnl{i})];
    
    dird = [dird,dirdata{i}];
end
% 
q1 = whiten([CAna;dird].');
q2 = whiten([CAnna;dird].');
q3 = whiten([CAla;dird].');
CAna = q1(:,1);
CAnna = q2(:,1);
CAla = q3(:,1);

Q = whiten([CAna,CAla]);

[c,p] = corr(Q(:,1),CAnna)
%% finding best N exponent only
R3 = [];
LN = [];
P3 = [];

for i = 1:10
    for r = 1:301
        LN = [];
        for ii = 1:2
            for t = 1:counter(i)
                l = flat{i,t,ii};
                n = fnor{i,t,ii};
                if length(l)<length(n)
                    n = n(1:length(l));
                end
                if length(l)>length(n)
                    l = l(1:length(n));
                end              
                g(r,t) = ((r-201)/100);
                ln = l.*(n.^(((r-201)/100)));
                LN(ii,t) = mean(ln);
            end
        end
        cord = isnan(LN(1,:)).*(1:length(LN));
        cord = cord(cord~=0);
        LN(1,cord) = LN(2,cord);
        cord = isnan(LN(2,:)).*(1:length(LN));
        cord = cord(cord~=0);
        LN(2,cord) = LN(1,cord);  
        cord = (~isnan(LN(1,:))).*(1:length(LN));
        cord = cord(cord~=0);
        [R3(i,r,1),P3(i,r,1)] = corr(mean(LN(:,cord))',(fsti{i}(cord)'.^2),'Type','Pearson');   
        [R3(i,r,2),P3(i,r,2)] = corr(mean(LN(:,cord))',fres{i}(cord)','Type','Pearson');
    end
end

%% finding the N exponent by allowing the relative scaling parameter vary
R4 = [];
LN = [];
P4 = [];
for i = 1:10
    i
    for r = 1:151
        for a = 1:150            
            LN = [];
            for ii = 1:2
                for t = 1:counter(i)
                    l = flat{i,t,ii};
                    n = fnor{i,t,ii};
                    %g(r,t) = ((r-101)/50);
                    ln = ((l.*(n.^((r-101)/50)))/(a/150))-(n.^(1 + ((r-101)/50)));
                    LN(ii,t) = mean(ln);
                end
            end
            cord = isnan(LN(1,:)).*(1:length(LN));
            cord = cord(cord~=0);
            LN(1,cord) = LN(2,cord);
            cord = isnan(LN(2,:)).*(1:length(LN));
            cord = cord(cord~=0);
            LN(2,cord) = LN(1,cord);
            cord = (~isnan(LN(1,:))).*(1:length(LN));
            cord = cord(cord~=0);
            [R4(i,r,a,1),P4(i,r,a,1)] = corr(mean(LN(:,cord))',(fsti{i}(cord)'.^2),'Type','Pearson');
            [R4(i,r,a,2),P4(i,r,a,2)] = corr(mean(LN(:,cord))',fres{i}(cord)','Type','Pearson');
        end
    end
end

tsize = 10;
for i = 1:10
    subplot(2,10,i);
    hold on
    imagesc(squeeze(R4(i,:,:,1)));
    set(gca, 'XTick',[1,75,150], 'XTickLabel', {0,.5,1})
    set(gca, 'YTick',[1,51,101,151], 'YTickLabel', {-2,-1,0,1})
    caxis([0 1]);
    cou1 = log10(squeeze(P4(i,:,:,1)));
    ot = contour(cou1,[-5.3,-5.3],'w');
    ot = contour(squeeze(R4(i,:,:,1)),[max(max(squeeze(R4(i,:,:,1))))-.02,max(max(squeeze(R4(i,:,:,1))))-.02],'k');
    set(gca,'FontSize',tsize)
    ylabel('$n$','Interpreter','latex','FontSize',tsize);
    xlabel('$\alpha$','Interpreter','latex','FontSize',tsize)
    title(strcat('s',num2str(i)),'Interpreter','latex','FontSize',tsize);  
    
    subplot(2,10,i+10);
    hold on
    imagesc(squeeze(R4(i,:,:,2)));
    caxis([0 1]);
    set(gca, 'XTick',[1,75,150], 'XTickLabel', {0,.5,1})
    set(gca, 'YTick',[1,51,101,151], 'YTickLabel', {-2,-1,0,1})
    cou1 = log10(squeeze(P4(i,:,:,2)));
    ot = contour(cou1,[-5.3,-5.3],'w');
    ot = contour(squeeze(R4(i,:,:,2)),[max(max(squeeze(R4(i,:,:,2))))-.025,max(max(squeeze(R4(i,:,:,2))))-.025],'k');
    set(gca,'FontSize',tsize)
    ylabel('$n$','Interpreter','latex','FontSize',tsize);
    xlabel('$\alpha$','Interpreter','latex','FontSize',tsize)
end


%% finding the N exponent by allowing the relative scaling parameter vary for the whole population
R5 = [];
P5 = [];
PP = [];

for r = 1:151
    r
    for a = 1:150
        PP = [];
        for i = 1:10
            LN = [];
            for ii = 1:2
                for t = 1:counter(i)
                    l = flat{i,t,ii};
                    n = fnor{i,t,ii};
                    %g(r,t) = ((r-101)/50);
                    ln = ((l.*(n.^((r-101)/50)))/(a/150))-(n.^(1 + ((r-101)/50)));
                    LN(ii,t) = mean(ln);
                end
            end
            cord = isnan(LN(1,:)).*(1:length(LN));
            cord = cord(cord~=0);
            LN(1,cord) = LN(2,cord);
            cord = isnan(LN(2,:)).*(1:length(LN));
            cord = cord(cord~=0);
            LN(2,cord) = LN(1,cord);
            cord = (~isnan(LN(1,:))).*(1:length(LN));
            cord = cord(cord~=0);
            PP = [PP;[mean(LN(:,cord))'/mean(mean(LN(:,cord))'),(fsti{i}(cord)'.^2),fres{i}(cord)']];
        end
        [R5(r,a,1),P5(r,a,1)] = corr(PP(:,1),PP(:,2));
        [R5(r,a,2),P5(r,a,2)] = corr(PP(:,1),PP(:,3));
    end
end

subplot(2,1,1);
hold on
imagesc(squeeze(R5(:,:,1)));
set(gca, 'XTick',[1,75,150], 'XTickLabel', {0,.5,1})
set(gca, 'YTick',[1,51,101,151], 'YTickLabel', {-2,-1,0,1})
caxis([0 1]);
ot = contour(squeeze(R5(:,:,1)),[max(max(squeeze(R4(:,:,1),1)))-.02,max(max(squeeze(R4(:,:,1),1)))-.02],'k');
set(gca,'FontSize',tsize)
ylabel('$n$','Interpreter','latex','FontSize',tsize);
xlabel('$\alpha$','Interpreter','latex','FontSize',tsize)


%% results for all subjects
% FWn curves
tsize = 10;
j = figure;
j.Position = [j.Position(1:2),[450,150]];
subplot(1,2,1)
x = linspace(-2,1,301);
hold on
for i = 1:10
    a = plot(x,R3(i,:,1).^2);
    a.Color = [.85 .85 .85];
    a.LineWidth = .75;
end
a = plot(v1,diag(R3(:,(v1*100)+201,1).^2),'.');

a = plot(x,mean(R3(:,:,1).^2));
a.Color = [1 0 0 .5];
a.LineWidth = 2;
a = plot([-1 -1],[0 1],':');
a.Color = [0 0 0];
a.LineWidth = .75;
axis([-2 1 -.01 1.01]);
set(gcf,'color','w');
xlabel('n','Interpreter','latex','FontSize',tsize)
title('$I_{out}^2$ vs $\mathrm{FW^{n}}$','Interpreter','latex','FontSize',tsize);
ylabel('$\mathrm{R^{2}}$','Interpreter','latex','FontSize',tsize);
xlabel('n','Interpreter','latex','FontSize',tsize)
set(gca,'FontSize',tsize)

subplot(1,2,2)
hold on
for i = 1:10
    a = plot(x,R3(i,:,2).^2);
    a.Color = [.85 .85 .85];
    a.LineWidth = .75;
end
a = plot(v2,diag(R3(:,(v2*100)+201,2).^2),'.');
a = plot(x,mean(R3(:,:,2).^2));
a.Color = [1 0 0 .5];
a.LineWidth = 2;
a = plot([-1 -1],[0 1],':');
a.Color = [0 0 0];
a.LineWidth = .75;
axis([-2 1 -.01 1.01]);
title('J vs $\mathrm{FW^{n}}$','Interpreter','latex','FontSize',tsize);
xlabel('n','Interpreter','latex','FontSize',tsize)
set(gca,'FontSize',tsize)

% FWn curves + p values curves


%% finding normal load
NM = [];
NS = [];
c = [];
for i = 1:10
    F = [];
    N = [];
    for ii = 1:2
        for t = 1:counter(i)
            f = flat{i,t,ii};
            n = fnor{i,t,ii};
            F(ii,t) = mean(f);
            N(ii,t) = mean(n);
        end
    end
    F(1,isnan(F(1,:))) = F(2,isnan(F(1,:)));
    F(2,isnan(F(2,:))) = F(1,isnan(F(2,:))); 
    F = mean(F);
    N(1,isnan(N(1,:))) = N(2,isnan(N(1,:)));
    N(2,isnan(N(2,:))) = N(1,isnan(N(2,:)));   
    N = N(:,~isnan(N(1,:)));    
    N = mean(N); 
    NM(i) = mean(N);
    NS(i) = std(N);
end

%% plot normal load related results
tsize = 10;

[v11,v1] = max(R3(:,:,1)');
v1 = (v1-201)/100;
[v22,v2] = max(R3(:,:,2)');
v2 = (v2-201)/100;

j = figure;
j.Position = [j.Position(1:2),[400,200]];
subplot(1,2,1)
hold on
for i = 1:10
    a = plot(NM(i),v1(i)+i/1000,'.');
    a.Color = [0 0 0];
    a.MarkerSize = 12;
    a = plot([NM(i)-NS(i),NM(i)+NS(i)],[v1(i)+i/1000,v1(i)+i/1000]);
    a.Color = [0 0 0];
end
ylabel('$n$ at max $R^2$','Interpreter','latex','FontSize',tsize);
xlabel('normal load across trials (N)','Interpreter','latex','FontSize',tsize);
[j,k,~] = corrcoef(NM',v1');
j = j(1,2);
k = k(1,2);
text(.85,-0,strcat('$r =\ $',num2str(round(j*100)/100)),'Interpreter','latex','FontSize',tsize);
text(.85,-.15,strcat('$p < .05$'),'Interpreter','latex','FontSize',tsize);
title('$I_{out}^2$ vs $FW^{n}$ fit','Interpreter','latex','FontSize',tsize);
axis([0 1.5 -1.1 .1]);
set(gca,'FontSize',tsize)

subplot(1,2,2)
hold on
for i = 1:10
    a = plot(NM(i),v2(i)+i/1000,'.');
    a.Color = [0 0 0];
    a.MarkerSize = 12;
    a = plot([NM(i)-NS(i),NM(i)+NS(i)],[v2(i)+i/1000,v2(i)+i/1000]);
    a.Color = [0 0 0];
end
xlabel('normal load across trials (N)','Interpreter','latex','FontSize',tsize);
[j,k,~] = corrcoef(NM',v2');
j = j(1,2);
k = k(1,2);
text(.85,-0,strcat('$r =\ $',num2str(round(j*100)/100)),'Interpreter','latex','FontSize',tsize);
text(.85,-.15,strcat('$p < .05$'),'Interpreter','latex','FontSize',tsize);
title('J vs $FW^{n}$ fit','Interpreter','latex','FontSize',tsize);
axis([0 1.5 -1.1 .1]);
set(gca,'FontSize',tsize)
set(gcf,'color','w');

%% plot normal load related results version 2

%% plot normal load related results
tsize = 10;
[v11,v1] = max(R3(:,:,1)');
v1 = (v1-201)/100;
[v22,v2] = max(R3(:,:,2)');
v2 = (v2-201)/100;

j = figure;
j.Position = [j.Position(1:2),[250,250]];
hold on

for i = 1:10
    a = plot(NM(i),v1(i)+i/1000,'.');
    a.Color = [1 0 0];
    a.MarkerSize = 14;
    a = plot([NM(i)-NS(i),NM(i)+NS(i)],[v1(i)+i/1000,v1(i)+i/1000],'-');
    a.Color = [1 0 0];
end
ylabel('n at max $R^2$','Interpreter','latex','FontSize',tsize);

for i = 1:10
    a = plot(NM(i),v2(i)+i/1000,'o');
    a.Color = [0 0 0];
    a.MarkerSize = 4;
    a.LineWidth = 1.02;
end

for i = 1:10
    a = plot([NM(i),NM(i)],[v1(i)+i/1000,v2(i)+i/1000],':');
    a.Color = [0 0 0];
    a.LineWidth = .5;
end

xlabel('normal load across trials (N)','Interpreter','latex','FontSize',tsize);
axis([0 2 -1.1 .1]);

[j,k,~] = corrcoef(NM',v1');
j = j(1,2);
k = k(1,2);

text(1,0,'$I_{out}^2$ vs $FW^{n}$ fit','Interpreter','latex','FontSize',tsize);
a = plot(.93,-.05,'.');
a.Color = [1 0 0];
a.MarkerSize = 14;
text(1,-.1,strcat('$r =\ $',num2str(round(j*100)/100)),'Interpreter','latex','FontSize',tsize);
text(1.7,-.1,strcat('$p < .05$'),'Interpreter','latex','FontSize',tsize);


[j,k,~] = corrcoef(NM',v2');
j = j(1,2);
k = k(1,2);
text(1,-.25,'J vs $FW^{n}$ fit','Interpreter','latex','FontSize',tsize);
a = plot(.93,-.3,'o');
a.Color = [0 0 0];
a.MarkerSize = 4;
a.LineWidth = 1.02;

text(1,-.35,strcat('$r =\ $',num2str(round(j*100)/100)),'Interpreter','latex','FontSize',tsize);
text(1.7,-.35,strcat('$p < .05$'),'Interpreter','latex','FontSize',tsize);

set(gca,'FontSize',tsize)
set(gcf,'color','w');

%% ploting v1 vs v2

col = jet;
tsize = 10;
j = figure;
j.Position = [j.Position(1:2),[250,250]];
hold on
for i = 1:10
    a = plot(v1(i)',v2(i)','.');
    a.Color = [0 0 0];
    a.MarkerSize = 14;
end
axis([-1.1 0.1 -1.1 0.1]);
set(gca, 'XTick',[-1,-.5,0], 'XTickLabel', {-1,-.5,0})
set(gca, 'YTick',[-1,-.5,0], 'YTickLabel', {-1,-.5,0})

text(-.35,-.85,strcat('$r = .79$'),'Interpreter','latex','FontSize',tsize);
text(-.35,-.95,strcat('$p < .01$'),'Interpreter','latex','FontSize',tsize);
set(gcf,'color','w');
a = plot([-2 2],[-2 2]);
a.Color = [.8 .8 .8];
xlabel('n at max $R^2$ for $I_{out}^2$ vs $\mathrm{FW^{n}}$ fit','Interpreter','latex','FontSize',tsize);
ylabel('n at max $R^2$ for J vs $\mathrm{FW^{n}}$ fit','Interpreter','latex','FontSize',tsize);

% ploting v1 vs v2 and normal load related stuff version 2

