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
[bl,al] = butter(2,2*100/1000); % filtering lateral force data
[bl2,al2] = butter(2,2*50/1000); % filtering lateral force data
[bn,an] = butter(2,2*25/1000); % filtering normal force data
times = {};
swipes = {};
time_data = {};
mu_data = {};
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
    lat = filtfilt(bl,al,data(:,1));
    lat = lat - mean(lat(1:2000,1));
    lat2 = filtfilt(bl2,al2,data(:,1));
    lat2 = lat2 - mean(lat2(1:2000,1));
    nor = filtfilt(bn,an,data(:,2));
    nor = nor - mean(nor(1:2000,1));
    cur = data(:,3);
    lat = lat/Lc;
    nor = nor/Nc;
    mu = lat./(nor);
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
    
    % extracting swipe information
    curr = cur>-4.8;
    for t = 1:length(time_vector)
        start = round(time_vector(t)) + time_fix{s}(t);
        endd = round(time_vector(t+1)) - time_fix{s}(t+1);
        endds = endd;
        
        % dividng up the trial into sections where the subject was in contact
        % with the device
        lat_d = lat(start:endd);
        cur_d = cur(start:endd);
        nor_d = nor(start:endd);
        nor_t = nor_d>0;
        lat_d = lat_d(nor_t);
        nor_d = nor_d(nor_t);
        cur_d = cur_d(nor_t);
        
        nor_data{s,t} = []; % initializing
        for n = 1:swipe_n{s}(t)
            start = times{s}(t,n,1);
            endd = times{s}(t,n,2);
            
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
            
            time_vec = [time_vec(diff(time_vec)>100),time_vec(end)];            
            time_vec1 = time_vec;
            time_vec = [time_vec(1:end-1)+round(diff(time_vec)/2)];
            time_vec2 = diff(time_vec1);
            
            % exacting first derivative values of mu and lat
            dmp = zeros(length(time_vec)-1,2); dmn = dmp; dln = dmp; dlp = dmp;
            v = [];
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
                mu_data{s,t,n,i,8} = dlp(i,:);
            end
            [sl,~] = size(dmn);
            for i = 1:sl
                mu_data{s,t,n,i,9} = dmn(i,:);
                mu_data{s,t,n,i,10} = dln(i,:);
            end
            
            ps = 0;
            ns = 0;
            
            for sw = 1:length(v)
                st = time_vec1(sw);
                e = time_vec1(sw+1);
                di = round((e - st)/4);
                
                m = 1;
                while ~((dmul(m+st)*dmul(m+st+1))<0 && abs(mul(m+st+1))>.25)
                    m = m + 1;
                end
                m = m + 1;
                if mul(st+m) > 0
                    ps = ps + 1;                    
                    [~,ma] = max(mul(st+m-10:st+m+10));                    
                    mu_data{s,t,n,ps,1} = mul(st+m-10+ma-1);
                    mu_data{s,t,n,ps,2} = latt(st+m-10+ma-1);
                    mu_data{s,t,n,ps,3} = norr(st+m-10+ma-1);
                else
                    ns = ns + 1;
                    [~,ma] = min(mul(st+m-10:st+m+10));                    
                    mu_data{s,t,n,ns,4} = mul(st+m-10+ma-1);
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
            for i = 1:time_data{s,t,n,3}(1)
                must(n,i,1) = mu_data{s,t,n,i,1};
                latt(n,i,1) = mu_data{s,t,n,i,2};
                nort(n,i,1) = mu_data{s,t,n,i,3};
            end
            for i = 1:time_data{s,t,n,3}(2)
                must(n,i,2) = mu_data{s,t,n,i,4};
                latt(n,i,2) = mu_data{s,t,n,i,5};
                nort(n,i,2) = mu_data{s,t,n,i,6};
            end
            swipen = swipen + [time_data{s,t,n,3},time_data{s,t,n,1}];
        end
        for d = 1:2
            if time_data{s,t,1,3}(d)>0
                temp = squeeze(abs(must(:,:,d)));
                temp = temp(temp>=.01);
                mus{s,t,d} = temp;
                temp = squeeze(abs(latt(:,:,d)));
                temp = temp(temp>=.01);
                lat{s,t,d} = temp;
                temp = squeeze(abs(nort(:,:,d)));
                temp = temp(temp>=.01);
                nor{s,t,d} = temp;
            end
        end
        swipe_times{s,t} = swipen;
    end
end

clearvars -except mus lat nor cur swipe_times sti res trials time_data

%% STATS

%% plot of stiction judgment as function of applied current (version 1)
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
    outliers{s} = ~isoutlier(allres{s},'mean');
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

% plotting
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
    a.Color = [.9 .9 .9];
    a.MarkerSize = 10;
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
text(1.9,2.45,'r = .74','Interpreter','latex','FontSize',11)
text(1.9,2.15,'$\rho$ = .8','Interpreter','latex','FontSize',11)
%text(2.5,3.15,'p $\ll$ $10^{-9}$','Interpreter','latex','FontSize',11)
xticks([0,2,3,4,5,6,7,8])
ha.EdgeColor = [0 0 0];
ha.Position(3) = ha.Position(3) + .02;
ha.Position(1) = ha.Position(1) + .01;
ha.Position(4) = ha.Position(4) + .02;

%% plot of stiction judgment as function of applied current (version 2)

j = figure;
j.Position = [j.Position(1:2),[500,400]];
L = length(subjects);
col = jet;
hold on
RES = []; res1 = [];
STI = [];
fun = @(x,xdata)x(1)*xdata + x(2);
shaps = {'^','s','v','p','x','.','<','>','d','o','*'};
ord = [1,2,3,4,5,8,6,7,9,10,11];
stims = [0,2,3,4,5,6,7,8];
for ii = [1,2,3,4,5,7,8,9,10,11]
    i = ord(ii);
    stii = sti{i};%(~isoutlier(res{i},'mean'));
    ress = res{i};%(~isoutlier(res{i},'mean'));
    STI = [STI; stii];
    RES = [RES; ress];
end
mm = [];
stims = [0,2,3,4,5,6,7,8];
for i = 1:8
    mm(i) = mean(RES(STI==stims(i)));
end
a = plot(stims,mm);
a.Color = [.8 .8 .8];
a.LineWidth = 2;
for ii = [1,2,3,4,5,7,8,9,10,11]
    i = ord(ii);
    stii = sti{i};%(~isoutlier(res{i},'mean'));
    ress = res{i};%(~isoutlier(res{i},'mean'));
    a = plot(sti{i}+(ord(ii) - 6)/15,res{i},shaps{ii});
    a.Color = [col(round(63/11*(ii-1)+1),:),0];
    a.LineWidth = 1.01;
    a.MarkerSize = 3;
    if ii>3
        a.MarkerSize = 4;
    end
end
[ha,hb,hc,hd] = legend('mean','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10');
hb(12).XData = [hb(12).XData(1) + .2,hb(12).XData(2) - .2];
ha.FontSize = 7;
ha.Location = 'northwest';
for ii = [1,2,3,4,5,6,8,7,9,10,11]
    i = ord(ii);
    if sum(isoutlier(res{i},'mean'))>0
        a = plot(sti{i}(isoutlier(res{i},'mean'))+(i - 6)/15,res{i}(isoutlier(res{i},'mean')),'o');
        a.Color = [0 0 0];
        a.MarkerSize = 10;
    end
end
set(gcf,'color','w');
xlabel('applied current (mA)','Interpreter','latex','FontSize',11)
ylabel('normalized stickiness judgment','Interpreter','latex','FontSize',11)
axis([-1 9 -.25 5])
text(1.9,3.95,'r = .69','Interpreter','latex','FontSize',11)
text(1.9,3.55,'$\rho$ = .74','Interpreter','latex','FontSize',11)
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
hh = 1;
resnew = {};
mnew = {};
M = [];
T = [];
N = [];
mus2 = {};
lat2 = {};
nor2 = {};
for i = 1:10
    M = [];
    for ii = 1:2
        for t = 1:counter(i)            
            m = fmus{i,t,ii};
            m = m(~isnan(m));  
            M(ii,t) = median(m);            
        end        
        mus2{i,ii} = M(ii,:);
    end
    co(i) = corr(mean(M)',fres{i}');
end
[~,cor] = sort(co);
%% mus vs judgment vs current 1

a = figure;
a.Position = [a.Position(1:2),[1500,300]];
L = length(subjects);
col = jet;
hold on

fun = @(x,xdata)x(1)*xdata + x(2);
shaps = {'s','o'};
P = zeros(2,2,10);
tit = {};
for i = 1:10
    for ii = 1:2
        subplot(2,10,i);
        hold on       
        
        M = mus2{i,ii};
        S = sti{i};
        a = plot(M,S-.3 + ii*.3,shaps{ii});
        if ii == 1
            a.Color = [0 0 1];
            a.MarkerSize = 4;
        else
            a.Color = [1 0 0];
            a.MarkerSize = 3;
        end
        a.LineWidth = 1.01;
        axis([0 1.6 -.5 9]);
        xticks([0,.50,1.0,1.5,2.0,2.5])
        yticks([0.0 2 4 6 8.0]);
        set(gca,'FontSize',9);
        [hh,jj,~] = corrcoef(M,S);
        P(1,ii,i) = jj(1,2);
        P(2,ii,i) = hh(1,2);
        if P(1,ii,i)<.001
            tit{ii} = strcat('r =',{' '},num2str(hh(1,2),' %0.2f'),' p $<$ .001');
        else
            tit{ii} = strcat('r =',{' '},num2str(hh(1,2),' %0.2f'),'  p =',{' '},num2str(jj(1,2),'%0.3f'));
        end
    end
    title({tit{1}{1},tit{2}{1}},'Interpreter','latex','FontSize',10);
end
set(gcf,'color','w');
subplot(2,10,1);
ylabel('applied current (mApp)','Interpreter','latex','FontSize',9);

for i = 1:10
    for ii = 1:2
        M = mnew{cor(i),ii};
        R = rnew{cor(i),ii};
        
        subplot(2,10,10+i);
        hold on
        a = plot(M,R/max(R) - .05 + ii*.05,shaps{ii});
        if ii == 1
            a.Color = [0 0 1];
            a.MarkerSize = 4;
        else
            a.Color = [1 0 0];
            a.MarkerSize = 3;
        end
        a.LineWidth = 1.01;
        axis([0 1.6 -.05 1.15]);
        xticks([0,.50,1.0,1.5,2.0,2.5])
        yticks([0.0 1]);
        set(gca,'FontSize',9);
        [hh,jj,~] = corrcoef(M,R);
        P(1,ii,i) = jj(1,2);
        P(2,ii,i) = hh(1,2);
        if P(1,ii,i)<.001
            tit{ii} = strcat('r =',{' '},num2str(hh(1,2),' %0.2f'),' p $<$ .001');
        else
            tit{ii} = strcat('r =',{' '},num2str(hh(1,2),' %0.2f'),'  p =',{' '},num2str(jj(1,2),'%0.3f'));
        end
    end
    title({tit{1}{1},tit{2}{1}},'Interpreter','latex','FontSize',10);
end
subplot(2,10,11);
ylabel('normalized judgement','Interpreter','latex','FontSize',9);
subplot(2,10,16);
xlabel('static coefficient of friction ($\mu_s$) ','Interpreter','latex','FontSize',9);

%% mus vs judgment vs current 2
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
        S = fsti{cor(i)};
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
        axis([0 2 -.5 9]);
        xticks([0,.50,1.0,1.5,2.0,2.5])
        yticks([0.0 2 4 6 8.0]);
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
    text(1,.75,tit{1},'Interpreter','latex','FontSize',9)
end


set(gcf,'color','w');
subplot(2,10,1);
ylabel('applied current (mApp)','Interpreter','latex','FontSize',9);

fun = @(x,xdata)x(1)*xdata + x(2);
for i = 1:10
    M = [];
    for ii = 1:2
        M(ii,1:length(mus2{cor(i),ii})) = mus2{cor(i),ii};
        R = fres{cor(i)}/max(fres{cor(i)});        
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
        a = plot([.1,1.5],[.1,1.5]*v(1)+v(2));
        
        if ii == 1
            a.Color = [0 0 1 .4];
        else
            a.Color = [1 0 0 .4];
        end
        
        a.LineWidth = 1.1;
        axis([0 2 -.1 1.15]);
        xticks([0,.50,1.0,1.5,2.0,2.5])
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
    text(1,0,tit{1},'Interpreter','latex','FontSize',9)
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

%% correlation based stats
m1 = {}; m2 = {}; s1 = {}; s2 = {}; M = {}; R ={};sn = {};st = {}; mN = {}; sN = {}; fs = {}; fr = {};
for i = 1:10
    m1{i} = filteredmus{1,i};
    m2{i} = filteredmus{2,i};
    s1{i} = filteredmus{3,i};
    s2{i} = filteredmus{4,i};
    R{i} = filteredres{i}/max(filteredres{i});
    st{i} = filteredst{i};
    sn{i} = filteredsn{i};
    mN{i} = filteredN{i,1};
    sN{i} = filteredN{i,2};
    fs{i} = filteredsti{i};
    fr{i} = filteredres{i};
    
    m2{i} = m2{i}(~isnan(m1{i}));
    R{i} = R{i}(~isnan(m1{i}));
    st{i} = st{i}(~isnan(m1{i}));
    sn{i} = sn{i}(~isnan(m1{i}));
    mN{i} = mN{i}(~isnan(m1{i}));
    sN{i} = sN{i}(~isnan(m1{i}));
    fs{i} = fs{i}(~isnan(m1{i}));
    fr{i} = fr{i}(~isnan(m1{i}));
    s1{i} = s1{i}(~isnan(m1{i}));
    s2{i} = s2{i}(~isnan(m1{i}));
    m1{i} = m1{i}(~isnan(m1{i}));
    
    
    
    st{i} = st{i}(~isnan(m2{i}));
    sn{i} = sn{i}(~isnan(m2{i}));
    R{i} = R{i}(~isnan(m2{i}));
    sN{i} = sN{i}(~isnan(m2{i}));
    mN{i} = mN{i}(~isnan(m2{i}));
    m1{i} = m1{i}(~isnan(m2{i}));
    fs{i} = fs{i}(~isnan(m2{i}));
    fr{i} = fr{i}(~isnan(m2{i}));
    s1{i} = s1{i}(~isnan(m2{i}));
    s2{i} = s2{i}(~isnan(m2{i}));
    m2{i} = m2{i}(~isnan(m2{i}));
    
    M{i} = (m1{i}+m2{i})/2;
end

%% correlation of goodness of judgment and time/swipes per trial
co = [];
fun = @(x,xdata)x(1)*xdata + x(2);
N = [];
T = [];
resid = {};
for i = 1:10
    v = lsqcurvefit(fun,[0,0],M{i},R{i});
    resid{i} = abs(R{i} - (M{i}*v(1)+v(2)));
    [A,B,~] = corrcoef(resid{i},sn{i}');
    co(1,1,i) = A(1,2);
    co(1,2,i) = B(1,2);
    [A,B,~] = corrcoef(resid{i},st{i}');
    co(2,1,i) = A(1,2);
    co(2,2,i) = B(1,2);
    N = [N,sn{i}];
    T = [T,st{i}];
end
%% direction sensitivity of stiction

H = [];
P = [];
D = {};
for i = 1:10
    [H(i),P(i),a,D{i}] = ttest2(m1{i},m2{i});
end


%% impact of applied normal force on discrimination
co = [];
for i = 1:10
    [A,B,~] = corrcoef(resid{i},1./mN{i});
    co(1,1,i) = A(1,2);
    co(1,2,i) = B(1,2);
    [A,B,~] = corrcoef(resid{i},sN{i});
    co(2,1,i) = A(1,2);
    co(2,2,i) = B(1,2);
end

%% effect of current on deviation of stiction values
co = [];
for i = 1:10
    [A,B,~] = corrcoef(resid{i},fs{i});
    co(1,i) = A(1,2);
    co(2,i) = B(1,2);
end

%% effect of deviation of stiction values on performance
co = [];

for i = 1:10
    [A,B,~] = corrcoef(resid{i},(s1{i} + s2{i}).^2);
    co(1,i) = A(1,2);
    co(2,i) = B(1,2);
end

%% effect of normal force on stiction deviation
co = [];

for i = 1:10
    ss = (s1{i} + s2{i}).^2;
    [A,B,~] = corrcoef(1./mN{i}(ss~=0),ss(ss~=0));
    co(1,i) = A(1,2);
    co(2,i) = B(1,2);
end


%% checking if judgement performance changed over time
tp = [];
ii = 1;
data = {};
fun = @(x,xdata)x(1)*xdata + x(2);
for i = [1,2,3,4,5,6,7,9,10,11]
    s = sti{i}(~isoutlier(res{i},'mean'));
    r = res{i}(~isoutlier(res{i},'mean'));
    o = lsqcurvefit(fun,[0,0],s,r);
    y = s*o(1) + o(2);
    e = (r - y).^2;
    l = round(length(e)/2);
    data{1} = e(1:l);
    data{2} = e(l+1:end);
    tp(ii,1:2) = unpairedttest(data);
    ii = ii + 1;
end

%% finding best N coefficient
R2 = [];
LN = [];
for i = 1:10
    for r = 1:301
        LN = [];
        for ii = 1:2
            for t = 1:counter(i)
                l = flat{i,t,ii};
                n = fnor{i,t,ii};
                ln = l.*(n.^((r-201)/100));
                LN(ii,t) = mean(ln);
            end
        end
        v = corrcoef(mean(LN)',fres{i});
        R2(i,r,1) = v(1,2);
        %v = corrcoef(mean(LN)',fsti{i},'Type','Spearman');
        %R2(i,r,2) = v(1,2);
        R2(i,r,2) = corr(mean(LN)',fsti{i}','Type','Spearman');        
    end
end

j = figure;
j.Position = [j.Position(1:2),[450,150]];
subplot(1,2,2)
x = linspace(-2,1,301);
hold on
for i = 1:10
    a = plot(x,R2(i,:,1));
    a.Color = [.8 .8 .8];
    a.LineWidth = .5;
end
a = plot(x,mean(R2(:,:,1)));
a.Color = [1 0 0 .5];
a.LineWidth = 2;
a = plot(x,std(R2(:,:,1)),'--');
a.Color = [0 0 0 .5];
a.LineWidth = 1;
axis([-2 1 -.1 1.1]);
title('judgment vs $FW^{k}$','Interpreter','latex','FontSize',10);
xlabel('k','Interpreter','latex','FontSize',10)
set(gca,'FontSize',10)


subplot(1,2,1)
hold on
for i = 1:10
    a = plot(x,R2(i,:,2));
    a.Color = [.8 .8 .8];
    a.LineWidth = .5;
end
a = plot(x,mean(R2(:,:,2)));
a.Color = [1 0 0 .5];
a.LineWidth = 2;
a = plot(x,std(R2(:,:,2)),'--');
a.Color = [0 0 0 .5];
a.LineWidth = 1;
axis([-2 1 -.1 1.1]);
set(gcf,'color','w');
xlabel('k','Interpreter','latex','FontSize',10)
title('current vs $FW^{k}$','Interpreter','latex','FontSize',10);
ylabel('Pearson''s r','Interpreter','latex','FontSize',10);
xlabel('k','Interpreter','latex','FontSize',10)
set(gca,'FontSize',10)