%% formulating filter for force 

b1 = [1 .25*(2*pi*255) (2*pi*255).^2];
a1 = [1 1*(2*pi*255) (2*pi*255).^2];
b2 = [0 0 (2*pi*250).^2];
a2 = [1 (2*pi*250) (2*pi*250).^2];

TF = tf(b1,a1)*tf(b1,a1)*tf(b2,a2)*tf(b2,a2);

[bb,aa] = tfdata(TF);
bb = cell2mat(bb);
aa = cell2mat(aa);

[bd,ad] = stoz(bb,aa,1000);

% formulating filter for force derivative
% 
% b1 = [1 .04*(2*pi*237) (2*pi*237).^2];
% a1 = [1 .1*(2*pi*237) (2*pi*237).^2];
% b2 = [0 0 (2*pi*140).^2];
% a2 = [1 sqrt(2.2)*(2*pi*140) (2*pi*140).^2];
% 
% TFd = tf(b2,a2)*tf(b2,a2)*tf(b1,a1)*tf(b1,a1);
% 
% [bb,aa] = tfdata(TFd);
% bb = cell2mat(bb);
% aa = cell2mat(aa);
% 
% [bdd,add] = stoz(bb,aa,10000);
% 
% FFTimpact = abs(fft(impact.')).';
% FFTimpactm = mean(FFTimpact(:,10:70).').';
% impact = impact./(FFTimpactm*ones(1,10000));
% 
% dimpact = impact;
% for i = 1:15
%     impact(i,1:2) = zeros(1,2);
%     impact(i,end-1:end) = zeros(1,2);
% end
% 
% for i = 1:15
%     dimpact(i,:) = derivR(impact(i,:),1,10000);
%     dimpact(i,1:2) = zeros(1,2);
%     dimpact(i,end-1:end) = zeros(1,2);
% end
% 
% hold on
% set(gca,'TickLabelInterpreter','latex','FontSize',10);
% a = plot(20*log10(mean(abs(fft(impact.')).')));
% a.Color = [0 0 0];
% a.LineWidth = 1;
% a = plot(20*log10(mean(abs(fft(filter(bd,ad,impact.'))).')));
% a.Color = [1 0 0];
% a.LineWidth = 1;
% set(gca, 'XScale', 'log');
% axis([2 1000 -25 25])
% a = legend('unfiltered','filtered');
% set(a,'Interpreter','latex','FontSize',10);
% a.Location = 'northwest';
% a.Box = 'off';
% set(gcf,'color','w');
% hold on
% set(gca,'TickLabelInterpreter','latex','FontSize',10);
% a = plot(20*log10(mean(abs(fft(dimpact.')).')),':');
% a.Color = [0 0 0];
% a.LineWidth = 1;
% a = plot(20*log10(mean(abs(fft(filter(bdd,add,dimpact.'))).')),':');
% a.Color = [1 0 0];
% a.LineWidth = 1;
% set(gca, 'XScale', 'log');
% axis([2 1000 -25 40])
% a = legend('unfiltered','filtered');
% set(a,'Interpreter','latex','FontSize',10);
% a.Location = 'northwest';
% a.Box = 'off';
% set(gcf,'color','w');