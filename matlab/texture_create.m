wavelengthx = logspace(log10(.5),log10(5),5);
wavelengthy = logspace(log10(.5),log10(5),5);

texts = {};
dim = [110,22];
res = .025;
xax = (0:res:dim(1));
yax = (0:res:dim(2));
for x = 1:length(wavelengthx)
    for y = 1:length(wavelengthy)
        lt = sin(2*pi/wavelengthx(x)*xax).';
        lt = lt*ones(1,length(yax));
        wt = sin(2*pi/wavelengthy(y)*yax).';
        wt = wt*ones(1,length(xax));                
        
        mat{x,y} = (wt.*lt.' + 1)/2;
    end
end
