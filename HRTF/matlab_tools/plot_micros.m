function plot_micros(fname,cx,cy);

if nargin == 1
      error('Au moins deux arguments...');
end

mic	= sigRead(fname);

long	= 2048;
nfft	= 2048;

mic	= mic(:,1:long)';
micf	= fft(mic,nfft);
micfm	= abs(micf(1:nfft/2+1));

if nargin == 2
      plot_gv(micfm(:,1),cx);
elseif nargin == 3
      plot_gv(micfm(:,1),cx);
      hold on;
      plot_gv(micfm(:,2),cy);
      hold off;
end

ylim([-30 0];
