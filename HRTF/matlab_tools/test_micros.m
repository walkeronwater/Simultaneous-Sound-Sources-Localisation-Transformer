function test_micros(suID,mesID)

if nargin == 0
      error('Rentrer au moins un argument...');
end

if nargin == 1
      mesID	= 0;
end

nom_ref	= sprintf('IRC_%d_M_KNO_0.wav',suID+1000);
ref	= sigRead(nom_ref);
ref	= ref(:,1:2048)';
ref_f	= fft(ref,2048);
ref_fm	= abs(ref_f(1:1025,:));

plot_gv(ref_fm(:,1));hold on;plot_gv(ref_fm(:,2),'r');
hold on;

if mesID ~= 0
      nom_mes	= sprintf('IRC_%d_M_KNO_%d.wav',suID+1000,mesID);
      mes	= sigRead(mes);
      mes	= mes(:,1:2048)';
      mes_f	= fft(mes,2048);
      mes_fm	= abs(mes_f(1:1025,:));
      plot_gv(mes_fm(:,1),'c');plot_gv(mes_fm(:,2),'m');
      legend('Ref_L','Ref_R','Before_L','Before_R');
else
      nom_mes1	= sprintf('IRC_%d_M_KNO_1.wav',suID+1000);
      mes1	= sigRead(mes1);
      mes1	= mes(:,1:2048)';
      mes1_f	= fft(mes1,2048);
      mes1_fm	= abs(mes1_f(1:1025,:));
      plot_gv(mes1_fm(:,1),'c');plot_gv(mes1_fm(:,2),'m');

      nom_mes2	= sprintf('IRC_%d_M_KNO_2.wav',suID+1000);
      mes2	= sigRead(mes2);
      mes2	= mes2(:,1:2048)';
      mes2_f	= fft(mes2,2048);
      mes2_fm	= abs(mes2_f(1:1025,:));
      plot_gv(mes2_fm(:,1),[0.34 0.45 1]);plot_gv(mes2_fm(:,2),[1 0.4 0.4]);
      legend('Ref_L','Ref_R','Before_L','Before_R','After_L','After_R');
end

ylim([-30 0]);
hold off;
