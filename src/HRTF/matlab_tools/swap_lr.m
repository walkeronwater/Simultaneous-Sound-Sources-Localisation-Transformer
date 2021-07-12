function echange_gd

rep	= '/u/salles/vanderno/Listen/Hrir';

az	= (0:15:345);
%az	= 0;
el	= (-45:15:45);
%el	= 90;

for i=1:length(az)
      for j=1:length(el)
	    fname_in	= sprintf('%s/IRC_1016_R_R0195_T%03d_P%03d.wav',rep,az(i),mod(el(j),360));
	    ri		= sigRead(fname_in);
	    ri_swap	= flipud(ri);
	    fname_out	= sprintf('%s/Temp/IRC_1016_R_R0195_T%03d_P%03d.wav',rep,az(i),mod(el(j),360));
	    sigWrite(ri_swap,fname_out,44100,'wav','int24');
      end
end
