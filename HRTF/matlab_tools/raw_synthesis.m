function [sound_m] = raw_synthesis(l_hrir_S,r_hrir_S,base_sound_v,trajectory_S);

% RAW_SYNTHESIS Convolves a sound by HRIRs on a given trajectory
%
% Usage
%   [sound_m] = raw_synthesis(l_hrir_S,r_hrir_S,base_sound_v,trajectory_S);
%
% Input
%   l_hrir_S     : left HRIRs ('FIR' data)
%   r_hrir_S     : right HRIRs ('FIR' data)
%   base_sound_v : a column vector of sound to be convoluted
%   trajectory_S : trajectory structure (see below)
%
% Output
%   sound_m : binaural output (2 columns)
%
% Trajectory
%   A trajectory structure contains 3 fields of same size:
% - time_v : a column vector of times (in samples).
% - elev_v, azim_v : column vectors of elevation and azimuth of the
% trajectory.
%
% Authors
%   Rio Emmanuel
%   (c) Ircam - May 2002

trajectory_S.time_v = round(trajectory_S.time_v);
% Various lengths
points_n = size(l_hrir_S.content_m,2);
base_length_n = length(base_sound_v);
convolution_n = base_length_n+points_n-1;
length_n = trajectory_S.time_v(end)+convolution_n-1;
positions_n = length(trajectory_S.time_v);
% Sounds allocation
l_sound_v = zeros(length_n,1);
r_sound_v = zeros(length_n,1);
% Convolution
wait_h = waitbar(0,'Synthesis...');
for i=1:positions_n
  time_n = trajectory_S.time_v(i);
  elev_n = trajectory_S.elev_v(i);
  azim_n = trajectory_S.azim_v(i);
  indice_n = find((l_hrir_S.elev_v==elev_n)&(l_hrir_S.azim_v==azim_n));
  if isempty(indice_n)
    disp(sprintf( ...
	  'RAW_SYNTHESIS : Unable to find couple : [%d %d]', ...
		elev_n,azim_n));
  else
    indice_n = indice_n(1);
    convolved_v = conv(base_sound_v,l_hrir_S.content_m(indice_n,:)');
    l_sound_v(time_n+[1:convolution_n]-1) = ...
	  l_sound_v(time_n+[1:convolution_n]-1)+convolved_v;
    convolved_v = conv(base_sound_v,r_hrir_S.content_m(indice_n,:)');
    r_sound_v(time_n+[1:convolution_n]-1) = ...
	  r_sound_v(time_n+[1:convolution_n]-1)+convolved_v;
  end;
  waitbar(i/positions_n,wait_h);
end;
close(wait_h);
% Concatenation
sound_m = [l_sound_v r_sound_v];
