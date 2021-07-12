function [l_ir_S,r_ir_S] = load_impulse_responses(...
    directory_s,name_write_fh,name_write_args_C,elev_v,azim_v,format_s);

% LOAD_IMPULSE_RESPONSES - Loads vectors from IR files
%
% Usage
%   [l_ir_S,r_ir_S] = load_impulse_responses(directory_s,name_write_fh,name_write_args_C,elev_v,azim_v,format_s);
%
% Input
%   directory_s       : directory from which impulse responses are loaded
%   name_write_fh     : a function handle of the 'write name' function
%   name_write_args_C : string arguments of the 'write name' function
%   elev_v            : vector of elevations (from -90 to 90)
%   azim_v            : vector of azimuths (from 0 to 360)
%   format_s          : a string giving the read format
%
% Output
%   l_ir_S : structure of impulse responses (left if 2 channels)
%   r_ir_S : right impulse responses (if exists)
%
% Formats
%   for supported formats, see LOAD_VECTOR
%
% See also LOAD_VECTOR
%
% Authors
%   Larcher Veronique (was LOADIMP)
%   Rio Emmanuel
%   (c) Ircam - June 2001

l_ir_m = zeros(length(azim_v),0);
r_ir_m = zeros(length(azim_v),0);
waitbar_h = waitbar(0,'loading Impulse Responses...');
sampling_hz = 0;
for i=1:length(azim_v)
  waitbar(i/length(azim_v),waitbar_h);
  [file_name_s,channel_n] = ...
	feval(name_write_fh,name_write_args_C,elev_v(i),azim_v(i));
  [impulse_response_v,sampling_hz] = ...
	load_vector([directory_s filesep file_name_s],format_s,channel_n);
  l_ir_m(i,1:length(impulse_response_v)) = ...
      impulse_response_v(1,:);
  if (size(impulse_response_v,1)==2)
    r_ir_m(i,1:length(impulse_response_v)) = ...
	  impulse_response_v(2,:);
  end
end
close(waitbar_h);
l_ir_S.type_s = 'FIR';
l_ir_S.elev_v = elev_v;
l_ir_S.azim_v = azim_v;
l_ir_S.sampling_hz = sampling_hz;
r_ir_S = l_ir_S;
l_ir_S.content_m = l_ir_m;
r_ir_S.content_m = r_ir_m;
if sampling_hz==0
  disp('LOAD_IMPULSE_RESPONSES : sampling rate unknown');
end
