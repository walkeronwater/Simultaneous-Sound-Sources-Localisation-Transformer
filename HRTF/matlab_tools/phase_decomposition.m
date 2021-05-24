function [magnitudes_S,excess_phases_S] = phase_decomposition(transfer_functions_S);

% PHASE_DECOMPOSITION Decomposes transfer functions into minimum and excess phases
%
% Usage
%   [magnitudes_S,excess_phases_S] = phase_decomposition(transfer_functions_S);
%
% Input
%   transfer_functions_S : 'TRANSFER_FUNCTION' data
%
% Output
%   magnitudes_S    : minimum phase part as 'MAGNITUDE' (only positive frequencies)
%   excess_phases_S : excess phase part as 'PHASE' (only positive frequencies)
%
% Number of points
%   since for real signals both magnitude and excess phase are symmetric,
%   only half of points are kept.
%   given N1 the FFT number of points, number of kept points is :
%     N2 = (N1/2)+1
%
% See also DATA_FORMATS, PHASE_RECOMPOSITION
%
% Authors
%   Jot Jean-Marc (was XSPHASE and XSPHASEX)
%   Rio Emmanuel
%   (c) Ircam - June 2001

% Todo in data formats : detail number of points (IR, TF, Mag, Eph)

if ~strcmp(transfer_functions_S.type_s,'TRANSFER_FUNCTION')
  disp(['PHASE_DECOMPOSITION : data type ' transfer_functions_S.type_s]);
  return;
end
magnitudes_S = rmfield(transfer_functions_S,'content_m');

fft_points_n = size(transfer_functions_S.content_m,2);
% index range for positive frequencies
fft_points_v = (1:fft_points_n/2+1);

% magnitude calculation
magnitudes_S.content_m = abs(transfer_functions_S.content_m);
magnitudes_S.content_m = max(5*eps,magnitudes_S.content_m);
if nargout>1
  excess_phases_S = rmfield(transfer_functions_S,'content_m');
  % phases computation
  phases_m = unwrap(angle(transfer_functions_S.content_m).').';
  minimum_phases_m = imag(hilbert(-log(magnitudes_S.content_m).').');
  excess_phases_S.content_m = phases_m-minimum_phases_m;
  % phase formatting
  excess_phases_S.content_m = excess_phases_S.content_m(:,fft_points_v);
  excess_phases_S.type_s = 'PHASE';
end;
% magnitude formatting
magnitudes_S.content_m = magnitudes_S.content_m(:,fft_points_v);
magnitudes_S.type_s = 'MAGNITUDE';
