function [transfer_functions_S] = phase_recomposition(magnitudes_S,excess_phases_S);

% PHASE_RECOMPOSITION Recomposes transfer functions from minimum and excess phases
%
% Usage
%   [transfer_functions_S] = phase_recomposition(magnitudes_S,excess_phases_S);
%
% Input
%   magnitudes_S    : minimum phase part as 'MAGNITUDE' (only positive frequencies)
%   excess_phases_S : excess phase part as 'PHASE' (only positive frequencies)
%
% Output
%   transfer_functions_S : 'TRANSFER_FUNCTION' data
%
% See also DATA_FORMATS, PHASE_DECOMPOSITION
%
% Authors
%   Jot Jean-Marc (was SXPHASE)
%   Rio Emmanuel
%   (c) Ircam - June 2001

% Todo detail number of points (IR, TF, Mag, Eph)

if ~exist('excess_phases_S')
    excess_phases_S = rmfield(magnitudes_S,'content_m');
    excess_phases_S.content_m = zeros(size(magnitudes_S.content_m));
    excess_phases_S.type_s = 'PHASE';
end
transfer_functions_S = rmfield(magnitudes_S,'content_m');
if ~strcmp(magnitudes_S.type_s,'MAGNITUDE')|~strcmp(excess_phases_S.type_s,'PHASE')
    disp(['PHASE_RECOMPOSITION : bad data types ' ...
            magnitudes_S.type_s ' and ' excess_phases_S.type_s]);
    return;
end

fft_points_n = (size(magnitudes_S.content_m,2)-1)*2;
magnitudes_S.content_m = max(5*eps,magnitudes_S.content_m);
magnitudes_S.content_m = [magnitudes_S.content_m magnitudes_S.content_m(:,fft_points_n/2:-1:2)];
excess_phases_S.content_m = [excess_phases_S.content_m -excess_phases_S.content_m(:,fft_points_n/2:-1:2)];
minimum_phases_m = imag(hilbert(-log(magnitudes_S.content_m).').');
phases_m = excess_phases_S.content_m+minimum_phases_m;
transfer_functions_S.content_m = magnitudes_S.content_m.*exp(sqrt(-1)*phases_m);
transfer_functions_S.type_s = 'TRANSFER_FUNCTION';
