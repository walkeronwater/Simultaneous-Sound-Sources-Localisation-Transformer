function [equalized_magnitudes_S,weighted_magnitude_v] = ...
    diffuse_field_equalize(magnitudes_S,weighting_method_s);

% DIFFUSE_FIELD_EQUALIZE Equalizes magnitudes by simulating a diffuse field.
%
% Usage
%   [equalized_magnitudes_S,weighted_magnitude_v] = diffuse_field_equalize(magnitudes_S,weighting_method_s);
%
% Input
%   magnitudes_S       : magnitudes to be equalized
%   weighting_method_s : see SPATIAL_SAMPLING_WEIGHTING
%
% Output
%   equalized_magnitudes_S : the result
%   weighted_magnitude_v   : the magnitude used in division
%
% See also SPATIAL_SAMPLING_WEIGHTING
%
% Authors
%   Larcher Veronique (was WINDIF)
%   Rio Emmanuel
%   (c) Ircam - June 2001

% Todo verify normalization

if ~strcmp(magnitudes_S.type_s,'MAGNITUDE')
    disp(['DIFFUSE_FIEL_EQUALIZE : bad data type ' magnitudes_S.type_s]);
    return;
end

weights_v = spatial_sampling_weighting(magnitudes_S.elev_v,magnitudes_S.azim_v,weighting_method_s);
weighted_magnitude_v = sum(magnitudes_S.content_m.*magnitudes_S.content_m...
    .*(weights_v*ones(1,size(magnitudes_S.content_m,2))),1);
weighted_magnitude_v = sqrt(weighted_magnitude_v);
weighted_magnitude_v = max(5*eps,weighted_magnitude_v);

equalized_magnitudes_S.content_m = magnitudes_S.content_m./(ones(size(magnitudes_S.content_m,1),1)*weighted_magnitude_v);
equalized_magnitudes_S.elev_v = magnitudes_S.elev_v;
equalized_magnitudes_S.azim_v = magnitudes_S.azim_v;
equalized_magnitudes_S.type_s = 'MAGNITUDE';
equalized_magnitudes_S.sampling_hz = magnitudes_S.sampling_hz;