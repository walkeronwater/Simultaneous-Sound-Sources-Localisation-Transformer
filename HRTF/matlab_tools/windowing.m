function [windowed_impulse_responses_S] = windowing(impulse_responses_S,window_extrema_v,zero_pad_s);

% WINDOWING Windowing of impulse responses
%
% Usage
%   [windowed_impulse_responses_S] = windowing(impulse_responses_S,window_extrema_v);
%
% Input
%   impulse_responses_S : impulse responses to window
%   window_extrema_v    : 2 elements vector (in samples)
%   zero_pad_s          : set to 'pad' if wanted
%
% Output
%   windowed_impulse_responses_S : corresponding result
%
% See also DATA_FORMATS
%
% Authors
%   Rio Emmanuel
%   (c) Ircam - June 2001

% Todo better zero_pad handle

if ~strcmp(impulse_responses_S.type_s,'FIR')
  disp(['WINDOWING : bad data type ' impulse_responses_S.type_s]);
  return;
end

windowed_impulse_responses_S = rmfield(impulse_responses_S,'content_m');

if exist('zero_pad_s')&strcmp(zero_pad_s,'pad')
  windowed_impulse_responses_S.content_m = ...
	zeros(size(impulse_responses_S.content_m));
  if length(window_extrema_v)==1
    windowed_impulse_responses_S.content_m(:,1:window_extrema_v) = ...
	  impulse_responses_S.content_m(:,1:window_extrema_v);
  else
    windowed_impulse_responses_S.content_m(:,window_extrema_v(1):window_extrema_v(2)) = ...
	  impulse_responses_S.content_m(:,window_extrema_v(1):window_extrema_v(2));  
  end
else
  if length(window_extrema_v)==1
    windowed_impulse_responses_S.content_m = ...
	  impulse_responses_S.content_m(:,1:window_extrema_v);
  else
    windowed_impulse_responses_S.content_m = ...
	  impulse_responses_S.content_m(:,window_extrema_v(1):window_extrema_v(2));
  end
end






