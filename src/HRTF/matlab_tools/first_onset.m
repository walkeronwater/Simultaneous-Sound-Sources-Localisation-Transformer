function [first_onset_n] = first_onset(l_hrir_S,r_hrir_S);

% FIRST_ONSET Detects first onset of impulse responses
%
% Usage
%   [first_onset_n] = first_onset(l_hrir_S,r_hrir_S);
%
% Input
%   l_hrir_S : left impulse responses structure (of type 'FIR')
%   r_hrir_S : right impulse responses structure (optional)
%
% Output
%   first_onset_n : indice of the first onset computed on both sets of
%                   impulse responses
%                   if r_hrir_S is not given, computation is done on
%                   l_hrir_S only.
%
% Authors
%   Rio Emmanuel
%   (c) Ircam - September 2002

if ~strcmp(l_hrir_S.type_s,'FIR')
  disp(['FIRST_ONSET : bad data type ' l_hrir_S.type_s]);
  first_onset_n = 0;
  return;
end;

first_onset_n = first_onset_1(l_hrir_S);
if exist('r_hrir_S')
  if ~strcmp(r_hrir_S.type_s,'FIR')
    disp(['FIRST_ONSET : bad data type ' r_hrir_S.type_s]);
    return;
  end;
  first_onset_n = min(first_onset_n,first_onset_1(r_hrir_S));
end;

function [first_onset_n] = first_onset_1(hrir_S)

[dummy_v,onsets_v] = max(abs(hrir_S.content_m) ...
    ==(max(abs(hrir_S.content_m),[],2) ...
    *ones(1,size(hrir_S.content_m,2))),[],2);
first_onset_n = min(onsets_v);
