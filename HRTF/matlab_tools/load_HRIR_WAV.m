function [l_hrir_S,r_hrir_S]=load_HRIR_WAV(subject_n,prompt_input_s,wav_directory_s,mat_directory_s);

%	[l_hrir_S,r_hrir_S]=load_HRIR_WAV(subject_n,prompt_input_s,wav_directory_s,mat_directory_s);
%
%	subject_n	: number of the subject (e.g., 1002)
%	prompt_input_s	: automatic saving of the Matlab files ('Auto')
%	wav_directory_s	: path of the HRIR WAV files (ending with separator)
%	mat_directory_s	: path of the HRIR unique Matlab file (ending with separator)
%
% Load HRIR from WAV files, and save a Matlab file

prompt_input_b=1;
if exist('prompt_input_s') & strcmp(prompt_input_s,'Auto')
  prompt_input_b=0;
end

if ~exist('wav_directory_s')
  wav_directory_s = ...
	'RAW/WAV/';
end

if ~exist('mat_directory_s')
  mat_directory_s = ...
	'RAW/MAT/';
end

% GENERAL PARAMETERS
if ~exist('subject_n')
  subject_n = input('Enter subject''s ID : ');
end;
subject_s = num2str(subject_n);
wav_directory_s = [wav_directory_s filesep subject_s];
write_token_fh = 'listen_write_token';
format_s = 'wav';
other_args_C = {'IRC' subject_n 'R' 1.95};
[elev_v,azim_v] = measure_set('Listen');

% LOADING
[l_hrir_S,r_hrir_S] = ...
      load_impulse_responses(wav_directory_s,write_token_fh,other_args_C,elev_v,azim_v,format_s);

% SAVE RAW MAT FILE
if prompt_input_b
  save_mat_b = -1;
  while (save_mat_b==-1)
    save_mat_b = input('Save raw mat files? [y] : ','s');
    switch save_mat_b
     case {'y','Y',''}
      save_mat_b = 1;
     case {'n','N'}
      save_mat_b = 0;
     otherwise
      save_mat_b = -1;
    end
  end
end

if save_mat_b
  save([mat_directory_s filesep subject_s '.mat'],'l_hrir_S','r_hrir_S');
end
