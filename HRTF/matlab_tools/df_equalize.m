function df_equalize(subject_n,prompt_input_s,raw_mat_directory_s,comp_wav_directory_s,comp_mat_directory_s);

%	df_equalize(subject_n,prompt_input_s,raw_mat_directory_s,comp_wav_directory_s,comp_mat_directory_s);
%
%	subject_n		: number of the subject (e.g., 1002)
%	prompt_input_s		: automatic saving of the Matlab files ('Auto')
%	raw_mat_directory_s	: path of the raw HRIR WAV files (ending with separator)
%	comp_wav_directory_s	: path of the compensated HRIR WAV files (ending with separator)
%	comp_mat_directory_s	: path of the compensated HRIR unique Matlab file (ending with separator)
%
% Load HRIR from Matlab file, window HRIR and perform diffuse field equalization,
% and save compensated HRIR (WAV files and unique Matlab file)


% DEFAULT VALUES
security_interval_n=30;
window_length_n=512;
prompt_input_b=1;
save_compensated_wav_b = 1;

if exist('prompt_input_s') & strcmp(prompt_input_s,'Auto')
  prompt_input_b=0;
end

if ~exist('raw_mat_directory_s')
  raw_mat_directory_s = ...
	'RAW/MAT/';
end

if ~exist('comp_wav_directory_s')
  comp_wav_directory_s = ...
	'COMPENSATED/WAV/';
end;

if ~exist('comp_mat_directory_s')
  comp_mat_directory_s = ...
	'COMPENSATED/MAT/';
end;

% GENERAL PARAMETERS
if ~exist('subject_n')
  subject_n = input('Enter subject''s ID : ');
end;
subject_s = num2str(subject_n);
[elev_v,azim_v] = measure_set('Listen');

% LOADING MAT FILES
load([raw_mat_directory_s filesep subject_s '.mat'],'l_hrir_S','r_hrir_S');

% WINDOWING
if prompt_input_b
  security_interval_input_n = input(sprintf( ...
	'  Security interval [%d] : ',security_interval_n));
  if ~isempty(security_interval_input_n)
    security_interval_n = round(max(0,security_interval_input_n));
  end;
  window_length_input_n = input(sprintf( ...
	'  Window length [%d] : ',window_length_n));
  if ~isempty(window_length_input_n)
    window_length_n = round(max(1,window_length_input_n));
  end;
end;

first_onset_n = first_onset(l_hrir_S,r_hrir_S);
window_extrema_v = first_onset_n-security_interval_n+[0 window_length_n-1];
l_hrir_S = windowing(l_hrir_S,window_extrema_v);
r_hrir_S = windowing(r_hrir_S,window_extrema_v);

if prompt_input_b
  view_windowed_b = -1;
  while (view_windowed_b==-1)
    view_windowed_b = input('Inspect windowed impulse responses? [y] : ','s');
    switch view_windowed_b
     case {'y','Y',''}
      view_windowed_b = 1;
     case {'n','N'}
      view_windowed_b = 0;
     otherwise
      view_windowed_b = -1;
    end;
  end;

  if view_windowed_b
    to_display_S.elev_v = l_hrir_S.elev_v;
    to_display_S.azim_v = l_hrir_S.azim_v;
    to_display_S.l_hrir_m = l_hrir_S.content_m;
    to_display_S.r_hrir_m = r_hrir_S.content_m;
    gui_plot_position(to_display_S);
    clear to_display_S;
  end;
end;

% DIFFUSE-FIELD EQUALIZATION
[l_mag_S,l_phase_S] = ...
      phase_decomposition(fast_fourier_transform(l_hrir_S));
[r_mag_S,r_phase_S] = ...
      phase_decomposition(fast_fourier_transform(r_hrir_S));
l_eq_mag_S = diffuse_field_equalize(l_mag_S,'Surface');
r_eq_mag_S = diffuse_field_equalize(r_mag_S,'Surface');
l_hrir_S = ...
      fast_fourier_transform(phase_recomposition(l_eq_mag_S,l_phase_S));
r_hrir_S = ...
      fast_fourier_transform(phase_recomposition(r_eq_mag_S,r_phase_S));

% NORMALIZATION & WRITING
if prompt_input_b
  save_compensated_wav_b = -1;
  while (save_compensated_wav_b==-1)
    save_compensated_wav_b = input('Save compensated wav files? [y] : ','s');
    switch save_compensated_wav_b
     case {'y','Y',''}
      save_compensated_wav_b = 1;
     case {'n','N'}
      save_compensated_wav_b = 0;
     otherwise
      save_compensated_wav_b = -1;
    end;
  end;
end;

if save_compensated_wav_b
  mkdir(comp_wav_directory_s,subject_s);
  comp_wav_directory_s = [comp_wav_directory_s filesep subject_s filesep];
  max_level_f = max(max(max(abs(l_hrir_S.content_m))), ...
	max(max(abs(r_hrir_S.content_m))))*1.01;
  l_hrir_S.content_m = l_hrir_S.content_m/max_level_f;
  r_hrir_S.content_m = r_hrir_S.content_m/max_level_f;
  other_args_C = {'IRC' subject_n 'C' 1.95};
  for index_n=1:length(elev_v)
    [file_name_s] = ...
	  listen_write_token(other_args_C,elev_v(index_n),azim_v(index_n));
     wavwrite_ext(...
	   [l_hrir_S.content_m(index_n,:); r_hrir_S.content_m(index_n,:)],44100,24,...
	   [comp_wav_directory_s file_name_s '.wav']);
    save([comp_mat_directory_s filesep subject_s '.mat'],'l_hrir_S','r_hrir_S');
  end;
end;
