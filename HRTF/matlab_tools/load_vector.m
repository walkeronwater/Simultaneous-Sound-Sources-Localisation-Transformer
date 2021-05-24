function [samples_v,sampling_hz] = load_vector(file_name_s,format_s,channel_n);

% LOAD_VECTOR - Loads a vector from a sound / IR file
%
% Usage
%   [samples_v,sampling_hz] = load_vector(file_name_s,format_s);
%
% Input
%   file_name_s : a string giving the file name (without the format suffix)
%   format_s    : a string giving the read format
%   channel_n   : channel in file
%
% Output
%   samples_v   : the vector of samples
%   sampling_hz : sampling rate in hertz (0 if unknown)
%
% Formats
%   case of format_s must be the same as in file name
%   possible values for format_s can be :
%   'snd' 'SND' : not implemented - NeXT sound file
%   'rif' 'RIF' : RIF/INF format (Ircam AMS format)
%   'fil' 'FIL' : FIL/IFF format (Ircam AMS format)
%   'gmk' 'GMK' : WAV/INF format (Ircam temporary GMK format)
%   'dat' 'DAT' : MIT Media Lab KEMAR Head measurements format
%   'mat' 'MAT' : Matlab file - not implemented
%   'asc' 'ASC' : ASCII format - not implemented
%   'wav' 'WAV' : WAV format
%
% See also LOAD_IMPULSE_RESPONSES
%
% Authors
%   Larcher Veronique (was LOADVAR)
%   Rio Emmanuel
%   (c) Ircam - August 2002

samples_v = [];
sampling_hz = 0;
switch format_s
 case {'rif','RIF','fil','FIL'}
  rif_suffix_s = format_s;
  switch format_s
   case 'rif'
    inf_suffix_s = 'inf';
   case 'RIF'
    inf_suffix_s = 'INF';
   case 'fil'
    inf_suffix_s = 'iff';
   case 'FIL'
    inf_suffix_s = 'IFF';
  end
  file_name_inf_s = [file_name_s '.' inf_suffix_s];
  file_identifier=fopen(file_name_inf_s,'r','ieee-le');
  fseek(file_identifier,161,'bof');
  sampling_hz = fread(file_identifier,1,'float');
  coeff_f = fread(file_identifier,1,'float');
  length_n = fread(file_identifier,1,'uint');
  fclose(file_identifier);
  
  file_name_rif_s = [file_name_s '.' rif_suffix_s];
  file_identifier = fopen(file_name_rif_s,'r','ieee-le');
  samples_v = fread(file_identifier,'short');
  fclose(file_identifier);
  samples_v = samples_v*coeff_f;
 case {'gmk','GMK'}
  file_name_wav_s = [file_name_s '.wav'];
  [samples_v,sampling_hz]=wavread(file_name_wav_s);
  samples_v = samples_v(:,channel_n);
  % gain correction
  info_name_s = [file_name_s '.inf'];
  info_fid = fopen(info_name_s,'r');
  info_v = fscanf(info_fid,'%d, %f;');
  fclose(info_fid);
  channels_v = info_v(1:2:end);
  coeffs_v = info_v(2:2:end);
  coeff_f = coeffs_v(find(channels_v==channel_n));
  samples_v = samples_v*coeff_f;
 case {'dat','DAT'}
  % MIT Media Lab KEMAR Head measurements format
  % see http://sound.media.mit.edu/KEMAR.html
  % Other measurements done in March 1996
  % Is the sampling frequency always the same? (44100 Hz)
  file_name_s = [file_name_s '.' format_s];
  file_identifier = fopen(file_name_s,'r','ieee-be');
  samples_v = fread(file_identifier,inf,'short');
  fclose(file_identifier);
  sampling_hz = 44100;
 case {'mat','MAT'}
  eval(['load ',file_name_s,' ',format_s]);
  slashes = [0 findstr(file_name_s,'/')];
  file_name_s = file_name_s((max(slashes)+1):length(file_name_s));
  dots = [findstr(file_name_s,'.') length(file_name_s)+1];
  file_name_s = file_name_s(1:(min(dots)-1));
  samples_v = eval(file_name_s);
 case {'wav','WAV'}
  file_name_s = [file_name_s '.wav'];
  file_fid = fopen(file_name_s,'r','ieee-le');
  % >RIFF<
  wav_riff_s = fgets(file_fid,4);
  wav_file_size_n = fread(file_fid,1,'uint')+8;
  wav_wave_s = fgets(file_fid,4);
  % >fmt <
  wav_ck_id_s = fgets(file_fid,4);                        % ckID
  wav_chunk_size_n = fread(file_fid,1,'uint');            % nChunkSize
  wav_format_tag_n = fread(file_fid,1,'ushort');          % wFormatTag
  channels_n = fread(file_fid,1,'ushort');                % nChannels
  sampling_hz = fread(file_fid,1,'uint');                 % nSamplesPerSec
  wav_average_bytes_per_sec_n = fread(file_fid,1,'uint'); % nAvgBytesPerSec
  wav_block_align_n = fread(file_fid,1,'ushort');         % nBlockAlign
  wav_nb_bits_per_samples = fread(file_fid,1,'ushort');   % nBitsPerSample
  sample_size_n = wav_block_align_n/channels_n;
  fseek(file_fid,wav_chunk_size_n-16,'cof'); % reamining data
  % >data<
  wav_data_s = fgets(file_fid,4);
  length_n = fread(file_fid,1,'uint')/sample_size_n/channels_n;
  byte_samples_v = fread(file_fid,inf,'uint8');
  byte_samples_v = ...
	reshape(byte_samples_v,sample_size_n*channels_n,length_n)';
  samples_v = zeros(length_n,channels_n);
  for i=1:channels_n
    samples_v(:,i) = ...
	  byte_samples_v(:,((i-1)*sample_size_n+1):(i*sample_size_n)) ...
	  * (2.^(8*[0:(sample_size_n-1)]'));
  end;
  samples_v = mod(samples_v+2^(8*sample_size_n-1),2^(8*sample_size_n)) ...
	- 2^(8*sample_size_n-1);
  samples_v = samples_v/(2^(8*sample_size_n-1));
  fclose(file_fid);
 otherwise
  disp(['LOAD_VECTOR : ' format_s 'files support not implemented']);
end
samples_v = samples_v.';





