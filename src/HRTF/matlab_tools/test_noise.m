function [sound_v] = test_noise(type_s,duration_n,sampling_hz,frequency_hz);

% TEST_NOISE Computes various noises for test purpose
%
% Usage
%   [sound_v] = test_noise(type_s,duration_n,sampling_hz,frequency_hz);
%
% Input
%   type_s      : 'White', 'Tone' or 'Modulated'
%   duration_n  : duration in samples
%   sampling_hz : sample rate in hertz
%        (mandatory for 'Tone' and 'Modulated')
%   frequency_hz : characteristic frequency of the sound
%        (optional)
%
% Output
%   sound_v : synthetized sound (column vector)
%
% Types
%   'White' : white noise
%   'Tone' : pure sinusoidal tone (at frequency frequency_hz)
%   'Modulated' : modulated pink noise (modulation frequency is
%   frequency_hz) 
% 
% Authors
%   Rio Emmanuel
%   (c) Ircam - May 2002

switch (type_s)
 case 'White'
  sound_v = 2*rand(duration_n,1)-1;
 case 'Tone'
  if ~exist('frequency_hz')
    frequency_hz=440;
  end;    
  sound_v = sin(2*pi*frequency_hz*[0:duration_n-1]'/sampling_hz);
 case 'Modulated'
  if ~exist('frequency_hz')
    frequency_hz=20;
  end;    
  sound_v = (2*rand(duration_n,1)-1) ...
      .*sin(2*pi*frequency_hz*[0:duration_n-1]'/sampling_hz);
end;
