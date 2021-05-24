function [output_S] = fast_fourier_transform(input_S,number_of_points_n);

% FAST_FOURIER_TRANSFORM Does FFT or IFFT depending on input data format
%
% Usage
%   [output_S] = fast_fourier_transform(input_S,number_of_points_n);
%
% Input
%   input_S            : input data (of type 'FIR' or 'TRANSFER_FUNCTION')
%   number_of_points_n : FFT number of points (optional)
%
% Output
%   output_S : transformed data
%
% See also DATA_FORMATS
%
% Authors
%   Rio Emmanuel
%   (c) Ircam - June 2001

% Todo real part / imaginary part of the IFFT

output_S = rmfield(input_S,'content_m');
switch input_S.type_s
 case 'FIR'
  if ~exist('number_of_points_n')
    number_of_points_n = size(input_S.content_m,2);
  end
  output_S.content_m = fft(input_S.content_m,number_of_points_n,2);
  output_S.type_s = 'TRANSFER_FUNCTION';
 case 'TRANSFER_FUNCTION'
  output_S.content_m = real(ifft(input_S.content_m,[],2));
  output_S.type_s = 'FIR';
 otherwise
  disp(['FAST_FOURIER_TRANSFORM : bad data type ' input_S.type_s]);
  output_S.content_m = [];
  output_S.type_s = '';
end
