function [file_name_s,channel_n] = listen_write_token(string_args_C,elev_n,azim_n);

% LISTEN_WRITE_TOKEN Listen style file name composition function
%
% Usage
%   [file_name_s] = listen_write_token(string_args_C,elev_n,azim_n);
%
% Principle
%   may be passed as a function handle to LOAD_IMPULSE_RESPONSES
%
% Input
%   string_args_C : string elements {author, subject, radius}
%   elev_n        : elevation to be written
%   azim_n        : azimuth to be written
%
% Output
%   file_name_s : composed file name
%   channel_n   : channel in file (not used)
%
% See also LOAD_IMPULSE_RESPONSES
%
% Authors
%   Rio Emmanuel
%   (c) Ircam - July 2002

file_name_s = sprintf('%s_%4.4d_%s_R%4.4d_T%3.3d_P%3.3d',...
    string_args_C{1},...
    string_args_C{2},...
    string_args_C{3},...
    string_args_C{4}*100,...
    azim_n,...
    round(mod(elev_n,360)));
channel_n = 0;
