function [weights_v] = spatial_sampling_weighting(elev_v,azim_v,method_s);

% SPATIAL_SAMPLING_WEIGHTING Returns weights for a given spatial distribution
%
% Usage
%   [weights_v] = spatial_sampling_weighting(elev_v,azim_v,method_s);
%
% Input
%   elev_v   : elevations (sorted in increasing order)
%   azim_v   : azimuths
%   method_s : 'Surface' or 'Angle'
%
% Output
%   weights_v : corresponding weights (see DIFFUSE_FIELD_EQUALIZATION)
%
% Methods
%   Weight corresponding to a specific elevation is given by the surface of
% the domain related to this elevation. The 'Surface' method consists in a
% whole tiling of the shere : the limit between domains of two different
% elevations is set at the middle elevation. The 'Angle' method computes
% arbitrary domains surrounding each elevation (+/- 5 degrees).
%
% See also DATA_FORMATS, DIFFUSE_FIELD_EQUALIZATION
%
% Authors
%   Rio Emmanuel
%   (c) Ircam - June 2001

% inits variables
weights_v = [];
prev_elev_n = -90;
index_n = 1;
% goes through elev_v vector
while index_n<=length(elev_v)
  % current elevation charcateristics
  % i.e. value, indices in elev_v & number of points
  elev_n = elev_v(index_n);
  indices_nv =  find(elev_v==elev_n);
  length_n = length(indices_nv);
  % computation of the inf (down_elev_n) and sup (up_elev_n)
  % limits of the surface
  switch method_s
   case 'Surface'
    % upper limit is half the distance to the next elevation
    if (index_n+length_n-1)<length(elev_v)
      next_elev_n = elev_v(index_n+length_n);
      up_elev_f = (elev_n+next_elev_n)/360*pi;
    else
      up_elev_f = pi/2;
    end
    if (index_n~=1)
      % down limit is half the distance to the previous elevation
      down_elev_f = (elev_n+prev_elev_n)/360*pi;
    else
      % for the first elevation, it is symmetric
      % to avoid a too big importance of lowest elevation
      down_elev_f = (3*elev_n-next_elev_n)/360*pi;
    end;
    prev_elev_n = elev_n;
   case 'Angle'
    % upper limit is current elevation + 5 degrees
    if elev_n<=85
      up_elev_f = (elev_n+5)/180*pi;
    else
      up_elev_f = pi/2;
    end
    % down limit is current elevation - 5 degrees
    down_elev_f = (elev_n-5)/180*pi;
  end
  % surface of the domain = a solid angle
  solid_angle_f = 2*pi*(sin(up_elev_f)-sin(down_elev_f));
  % a vector of boolean get rid of the 360 degrees azimuth point
  weighting_bv = ones(length_n,1);
  if azim_v(index_n)==0 & azim_v(index_n+length_n-1)==360
    weighting_bv(end) = 0;
  end
  % weights for the current indices
  weights_v(indices_nv,1) = solid_angle_f*weighting_bv/sum(weighting_bv);
  % prepares next elevation
  index_n = index_n+length_n;
end
% normalization
weights_v = weights_v/sum(weights_v);

