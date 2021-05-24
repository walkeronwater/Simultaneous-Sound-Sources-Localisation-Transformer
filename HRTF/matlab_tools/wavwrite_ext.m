function wavwrite_24(y,fs,nbits,wavefile)

% WAVWRITE_24 : write 24 bits WAV audio files
%
%	wavwrite_24(y,fs,nbits,wavefile)
% 
%	x	: vector to be output
%	sr      : sampling rate
%	res     : resolution : 8, 16(default), 24 or 32
%	name	: filename with or without extension. Complete filename must be 
%		  provided (path included)
%
% (c) Guillaume VANDERNOOT - IRCAM - 2002


filedata	= y;
fileName	= wavefile;
fileSR		= fs;
switch nbits
      case 8
	    fileRes	= 'int8';
      case 16
	    fileRes	= 'int16';
      case 24
	    fileRes	= 'bit24';
      case 32
	    fileRes	= 'int32';
      otherwise
	    fprintf(1,'Data types incompatible with WAV file format or not yet supported\n');
	    fprintf(1,'Using default resolution : 16 bits\n');
	    fileRes	= 'int16';
end

% Multichannel
[numch,n]		= size(filedata);
if numch > n
      filedata	= filedata';
      [numch,n]	= size(filedata);
end

% Nomber of bits per sample
switch fileRes
      case 'int8'
	    samplesize	= 8;
      case 'int16'
	    samplesize	= 16;
      case 'bit24'
	    samplesize	= 24;
      case 'int32'
	    samplesize	= 32;
end

% Quantization
tempvec	= quantize(filedata,samplesize);

% Opening the target file
fid	= fopen(fileName,'wb','ieee-le');
if (fid == -1)
      fprintf(1,'There was a problem creating the output file. Check your machine,\n') 
      fprintf(1,'or have a serious talk with the system administrator !\n');
else
      fprintf(1,'Writing WAV file : %s\n',fileName);
      % ===================== Format RIFF =====================
      nsamples	= numch*n;
      riffsize	= 36 + (nsamples*samplesize/8);

      % write riff chunk
      fwrite(fid,'RIFF','uchar');
      fwrite(fid,riffsize,'ulong');
      fwrite(fid,'WAVE','uchar');

      % write format sub-chunk
      fwrite(fid,'fmt ','uchar');
      fwrite(fid,16,'ulong');
      fwrite(fid,1,'ushort');			% PCM format 
      fwrite(fid,numch,'ushort');		% number of channel
      fwrite(fid,fileSR,'ulong');		% number of samples per second
      fwrite(fid,fileSR*samplesize/8,'ulong');	% number of bytes per second
      fwrite(fid,samplesize/8*numch,'ushort');	% block alignment
      fwrite(fid,samplesize,'ushort');		% number of bits per sample

      % write data sub-chunck
      fwrite(fid,'data','uchar');
      fwrite(fid,nsamples*samplesize/8,'ulong');% total number of bytes
      fwrite(fid,tempvec(:),fileRes);

      fclose(fid);
end

% --------------------------------------------------------------------------------

function y = quantize(x,samplesize)

% Number of channels
ch	= size(x,1);

% Limits
MAX	= pow2(samplesize  - 1);
LIM	= [-1 ; 1] .* MAX; LIM(2) = LIM(2) - 1;

% Integer conversion
y	= round(x .* MAX);

% Clipping
for i=1:ch
      clipmsg= '';
      ipb	= find(y(i,:) < LIM(1));
      if ~isempty(ipb)
	    clipmsg	= 'Data clipped during write to file (min < -1) !\n';
	    y(i,ipb)	= LIM(1);
      end
      ipb	= find(y(i,:) > LIM(2));
      if ~isempty(ipb)
	    clipmsg	= 'Data clipped during write to file (max >= 1) !\n';
	    y(i,ipb)	= LIM(2);
      end
end

if ~isempty(clipmsg)
      fprintf(1,clipmsg);
end
