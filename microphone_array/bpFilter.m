function y = bpFilter(x)
%DOFILTER Filters input x and returns output y.

% MATLAB Code
% Generated by MATLAB(R) 9.1 and the DSP System Toolbox 9.3.
% Generated on: 21-Apr-2019 22:22:00

persistent Hd;

if isempty(Hd)
    
    Fstop1 = 300;    % First Stopband Frequency
    Fpass1 = 500;    % First Passband Frequency
    Fpass2 = 3200;   % Second Passband Frequency
    Fstop2 = 3400;   % Second Stopband Frequency
    Astop1 = 60;     % First Stopband Attenuation (dB)
    Apass  = 1;      % Passband Ripple (dB)
    Astop2 = 60;     % Second Stopband Attenuation (dB)
    Fs     = 44100;  % Sampling Frequency
    
    h = fdesign.bandpass('fst1,fp1,fp2,fst2,ast1,ap,ast2', Fstop1, Fpass1, ...
        Fpass2, Fstop2, Astop1, Apass, Astop2, Fs);
    
    Hd = design(h, 'equiripple', ...
        'MinOrder', 'any');
    
    
    
    set(Hd,'PersistentMemory',true);
    
end

y = filter(Hd,x);

