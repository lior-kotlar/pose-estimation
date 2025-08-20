function [ampSpec, f_half] = mySpectrum(x, Fs, plotFlag, saveFigFlag) 
%
% Calculate the Fourier amplitude spectrum of the signal x
%
% inputs
%   x           - a signal (assumed to consist of real numbers)
%   Fs          - sampling frequency of x
%   plotFlag    - true/false flag, whether to generate plots
%   saveFigFlag - true/false flag, whether to save fig to png
%
% outputs
%   ampSpec - the ampliude spectrum of x
%   f_half  - the frequency vector corresponding to ampSpec
%
% Tsevi Beatus, HUJI Course no. 78852.
%
if ~exist('plotFlag','var')
    plotFlag = true ;
end

if ~exist('saveFigFlag','var')
    saveFigFlag = false ;
end

N      = length(x) ;
modes  = 0:N-1 ;
% find number of the highest mode (account for odd/even N)
N_fastest = ceil((N+1)/2) ;  

% set mode number of negative modes (account for odd/even N)
modes(N_fastest+1 : end) = modes(N_fastest+1 : end) - N ;
f = modes * Fs / N ;  % full frequency vector

% frequencies corresponding to spectrum (i.e. to modes with n>=0)
f_half = f(1:N_fastest) ; 

Y = fft(x); % at last, perform the Fourier Transform

P2 = abs(Y/N);  % calc amplitude of entire Y and normalize by 1/N
ampSpec = P2(1:N_fastest); % take the n>=0 half of the spectrum

d = mod(N,2)==1 ;
ampSpec(2:end-d) = 2*ampSpec(2:end-d);
% explanation for the above two lines:
% multiply by 2 only those modes that appear twice in fft
% either way, do not x2 the first term. 
% the last term depends whether N is odd/even
% the next two lines can handle both odd and even N 
% if N is ODD we get d=1, then we do not x2 last term
% if N is EVEN we get d=0, then we x2 all terms


%{
% the above line is equivalent to
if mod(N,2)==0
    ampSpec(2:end-1) = 2*ampSpec(2:end-1); % for even N 
else
    ampSpec(2:end) = 2*ampSpec(2:end);     % for odd N 
end
%}

if plotFlag
    
    ind = 1:N_fastest;
    t_vec = (0:N-1) / Fs ; % time vector
    
    figure('color','w','position',[500 100  500 400]) ; hold on ;
    plot(f_half, ampSpec) ; set(gca,'fontsize',14) ;
    xlabel('f [Hz]'); ylabel('Amplitude') ; title('Amplitude spectrum') ;
    box on ; grid on ;
    if saveFigFlag
        print(gcf,'Fig_1_Amplitude_Spectrum','-dpng','-r0') ;
    end
    
    
    figure('color','w','position',[200 100 500 600]) ;
    subplot(2,1,1) ; plot(real(Y),'b-') ; set(gca,'fontsize',14) ;
    xlabel('index') ; ylabel('Real part') ; title('FFT output') ;
    ylim([-200,200]) ;
    subplot(2,1,2) ; plot(imag(Y),'r-') ; set(gca,'fontsize',14) ;
    xlabel('index') ; ylabel('Imaginary part') ; title('FFT output') ;
    if saveFigFlag
        print(gcf,'Fig_2_Raw_FFT','-dpng','-r0') ;
    end
    
    figure('color','w','position',[300 100 500 600]) ;
    subplot(2,1,1) ; plot(f_half, real(Y(ind)),'b-') ; set(gca,'fontsize',14) ;
    xlabel('Frequency [Hz]') ; ylabel('Real part') ; title('FFT output') ;
    subplot(2,1,2) ; plot(f_half, imag(Y(ind)),'r-') ; set(gca,'fontsize',14) ;
    xlabel('Frequency [Hz]') ; ylabel('Imaginary part') ; title('FFT output') ;
    if saveFigFlag
        print(gcf,'Fig_3_Raw_FFT_vs_Freq','-dpng','-r0') ;
    end
    
    h=figure('color','w','position',[600 100  500 600]) ; 
    subplot(2,1,1) ; plot(t_vec, x) ; set(gca,'fontsize',14) ;
    xlabel('Time [sec]'); ylabel('Signal') ; title('Signal') ;
    box on ; grid on ;
    mm = min ([length(f_half), 500]) ;
    subplot(2,1,2) ; plot(f_half(1:mm), ampSpec(1:mm)) ; set(gca,'fontsize',14) ;
    xlabel('f [Hz]'); ylabel('Amplitude') ; title('Amplitude spectrum') ;
    box on ; grid on ;
    if saveFigFlag
        print(gcf,'Fig_4_Signal_and_Amplitude_Spectrum','-dpng','-r0') ;
    end  
end
return

