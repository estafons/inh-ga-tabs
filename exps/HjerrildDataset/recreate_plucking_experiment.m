%%
% This script will recreate the final experiment that estimates guitar string,
% fret and plucking position on a recording with sudden changes of all these positions.
% The script serves the purpose of clearifying the implementation of the
% the proposed method entitled 
% "Estimation of guitar string, fret and plucking position using
% parametric pitch estimation" published at IEEE ICASSP 2019.
%
% It should help the reader understand how to implement the method and
% explain details not mentioned in the theory in the paper.
%
% Along with the script follows a dataset containing recording of electric
% and acoustic guitar, which was used for training and testing in the paper.
%
% If you find this code or data set useful, please cite the paper:
%
% ------------------------------------------------------------------------
% LaTeX citation format:
%
% @inproceedings{hjerrild_icassp2019,
%  title={Estimation of guitar string, fret and plucking position using parametric pitch estimation},
%  author={Hjerrild, Jacob M{\o}ller and Christensen, {Mads Gr\ae sb\o ll}},
%  booktitle= {Proc.\ IEEE Int.\ Conf.\ Acoust., Speech, Signal Process.},
%  year={2019}
%}
% ------------------------------------------------------------------------
% Implemented by Jacob Møller at the Audio Analysis Lab, Aalborg University
%  October 2018.
%%
clear all;
addpath(genpath('util'));
%mirverbose(0);
addpath mats

%% Load trained model (training data was captured from the given fret)
trainingFret = 4; % user can set the fret from which the audio is captured for training a model of the guitar strings.
modelFile = sprintf('trained_model_of_firebird_from_%1.0fth_fret',trainingFret);
load(modelFile);
% the model will contain the parameters of interest, which is:
% mu: the expected value of each class.(normalized and with dimension: K by
% 2), see i.e. eq. (19) and eq. (20)
% normalizationConstant: these are the values that has been used for
% normalizing mu. These two constants is the maximum value of the mean of
% the observations of omega0 and B.
% sigma: the mean of all variances in the normalized feature space.
% we note that the prior is uniform and is not included here.
%
% all other parameters in the loaded model file can potentially be of
% interest, but is not required for the proposed classifier.

%% Initialize implementation constants 
segmentDuration = 40e-3; % segment duration in seconds.
LOpen = 64.3; % assumed length of all open strings.
M = 25; % assumed high number of harmonics (M>>1). Used for inharmonic pitch estimation and for estimation of plucking position on the fretted string.
MInitial = 5; % number of harmonics for initial harmonic estimate (B=0).
f0Limits = [75 700]; % boundaries for f0 search grid in Hz.
nFFT = 2^19; % Length of  zero-padded FFT.
BRes = 1e-7; % resolution of search grid for B in m^(-2)
plausibilityFilterFlag = 0; % The user can apply a plausibility filter by setting this flag to 1.

%% read in the observed guitar recording and do onset detection
% util/recording_of_plucking_with_sudden_changes.wav

% frets = [0 1 2 3 4 5 6 7 8 9 10 11 12];
% strings = [1 2 3 4 5 6];
% recs = [0 1 2 3 4 5 6 7 8 9 10];

for rec = 1:10
    recording = int2str(rec);
    martin_i = strcat('/','martin',recording);
    for s = 1:6
        string = int2str(s)
        stringdir = strcat(martin_i,'/','string',string)
        for n = 0:12
            fret = int2str(n)
            wav = strcat(stringdir,'/',fret,'.wav');
            [recordedSignal,fs]=audioread(strcat('hjerrild_dataset/', wav));
            % segment the signal from every onset event (i.e. 40 ms)
            [segments, onsetsInSeconds] = icassp19_segment_from_all_onsets(recordedSignal,fs,segmentDuration); 

            fileID = fopen(strcat('hjerrild_dataset/',stringdir,'/',fret,'.txt'), 'w');
            fprintf(fileID,num2str(onsetsInSeconds(1)));
            fclose(fileID);
        end
    end
end
    
% [recordedSignal,fs]=audioread('hjerrild_dataset/martin2/string3/3.wav');
% [segments, onsetsInSeconds] = icassp19_segment_from_all_onsets(recordedSignal,fs,segmentDuration); 
% fileID = fopen(strcat('hjerrild_dataset//martin2/string3/3.txt'), 'w');
% disp(onsetsInSeconds(1))
% fprintf(fileID,num2str(onsetsInSeconds));
% fclose(fileID);

% 
% %% Estimate string, fret and plucking position on every 40 ms. segment
% for n = 1:size(segments,2)
%     % Hilbert transform and windowing
%     x = icassp19_hilbert_transform(segments(:,n),fs);
%     x = icassp19_apply_gaussian_window(x);
% 
%     %% Feature extraction with the inharmonic pitch estimator
%     % The implementation of Eq. (17) is done with one FFT, since it is fast
%     % Hence, it is equivalent to harmonic summation which the in the proposed 
%     % method is extended to inharmonic summation.
%     % See details on harmonic summation in Christensen and Jakobsson [27].
%     [~,X, ~] = icassp19_fft(x, fs, nFFT);
%     f0Initial = icassp19_harmonic_summation(X, f0Limits, MInitial, fs);
%     [pitchEstimatePhi, BEstimatePhi] = icassp19_inharmonic_summation(X, f0Initial, M, fs, BSearchGrid,nFFT);
%     disp(pitchEstimatePhi)
%     disp(BEstimatePhi)
%     % feature vector computed from the observation and normalized for
%     % euclidean distance. We use the trained model as part of normalization.
%     phi = [pitchEstimatePhi BEstimatePhi]./normalizationConstant;
%     % normalizationConstant: these are the values that has been used for normalizing mu.
% end
% %{
% 
%     %% Classifation of String and Fret (maximum likelihood w. uniform prior)
%     % the log is taken on frequencies to linearize the frequency wise distance between
%     % consequtive notes.
%     if plausibilityFilterFlag ~= 1,
%         % this is the classifier
%         euclideanDistance   =  sqrt( (log(phi(:,1))-log(mu(:,1))).^2 + (phi(:,2)-mu(:,2)).^2 );
%         [C,I] = min(euclideanDistance);
%         fretEstimate(n) = mod(I,13)-1; % <-- due to a matrix structure (13x6)
%         stringEstimate(n) = floor((I+13)/13);
%         if fretEstimate(n) == -1, fretEstimate(n)=fretOptions(end-1); stringEstimate(n)=stringEstimate(n)-1;end
%     
%     else
%         % this is the classifier with plausability filter (see Abesser et al. [7])
%         [sC, fC] = icassp19_obtain_pitch_candidates(pitchEstimatePhi,pitchModelOriginalUnits(:,:,trainingFret)');
%         ndx = (sC*13)-12+fC;
%         euclideanDistance   =  sqrt( (log(phi(:,1))-log(mu([ndx],1))).^2 + (phi(:,2)-mu([ndx],2)).^2 );
%         [C,I] = min(euclideanDistance);
%         stringEstimate(n) = sC(I);
%         fretEstimate(n) = fC(I);
%     end
%     %% Estimate the amplitudes (alpha vector)
%     Z = icassp19_Z(pitchEstimatePhi,length(x),fs,M,BEstimatePhi);
%     alpha = inv(Z'*Z)*Z'*x;
%     amplitudesAbs = abs(alpha)'; % absolute values for the estimator
% 
%     %% Plucking Position Estimator (minimizer of log spectral distance)
%     L = LOpen * 2^(-fretEstimate(n)/12); % length of vibrating part of the estimated string
%     pluckCmFromBridge(n) = icasssp19_plucking_position_estimator_LSD(amplitudesAbs,L);
%     fprintf('\n Estimating string, fret and plucking position for segment %1.0f out of %1.0f segments',n,size(segments,2));
% end
% 
% %% Plot the results and compare audio playback
% recordDuration = length(recordedSignal)/fs;
% timeAxis = [0:1/fs:recordDuration-1/fs];
% 
% figure(10); clf
% subplot(3,1,1)
% % plot the recorded signal in time domain
% plot(timeAxis,recordedSignal); ylabel('Ampl.'); 
% grid minor;
% set(gca,'xticklabel',[]);
% subplot(3,1,2)
% % plot the plucking poition estimates as distance from the bridge.
% plot(onsetsInSeconds(1:n),pluckCmFromBridge(1:n),'x');
% ylim([4 32])
% set(gca,'xticklabel',[]); ylabel('$\hat{P}$[cm]','interpreter','latex')
% grid minor;
% subplot(3,1,3)
% for p=1:6
%     % plot the backrground lines that represents guitar strings
%     plot([0,recordDuration],[p,p], 'Color', [0.4 0.4 0.4], 'linewidth',1); hold on;
% end
% ylim([0.1 6.9]);
% for nn = 1:n
%     % plot the estimated string and fret positions
%     text(onsetsInSeconds(nn)-0.2,stringEstimate(nn),sprintf('%1.0f',fretEstimate(nn)),'fontsize', 18)
% end
% set(gca,'ytick',[1 2 3 4 5 6])
% yticklabels({'1','2','3','4','5','6'})
% ylabel('String Est.'); xlabel('Time [sec]');
% 
% % listen to the recorded signal and take a look at the plot :)
% soundsc(recordedSignal,fs);
% %}