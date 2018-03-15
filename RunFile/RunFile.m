% To Run the File your current directory has to be the RunFile Directory as all the paths are defined relative to it

clc
clear
close all
%% Load the Data
% Enter the path to your data
DataPath = '../SampleData/';
addpath('../QuickNAT_Networks');
% FileName
FileName = 'SampleData.mgz'; % This is a sample Data from OASIS Dataset not used in training the model
% Use your Loader to load MRI of isotropic resolution (I use MRIread from FreeSurfer)
% The volume has to be resampled to isotropic resolution (256*256*256) with Coronal axis as dimension one i.e (dimension 1 and 2 provides coronal slices)
% I recommend 'mri-convert --conform' routine in FreeSurfer to pre-process which takes less than a second.
DataVol = MRIread([DataPath,FileName]);
Data = DataVol.vol;
% Run QuickNAT
[Predictions, SegTime] = SegmentVol(Data,70); % Change the 70 based on your GPU RAM size
disp(['----Processing Over. Segmentation Time is ',num2str(SegTime)]);
% Copy the header of original File to Pred File for consistency
PredVol = DataVol;
PredVol.vol = Predictions-1;

% save to Directory
err = MRIwrite(PredVol, [DataPath,FileName(1:end-4),'_Pred.mgz']);

