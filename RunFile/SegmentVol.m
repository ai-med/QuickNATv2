function [Predictions, SegTime] = SegmentVol(DataVol,NumFrames)

% The segmentation is done 2D slice-wise. Speed is dependent on number of frames you can push 'NumFrames'. This is dependent on GPU size. Please 
% try different values to optimize this for your GPU. In Titan Xp 12GB, 70 slices were pushed giving segmentation time of 20secs.

warning('off', 'all');
% Load the Trained Models
load('../TrainedModels/CoronalNet.mat'); % CoronalNet
fnet = dagnn.DagNN.loadobj(net);
load('../TrainedModels/AxialNet.mat'); % AxialNet
fnet2 = dagnn.DagNN.loadobj(net);
load('../TrainedModels/SagittalNet.mat') % SagittalNet
fnet3 = dagnn.DagNN.loadobj(net);

% Prepare the data for deployment in QuickNAT
sz = size(DataVol);
DataVol_Ax = permute(DataVol, [3,2,1]);
DataVol_Sag = permute(DataVol, [3,1,2]);
DataSelect = single(reshape(mat2gray(DataVol(:,:,:)),[sz(1), sz(2), 1, sz(3)]));
DataSelect_Ax = single(reshape(mat2gray(DataVol_Ax(:,:,:)),[sz(1), sz(2), 1, sz(3)]));
DataSelect_Sag = single(reshape(mat2gray(DataVol_Sag(:,:,:)),[sz(1), sz(2), 1, sz(3)]));
PredictionsFinal_Ax = [];
PredictionsFinal_Cor = [];
PredictionsFinal_Sag = [];

% ---- start of segmentation
tic
% 70 slices in one pass restricted by GPU space
for j= 1:NumFrames:256
    if(j==211)
        k=256;
    else
        k=j+NumFrames-1;
    end
    fnet3.move('cpu'); % GPU-CPU handshaking for space
    fnet.mode = 'test'; fnet.move('gpu');
    %     fnet.conserveMemory = 0;
    fnet.eval({'input', gpuArray(DataSelect(:,:,:,j:k))});
    reconstruction = fnet.vars(fnet.getVarIndex('prob')).value;
    reconstruction = gather(reconstruction);
    Predictions1 = squeeze(reconstruction);
    PredictionsFinal_Cor = cat(4,PredictionsFinal_Cor, Predictions1);
    
    fnet.move('cpu');
    fnet2.mode = 'test'; fnet2.move('gpu');
    %     fnet2.conserveMemory = 0;
    fnet2.eval({'input', gpuArray(DataSelect_Ax(:,:,:,j:k))});
    reconstruction = fnet2.vars(fnet2.getVarIndex('prob')).value;
    reconstruction = gather(reconstruction);
    Predictions1 = squeeze(reconstruction);
    PredictionsFinal_Ax = cat(4,PredictionsFinal_Ax, Predictions1);
    
    fnet2.move('cpu');
    fnet3.mode = 'test'; fnet3.move('gpu');
    %     fnet3.conserveMemory = 0;
    fnet3.eval({'input', gpuArray(DataSelect_Sag(:,:,:,j:k))});
    reconstruction = fnet3.vars(fnet3.getVarIndex('prob')).value;
    reconstruction = gather(reconstruction);
    Predictions1 = squeeze(reconstruction);
    PredictionsFinal_Sag = cat(4,PredictionsFinal_Sag, Predictions1);
    
end
PredictionsFinal_Ax = permute(PredictionsFinal_Ax, [4,2,3,1]);
PredictionsFinal_Sag = permute(PredictionsFinal_Sag, [2,4,3,1]);
% 16 class SagittalNet predictions converted to 28 class for consistency
PredictionsFinal_Sag = ReMapSagProbMap(PredictionsFinal_Sag);
% Multi-view Aggregation Stage
PredictionsFinal = (0.4*PredictionsFinal_Ax + 0.4*PredictionsFinal_Cor + 0.2*PredictionsFinal_Sag);
% Arg Max Stage for dense labelling
[~, Predictions] = max(PredictionsFinal,[],3);
Predictions = squeeze(Predictions);
SegTime = toc;
%---- end of Segmentation


