function YTV = vl_nnloss_TVReg(in2, out2, weights, dzdy)
% VL_NNLOSS  Loss with L2 + SSIM edge weighted with TV Regularization
%    Y = VL_NNLOSS(X, C) applies the the logistic loss to the data
%    X. X has dimension H x W x D x N, packing N arrays of W x H
%    D-dimensional vectors.
%
%    C contains the class labels, which should be integers in the range
%    1 to D. C can be an array with either N elements or with dimensions
%    H x W x 1 x N dimensions. In the fist case, a given class label is
%    applied at all spatial locations; in the second case, different
%    class labels can be specified for different locations.
%
%    DZDX = VL_NNLOSS(X, C, DZDY) computes the derivative DZDX of the
%    function projected on the output derivative DZDY.
%    DZDX has the same dimension as X.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


% no division by zero
in = in2 + single(1e-4) ;
out2 = out2 + single(1e-4) ;

sz = size(in);

% [~, in] = max(in2,[],3);

n = sz(4);
% For TV Regularization Cost
inPost = padarray(in, [1 1 0 0],'replicate','post');
inPre  = padarray(in, [1 1 0 0],'replicate','pre');
diffXPost = diff(inPost); diffYPost = diff(inPost,1,2);
diffXPost = diffXPost(:,1:end-1,:,:);
diffYPost = diffYPost(1:end-1,:,:,:);
t = sqrt(diffXPost.^2 + diffYPost.^2);
diffXPre = diff(inPre); diffYPre = diff(inPre,1,2);
diffXPre = diffXPre(:,1:end-1,:,:);
diffYPre = diffYPre(1:end-1,:,:,:);
tPreX = padarray(t, [1 0 0 0],'replicate','pre');
tPreX = tPreX(1:end-1,:,:,:);
tPreY = padarray(t, [0 1 0 0],'replicate','pre');
tPreY = tPreY(:,1:end-1,:,:);
if nargin <=3
    t(isnan(t)|isinf(abs(t))) = 0;
    YTV = weights.*sum(t(:))/numel(t);
    YTV(isnan(YTV)|isinf(YTV)) = 0;
else
    IWmat = out2(:,:,2,:);
    IWmat = repmat(IWmat, [1,1,sz(3),1]);
    YTV_ = (-(diffXPost + diffYPost)./t) + (diffXPre./tPreX) + (diffYPre./tPreY);
    YTV_(isnan(YTV_)|isinf(abs(YTV_))) = 0;
    YTV  =  weights.*IWmat.*single((YTV_.*dzdy) / n) ;
    YTV(isnan(YTV)|isinf(YTV)) = 0;
	%disp(['   TV gradient is ',num2str(mean(abs(YTV(:))))]);
end


end

