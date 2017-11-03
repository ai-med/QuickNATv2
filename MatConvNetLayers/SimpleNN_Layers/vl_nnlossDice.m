function Y = vl_nnlossDice(in, out, weights, dzdy)
% VL_NNLOSS  CNN log-loss
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
in = in + single(1e-4) ;
sz1 = size(in);
if(size(out,3)>1)
    IWmat = repmat(out(:,:,2,:), [1,1,sz1(3),1]);
    out = out(:,:,1,:);
end
sz2 = size(out);
outMod = zeros(sz2(1), sz2(2), sz1(3), sz2(4), 'like', in);
for i = 1:sz1(3)
    outMod(:,:,i,:) = single(out == i);
end


DiceNum = sum(sum((in.*outMod),1),2);
DiceDenom = sum(sum((in),1),2) + sum(sum((outMod),1),2) + eps;


if nargin <= 3
	Y = 1 - mean(mean((2*DiceNum)./DiceDenom)) ;
    Y(isinf(Y)|isnan(Y)) = 0;
else
    DiffNum = repmat(DiceDenom, [sz1(1), sz1(2), 1, 1]).*outMod - 2*repmat(DiceNum, [sz1(1), sz1(2), 1, 1]).*in;
    DiffDenom = repmat(DiceDenom.^2, [sz1(1), sz1(2), 1, 1]) + eps;
% 	Diff = -2*weights*IWmat.*(DiffNum./DiffDenom);
    Diff = -2*weights*(DiffNum./DiffDenom);
	Y = Diff.*dzdy ;
    Y(isinf(Y)|isnan(Y)) = 0;
   %disp(['  Dice gradient is ',num2str(mean(abs(Y(:))))]);
end
