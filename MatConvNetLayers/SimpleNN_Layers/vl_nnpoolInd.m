function [Y, Ind, s, s2] = vl_nnpoolInd(X, POOL, Ind, pad, Kernel, dzdx, varargin)
	% forward mode
	if(isempty(dzdx))
		% initial padding
        if(sum(pad) > 0)
            X1 = [gpuArray.zeros(pad(1), size(X, 2), size(X, 3), size(X, 4), 'single'); X; gpuArray.zeros(pad(2), size(X, 2), size(X, 3), size(X, 4), 'single')];        
        
            XP = [gpuArray.zeros(size(X1, 1), pad(3), size(X1, 3), size(X1, 4), 'single'), X1, gpuArray.zeros(size(X1, 1), pad(4), size(X1, 3), size(X1, 4), 'single')];
            Y = gpuArray.zeros(ceil(size(XP, 1)/POOL), ceil(size(XP, 2)/POOL), size(XP, 3), size(XP, 4), 'single');
            Ind = gpuArray.zeros(numel(Y), 1, 'int32');

            [Y, Ind] = feval(Kernel, Y, Ind, XP, int32(size(XP, 1)), int32(size(XP, 2)), int32(size(XP, 3)), int32(size(XP, 4)), POOL);
        else
            Y = gpuArray.zeros(ceil(size(X, 1)/POOL), ceil(size(X, 2)/POOL), size(X, 3), size(X, 4), 'single');
            Ind = gpuArray.zeros(numel(Y), 1, 'int32');

            %size(Y)
            %size(X)
            %size(Ind)
            [Y, Ind] = feval(Kernel, Y, Ind, X, int32(size(X, 1)), int32(size(X, 2)), int32(size(X, 3)), int32(size(X, 4)), POOL);
     
        end
            
        s = size(X);
	s2 = size(Y);
	% backward mode
    else

		Y = gpuArray.zeros(size(X), 'single');
		Y(Ind+1) = dzdx;
		Y = (Y((pad(1)+1):(end-pad(2)), (pad(3)+1):(end-pad(4)), :, :));
		Ind = [];
        s = [];
	s2 = [];
	end
end
