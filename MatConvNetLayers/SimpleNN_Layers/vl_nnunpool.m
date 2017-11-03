function [Y] = vl_nnunpool(X, Ind, sizeY, varargin)
	% forward mode
	if(isempty(varargin))

        Y = gpuArray.zeros(sizeY(1), sizeY(2), size(X, 3), size(X, 4), 'single');
		Y(real(Ind+1)) = X;
		
	% backward mode
	else
		DZDX = (varargin{1});
		Y = reshape(DZDX(Ind+1), sizeY);
	end
end
