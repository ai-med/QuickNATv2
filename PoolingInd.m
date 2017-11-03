classdef PoolingInd < dagnn.Filter
  properties
    method = 'max'
    poolSize = [2 2]
    opts = {'cuDNN'}
    kernel = parallel.gpu.CUDAKernel('kernel_pooling.ptx','kernel_pooling.cu','my_poolingIndices');
    lastIndices = [];
  end

  methods

    function outputs = forward(self, inputs, params)
      % kernel setup
      self.kernel = parallel.gpu.CUDAKernel('kernel_pooling.ptx','kernel_pooling.cu','my_poolingIndices');
      self.kernel.GridSize(1) = size(inputs{1}, 3);
      self.kernel.GridSize(2) = size(inputs{1}, 4);
      self.kernel.ThreadBlockSize = [512, 1, 1];
      [outputs{1}, outputs{2}, outputs{3}, outputs{4}] = vl_nnpoolInd(inputs{1}, self.poolSize(1), [], self.pad, self.kernel, [], self.opts{:});
      self.lastIndices = outputs{2};
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      self.kernel = parallel.gpu.CUDAKernel('kernel_pooling.ptx','kernel_pooling.cu','my_poolingIndices');
      self.kernel.GridSize(1) = size(inputs{1}, 3);
      self.kernel.GridSize(2) = size(inputs{1}, 4);
      self.kernel.ThreadBlockSize = [512, 1, 1];
      derInputs{1} = vl_nnpoolInd(inputs{1}, self.poolSize(1), self.lastIndices, self.pad, self.kernel, derOutputs{1}, self.opts{:});
      derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = inputSizes{1}(3) ;
    end

    function obj = PoolingInd(varargin)
      obj.load(varargin) ;
    end
  end
end
