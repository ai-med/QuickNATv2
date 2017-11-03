classdef Unpooling < dagnn.Filter
  properties
    ID = -1;
  end

  methods

    function outputs = forward(self, inputs, params)
      [outputs{1}] = vl_nnunpool(inputs{1}, inputs{2}, inputs{3});
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnunpool(inputs{1}, inputs{2}, inputs{4},  derOutputs{1});
      derInputs{2} = 0*inputs{2};
      derInputs{3} = 0*inputs{3};
      derInputs{4} = 0*inputs{4};
      derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = inputSizes{1}(3) ;
    end

    function obj = Unpooling(varargin)
      obj.load(varargin) ;
    end
  end
end
