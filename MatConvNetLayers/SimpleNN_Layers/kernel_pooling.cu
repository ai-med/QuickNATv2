#include <stdio.h>

__global__ void my_pooling(float* const out, float const* const data, size_t const H, size_t const W, size_t const C, size_t const N, size_t const stride) 
{
	//channel
	const size_t d = blockIdx.x;
	//image number	
	const size_t n = blockIdx.y;

	// spatial dimensions of the output array
	const size_t H_out = (size_t) ceilf(H/stride);
	const size_t W_out = (size_t) ceilf(W/stride);
	const size_t R = H_out * W_out;
	const size_t regTi = (size_t) ceilf(R / blockDim.x);


	printf("regTi=%d\n", regTi);

	// for each of the regions assigned to the current thread
	for(size_t reg=regTi*threadIdx.x; reg < min((int)regTi*(threadIdx.x+1),(int) R); reg++)
	{
		//get the base (v,u) positions of the input Image
		size_t vIN = stride * ( reg % H_out ); 
		size_t uIN = stride * floorf( reg / H_out );

		// -inf
		float max_ =  __int_as_float(0xff800000);
		for(size_t v_ = vIN; v_ < min(int(vIN + stride), int(H)); v_++)
		{
			for(size_t u_ = uIN; u_ < min(int(uIN + stride), int(W)); u_++)
			{
				size_t indIN = n*(H*W*C) + d*(H*W) + uIN*(H) + vIN;
				if(data[indIN] > max_)
					max_ = data[indIN];
			}
		}

		//assign result to output
		size_t vOUT = reg % H_out;
		size_t uOUT = floorf(reg / H_out);
		size_t indOUT = n*(H_out*W_out*C) + d*(H_out*W_out) + uOUT*(H_out) + vOUT;
		out[indOUT] = max_;
	}
}

__global__ void my_poolingIndices(float* const outMax, int* const outIndices, float const* const data, size_t const H, size_t const W, size_t const C, size_t const N, size_t const stride) 
{
	//channel
	const size_t d = blockIdx.x;
	//image number	
	const size_t n = blockIdx.y;

	// spatial dimensions of the output array
	const size_t H_out = (size_t) ceil(double(H)/double(stride));
	const size_t W_out = (size_t) ceil(double(W)/double(stride));
	const size_t R = H_out * W_out;
	const size_t regTi = (size_t) ceil(double(R) / double(blockDim.x));

	// for each of the regions assigned to the current thread
	for(size_t reg=regTi*threadIdx.x; reg < min((int)regTi*(threadIdx.x+1),(int) R); reg++)
	{
		//get the base (v,u) positions of the input Image
		size_t vIN = stride * ( reg % H_out ); 
		size_t uIN = (size_t)stride * floor( double(reg) / double(H_out) );

		// -inf
		float max_ =  __int_as_float(0xff800000);
		int maxIdx_ = -1;
		for(size_t v_ = vIN; v_ < min(int(vIN + stride), int(H)); v_++)
		{
			for(size_t u_ = uIN; u_ < min(int(uIN + stride), int(W)); u_++)
			{
				size_t indIN = n*(H*W*C) + d*(H*W) + u_*(H) + v_;
				if(data[indIN] > max_)
				{
					max_ = data[indIN];
					maxIdx_ = indIN;
				}
			}
		}

		//assign result to output
		size_t vOUT = reg % H_out;
		size_t uOUT = (size_t)floor(double(reg) / double(H_out));

		size_t indOUT = n*(H_out*W_out*C) + d*(H_out*W_out) + uOUT*(H_out) + vOUT;
		outMax[indOUT] = max_;
		outIndices[indOUT] = maxIdx_;
	}
}


__global__ void test(int* const deb, float const* const data)
{
		size_t const stride = 2;
		size_t reg = 93;
		const size_t H_out = (size_t) ceil(double(360)/double(stride));
		size_t uIN = (size_t)stride * floor( double(reg) / double(H_out) );

		printf("uIN = %d, H_out = %d\n", uIN, H_out);

		deb[0] = uIN;
		deb[1] = H_out;
}

