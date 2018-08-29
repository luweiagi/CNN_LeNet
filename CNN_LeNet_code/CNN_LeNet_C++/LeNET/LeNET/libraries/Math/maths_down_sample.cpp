#include <maths_down_sample.h>
#include <maths.h>


// 采样的尺寸变化规则是：5x5的图片，以2的倍率降采样，其实相当于4x4的图片来降采样。
Array3Dd down_sample(const Array3Dd &vector_array, const int sample_num, down_sample_type sample_type)
{
	switch (sample_type)
	{
		case MeanPooling:
		{
			return down_sample_mean_pooling(vector_array, sample_num);
			break;
		}
		case MaxPooling:
		{
			return down_sample_max_pooling(vector_array, sample_num);
			break;
		}
		case StochasticPooling:
		{
			return down_sample_stochastic_pooling(vector_array, sample_num);
			break;
		}
		default:
		{
			cout << "down sample type is not defined yet!" << endl << "down_sample() failed!" << endl;
			Array3Dd temp;
			return temp;
			break;
		}
	}
}


Array3Dd down_sample_mean_pooling(const Array3Dd &vector_array, const int sample_num)
{
	Array2Dd mean_pooling_convolution_kernel(sample_num, sample_num, double(1.0 / pow(sample_num, 2)));

	Array3Dd Z = convolution_n_dim(vector_array, mean_pooling_convolution_kernel);

	Array3Dd Y = Z.sampling(sample_num);

	return Y;
}


Array3Dd down_sample_max_pooling(const Array3Dd &vector_array, const int sample_num)
{
	cout << "MaxPooling type down sample is not defined yet!" << endl << "down_sample_max_pooling() has stoped!" << endl;
	Array3Dd temp;
	return temp;
}


Array3Dd down_sample_stochastic_pooling(const Array3Dd &vector_array, const int sample_num)
{
	cout << "StochasticPooling type down sample is not defined yet!" << endl << "down_sample_stochastic_pooling() has stoped!" << endl;
	Array3Dd temp;
	return temp;
}


