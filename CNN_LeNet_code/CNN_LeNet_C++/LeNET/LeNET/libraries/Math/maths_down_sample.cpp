#include <maths_down_sample.h>
#include <maths.h>


// 采样的尺寸变化规则是：5x5的图片，以2的倍率降采样，其实相当于4x4的图片来降采样。
vector<array2D> down_sample(const vector<array2D> &vector_array, const int sample_num, down_sample_type sample_type)
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
			cout << "down sample type is not defined yet!" << endl << "down_sample() has stoped!" << endl;
			vector<array2D> temp;
			return temp;
			break;
		}
	}
}


vector<array2D> down_sample_mean_pooling(const vector<array2D> &vector_array, const int sample_num)
{
	array2D mean_pooling_convolution_kernel = create_array2D(sample_num, sample_num, double(1 / pow(sample_num, 2)));

	vector<array2D> Z = convolution_n_dim(vector_array, mean_pooling_convolution_kernel);

	vector<array2D> Y = equal_interval_sampling_vector_array2D(Z, sample_num);

	return Y;
}


vector<array2D> down_sample_max_pooling(const vector<array2D> &vector_array, const int sample_num)
{
	cout << "MaxPooling type down sample is not defined yet!" << endl << "down_sample_max_pooling() has stoped!" << endl;
	vector<array2D> temp;
	return temp;
}


vector<array2D> down_sample_stochastic_pooling(const vector<array2D> &vector_array, const int sample_num)
{
	cout << "StochasticPooling type down sample is not defined yet!" << endl << "down_sample_stochastic_pooling() has stoped!" << endl;
	vector<array2D> temp;
	return temp;
}


