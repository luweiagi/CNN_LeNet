#include <CNN.h>
#include <maths.h>
#include <time.h>
#include <iostream>


using namespace std;


// 初始化CNN类
CNN::CNN(const vector<Layer> &layers, float alpha, float eta, int batchsize, int epochs, activation_function_type activ_func_type, down_sample_type down_samp_type)
	:_layers(layers), 
	_alpha(alpha), 
	_eta(eta), 
	_batchsize(batchsize), 
	_epochs(epochs), 
	_activation_func_type(activ_func_type), 
	_down_sample_type(down_samp_type)
{
	// 依据网络结构设置CNN.layers, 初始化一个CNN网络
	init();

	_ERR.assign(_epochs, 0);// 将历次迭代的均方误差初始化为0
	_err = 0;// 将当前轮的当前批次的均方误差初始化为0

	cout << "CNN has initialised!" << endl;
}


// CNN网络，训练
void CNN::train(const Array3Dd &train_x, const Array2Dd &train_y)
{
	cout << "begin to train" << endl;

	if (train_x.size() != train_y.size())
	{
		cout << "train_x size is not same as train_y size!" << endl << "CNN.train() failed!" << endl;
		return;
	}

	int m = train_x.size();// 训练样本个数，比如1000个
	int numbatches = ceil(m / _batchsize);// "训练集整体迭代一次" 网络权值更新的次数，比如1000/10=100

	// 对训练集整体迭代次数做循环
	for (int I = 0; I < _epochs; I++)// 比如整体重复迭代25次
	{
		// 显示进度
		cout << "epoch " << I+1 << "/" << _epochs << endl;

		clock_t tic = clock(); //获取毫秒级数目
		
		// 打乱训练样本顺序，实现洗牌的功能
		const vector<int> kk = randperm_vector(m);

		// ********************************************************************************************* //
		// 对"训练集整体迭代一次"网络权值更新的次数做循环

		double mse = 0;// 当次训练集整体迭代时的均方误差

		// 整体训练一次网络更新的次数,
		// 即：假设训练样本数为1000个，批处理数为10个，那整体训练一次就得1000/10=100次
		for (int L = 0; L < numbatches; L++)
		{
			// 取出打乱顺序后的batchsize个样本和对应的标签
			Array3Dd batch_train_x;
			Array2Dd batch_train_y;

			for (int i = L*_batchsize; i < min((L + 1)*_batchsize, m); i++)// (L+1)*_batchsize在最后一次循环中可能会大于m
			{
				batch_train_x.push_back(train_x.at(kk.at(i)));
				batch_train_y.push_back(train_y.at(kk.at(i)));
			}
			// 显示当前正在处理的批量图片（只显示10张，当批处理图片的数量小于10时会报错，这时需要修改显示的张数）
			//string window_title = "Images from " + to_string(L*_batchsize) + " to " + to_string(min((L + 1)*_batchsize, m));
			//batch_train_x.show_specified_images_64FC1(window_title, CvSize(5, 2), CvSize(32, 32), 150);

			// 在当前的网络权值和网络输入下计算网络的输出(正向计算)
			clock_t tic_ff = clock();
			feed_forward(batch_train_x);
			clock_t toc_ff = clock();
			cout << "                          batches " << L+1 << " feedforward time has elapsed: " << (double)(toc_ff - tic_ff) / 1000 << " seconds" << endl;
		
			// 得到上面的网络输出后，通过对应的样本标签用bp算法来得到误差对网络权值(反向传播) 的导数
			clock_t tic_bp = clock();
			back_propagation(batch_train_y);
			clock_t toc_bp = clock();
			cout << "                          batches " << L + 1 << " back propagation time has elapsed: " << (double)(toc_bp - tic_bp) / 1000 << " seconds" << endl;

			// 得到误差对权值的导数后，就通过权值更新方法去更新权值
			update();

			mse += _err;
		}

		mse /= numbatches;
		_ERR.at(I) = mse; // 记录第I次训练集整体迭代时的均方误差

		// ********************************************************************************************* //

		clock_t toc = clock(); //获取毫秒级数目
		cout << "epochs " << I+1 << " time has elapsed: " << (double)(toc - tic) / 1000 << " seconds" << endl;
	}
	//*/
	cout << "train has finished!" << endl;
}


// CNN网络，测试，返回错误率
double CNN::test(const Array3Dd &test_x, const Array2Dd &test_y)
{
	cout << "begin to test" << endl;
	return 0;
}


// 依据网络结构设置CNN.layers, 初始化一个CNN网络
void CNN::init()
{
	// CNN网络层数
	int n = _layers.size();

	cout << "input layer " << 1 << " has initialised!" << endl;

	// 对CNN网络层数做循环
	for (int L = 1; L < n; L++)// 注意:这里实际上可以从第2层开始，所以L初始值是1不是0
	{
		// ======================================================================
		// 以下代码仅对第2, 4层(卷积层)有效

		if (_layers.at(L).type == 'c')
		{
			// 由前一层图像行列数,和本层卷积核尺度,计算本层图像行列数,二维向量
			_layers.at(L).iSizePic[0] = _layers.at(L - 1).iSizePic[0] - _layers.at(L).iSizeKer + 1;
			_layers.at(L).iSizePic[1] = _layers.at(L - 1).iSizePic[1] - _layers.at(L).iSizeKer + 1;

			// "前一层任意一个通道", 对应"本层所有通道"卷积核权值W(可训练参数)个数, 不包括加性偏置
			int fan_out = _layers.at(L).iChannel * pow(_layers.at(L).iSizeKer, 2);// 比如 4*5^2=4*25

			// "前一层所有通道", 对应"本层任意一个通道"卷积核权值W(可训练参数)个数, 不包括加性偏置
			int fan_in = _layers.at(L-1).iChannel * pow(_layers.at(L).iSizeKer, 2);

			// 对当前层的Ker和Ker_delta初始化维数，并赋初值
			_layers.at(L).Ker.resize(_layers.at(L - 1).iChannel);
			_layers.at(L).Ker_delta.resize(_layers.at(L - 1).iChannel);

			// 对本层输入通道数做循环
			for (int I = 0; I < _layers.at(L - 1).iChannel; I++)
			{
				_layers.at(L).Ker.at(I).resize(_layers.at(L).iChannel);
				_layers.at(L).Ker_delta.at(I).resize(_layers.at(L).iChannel);

				// 对本层输出通道数做循环
				for (int J = 0; J < _layers.at(L).iChannel; J++)
				{
					double maximum = (double)sqrt(6.0f / (fan_in + fan_out));

					// "前一层所有通道",对"本层所有通道",层对层的全连接,卷积核权值W,进行均匀分布初始化,范围为:[-1,1]*sqrt(6/(fan_in+fan_out))
					_layers.at(L).Ker[I][J].set_rand(_layers.at(L).iSizeKer, _layers.at(L).iSizeKer, -maximum, maximum);
					_layers.at(L).Ker_delta[I][J].set_zero_same_size_as(_layers.at(L).Ker[I][J]);
				}
			}

			// 对本层输出通道加性偏置进行0值初始化
			_layers.at(L).B.assign(_layers.at(L).iChannel, 0);
			_layers.at(L).B_delta.assign(_layers.at(L).iChannel, 0);

			cout << "convolutional layer " << L + 1 << " has initialised!" << endl;
		}

		// ======================================================================
		// 以下代码对第3,5层(下采样层)有效

		if (_layers.at(L).type == 's')
		{
			_layers.at(L).iSizePic[0] = floor((_layers.at(L - 1).iSizePic[0] + _layers.at(L).iSample - 1) / _layers.at(L).iSample);
			_layers.at(L).iSizePic[1] = floor((_layers.at(L - 1).iSizePic[1] + _layers.at(L).iSample - 1) / _layers.at(L).iSample);
			_layers.at(L).iChannel = _layers.at(L - 1).iChannel;

			// 以下代码用于下采样层的计算

			// 对本层输出通道乘性偏置进行1值初始化
			_layers.at(L).Beta.assign(_layers.at(L).iChannel, 1);
			_layers.at(L).Beta_delta.assign(_layers.at(L).iChannel, 0);

			// 对本层输出通道加性偏置进行0值初始化
			_layers.at(L).B.assign(_layers.at(L).iChannel, 0);
			_layers.at(L).B_delta.assign(_layers.at(L).iChannel, 0);

			cout << "subsampling layer " << L + 1 << " has initialised!" << endl;
		}

		// ======================================================================
		// 本层是全连接层的前提下，三种情况：前一层是下采样层，前一层是卷积层，前一层是输入层

		if (_layers.at(L).type == 'f')
		{
			if (_layers.at(L - 1).type == 's' || _layers.at(L - 1).type == 'c' || _layers.at(L - 1).type == 'i')
			{
				// ------------------------------------------------------------------
				// 以下代码对第6层(过渡全连接层)有效

				// 当前层全连接输入个数 = 上一层每个通道的像素个数 * 上一层输入通道数
				int fvnum = _layers.at(L - 1).iSizePic[0] * _layers.at(L - 1).iSizePic[1] * _layers.at(L - 1).iChannel;
				// 当前输出层类别个数
				int onum = _layers.at(L).iChannel;

				double maximum = (double)sqrt(6.0f / (onum + fvnum));
				// 初始化当前层与上一层的连接权值
				_layers.at(L).W.set_rand(fvnum, onum, -maximum, maximum);// 注意是W[I列][J行],I为当前层全连接输入个数，J为当前层数目
				_layers.at(L).W_delta.set_zero_same_size_as(_layers.at(L).W);

				// 对本层输出通道加性偏置进行0值初始化
				_layers.at(L).B.assign(onum, 0);
				_layers.at(L).B_delta.assign(onum, 0);
			}
			else if (_layers.at(L - 1).type == 'f')
			{
				// ------------------------------------------------------------------
				// 以下代码对第7层(全连接层)有效。 对第8层也有效吧？

				// 当前层全连接输入个数 = 上一层输入通道数
				int fvnum = _layers.at(L - 1).iChannel;
				// 当前输出层类别个数
				int onum = _layers.at(L).iChannel;

				double maximum = (double)sqrt(6.0f / (onum + fvnum));
				// 初始化当前层与上一层的连接权值
				_layers.at(L).W.set_rand(fvnum, onum, -maximum, maximum);// 注意是W[列I][行J],I为上一层的数目，J为当前层数目
				_layers.at(L).W_delta.set_zero_same_size_as(_layers.at(L).W);

				// 对本层输出通道加性偏置进行0值初始化
				_layers.at(L).B.assign(onum, 0);
				_layers.at(L).B_delta.assign(onum, 0);
			}

			cout << "fully connected layer " << L + 1 << " has initialised!" << endl;
		}
	}
}


// CNN网络,正向计算(批处理算法,核心是convn用法,和输出层批量映射)
void CNN::feed_forward(const Array3Dd &train_x)
{
	// CNN网络层数
	int n = _layers.size();

	_layers.at(0).X.resize(1);
	_layers.at(0).X.at(0) = train_x;

	for (int L = 1; L < n; L++)
	{
		// ======================================================================
		// 以下代码仅对第2,4层(卷积层)有效

		if (_layers.at(L).type == 'c')
		{
			// 特别注意:
			// 卷积层涉及到三个运算 : (1)卷积, (2)偏置(加), (3)sigmoid映射

			// 对当前层的输出做初始化
			_layers.at(L).X.resize(_layers.at(L).iChannel);// 即为当前层每一个通道分配一个输出图

			// 对本层输出通道数做循环
			for (int J = 0; J < _layers.at(L).iChannel; J++)
			{
				// 对当前层第J个通道的对上一层的所有卷积之和z，进行初始化(batchsize幅输入同时处理)
				Array3Dd z;
				bool conv_first_time = true;

				// 1.卷积
				for (int I = 0; I < _layers.at(L - 1).iChannel; I++)
				{
					// 特别注意:
					// _layers.at(L - 1).X(I)为_batchsize幅输入, 为三维矩阵
					// _layers.at(L).Ker[I][J]为二维卷积核矩阵
					// 这里采用了函数convn, 实现多个样本输入的同时处理
					// convn是三维卷积，此处是关键
					if (conv_first_time)
					{
						z = convolution_n_dim(_layers.at(L - 1).X.at(I), _layers.at(L).Ker.at(I).at(J));
						conv_first_time = false;
					}
					else
					{
						z.add(convolution_n_dim(_layers.at(L - 1).X.at(I), _layers.at(L).Ker.at(I).at(J)));
					}
				}

				// 2.偏置(加)
				_layers.at(L).X.at(J) = z + _layers.at(L).B.at(J);
				
				// 3.sigmoid映射
				_layers.at(L).X.at(J) = activation_function(_layers.at(L).X.at(J), _activation_func_type);
			}
		}

		// ======================================================================
		// 以下代码对第3,5层(下采样层)有效

		if (_layers.at(L).type == 's')
		{
			// 特别注意:
			// 下采样层仅涉及两个运算 : (1)下采样, (2)偏置(乘和加)
			// 这里没有"sigmoid映射"

			// 对当前下采样层的输入输出做初始化
			_layers.at(L).X_down.resize(_layers.at(L).iChannel);// 即为当前层每一个通道分配一个输入图
			_layers.at(L).X.resize(_layers.at(L).iChannel);// 即为当前层每一个通道分配一个输出图

			// 对本层输出通道数做循环(输入输出通道数相等)
			for (int J = 0; J < _layers.at(L).iChannel; J++)
			{
				// 图片下采样函数, 行列采样倍数为iSample
				// 以下代码用于下采样层的计算

				// (1)下采样
				_layers.at(L).X_down.at(J) = down_sample(_layers.at(L - 1).X.at(J), _layers.at(L).iSample, MeanPooling);

				// (2)偏置(乘和加)
				_layers.at(L).X.at(J) = _layers.at(L).X_down.at(J) * _layers.at(L).Beta.at(J) + _layers.at(L).B.at(J);
			}
		}

		// ======================================================================
		// 以下代码对第6,7,8层（全连接层）有效

		if (_layers.at(L).type == 'f')
		{
			if ((_layers.at(L - 1).type == 's') || (_layers.at(L - 1).type == 'c') || (_layers.at(L - 1).type == 'i'))
			{
				// ------------------------------------------------------------------
				// 以下代码对第6层(过渡全连接层)有效

				// 特别注意:
				// 全连接输出层涉及到三个运算 : (1)加权, (2)偏置(加), (3)sigmoid映射

				_layers.at(L - 1).X_Array.clear();

				// 对前一层输出通道数做循环
				// 计算用于本层输入的一维向量（前一层的所有通道输出图合并为一个一维向量，以便计算）
				for (int J = 0; J < _layers.at(L - 1).iChannel; J++)
				{
					// 第j个特征map的大小(实际上每个j都相等)
					int sa_page = _layers.at(L - 1).X.at(J).size();
					int sa_col = _layers.at(L - 1).X.at(J).at(0).size();
					int sa_row = _layers.at(L - 1).X.at(J).at(0).at(0).size();

					// 将所有的特征map拉成一条列向量。还有一维就是对应的样本索引。每个样本一列，每列为对应的特征向量，此处非常巧妙！
					// 会将每幅图像的按照列向量的形式抽取为1行，然后再将不同样本的列向量串联起来
					_layers.at(L - 1).X_Array.append_along_row( _layers.at(L - 1).X.at(J).reshape_to_Array2D() );
				}

				// 计算网络的最终输出值。sigmoid(W*X + b)，注意是同时计算了batchsize个样本的输出值

				int col_batchsize = _layers.at(L - 1).X_Array.size();
				// in = W*X + B (1)加权, (2)偏置(加)
				Array2Dd fcl_map_in = _layers.at(L).W.product(_layers.at(L - 1).X_Array) + Array2Dd::repmat(_layers.at(L).B, 1, col_batchsize);
				// out = activ(in) (3)sigmoid映射
				_layers.at(L).X_fcl = activation_function(fcl_map_in, _activation_func_type);

				// 特别注意:
				// 全连接输出层涉及到三个运算 : (1)加权, (2)偏置(加), (3)sigmoid映射
			}
			else if (_layers.at(L - 1).type == 'f')
			{
				// ------------------------------------------------------------------
				// 以下代码对第7,8层(全连接层)有效

				// 特别注意:
				// 全连接输出层涉及到三个运算 : (1)加权, (2)偏置(加), (3)sigmoid映射

				// 计算网络的最终输出值。sigmoid(W*X + b)，注意是同时计算了batchsize个样本的输出值

				int col_batchsize = _layers.at(L - 1).X_fcl.size();
				// in = W*X + B (1)加权, (2)偏置(加)
				Array2Dd fcl_map_in = _layers.at(L).W.product(_layers.at(L - 1).X_fcl) + Array2Dd::repmat(_layers.at(L).B, 1, col_batchsize);
				// out = activ(in) (3)sigmoid映射
				_layers.at(L).X_fcl = activation_function(fcl_map_in, _activation_func_type);

				// 特别注意:
				// 全连接输出层涉及到三个运算 : (1)加权, (2)偏置(加), (3)sigmoid映射
			}
		}
	}

	// 将最后一层（全连接层）的输出结果喂给_Y，作为神经网络的输出
	_Y = _layers.at(n-1).X_fcl;
}


// CNN网络,反向传播(批处理算法)
void CNN::back_propagation(const Array2Dd &train_y)
{
	// CNN网络层数
	int n = _layers.size();

	// 输出误差: 预测值-期望值
	Array2Dd E = _layers.at(n - 1).X_fcl - train_y;

	// 输出层灵敏度(残差)
	// 注意，这里需要说明下，这里对应的公式是 delta = (y - t).*f'(u),但是这里为什么是f'(x)呢？
	// 因为这里其实是sigmoid求导，f'(u) = x*(1-x)，所以输入的就是x了。
	// 其中，u表示当前层输入，x表示当前层输出。
	_layers.at(n - 1).Delta_fcl = E * derivation(_layers.at(n - 1).X_fcl, _activation_func_type);

	// 代价函数是均方误差,已对样本数做平均
	_err = 0.5 * E.pow(2).sum() / E.size();// 当前轮的当前批次的均方误差

	// ************** 灵敏度(残差)的反向传播 ******************************

	int tmp;
	if (_layers.at(1).type == 'f')
	{
		// 当第二层就是全连接层时,相当于输入图片拉成一个特征矢量形成的BP网络,
		// 考虑到必须计算net.layers{1}.X_Array,所以L的下限必须到0
		tmp = 0;
	}
	else
	{
		// 其它情况L下限是1就可以
		tmp = 1;
	}

	for (int L = (n - 2); L >= tmp; L--)
	{
		// =====================================================================
		// 以下代码对“下一层”为“全连接层”时有效

		if (_layers.at(L + 1).type == 'f')
		{
			if (_layers.at(L).type == 'f')
			// ------------------------------------------------------------------
			// 以下代码对第6(过渡全连接层),7层(全连接层)有效
			{
				// 典型的BP网络输出层对隐层的灵敏度(残差)的反向传播公式
				_layers.at(L).Delta_fcl = _layers.at(L + 1).W.transpose().product(_layers.at(L + 1).Delta_fcl) * derivation(_layers.at(L).X_fcl, _activation_func_type);
				// 作为参考，当L=6（倒数第二层）时，上式的维度如下行所示：
				// _layers.at(L).Delta = [84, 10] [行 列]
				// _layers.at(L + 1).W = [10, 84]
				// _layers.at(L + 1).W.transpose() = [84 10]
				// _layers.at(L + 1).Delta = [10 10]
				// _layers.at(L).X_fcl = [84 10]
			}
			else if (_layers.at(L).type == 's' || _layers.at(L).type == 'c' || _layers.at(L).type == 'i')
			// ------------------------------------------------------------------
			// 以下代码对第5层(降采样层)有效，其“下一层”为过渡全连接层
			{
				// 每个输出通道图像尺寸(三维矢量,  第三维是批处理样本个数，最后两维是尺寸)
				int SizePic_0 = _layers.at(L).X.at(0).at(0).size();
				int SizePic_1 = _layers.at(L).X.at(0).at(0).at(0).size();

				// 输出图像像素个数
				int fvnum = SizePic_0 * SizePic_1;                                  

				// 典型的BP网络输出层对隐层的灵敏度(残差)的反向传播公式

				// 若当前层是降采样层，或输入层
				int col1 = _layers.at(L + 1).W.size();
				int row1 = _layers.at(L + 1).W.at(0).size();

				int col2 = _layers.at(L + 1).Delta_fcl.size();
				int row2 = _layers.at(L + 1).Delta_fcl.at(0).size();

				_layers.at(L).Delta_Array = _layers.at(L + 1).W.transpose().product(_layers.at(L + 1).Delta_fcl);
				// 作为参考，当L=4（第二个降采样层，下一层为全连接层）时，上式的维度如下行所示：
				// _layers.at(L).Delta_Array = [100 10]
				// _layers.at(L + 1).W = [120 100]
				// _layers.at(L + 1).W.transpose() = [100 120]
				// _layers.at(L + 1).Delta = [120 10]

				// 若当前层是卷积层
				if (_layers.at(L).type == 'c')
				{
					// 由于卷积层存在激活函数，则还需要点乘当前层激活函数的导数，才是当前层的灵敏度
					_layers.at(L).Delta_Array.dot_product(derivation(_layers.at(L).X_Array, _activation_func_type));
				}

				for (int J = 0; J < _layers.at(L).iChannel; J++)
				{
					// 此处也是批处理的
					// 将本层的长矢量灵敏度(残差), 每一列为一个样本, reshape成通道表示(矢量化全连接->通道化全连接)
				}
			}
		}

		// =====================================================================
		// 以下代码对“下一层”为“下采样层”时有效

		if (_layers.at(L + 1).type == 's')
		{

		}

		// =====================================================================
		// 以下代码对“下一层”为“卷积层”时有效

		if (_layers.at(L + 1).type == 'c')
		{

		}

		// =====================================================================
	}

	// ****************** 求训练参数的梯度 **************************************

	// 这里与《Notes on Convolutional Neural Networks》中不同，
	// 这里的“子采样”层没有参数，也没有激活函数，
	// 所以在子采样层是没有需要求解的参数的

	// 对CNN网络层数做循环(注意:这里实际上可以从第2层开始)
	for (int L = 1; L < n; L++)
	{
		// =====================================================================
		// 以下代码用于第2,4层(卷积层)的计算

		if (_layers.at(L).type == 'c')
		{

		}

		// =====================================================================
		// 以下代码用于第3,5层(下采样层)的计算

		if (_layers.at(L).type == 's')
		{

		}

		// =====================================================================
		// 以下代码用于第6,7,8层(全连接层)的计算

		if (_layers.at(L).type == 'f')
		{

		}

		// =====================================================================
	}
}


// CNN网络,卷积层和输出层的权值更新(附加惯性项)
void CNN::update(void)
{
	;
}
