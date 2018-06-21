
clear all; close all; clc;
% CNN - 主程序

% addpath('../data');                                     % 数据目录
% addpath('../util');                                     % 子函数目录

%--------------------------------------------------------------------------

% % train_x - uint8类型, 60000*784矩阵, 每一行一个样本28*28=784, 共60000个
% % test_x - uint8类型, 10000*784矩阵, 每一行一个样本28*28=784, 共10000个
% % train_y - uint8类型, 60000*10矩阵, 每一行一个BP目标矢量, 共60000个
% % test_y - uint8类型, 10000*10矩阵, 每一行一个BP目标矢量, 共10000个

% load mnist_uint8;
% 
% train_x = double(reshape(train_x',28,28,60000))/255;    % 归一化到0-1之间, 每一层一个28*28的样本
% train_y = double(train_y');                             % 每一列一个BP目标矢量
% 
% test_x = double(reshape(test_x',28,28,10000))/255;
% test_y = double(test_y');

%--------------------------------------------------------------------------

%load('TrainTestSample.mat','SAMPLE','TARBP','TARSVM');
% SAMPLE是1000张32x32的图片
% TARBP是1000个图片对应的10维BP目标矢量
% TARSVM是1000张图片SVM处理的结果
train_x = normalize_lzb(double(SAMPLE));                            % 输入图片归一化到[0 1], double型
% train_x = 32x32x1000
train_y = double(TARBP);
% train_y = 10x1000

test_x = train_x;
test_y = train_y;

%--------------------------------------------------------------------------

rng(1);
kk = randperm(size(train_x,3));                                                   % 打乱训练样本顺序
% kk是1000张图像打乱顺序后的排列

figure;
for I=1:25% 抽取打乱顺序后的前25张图片
    i = kk(I);
    Y1 = train_x(:,:,i)*255;                                      % 特别注意: 原图为仅有0,255的二值化图像
    Y2 = Y1;                                                     % 原始数据按C语言行方向存储,这里显示需要转置
    t = find(train_y(:,i))-1;                                     % 目标值,依次从0-9正交编码
    subplot(5,5,I); imshow(uint8(Y2)); title(num2str(t));
end

%--------------------------------------------------------------------------
% 初始化一个CNN网络

% net网络结构设置
net.layers = {
    struct('type','i','iChannel',1,'iSizePic',[32 32])          % 输入层:         'i',iChannel个输出通道，输入图片大小iSizePic
    struct('type','c','iChannel',2,'iSizeKer',5)                % 卷积层:         'c',iChannel个输出通道，卷积核大小[iSizePic iSizePic]
    struct('type','s','iSample',2)                              % 下采样层:       's',行列下采样率[iSample iSample]
    struct('type','c','iChannel',4,'iSizeKer',5)                % 卷积层:         'c',iChannel个输出通道，卷积核大小[iSizePic iSizePic]
    struct('type','s','iSample',2)                              % 下采样层:       's',行列下采样率[iSample iSample]
    struct('type','f','iChannel',120)                           % 全连接层:       'f',iChannel个输出节点    
    struct('type','f','iChannel',84)                            % 全连接层:       'f',iChannel个输出节点
    struct('type','f','iChannel',10)                            % 全连接层:       'f',iChannel个输出节点    
              };
net.alpha = 2;                                                  % 学习率[0.1,3]
net.eta = 0.5;                                                  % 惯性系数[0,0.95],>=1不收敛，==0为不用惯性项
net.batchsize = 10;                                             % 每次用batchsize个样本计算一个delta调整一次权值，每十个样本做平均进行调节；
net.epochs = 25;                                                % 训练集整体迭代次数        

%--------------------------------------------------------------------------
% 依据网络结构设置net.layers,初始化一个CNN网络

net = cnninit(net);

%--------------------------------------------------------------------------
% CNN网络训练

net = cnntrain(net, train_x, train_y);

%--------------------------------------------------------------------------
% CNN网络测试

% 然后就用测试样本来测试
[er, bad] = cnntest(net, test_x, test_y);

%--------------------------------------------------------------------------

figure;
%plot mean squared error
plot(net.ERR);
%show test error
disp([num2str(er*100) '% error']);

