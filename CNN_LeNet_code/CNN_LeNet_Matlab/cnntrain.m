function net = cnntrain(net, x, y)
% CNN网络,训练

m = size(x, 3);                                                 % 训练样本个数
numbatches = ceil(m / net.batchsize);                           % "训练集整体迭代一次"网络权值更新的次数

net.ERR = zeros(1,net.epochs);
% net.ERR = [];

for I = 1 : net.epochs                                          % 对训练集整体迭代次数做循环
    
    disp(['epoch ' num2str(I) '/' num2str(net.epochs)]);        % 显示进度
    tic;
    
    kk = randperm(m);                                           % 打乱训练样本顺序，实现洗牌的功能
    
    %----------------------------------------------------------------------
    % 对"训练集整体迭代一次"网络权值更新的次数做循环
    
    mse = 0;
    for L = 1 : numbatches                                      % 整体训练一次网络更新的次数
        
        % 取出打乱顺序后的batchsize个样本和对应的标签
        batch_x = x(:, :, kk((L - 1) * net.batchsize + 1 : min(L * net.batchsize, m)));    %min是进行保护的，保证不会超过m
        batch_y = y(:,    kk((L - 1) * net.batchsize + 1 : min(L * net.batchsize, m)));
        
        % 在当前的网络权值和网络输入下计算网络的输出(正向计算)
        net = cnnff(net, batch_x);                              % CNN正向计算(批处理算法,核心是convn用法,和输出层批量映射)
        
        % 得到上面的网络输出后，通过对应的样本标签用bp算法来得到误差对网络权值(反向传播) 的导数？
        net = cnnbp(net, batch_y);
        
        % 得到误差对权值的导数后，就通过权值更新方法去更新权值
        net = cnnupdate(net);
        
        mse = mse+net.err;
%         if isempty(net.ERR)
%             net.ERR(1) = net.err;                                       % 代价函数值，也就是误差值
%         end
%         net.ERR(end + 1) = 0.99 * net.ERR(end) + 0.01 * net.err;        % 保存历史的误差值，以便画图分析
    end
    mse = mse / numbatches
%     mse = net.ERR(end)
    
    net.ERR(I) = mse;
    
    %----------------------------------------------------------------------
    
    toc;
end

%==========================================================================

end
