function [er, bad] = cnntest2(net, x, y)
% CNN网络,测试(C串行风格)

m = size(x, 3);                                                 % 训练样本个数
h = zeros(1,m);
a = zeros(1,m);

for I = 1 : m
    
    net = cnnff2(net, x(:,:,I));                                                % 前向传播得到输出
    
    % [Y,I] = max(X) returns the indices of the maximum values in vector I
    [~, h(I)] = max(net.Y);                                                % 找到最大的输出对应的标签
    
    [~, a(I)] = max(y(:,I));                                          % 找到最大的期望输出对应的索引
    
end

bad = find(h ~= a);                                                 % 找到他们不相同的个数，也就是错误的次数
er = numel(bad) / size(y, 2);                                       % 计算错误率    

end
