function net = cnnff2(net, x)
% CNN网络,正向计算(批处理算法,核心是convn用法,和输出层批量映射)(C串行风格)

n = numel(net.layers);                                          % CNN网络层数
net.layers{1}.X{1} = x;                                         % 网络的第一层就是输入(opts.batchsize幅输入同时处理)

for L = 2 : n                                                   % 对CNN网络层数做循环(注意:这里实际上可以从第2层开始)
    
    %======================================================================
    % 以下代码仅对第2,4层(卷积层)有效
    
    if (strcmp(net.layers{L}.type, 'c'))        
        
        for J = 1 : net.layers{L}.iChannel                    % 对本层输出通道数做循环
            
            % 对本层一个通道输出0值初始化(opts.batchsize幅输入同时处理)
            z = zeros(size(net.layers{L - 1}.X{1}) - [net.layers{L}.iSizeKer - 1, net.layers{L}.iSizeKer - 1]);
%             z = zeros(size(net.layers{L - 1}.X{1}) - [net.layers{L}.iSizeKer - 1, net.layers{L}.iSizeKer - 1, 0]);

            for I = 1 : net.layers{L-1}.iChannel                             % 对本层输入通道数做循环
                
                % 特别注意:
                % net.layers{L - 1}.X{I}为opts.batchsize幅输入,为三维矩阵
                % net.layers{L}.Ker{I}{J}为二维卷积核矩阵
                % 这里采用了函数convn,实现多个样本输入的同时处理
                
                z = z + conv2(net.layers{L - 1}.X{I}, net.layers{L}.Ker{I}{J}, 'valid');
%                 z = z + convn(net.layers{L - 1}.X{I}, net.layers{L}.Ker{I}{J}, 'valid');

            end
            
            % 本层一个通道输出对应一个加性偏置net.layers{L}.B{J},再采用sigmoid激活函数
            net.layers{L}.X{J} = z + net.layers{L}.B{J};                % 本层输出计算
            net.layers{L}.X{J} = fx(net.layers{L}.X{J});                % 本层输出计算            
            
            % 特别注意:
            % 卷积层涉及到三个运算:(1)卷积,(2)偏置(加),(3)sigmoid映射
        end
        
    end
    
    %======================================================================
    % 以下代码对第3,5层(下采样层)有效
    
    if (strcmp(net.layers{L}.type, 's'))        
        
        for J = 1 : net.layers{L-1}.iChannel                    % 对本层输入通道数做循环(输入输出通道数相等)
            
            % 图片下采样函数,行列采样倍数为iSample
            
            %##################################################################
            % 后改，以下代码用于下采样层的计算
            net.layers{L}.X_down{J} = down(net.layers{L - 1}.X{J}, net.layers{L}.iSample);
            net.layers{L}.X{J} = net.layers{L}.Beta{J} * net.layers{L}.X_down{J} + net.layers{L}.B{J};
%             net.layers{L}.X{J} = fx(net.layers{L}.X{J});                % 本层输出计算
         
            % 特别注意:
            % 下采样层仅涉及两个个运算:(1)下采样, (2)偏置(乘或加),"sigmoid映射"这里都没有
        end
        
    end
    
    %======================================================================
    
    if (strcmp(net.layers{L}.type, 'f'))
        
        if (strcmp(net.layers{L-1}.type, 's') || strcmp(net.layers{L-1}.type, 'c') || strcmp(net.layers{L-1}.type, 'i'))
        %------------------------------------------------------------------
        % 以下代码对第6层(过渡全连接层)有效
        
            net.layers{L-1}.X_Array = [];
            for J = 1 : net.layers{L-1}.iChannel                            % 对前一层输出通道数做循环
                
                sa = size(net.layers{L-1}.X{J});                            % 第j个特征map的大小(实际上每个j都相等)
                
                % 将所有的特征map拉成一条列向量。还有一维就是对应的样本索引。每个样本一列，每列为对应的特征向量
                net.layers{L-1}.X_Array = [net.layers{L-1}.X_Array; reshape(net.layers{L-1}.X{J}, sa(1) * sa(2), 1)];
%                 net.layers{L-1}.X_Array = [net.layers{L-1}.X_Array; reshape(net.layers{L-1}.X{J}, sa(1) * sa(2), sa(3))];

            end
            
            % 计算网络的最终输出值。sigmoid(W*X + b)，注意是同时计算了batchsize个样本的输出值
            net.layers{L}.X = fx(net.layers{L}.W * net.layers{L-1}.X_Array + net.layers{L}.B);
%             net.layers{L}.X = fx(net.layers{L}.W * net.layers{L-1}.X_Array + repmat(net.layers{L}.B, 1, size(net.layers{L-1}.X_Array, 2)));
            
            % 特别注意:
            % 输出层涉及到三个运算:(1)加权,(2)偏置(加),(3)sigmoid映射
            
        elseif (strcmp(net.layers{L-1}.type, 'f'))
        %------------------------------------------------------------------
        % 以下代码对第7层(全连接层)有效            
            
            net.layers{L}.X = fx(net.layers{L}.W * net.layers{L-1}.X + net.layers{L}.B);
%             net.layers{L}.X = fx(net.layers{L}.W * net.layers{L-1}.X + repmat(net.layers{L}.B, 1, size(net.layers{L-1}.X, 2)));
            
        %------------------------------------------------------------------            
        end
    end
    
    %======================================================================
    
end

net.Y = net.layers{n}.X;

end
