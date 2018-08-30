function net = cnninit(net)
% CNN网络,初始化

n = numel(net.layers);                                          % CNN网络层数，numel返回数组的个数

for L = 2 : n                                                   % 对CNN网络层数做循环(注意:这里实际上可以从第2层开始)

    %======================================================================
    % 以下代码仅对第2,4层(卷积层)有效
    
    if (strcmp(net.layers{L}.type, 'c'))
        
        % 由前一层图像行列数,和本层卷积核尺度,计算本层图像行列数,二维向量
        net.layers{L}.iSizePic = net.layers{L-1}.iSizePic - net.layers{L}.iSizeKer + 1;
        
        % "前一层任意一个通道",对应"本层所有通道"卷积核权值W(可训练参数)个数,不包括加性偏置
        fan_out = net.layers{L}.iChannel * net.layers{L}.iSizeKer ^ 2;
        
        % "前一层所有通道",对应"本层任意一个通道"卷积核权值W(可训练参数)个数,不包括加性偏置
        fan_in = net.layers{L-1}.iChannel * net.layers{L}.iSizeKer ^ 2;
        
        for J = 1 : net.layers{L}.iChannel                      % 对本层输出通道数做循环
            
            for I = 1 : net.layers{L-1}.iChannel                % 对本层输入通道数做循环
                
                % "前一层所有通道",对"本层所有通道",层对层的全连接,卷积核权值W,进行均匀分布初始化,范围为:[-1,1]*sqrt(6/(fan_in+fan_out))
                net.layers{L}.Ker{I}{J} = (rand(net.layers{L}.iSizeKer) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                net.layers{L}.Ker_delta{I}{J} = zeros(size(net.layers{L}.Ker{I}{J}));                                    %w后续会使用
            end
            net.layers{L}.B{J} = 0;                             % 对本层输出通道加性偏置进行0值初始化
            net.layers{L}.B_delta{J} = 0;
        end
        
    end
    
    %======================================================================
    % 以下代码对第3,5层(下采样层)有效
    
    if (strcmp(net.layers{L}.type, 's'))        
        
        % 由前一层图像行列数,和本层下采样尺度,计算本层图像行列数,二维向量
        net.layers{L}.iSizePic = floor(net.layers{L-1}.iSizePic / net.layers{L}.iSample);
        net.layers{L}.iChannel = net.layers{L-1}.iChannel;
        
        %##################################################################
        % 后改，以下代码用于下采样层的计算
        for J = 1 : net.layers{L}.iChannel
            
            net.layers{L}.Beta{J} = 1;                          % 对本层输出通道乘性偏置进行1值初始化
            net.layers{L}.Beta_delta{J} = 0;            
            
            net.layers{L}.B{J} = 0;                             % 对本层输出通道加性偏置进行0值初始化
            net.layers{L}.B_delta{J} = 0;            
            
        end
    end
    
    %======================================================================
    % 本层是全连接层的前提下，三种情况：前一层是下采样层，前一层是卷积层，前一层是输入层
    
    if (strcmp(net.layers{L}.type, 'f'))

        if (strcmp(net.layers{L-1}.type, 's') || strcmp(net.layers{L-1}.type, 'c') || strcmp(net.layers{L-1}.type, 'i'))
        %------------------------------------------------------------------
        % 以下代码对第6层(过渡全连接层)有效            

            fvnum = prod(net.layers{L-1}.iSizePic) * net.layers{L-1}.iChannel;          % 每个通道的像素个数 * 输入通道数 = 下一层全连接输入个数
            onum = net.layers{L}.iChannel;                                              % 输出层类别个数
            
            net.layers{L}.W = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum)); % 输出层加性偏置0值初始化
            net.layers{L}.W_delta = zeros(size(net.layers{L}.W));                       % 输出层加性偏置0值初始化            
            
            net.layers{L}.B = zeros(onum, 1);                                           % 输出层加性偏置0值初始化
            net.layers{L}.B_delta = zeros(onum, 1);                                     % 输出层加性偏置0值初始化
        
        elseif (strcmp(net.layers{L-1}.type, 'f'))
        %------------------------------------------------------------------
        % 以下代码对第7层(全连接层)有效
            
            fvnum = net.layers{L-1}.iChannel;                                           % 每个通道的像素个数 * 输入通道数 = 下一层全连接输入个数
            onum = net.layers{L}.iChannel;                                              % 输出层类别个数
            
            net.layers{L}.W = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum)); % 输出层加性偏置0值初始化
            net.layers{L}.W_delta = zeros(size(net.layers{L}.W));                       % 输出层加性偏置0值初始化
            
            net.layers{L}.B = zeros(onum, 1);                                           % 输出层加性偏置0值初始化
            net.layers{L}.B_delta = zeros(onum, 1);                                     % 输出层加性偏置0值初始化
            
        %------------------------------------------------------------------            
        end
        
    end
    
    %======================================================================
    
end

end
