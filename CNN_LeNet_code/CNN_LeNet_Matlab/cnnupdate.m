function net = cnnupdate(net)
% CNN网络,卷积层和输出层的权值更新(附加惯性项)

n = numel(net.layers);                                              % CNN网络层数

for L = 2 : n                                                       % 对CNN网络层数做循环(注意:这里实际上可以从第2层开始)
    
    %======================================================================
    
    if (strcmp(net.layers{L}.type, 'c'))        
        
        for J = 1 : net.layers{L}.iChannel                          % 对本层输出通道数做循环
            for I = 1 : net.layers{L-1}.iChannel                    % 对上一层输出通道数做循环
                
                % 这里没什么好说的，就是普通的权值更新的公式：W_new = W_old - alpha * de/dW（误差对权值导数）
                net.layers{L}.Ker_delta{I}{J} = net.eta * net.layers{L}.Ker_delta{I}{J} - net.alpha * net.layers{L}.Ker_grad{I}{J};
                net.layers{L}.Ker{I}{J} = net.layers{L}.Ker{I}{J} + net.layers{L}.Ker_delta{I}{J};
                
            end
            
            % 本层一个通道输出对应一个加性偏置net.layers{L}.B{J}
            net.layers{L}.B_delta{J} = net.eta * net.layers{L}.B_delta{J} - net.alpha * net.layers{L}.B_grad{J};
            net.layers{L}.B{J} = net.layers{L}.B{J} + net.layers{L}.B_delta{J};
            
        end
        
        % 特别注意(zouxy09源代码勘误,外文源代码正确)：
        % 下面这一行放在这里是错误的，应该放在上面循环J内部！
        % net.layers{L}.B{J} = net.layers{L}.B{J} - net.alpha * net.layers{L}.B_grad{J};

    end
    
    %######################################################################
    % 后改，以下代码用于下采样层的计算   
    
    if (strcmp(net.layers{L}.type, 's')) 
        
        for J = 1 : net.layers{L}.iChannel                          % 对本层输出通道数做循环
        
            % 本层一个通道输出对应一个加性偏置net.layers{L}.B{J}
            net.layers{L}.B_delta{J} = net.eta * net.layers{L}.B_delta{J} - net.alpha * net.layers{L}.B_grad{J};
            net.layers{L}.B{J} = net.layers{L}.B{J} + net.layers{L}.B_delta{J};        
            
            net.layers{L}.Beta_delta{J} = net.eta * net.layers{L}.Beta_delta{J} - net.alpha * net.layers{L}.Beta_grad{J};
            net.layers{L}.Beta{J} = net.layers{L}.Beta{J} + net.layers{L}.Beta_delta{J};        
        
        end
        
    end
    
    %======================================================================
    
    if (strcmp(net.layers{L}.type, 'f'))
        
        % 本层一个通道输出对应一个加性偏置net.layers{L}.B{J}
        net.layers{L}.W_delta = net.eta * net.layers{L}.W_delta - net.alpha * net.layers{L}.W_grad;
        net.layers{L}.W = net.layers{L}.W + net.layers{L}.W_delta;
        
        
        % 本层一个通道输出对应一个加性偏置net.layers{L}.B{J}
        net.layers{L}.B_delta = net.eta * net.layers{L}.B_delta - net.alpha * net.layers{L}.B_grad;
        net.layers{L}.B = net.layers{L}.B + net.layers{L}.B_delta;
        
    end
    
    %======================================================================    
    
end

end
