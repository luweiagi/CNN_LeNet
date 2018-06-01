function net = cnngradzero(net)
% CNN网络,训练参数置0(C串行风格专有)
% 
% 特别注意：以下训练参数改C时要提前设置所占内存空间
% net.layers{L}.Ker_grad{I}{J}
% net.layers{L}.B_grad{J}
% net.layers{L}.W_grad
% net.layers{L}.B_grad

n = numel(net.layers);                                              % CNN网络层数

for L = 2 : n                                                       % 对CNN网络层数做循环(注意:这里实际上可以从第2层开始)
    
    %======================================================================
    
    if (strcmp(net.layers{L}.type, 'c'))        
        
        for J = 1 : net.layers{L}.iChannel                          % 对本层输出通道数做循环
            for I = 1 : net.layers{L-1}.iChannel                    % 对上一层输出通道数做循环
                
                % 这里没什么好说的，就是普通的权值更新的公式：W_new = W_old - alpha * de/dW（误差对权值导数）
                net.layers{L}.Ker_grad{I}{J} = 0;
                
            end
            
            % 本层一个通道输出对应一个加性偏置net.layers{L}.B{J}
            net.layers{L}.B_grad{J} = 0;
            
        end
        
        % 特别注意(zouxy09源代码勘误,外文源代码正确)：
        % 下面这一行放在这里是错误的，应该放在上面循环J内部！
        % net.layers{L}.B{J} = net.layers{L}.B{J} - net.alpha * net.layers{L}.B_grad{J};

    end
    
    %######################################################################
    % 后改，以下代码用于下采样层的计算     
    
    if (strcmp(net.layers{L}.type, 's'))   
       
        for J = 1 : net.layers{L}.iChannel                          % 对本层输出通道数做循环
            
            net.layers{L}.Beta_grad{J} = 0;            
            
            % 对所有net.layers{L}.Delta{J}的叠加,结果要对样本数做平均
            net.layers{L}.B_grad{J} = 0;            
        end
        
    end    
    
    %======================================================================
    
    if (strcmp(net.layers{L}.type, 'f'))
        
        % 本层一个通道输出对应一个加性偏置net.layers{L}.B{J}
        net.layers{L}.W_grad = 0;
       
        
        % 本层一个通道输出对应一个加性偏置net.layers{L}.B{J}
        net.layers{L}.B_grad = 0;
        
    end
    
    %======================================================================    
    
end

end
