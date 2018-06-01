function net = cnnbp2(net, y)
% CNN网络,反向传播(批处理算法)(C串行风格)

n = numel(net.layers);                                              % CNN网络层数

E = net.layers{n}.X - y;                                            % 输出误差: 预测值-期望值
net.layers{n}.Delta = E .* dy(net.layers{n}.X);                     % 输出层灵敏度(残差)

net.err = 1/2* sum(E(:) .^ 2);                                      % 代价函数是均方误差,已对样本数做平均
% net.err = 1/2* sum(E(:) .^ 2) / size(E, 2);                         % 代价函数是均方误差,已对样本数做平均

%% 灵敏度(残差)的反向传播

if strcmp(net.layers{2}.type, 'f')
   % 当第二层就是全连接层时,相当于输入图片拉成一个特征矢量形成的BP网络,考虑到必须计算net.layers{1}.X_Array,所以L的下限必须到1
   tmp = 1;                                                         
else
   % 其它情况L下限是2就可以 
   tmp = 2;
end

for L = (n - 1) : -1 : tmp
    
    %======================================================================
    % 以下代码对“下一层”为“全连接层”时有效
    
    if (strcmp(net.layers{L+1}.type, 'f'))

        if (strcmp(net.layers{L}.type, 'f'))
        %------------------------------------------------------------------
        % 以下代码对第7层(全连接层)有效
            
            % 典型的BP网络输出层对隐层的灵敏度(残差)的反向传播公式
            net.layers{L}.Delta = (net.layers{L+1}.W' * net.layers{L+1}.Delta) .* dy(net.layers{L}.X);
        
        elseif (strcmp(net.layers{L}.type, 's') || strcmp(net.layers{L}.type, 'c') || strcmp(net.layers{L}.type, 'i'))
        %------------------------------------------------------------------
        % 以下代码对第6层(过渡全连接层)有效            

            sa = size(net.layers{L}.X{1});                          % 每个输出通道图像尺寸(三维矢量,前两维是尺寸,第三维是批处理样本个数)
            fvnum = sa(1) * sa(2);                                  % 输出图像像素个数
            
            % 典型的BP网络输出层对隐层的灵敏度(残差)的反向传播公式
            net.layers{L}.Delta_Array = (net.layers{L+1}.W' * net.layers{L+1}.Delta);
            if strcmp(net.layers{L}.type, 'c')
                net.layers{L}.Delta_Array = net.layers{L}.Delta_Array .* dy(net.layers{L}.X_Array);
            end            
            
            for J = 1 : net.layers{L}.iChannel
                
                % 将本层长矢量灵敏度(残差),每一列为一个样本,reshape成通道表示 (矢量化全连接 -> 通道化全连接)
                net.layers{L}.Delta{J} = reshape(net.layers{L}.Delta_Array (((J - 1) * fvnum + 1) : J * fvnum, :), sa(1), sa(2));
%                 net.layers{L}.Delta{J} = reshape(net.layers{L}.Delta_Array (((J - 1) * fvnum + 1) : J * fvnum, :), sa(1), sa(2), sa(3));

            end
            
        %------------------------------------------------------------------            
        end
    end
    
    %======================================================================
    % 以下代码对“下一层”为“下采样层”时有效
    
    if (strcmp(net.layers{L+1}.type, 's'))        
        
        for J = 1 : net.layers{L}.iChannel                          % 对本层输出通道数做循环
            
            tmp1 = dy(net.layers{L}.X{J});                          % 为本层导数
            
            % 为上采样函数,这里用expand函数代替文献中的kron(kron仅适用二维情况)
            tmp2 = up(net.layers{L + 1}.Delta{J}, net.layers{L + 1}.iSample);
%             tmp2 = expand(net.layers{L + 1}.Delta{J}, [net.layers{L + 1}.iSample,net.layers{L + 1}.iSample,1]);   
            
            % net.layers{L}.Delta{J} = net.layers{L}.X{J} .* (1 - net.layers{L}.X{J}) .* (expand(net.layers{L + 1}.Delta{J}, [net.layers{L + 1}.iSample net.layers{L + 1}.iSample 1]) / net.layers{L + 1}.iSample ^ 2);
            % 与上式相比, 这里我认为最好不除以net.layers{L + 1}.iSample ^ 2, 因为在CNN正向计算函数cnnff中只做了下采样处理, 可以认为灵敏度(残差)是直接复制过去的.
            
            %##############################################################
            % 后改，以下代码用于下采样层的计算       
            net.layers{L}.Delta{J} = tmp1 .* tmp2;              
            net.layers{L}.Delta{J} = net.layers{L+1}.Beta{J} * net.layers{L}.Delta{J}; 
            
        end

    end
    
    %======================================================================
    % 以下代码对“下一层”为“卷积层”时有效
    
    if (strcmp(net.layers{L+1}.type, 'c'))        
        
        for I = 1 : net.layers{L}.iChannel                          % 对本层输出通道数做循环
            
            z = zeros(size(net.layers{L}.X{1}));
            for J = 1 : net.layers{L+1}.iChannel                    % 对下一层输出通道数做循环
                
                % 当前层灵敏度(残差)net.layers{L}.Delta{J}计算
                z = z + conv2(net.layers{L + 1}.Delta{J}, rot180(net.layers{L + 1}.Ker{I}{J},2), 'full');
%                 z = z + convn(net.layers{L + 1}.Delta{J}, rot180(net.layers{L + 1}.Ker{I}{J},2), 'full');

            end
            net.layers{L}.Delta{I} = z;
%             net.layers{L}.Delta{I} = dy(net.layers{L}.X{I}) .* net.layers{L}.Delta{I};            
        end        

    end
    
    %======================================================================
    
end

%% 求训练参数的梯度

% 特别注意：与Matlab并行版本不同，由于C采用是串行机制，以下训练参数在这里需要累加，之后在函数cnngradmean()中对一批样本做平均
% net.layers{L}.Ker_grad{I}{J}
% net.layers{L}.B_grad{J}
% net.layers{L}.W_grad
% net.layers{L}.B_grad
%
% 这里与 Notes on Convolutional Neural Networks 中不同，这里的 子采样 层没有参数，也没有
% 激活函数，所以在子采样层是没有需要求解的参数的

for L = 2 : n                                                       % 对CNN网络层数做循环(注意:这里实际上可以从第2层开始)
    
    %======================================================================
    
    if (strcmp(net.layers{L}.type, 'c'))     
        
        for J = 1 : net.layers{L}.iChannel                          % 对本层输出通道数做循环
            
            for I = 1 : net.layers{L-1}.iChannel                    % 对上一层输出通道数做循环
                
                % 特别注意:
                % (1)等价关系 rot180(conv2(a,rot180(b),'valid')) = conv2(rot180(a),b,'valid')
                % (2)若ndims(a)=ndims(b)=3,则convn(filpall(a),b,'valid')表示三个维度上同时相关运算
                % (3)若size(a,3)=size(b,3),则上式输出第三维为1,表示参与训练样本的叠加和(批处理算法),结果要对样本数做平均
                
                net.layers{L}.Ker_grad{I}{J} = net.layers{L}.Ker_grad{I}{J} + conv2(rot180(net.layers{L - 1}.X{I},2), net.layers{L}.Delta{J}, 'valid');
%                 net.layers{L}.Ker_grad{I}{J} = convn(rot180(net.layers{L - 1}.X{I},3), net.layers{L}.Delta{J}, 'valid') / size(net.layers{L}.Delta{J}, 3);
                
            end
            
            % 对所有net.layers{L}.Delta{J}的叠加,结果要对样本数做平均
            net.layers{L}.B_grad{J} = net.layers{L}.B_grad{J} + sum(net.layers{L}.Delta{J}(:));
%             net.layers{L}.B_grad{J} = sum(net.layers{L}.Delta{J}(:)) / size(net.layers{L}.Delta{J}, 3);
            
        end
              
    end
    
    %######################################################################
    % 后改，以下代码用于下采样层的计算     
    
    if (strcmp(net.layers{L}.type, 's'))   
       
        for J = 1 : net.layers{L}.iChannel                          % 对本层输出通道数做循环
            
            net.layers{L}.Beta_grad{J} = net.layers{L}.Beta_grad{J} + sum(net.layers{L}.Delta{J}(:) .* net.layers{L}.X_down{J}(:));
%             net.layers{L}.Beta_grad{J} = sum(net.layers{L}.Delta{J}(:) .* net.layers{L}.X_down{J}(:)) / size(net.layers{L}.Delta{J}, 3);            
            
            % 对所有net.layers{L}.Delta{J}的叠加,结果要对样本数做平均
            net.layers{L}.B_grad{J} = net.layers{L}.B_grad{J} + sum(net.layers{L}.Delta{J}(:));      
%             net.layers{L}.B_grad{J} = sum(net.layers{L}.Delta{J}(:)) / size(net.layers{L}.Delta{J}, 3);  
        end
        
    end    
    
    %======================================================================
    
    if (strcmp(net.layers{L}.type, 'f'))

        if (strcmp(net.layers{L-1}.type, 's') || strcmp(net.layers{L-1}.type, 'c') || strcmp(net.layers{L-1}.type, 'i'))
        %------------------------------------------------------------------
        % 以下代码对第6层(过渡全连接层)有效           
        
            % 权值矩阵梯度,结果要对样本数做平均
            net.layers{L}.W_grad = net.layers{L}.W_grad + net.layers{L}.Delta * (net.layers{L-1}.X_Array)';
%             net.layers{L}.W_grad = net.layers{L}.Delta * (net.layers{L-1}.X_Array)' / size(net.layers{L}.Delta, 2);

            % 输出层灵敏度(残差)就是偏置(加性)的梯度,这里也要对样本数做平均              
            net.layers{L}.B_grad = net.layers{L}.B_grad + net.layers{L}.Delta;
%             net.layers{L}.B_grad = mean(net.layers{L}.Delta, 2);
        
        elseif (strcmp(net.layers{L-1}.type, 'f'))
        %------------------------------------------------------------------
        % 以下代码对第7层(全连接层)有效
        
            % 权值矩阵梯度,结果要对样本数做平均
            net.layers{L}.W_grad = net.layers{L}.W_grad + net.layers{L}.Delta * (net.layers{L-1}.X)';
%             net.layers{L}.W_grad = net.layers{L}.Delta * (net.layers{L-1}.X)' / size(net.layers{L}.Delta, 2);            

            % 输出层灵敏度(残差)就是偏置(加性)的梯度,这里也要对样本数做平均              
            net.layers{L}.B_grad = net.layers{L}.B_grad + net.layers{L}.Delta;
%             net.layers{L}.B_grad = mean(net.layers{L}.Delta, 2);
        
        %------------------------------------------------------------------            
        end
    end
    
    %======================================================================
    
end

end
