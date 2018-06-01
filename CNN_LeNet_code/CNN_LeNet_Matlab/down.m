function [Y] = down(X,iSample)
% 图片下采样函数,行列采样倍数为iSample

if ndims(X)==2
    
    % 以尺度ones(net.layers{L}.iSample)对上一层进行卷积运算,相当于均值滤波
    Z = conv2(X, ones(iSample) / iSample ^ 2, 'valid');
    
    % 以间隔net.layers{L}.iSample对以上结果行列下采样
    Y = Z(1:iSample:end, 1:iSample:end);
    
elseif ndims(X)==3
    
    % 以尺度ones(net.layers{L}.iSample)对上一层进行卷积运算,相当于均值滤波
    Z = convn(X, ones(iSample) / iSample ^ 2, 'valid');
    
    % 以间隔net.layers{L}.iSample对以上结果行列下采样
    Y = Z(1:iSample:end, 1:iSample:end, :);
    
end

end

