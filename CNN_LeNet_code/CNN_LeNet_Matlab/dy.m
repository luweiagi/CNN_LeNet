function [d] = dy(y)
% 非线性映射Sigmoid函数y=f(x)=1/(1+e(-x)),输入为y,输出为y对x的导数

d = y.*(1-y);

end

