function X = rot180(X,dim)
% 对多维矩阵第1->dim维数据翻转180度

if nargin<2
    dim = 2;
end

for i=1:min(ndims(X),dim);
    X = flipdim(X,i);
end

end