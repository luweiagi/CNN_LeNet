function [] = convertData2Img(SAMPLE,path)

 if ~exist(path) 
    mkdir(path)         % 若不存在，在当前目录中产生一个子目录‘Figure’
 end 

[m,n,l] = size(SAMPLE);
image = zeros(m,n);
for iIndex = 1:l    
    image = SAMPLE(:,:,iIndex); 
    %iNameIndex = int2str(iIndex);
    name = [path, '/', int2str(iIndex),'.bmp'];    
    imwrite(image,name);
end
    


