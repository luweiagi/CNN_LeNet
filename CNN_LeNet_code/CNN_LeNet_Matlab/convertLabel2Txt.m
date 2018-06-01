function [] = convertLabel2Txt(TARSVM,path)

%length = size(TARSVM);
fid = fopen(path,'wt');
%for i = 1:length
    fprintf(fid,'%g\n',TARSVM);
%end
fclose(fid);
    