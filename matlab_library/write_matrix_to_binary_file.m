function write_matrix_to_binary_file(A, bin_file)
% write matrix to binary file, column major format
[m,n] = size(A);

fprintf('write matrix A to binary file\n');

fp = fopen(bin_file,'w');
fwrite(fp,m,'int32');
fwrite(fp,n,'int32');
% write nnzs, each column at a time
for j=1:n
    fprintf('writing column %d of %d\n', j,n);
    for i=1:m
        fwrite(fp,A(i,j),'double');
    end
end
fclose(fp);

end

