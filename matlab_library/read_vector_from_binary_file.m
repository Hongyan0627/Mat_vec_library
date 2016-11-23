function [v] = read_vector_from_binary_file(bin_file)
% read vector from binary file

fp = fopen(bin_file,'r');

n = fread(fp,1,'int32');
v = zeros(n,1);
% write nnzs
for j=1:n
	v(j) = fread(fp,1,'double');
end
fclose(fp);

end

