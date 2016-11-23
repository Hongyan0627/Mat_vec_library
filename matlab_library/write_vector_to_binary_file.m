function write_vector_to_binary_file(v, bin_file)
% write vector to binary file

m = length(v);

fp = fopen(bin_file,'w');

fwrite(fp,m,'int32');

% write nnzs
for j=1:m
	fwrite(fp,v(j),'double');
end
fclose(fp);

end

