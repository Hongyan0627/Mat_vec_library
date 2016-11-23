function [A] = make_spd_matrix(n)
% make a SPD matrix

M = make_matrix(1000,n);

A = M' * M;

end

