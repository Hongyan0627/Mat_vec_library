function [x, b] = make_vec_x_noise_b(A)
% make random vectro x, and Ax = b, b has noise

n = size(A,2);
x = randn(n,1);

b = A*x; 
% can also add some noise to b
% add some Gaussian noise
temp=randn(length(b),1);
noise = noise_frac*norm(b)/norm(temp)*temp;
b = b + noise;

end

