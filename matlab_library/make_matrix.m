function [M] = make_matrix(m, n)
    % make a matrix with rapdily decaying singular values
    fprintf('making matrix..\n');

    p = min(m,n);
    if m >= n
       [U, temp] = qr(randn(m,n),0);
       [V, temp] = qr(randn(n));
    else
       [U, temp] = qr(randn(m));
       [V, temp] = qr(randn(n,m),0);
    end


    if p>1
        S = logspace(1,-2,p);
    else
        S = [1];
    end
    S = diag(S);
    M = U*S*V';

end

