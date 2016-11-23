function [M] = make_matrix_specified_decay(m,n,S)
    % makes a matrix of given dimensions and with singular values 
    % specified by the diagonal matrix S
    fprintf('making matrix..\n');
    p = min(m,n);
    if m >= n
        [U, temp] = qr(randn(m,n),0);
        [V, temp] = qr(randn(n));
    else
        [U, temp] = qr(randn(m));
        [V, temp] = qr(randn(n,m),0);
    end

    size(U)
    size(V)

    M = U*S*V';
end

