function [W b] = hinge_loss_square_max_margin(X, y, C)
    [m n] = size(X);
    cvx_begin
        variables W(n) b
        minimize(0.5*W'*W + (C/m)*sum(max(0,(1-y.*(X*W-b)).^2)))
    cvx_end
end
