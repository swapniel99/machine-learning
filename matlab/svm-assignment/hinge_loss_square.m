function [W b] = hinge_loss_square(X, y)
    [m n] = size(X);
    cvx_begin
        variables W(n) b
        minimize((1/m)*sum(max(0,(1-y.*(X*W-b)).^2)))
    cvx_end
end
