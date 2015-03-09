function [W b] = hinge_loss_max_margin(X, y, C)
    [m n] = size(X);
    cvx_begin
        variables W(n) b
        minimize(0.5*W'*W + (C/m)*sum(max(0,y.*(X*W-b))))
    cvx_end
end
