function [W b] = hinge_loss(X, y)
    [m n] = size(X);
    cvx_begin
        variables W(n) b
        minimize((1/m)*sum(max(0,y.*(X*W-b))))
    cvx_end
end
