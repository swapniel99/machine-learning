function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = [];
iter=1;
cost1=computeCostMulti(X,y,theta);

while true,

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    J_history=[J_history;cost1];
    theta=theta-alpha*(X'*(X*theta - y))/m;
    cost2=computeCostMulti(X,y,theta);
    if abs(cost2-cost1)<0.000001,
	printf('Convergence at iteration %d\n',iter);
	break;
    end;
    iter=iter+1;
    cost1=cost2;

    % ============================================================

    % Save the cost J in every iteration    
%    J_history(iter) = computeCostMulti(X, y, theta);

end

end
