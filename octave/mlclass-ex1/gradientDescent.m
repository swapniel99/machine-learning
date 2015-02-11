function [theta,J_history] = gradientDescent(X, y, theta, alpha)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history=[];
cost1=computeCost(X,y,theta);
iter=1;
while true

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    J_history=[J_history;[theta;cost1]'];
    theta=theta-alpha*(X'*(X*theta - y))/m;
    cost2=computeCost(X,y,theta);
    if (abs(cost2-cost1)<0.000001),
	printf('Convergence at iteration %d.\n',iter);
	break;
    end;
    iter=iter+1;
    cost1=cost2;
    % ============================================================

    % Save the cost J in every iteration
%    J_history(iter) = computeCost(X, y, theta);

end

end
