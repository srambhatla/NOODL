function [A] = FISTA_with_init(D,Y,lam, Xinit, tol)
% Implementation of FISTA procedure with warm start
% to solve the subproblem
%
% X_est = argmin_X ||D*A-Y||_F^2  + lam*||A||_1

n = size(D,2);
m = size(Y,2);

% Number of iterations
k = 75;

% Lipschitz constant
L = 2*max(eig(D'*D));

eps = tol; 

% Initialize parameters
%Xinit = zeros(n,m);
y_s = Xinit;
x_sm1 = y_s;
t_s = 1;

der = @(x) 2*D'*(D*x-Y);
F = @(x) norm(D*x-Y)^2 + lam*norm(x,1); 

err = [];
%% FISTA Iterations
stopval = 0;
its = 0;

while (its < k) && (stopval == 0)

    % Take a (proximal) gradient step 
    z_s= softThr(y_s-(1/L)*der(y_s),lam/L);
  
    % Ensure descent
    if(F(z_s) < F(x_sm1))
        x_s = z_s;
    else
        x_s = x_sm1;
    end
    
    % Calculate new parmaters
    t_sp1 = (1/2)*(1 + sqrt(1+4*t_s^2));
    
    % Update y
    y_sp1 = x_s + (t_s / t_sp1) * (z_s - x_s) + ((t_s - 1)/t_sp1) * (x_s - x_sm1);
    
    NN = norm(y_s - y_sp1, 'fro') / norm(y_sp1, 'fro');
    stopval = (NN < eps) || isnan(NN);
    
    err=[err  NN];
    % err = [ err norm(Y - D*x_s)/norm(Y)];
    % semilogy(err)
    % drawnow
   
    % Set up quantities for next iteration
    z_sm1 = z_s;
    y_s = y_sp1;
    t_s = t_sp1;
    x_sm1 = x_s;
    
    % Update iteration count
    its = its + 1;
end
A = x_s;


