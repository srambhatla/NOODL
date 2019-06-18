function [A] = FISTA_with_init_stochastic(D, Y, lam, Xinit, tol)
% Implementation of Stochastic ISTA procedure with warm start
% to solve the subproblem
% X_est = argmin_X ||D*A-Y||_F^2  + lam*||A||_1
% (please pardon the somewhat misleading name of the file!)

n = size(D,2);
m = size(Y,2);

% Number of iterations
k = 100;

% Lipschitz constant
L=2*max(eig(D'*D));

eps = tol; 

% Initialize parameters
%Xinit = zeros(n,m); 
y_s = Xinit;
x_sm1 = y_s;
t_s = 1;

der = @(x) 2*D'*(D*x-Y);
F = @(x) norm(D*x-Y)^2 + lam*norm(x,1); 
der_sto = @(x,y) 2*(D')*D*x-2*D'*y;

err = [];
%%
% FISTA Iterations
stopval = 0;
its = 0;
batch_siz = 1000;

while (its < k) && (stopval == 0)
    
    id = randi([1,m],1,batch_siz);
    Y_sto = Y(:, id);
    y_s_sto = y_s(:, id);
    y_sp1 = y_s;
    gr = kron(ones(1,m), mean(der_sto(y_s_sto,Y_sto),2)) ;
    
    % Take a (proximal) gradient step
    y_s= softThr(y_s-(0.5/L)*gr, lam/L);
    
    % err = [ err norm(Y - D*x_s)/norm(Y)];
    % semilogy(err)
    % drawnow
    
    its = its + 1;
end
A = y_s;
