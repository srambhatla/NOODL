% Function to find the best sparse approximation using FISTA or stochastic
% version of ISTA depending upon how big the task is. This also scans
% across the range of hyperparameters to find the best approximation. 

function [X_fi_best, i, t_f] = best_fista_result(A, Y_s, X_fi_init, X_o_fi, tol, i_max)
tic
err = [];

lam_max = 2*max(max(abs(A'*Y_s)));
bestErr = 1000000;

display('In Fista, will take some time...')

    for i = 1:i_max
    % Gradually scan the space of regularization parameter
    lam = (0.7^(i_max-i))*lam_max;

    % Decide whether to run stochastic-ISTA or FISTA
    if (size(Y_s,2)>2000)
        X_fi_est = FISTA_with_init_stochastic(A, Y_s, lam, X_fi_init, tol);
    else
        X_fi_est = FISTA_with_init(A, Y_s, lam, X_fi_init, tol);
    end
        err = [err norm(X_fi_est - X_o_fi,'fro')/norm(X_o_fi,'fro')]; 
  
    if(err(end) < bestErr )
        bestErr = err(end);
        best_X_fi_est =  X_fi_est;
        bestId = i; 
    end
    
    if(err(end) <= tol) 
        X_fi_best = X_fi_est;
        t_f = toc;  
         break; 
    end
    X_fi_init = X_fi_est;
   
     % plot(err)
     % drawnow
    end

    if (i == i_max)
         err = bestErr;
         X_fi_est =  best_X_fi_est; 
    end
        
  X_fi_best = X_fi_est;
t_f = toc;  
end
