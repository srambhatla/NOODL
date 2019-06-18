% NOODL: Neurally Plausible alternating Optimization-based Online Dictionary Learning 

% Sirisha Rambhatla, March 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: NOODL_dist.m
%
% Description: This file implements distributed NOODL. It uses MATLAB's
% spmd for distributed implementation. 

% Note 1: Consider the trade-off between latency of communication vs. 
% processing to decide how to distribute the samples across workers.
%
% Note 2: This ties in with the main.m function and assumes that an initial
% estimate of the dictionary is known. Also, for these synthetic
% simulations it assumes that the true dictionary is known as well. 
%
% Note 3: For real-world applications where the true dictionary (A_o) is
% not known, the outer loop, which currently depends on "change_A" variable
% (it measures how close our estimate A_our is to A_o), can be made to
% run for a specified number of outer-loop iterations T. Instead of the
% while loop one can use a for loop that runs for T iterations. The input
% to the algorithm also will change accordingly.

function [A_our, errA, errX, err, Y_last, X_last, X_last_o] = NOODL_dist(A, A_o, k, p, eta_x, thr, C, eta_A, tol_X, tol_A, show)

display('II. Making Noodl: Refining Dictionary via the Noodl Algorithm .....')


n = size(A,1);
m = size(A,2);

% Parameters for generating coefficient matrices
C_x = C; 

% Set our initial estimate to input initialization
A_our = A;

errA = []; errX = []; err = []; 

i = 0;
change_A = norm(A - A_o, 'fro')/norm(A_o,'fro');

if show
figure('pos',[1000 1500 1200 600])
end

% Begin Alternating Optimization
while(change_A>tol_A)   
    
   % Generate new samples 
   X_new = simple_gen_coeff_mat_X_upto(m, k, p, C_x);
   Y_m = A_o*X_new;
   
   % Hard Thresholding
   AtY = A_our'*Y_m;
   XS = AtY.*(abs(AtY)>=C/2);
   AtA_our = A_our'*A_our; % store for easy calculations later on
   
   % Clear some parameters which check the progress of the alg.
   clear change_X change_X_g
   
   % Set thing to be able to run MATLAB's spmd for distributed processing
   % of samples.
   
   change_X_g= Composite();
   err_int_X_g = Composite();
   
   
   for lab = 1:length(change_X_g)
       change_X_g{lab} = 1;
       err_int_X_g{lab} = [];
   end
   
   ii = 0;
   tic % to note how much time it takes for distributed processing; See Note 1.
   spmd
       
       % Distribute variabled to prep data for spmd command
       XS_dist = codistributed(XS);
       X_new_dist = codistributed(X_new, getCodistributor(XS_dist));
       Y_m_dist = codistributed(Y_m, getCodistributor(XS_dist));
       
       change_X = codistributed(change_X_g);
       err_int_X = codistributed(err_int_X_g);
      
       local_XS = getLocalPart(XS_dist);
       dist = getCodistributor(XS_dist);
       local_X_new = getLocalPart(X_new_dist);
       local_Y_m = getLocalPart(Y_m_dist);
    
       err_int_X =  norm(local_XS(:) -  local_X_new(:))/norm( local_X_new(:));
     
       % Begin Coefficient Update
       while(change_X > tol_X)

           % Take IHT step
           local_XS = local_XS - eta_x*((AtA_our*local_XS) - A_our'*local_Y_m ); 
           local_XS(abs(local_XS) <= thr) = 0;
 
           err_int_X = [err_int_X norm(local_XS(:) - local_X_new(:))/norm(local_X_new(:))];
           change_X = abs(err_int_X(end) - err_int_X(end-1));
        
           % Display progress
           if(show)
               subplot(121)
               semilogy(err_int_X)
               title({'Convergence of Current Coefficient Learning Step', ['n = ', num2str(n), ' m = ', num2str(m), ' k = ', num2str(k)]})
               ylabel('Relative error in current coefficient estimate')
               xlabel('Iterations')
               set(gca, 'FontSize',16)
               drawnow
           end
   
           ii = ii + 1;
     
       end
       
       % Collect processed data
       XS_dist = codistributed.build(local_XS, dist, 'noCommunication');
       X_new_dist = codistributed.build(local_X_new, dist, 'noCommunication');
       Y_m_dist = codistributed.build(local_Y_m, getCodistributor(Y_m_dist), 'noCommunication');
       labBarrier;
   end
   
   % Collect processed data
   XS = gather(XS_dist);
   X_new = gather(X_new_dist);
   Y_m = gather(Y_m_dist);

   toc % time for spmd
   
 
   % Dictionary Update
   
   % Form the gradient estimate for dictionary update 
   gr = double(1/p)*(Y_m - (A_our*XS))*sign(XS)';

   % Descend
   A_our = A_our + (eta_A)*gr;
   A_our = nrmc(A_our);
     
   % Set errors
   err = [err norm((Y_m - A_our*XS),'fro')/norm(Y_m,'fro')];  
   errA = [errA norm(A_our - A_o,'fro')/norm(A_o,'fro')];
   errX = [errX  norm(XS - X_new,'fro')/norm(X_new,'fro')];
   
   % Some checks to see if everything works properly
   if(isnan(errX(end))||isnan(errA(end)))
       break;
   end
   
   % Need to exit?
   change_A = errA(end);
   
   % Plot
   if(show)

       subplot(122)
       semilogy(errA,  'LineWidth',2)
       hold all
       semilogy(errX,  'LineWidth',2)
       semilogy(err, 'LineWidth',2)
       legend('Error in Dictionary', 'Error in Current Coefficients','Fit Error')
       set(gca, 'FontSize',16)
       grid on
       title({'Convergence of Online Dictionary Learning Algorithm', ['n = ', num2str(n), ' m = ', num2str(m), ' k = ', num2str(k)]})
       ylabel('Cost')
       xlabel('Dictionary Iterations (per fresh sample batch)')
       hold off

       drawnow

   end
   i = i + 1;
   display(['iter = ', num2str(i), '   ,errA = ', num2str(errA(end)), '   ,errX = ', num2str(errX(end))])
end
 
% Store the data and the estimates corresponding to the last iterate of
% the online algorithm. 
Y_last = Y_m;
X_last = XS;
X_last_o = X_new;
 