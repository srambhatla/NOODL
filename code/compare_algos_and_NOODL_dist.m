% Comparing performance of NOODL with other techniques
%
% Sirisha Rambhatla, March 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: compare_algos_and_NOODL_dist.m
%
% Description: This function compares NOODL with other related techniques. 
% Specifically, we compare with techniques in Arora '15 and the online
% dictionary learning (ODL) algorithm by Mairal et. al.
%
% Note 1: This ties in with the main.m function and assumes that an initial
% estimate of the dictionary is known. Also, for these synthetic
% simulations it assumes that the true dictionary is known as well. 
%
% Note 2: For real-world applications where the true dictionary (A_o) is
% not known, the outer loop, which currently depends on "change_A" variable
% (it measures how close our estimate A_our is to A_o), can be made to
% run for a specified number of outer-loop iterations T. Instead of the
% while loop one can use a for loop that runs for T iterations. The input
% to the algorithm also will change accordingly.

% Descriptions of the outputs

% Set 'out_folder = []' if you don't want to save anything

% X_last_o -- the ground truth X matrix at the last iteration
% Y_last -- the corresponding data matrix at the last iteration 

% NOODL
% A_our -- NOODL estimate of factor A
% X_last -- the final estimate of X_last_o
% errA -- trajectory of the error in factor A
% errX -- trajectory of the error in X (sparse matrix)
% err -- trajectory of fit error
% time_our -- time taken by TensorNOODL at each iteration

% Arora (biased)
% A_arora -- Arora(biased) estimate of factor A
% errA_arora -- trajectory of the error in factor A
% errX_arora -- trajectory of the error in X (sparse matrix)
% err_arora -- trajectory of fit error
% time_arora -- time taken at each iteration

% Arora (unbiased)
% A_arora_red -- Arora(unbiased) estimate of factor A
% errA_arora_red -- trajectory of the error in factor A
% errX_arora_red -- trajectory of the error in X (sparse matrix)
% err_arora_red -- trajectory of fit error
% time_arora_red -- time taken at each iteration

% Mairal's Online Dictionary Learning (ODL)
% A_odl -- Mairal's estimate of factor A
% errA_odl -- trajectory of the error in factor A
% errX_odl -- trajectory of the error in X (sparse matrix)
% err_odl -- trajectory of fit error
% time_odl -- time taken at each iteration

%%
function [A_our, A_arora, A_arora_red, errA, errX, err,  err_arora, errA_arora, errX_arora, ...
    err_arora_red, errA_arora_red, errX_arora_red, Y_last, X_last, X_last_o, X_arora_last, X_arora_red_last, ...
    time_our, time_arora_red, time_arora]...
    = compare_algos_and_NOODL_dist(A, A_o, k, p, eta_x, thr, C, eta_A, eta_A_arora, eta_A_arora_red, tol_X, tol_A, out_folder, show)

display('II. Making Noodl: Refining Dictionary via the Noodl Algorithm .....')

n = size(A,1);
m = size(A,2);

% Parameters for generating coefficient matrices
C_x = C; 

% Initialize all dictionaries with our initial estimate of A_o 
A_our = A;
A_arora = A;
A_arora_red = A;
A_odl = A;

% Initialize all errors 
errA = []; errA_arora = []; errA_arora_red = []; errA_odl = [];
errX = []; errX_arora = []; errX_arora_red = []; errX_odl = [];
err = []; err_arora = []; err_arora_red = []; err_odl = [];

% Initilize timings 
time_our = [];
time_arora_red = [];
time_arora = [];
time_odl = [];
time_odl_coeff = [];

% ODL (Mairal) Specific
XS_odl = zeros(m, p);
E = zeros(m, m);
F = zeros(n, m);

i = 1;
change_A = norm(A - A_o, 'fro')/norm(A_o,'fro');

if show
figure('pos',[1000 1500 1200 600])
end

display('Begin comparing...')

% Begin Alternating Optimization
while((change_A>tol_A))   
    
   % Generate new samples
   X_new = simple_gen_coeff_mat_X_upto(m, k, p, C_x);
   Y_m = A_o*X_new;

   tic
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
   
   % Start distributed processing
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
   
   % Dictionary Update
   
   % Form the gradient estimate for dictionary update 
   gr = double(1/p)*(Y_m - (A_our*XS))*sign(XS)';

   % Descend
   A_our = A_our + (eta_A)*gr;
   A_our = nrmc(A_our);
   
   t_o = toc; % time to NOODL
   
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
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Arora "Unbiased"
   
   tic % Start time for Arora "Unbiased"
   B = zeros(n,m,m);
   for iii = 1:m
       dict_ele = setdiff([1:m],[iii]);
       B_temp = A_arora_red(:,dict_ele) - (kron((A_arora_red(:,iii)'*A_arora_red(:,dict_ele)),A_arora_red(:,iii)));
       B(:,dict_ele,iii) = B_temp;
       B(:,iii, iii) = A_arora_red(:,iii);
   end
   
   % Hard Thresholding
   AtY = A_arora_red'*Y_m;
   XS_arora_red = AtY.*(abs(AtY)>=C/2);
   
   % Gradient Estimate
   gr_arora_red = zeros(size(A));
   for iii = 1:m
       temp_x_i = squeeze(B(:,:,iii))'*Y_m;
       temp_x_i = temp_x_i.*(abs(temp_x_i)>=C/2);
       gr_arora_red(:,iii) =  mean((Y_m - squeeze(B(:,:,iii))*temp_x_i).*kron(sign(XS_arora_red(iii,:)), ones(n,1)), 2);
   end
   
   % Update Dictionary (Descend)
   A_arora_red = A_arora_red + (eta_A_arora_red)*gr_arora_red;
   A_arora_red = nrmc(A_arora_red);
  
   t_ar1 = toc; % Time to Arora "Unbiased"
   
   % Set errors
   err_arora_red = [err_arora_red norm((Y_m - A_arora_red*XS_arora_red),'fro')/norm(Y_m,'fro')];  
   errA_arora_red = [errA_arora_red norm(A_arora_red - A_o,'fro')/norm(A_o,'fro')];
  
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Arora "Biased"
   
   tic % Start time for Arora "Biased"
   
   % Hard Thresholding
   AtY = A_arora'*Y_m;
   XS_arora = AtY.*(abs(AtY)>=C/2);
  
   % Gradient
   gr_arora = (1/p)*(Y_m - (A_arora*XS_arora))*sign(XS_arora)';

   % Update Dictionary
   A_arora = A_arora + (eta_A_arora)*gr_arora;
   A_arora = nrmc(A_arora);
   
   t_a1 = toc; % Time to Arora "Biased"

   % Set errors
   err_arora = [err_arora norm((Y_m - A_arora*XS_arora),'fro')/norm(Y_m,'fro')];  
   errA_arora = [errA_arora norm(A_arora - A_o,'fro')/norm(A_o,'fro')];

% Uncomment to estimate coefficients at every step (for Arora '15 methods)

% display('Finding good coefficients') 
  
%  % Output Best Coefficient Estimates
%   spmd
%     if(labindex ==1)
%        tic
%         [XS_arora_red_ol i_ar] = best_fista_result(A_arora_red, Y_m, XS_arora_red, X_new, tol_X, 10);
%         t_ar2 = toc;
%     elseif(labindex ==2)
%        tic
%        [XS_arora_ol i_a] = best_fista_result(A_arora, Y_m, XS_arora, X_new, tol_X, 10);
%        t_a2 = toc;
%     end
%   end
%   XS_arora_red_ol = XS_arora_red_ol{1};
%   XS_arora_ol = XS_arora_ol{2};

 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Mairal's ODL
   
   % Estimate coefficients (we use FISTA to solve Lasso)
   [XS_odl, i_odl, t_odl2] = best_fista_result(A_odl, Y_m, zeros(m, p), X_new, tol_X, 10);

   % ODL dictionary update 
   
   tic % Start time for ODL
   if i < p
       theta = i*p;
   else
       theta = p^2 + i - p; 
   end
   
   beta = (theta + 1 - p)/(theta + 1);
       
   E = beta*E + XS_odl*XS_odl';
   F = beta*F + Y_m*XS_odl';
   
   U = repmat(1./(diag(E)'),n,1).*(F - A_odl*E) + A_odl;
   A_odl = nrmc(U);
   
   t_odl1 = toc; % Time to Mairal's ODL
   
   % Set errors for ODL
   errA_odl = [errA_odl norm(A_odl - A_o,'fro')/norm(A_o,'fro')];
   err_odl = [err_odl norm((Y_m - A_odl*XS_odl),'fro')/norm(Y_m,'fro')];   
   
   % Set errors for coefficients estimates for Arora '15 and odl
   errX_arora_red = [errX_arora_red  norm(XS_arora_red - X_new,'fro')/norm(X_new,'fro')];
   errX_arora = [errX_arora  norm(XS_arora - X_new,'fro')/norm(X_new,'fro')];
   errX_odl = [errX_odl norm(XS_odl - X_new,'fro')/norm(X_new,'fro')];
   
   % Calculate time it takes for each method
   
   % uncomment for coefficient estimation by for Arora '15
   %t_ar = t_ar1 + t_ar2/i_ar{1};
   %t_a = t_a1 + t_a2/i_a{2};
   t_odl = t_odl1 + t_odl2/i_odl;
   t_ar = t_ar1;
   t_a = t_a1;
   
   time_our = [time_our t_o];
   time_arora_red = [time_arora_red t_ar];
   time_arora = [time_arora t_a];
   time_odl = [time_odl t_odl];
   time_odl_coeff = [time_odl_coeff t_odl2/i_odl];   
   
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
  
   display(['Our: iter = ', num2str(i), '   ,errA = ', num2str(errA(end)), '   ,errX = ', num2str(errX(end)), ', time = ', num2str(time_our(end))])
   display(['Arora "Unbiased": iter = ', num2str(i), '   ,errA = ', num2str(errA_arora_red(end)), '   ,errX = ', num2str(errX_arora_red(end)),', time = ', num2str(time_arora_red(end))])
   display(['Arora "Biased": iter = ', num2str(i), '   ,errA = ', num2str(errA_arora(end)), '   ,errX = ', num2str(errX_arora(end)), ', time = ', num2str(time_arora(end))])
 display(['ODL Sapiro: iter = ', num2str(i), '   ,errA = ', num2str(errA_odl(end)), '   ,errX = ', num2str(errX_odl(end)), ', time = ', num2str(time_odl(end)), ', time coeff = ', num2str(time_odl_coeff(end))])
   i = i + 1;
end


  % Calculate Best Coefficient Estimates for Arora '15 methods at the end
  spmd
    if(labindex ==1)
        [XS_arora_red_ol, i_ar, t_ar_red_coeff] = best_fista_result(A_arora_red, Y_m, XS_arora_red, X_new, tol_X, 10);
    elseif(labindex ==2)
        [XS_arora_ol, i_a, t_ar_coeff] = best_fista_result(A_arora, Y_m, XS_arora, X_new, tol_X, 10);
    end
  end
  
  XS_arora_red_ol = XS_arora_red_ol{1};
  XS_arora_ol = XS_arora_ol{2};

  t_ar2 = t_ar_red_coeff{1}/i_ar{1};
  t_a2 = t_ar_coeff{2}/i_a{2};


display(['Err Coeff "unbiased" after lasso ',num2str(norm(XS_arora_red_ol - X_new,'fro')/norm(X_new,'fro')),' time coeff',t_ar_red_coeff{1},'iter',i_ar{1}])
display(['Err Coeff "biased" after lasso ',num2str(norm(XS_arora_ol - X_new,'fro')/norm(X_new,'fro')),' time coeff',t_ar_coeff{2},'iter coeff',i_a{2}])

 Y_last = Y_m;
 X_last = XS;

 X_last_o = X_new;
 X_arora_last = XS_arora_ol;
 X_arora_red_last = XS_arora_red_ol;
 X_odl_last = XS_odl;

if (~isempty(out_folder))
    name = strcat(out_folder,'res_coeff_end_n_',num2str(n),'_m_',num2str(m),'_k_',...
    strrep(num2str(k),'.','_'),'_etaA_',strrep(num2str(eta_A),'.','_'),'_p_',num2str(uint64(p)),'.mat')
    save(name, 'A_our', 'A_arora', 'A_arora_red','A_odl', 'errA', 'errX', 'err', 'err_arora', 'errA_arora', 'errX_arora', ...
        'err_arora_red', 'errA_arora_red', 'errX_arora_red', 'errA_odl', 'errX_odl', 'err_odl','Y_last', 'X_last', 'X_last_o', 'X_arora_last','X_arora_red_last','X_odl_last', 'A', 'A_o', 'k_x', 'pf', 'c', 'thr_c', 'eta', 'eta_arora', 'eta_arora_red', ...
        'tol_X', 'tol_A', 'time_our', 'time_arora_red', 'time_arora','time_odl_coeff','time_odl','t_ar2','t_a2')
end

