% NOODL: Neurally Plausible alternating Optimization-based Online Dictionary Learning 
% Sirisha Rambhatla, March 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: main.m
%
% Description: Use this as your entry point to NOODL. This file implements
% various types of NOODL functions. Both distributed and non-distributed
% versions. In addition, we also show distributed and non-distributed
% versions of the comparisons with other algorithms (Arora '15 and Mairal
% et. al.).
%
% Note: Here we generate an initial dictionary estimate which is epsilon
% close to the true dictionary. In practice, initial dictionary estimate can
% be recovered using the initialization algorithms such as those presented 
% in Arora '15.

clc
close all 
clear all
display('Welcome, Let''s make some Noodl.....')
  
% Set random seed for reproducibility
rng(42)
verbose = 1;
success = 0;

%% Parameters

% Data Generation 
n= 200;
m = 250;

% Set parameters for sparse coefficients
k = 8; C_x = 1; 
epsDict = 0.4;

%% Prep Noodl : Initialize dictionary

% The true dictionary
A_o = randn(n, m); A_o = nrmc(A_o);

% Add noise
noise =  nrmc(randn(size(A_o))); noise = 2*noise*(1/log(n));
A = A_o + noise; 

% Normalize columns
A = nrmc(A);

%% Make Noodl: Alternating Minimization to refine the Dictionary: non-distributed
show = 1; eta_A = 50; p = 500 ;C = 1;
eta_x = 0.2; thr = 0.1;
tol_X = 1e-7;
tol_A = 1e-6;

close all
tic
[A, errA, errX, err, Y_last, X_last, X_last_o] = NOODL(A, A_o, k, p, eta_x, thr, C, eta_A, tol_X, tol_A, show);
learn_time = toc;
display(['Noodl is ready! It took ', num2str(learn_time/60, 3), ' minutes to make Noodl.'])
done = 1

%% Make Noodl: Alternating Minimization to refine the Dictionary: distributed

% Run parpool if it is not already running
if(isempty(gcp('nocreate')))
    parpool('local',5);
end

show = 0; eta_A = 15; p = 500 ;C = 1;
eta_x = 0.2; thr = 0.1;
tol_X = 1e-7;
tol_A = 1e-6;

close all
tic;
[A, errA, errX, err, Y_last, X_last, X_last_o] = NOODL_dist(A, A_o, k, p, eta_x, thr, C, eta_A, tol_X, tol_A, show);
learn_time = toc;
display(['Noodl is ready! It took ', num2str(learn_time/60, 3), ' minutes to make Noodl.'])
done = 1

%% Make Noodl: Alternating Minimization to refine the Dictionary: distributed

% Run parpool if it is not already running
if(isempty(gcp('nocreate')))
    parpool('local',5);
end

show = 0; eta_A = 15; p = 500 ;C = 1;
eta_x = 0.2; thr = 0.1;
tol_X = 1e-7;
tol_A = 1e-6;
out_folder = []

eta_A_arora = eta_A;
eta_A_arora_red = eta_A;

close all
tic;
[A_our, A_arora, A_arora_red, errA, errX, err,  err_arora, errA_arora, errX_arora, err_arora_red, ...
 errA_arora_red, errX_arora_red, Y_last, X_last, X_last_o, X_arora_last, X_arora_red_last, ...
 time_our, time_arora_red, time_arora]...
    = compare_algos_and_NOODL_dist(A, A_o, k, p, eta_x, thr, C, eta_A, eta_A_arora, eta_A_arora_red, tol_X, tol_A, out_folder, show);
learn_time = toc;
display(['NoodL is ready! It took ', num2str(learn_time/60, 3), ' minutes to make and compare NOODL.'])
done = 1

%% Make Noodl: Alternating Minimization to refine the Dictionary: distributed

% Run parpool if it is not already running
if(isempty(gcp('nocreate')))
    parpool('local',5);
end

% Set parameters
show = 0; eta_A = 15; p = 500 ;C = 1;
eta_x = 0.2; thr = 0.1;
tol_X = 1e-2;
tol_A = 1e-1;
out_folder = []

eta_A_arora = eta_A;
eta_A_arora_red = eta_A;

close all
tic;
[A_our, A_arora, A_arora_red, errA, errX, err,  err_arora, errA_arora, errX_arora, err_arora_red, ...
 errA_arora_red, errX_arora_red, Y_last, X_last, X_last_o, X_arora_last, X_arora_red_last, ...
 time_our, time_arora_red, time_arora]...
    = compare_algos_and_NOODL(A, A_o, k, p, eta_x, thr, C, eta_A, eta_A_arora, eta_A_arora_red, tol_X, tol_A, out_folder, show);
learn_time = toc;
display(['NoodL is ready! It took ', num2str(learn_time/60, 3), ' minutes to make and compare NOODL.'])
done = 1

%% Sample Plots
close all

subplot(211)
semilogy(errA)
hold all
semilogy(errA_arora)
semilogy(errA_arora_red)

subplot(212)
semilogy(errX)
hold all
semilogy(errX_arora)
semilogy(errX_arora_red)


