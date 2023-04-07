%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demonstration of the Linear Kalman Filter
%
%   Author: C.C. de Visser, Delft University of Technology, 2013
%   email: c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
close all;
clear all;
randn('seed', 7);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set simulation parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n               = 1;        % number of states
nm              = 1;        % number of measurements
m               = 1;        % number of inputs
dt              = 0.01;     % time step (s)
N               = 1000;     % sample size
epsilon         = 1e-10;
doIEKF          = 1;        % set 1 for IEKF and 0 for EKF
maxIterations   = 100;

printfigs = 0;
figpath = '';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set initial values for states and statistics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
E_x_0       = 10;   % initial estimate of optimal value of x_k1_k1
x_0         = 5;    % initial state

B           = 1;    % input matrix
G           = 1;    % noise input matrix

% Initial estimate for covariance matrix
std_x_0     = 10;
P_0         = std_x_0^2;

% System noise statistics:
E_w         = 0;                            % bias of system noise
std_w       = 1;                            % standard deviation of system noise
Q           = std_w^2;                      % variance of system noise
w_k         = std_w * randn(n, N) + E_w;    % system noise

% Measurement noise statistics:
E_v         = 0;                            % bias of measurement noise
std_v       = 5;                            % standard deviation of measurement noise
R           = std_v^2;                      % variance of measurement noise
v_k         = std_v * randn(n, N) + E_v;    % measurement noise

% for numerical demo only
w_k(1)      = -2;
v_k(1)      = 1;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate batch with measurement data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% True state and measurements data:
x           = x_0;              % initial true state
X_k         = zeros(n, N);      % true state
Z_k         = zeros(nm, N);     % measurement
U_k         = zeros(m, N);      % inputs
for i = 1:N
    dx          = kf_calc_f(0, x, U_k(:,i));            % calculate noiseless state derivative     
    x           = x + (dx + w_k(:,i)) * dt;             % calculate true state including noise
    X_k(:,i)    = x;                                    % store true state
    Z_k(:,i)    = kf_calc_h(0, x, U_k(:,i)) + v_k(:,i); % calculate and store measurement 
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize Iterated Extended Kalman filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t_k             = 0; 
t_k1            = dt;

XX_k1_k1        = zeros(n, N);
PP_k1_k1        = zeros(n, N);
STD_x_cor       = zeros(n, N);
STD_z           = zeros(n, N);
ZZ_pred         = zeros(nm, N);
IEKFitcount     = zeros(N, 1);

x_k1_k1         = E_x_0;    % x(0|0)=E{x_0}
P_k1_k1         = P_0;      % P(0|0)=P(0)

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run the Extended Kalman filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;

% Run the filter through all N samples
for k = 1:N
    % x(k+1|k) (prediction)
    [t, x_k1_k]     = rk4(@kf_calc_f, x_k1_k1,U_k(:,k), [t_k t_k1]); 

    % z(k+1|k) (predicted observation)
    z_k1_k          = kf_calc_h(0, x_k1_k, U_k(k));     % prediction of observation
    ZZ_pred(k)      = z_k1_k;   % store this observation prediction, since later prediction observations
                                % are corrected using the actual observation
        
    % Calc Jacobians Phi(k+1,k) and Gamma(k+1, k)
    Fx              = kf_calc_Fx(0, x_k1_k, U_k(:,k)); % perturbation of f(x,u,t)
    % the continuous to discrete time transformation of Df(x,u,t) and G
    [dummy, Psi]    = c2d(Fx, B, dt);   
    [Phi, Gamma]    = c2d(Fx, G, dt);   
    
    % P(k+1|k) (prediction covariance matrix)
    P_k1_k          = Phi*P_k1_k1*Phi.' + Gamma*Q*Gamma.'; 
    
    % Run the Iterated Extended Kalman filter (if doIEKF = 1), else run standard EKF
    if (doIEKF)
        % do the iterative part
        eta2    = x_k1_k;
        err     = 2*epsilon;

        itts    = 0;
        while (err > epsilon)
            if (itts >= maxIterations)
                fprintf('Terminating IEKF: exceeded max iterations (%d)\n', maxIterations);
                break
            end
            itts    = itts + 1;
            eta1    = eta2;

            % Construct the Jacobian H = d/dx(h(x))) with h(x) the observation model transition matrix 
            Hx       = kf_calc_Hx(0, eta1, U_k(:,k)); 
            
            % Check observability of state
            if (k == 1 && itts == 1)
                rankHF  = kf_calcObsRank(Hx, Fx);
                if (rankHF < n)
                    warning('The current state is not observable; rank of Observability Matrix is %d, should be %d', rankHF, n);
                end
            end
         
            % Observation and observation error predictions
            z_k1_k      = kf_calc_h(0, x_k1_k, U_k(k));     % prediction of observation 
            P_zz        = (Hx*P_k1_k*Hx.' + R);             % covariance matrix of observation error
            std_z       = sqrt(diag(P_zz));                 % standard deviation of observation error (for validation) 

            % calculate the Kalman gain matrix
            K           = P_k1_k*Hx'/P_zz;
            % new observation state
            z_p         = kf_calc_h(0, eta1, U_k(:,k));

            eta2        = x_k1_k + K*(Z_k(:,k) - z_p - Hx*(x_k1_k - eta1));
            err         = norm((eta2 - eta1), inf)/norm(eta1, inf);
        end

        IEKFitcount(k)  = itts;
        x_k1_k1         = eta2;
    else
        % Correction
        Hx              = kf_calc_Hx(0, x_k1_k, U_k(:,k));  % perturbation of h(x,u,t)
        
        % P_zz(k+1|k) (covariance matrix of innovation)
        P_zz            = (Hx*P_k1_k * Hx.' + R);           % covariance matrix of observation error
        std_z           = sqrt(diag(P_zz));                 % standard deviation of observation error (for validation)    

        % K(k+1) (gain)
        K               = P_k1_k * Hx.'/P_zz;
        
        % Calculate optimal state x(k+1|k+1) 
        x_k1_k1         = x_k1_k + K*(Z_k(:,k) - z_k1_k); 
    end    
   
    P_k1_k1         = (eye(n) - K*Hx) * P_k1_k * (eye(n) - K*Hx).' + K*R*K.';  
    std_x_cor       = sqrt(diag(P_k1_k1));

    % Next step
    t_k             = t_k1; 
    t_k1            = t_k1 + dt;
    
    % store results
    XX_k1_k1(k)     = x_k1_k1;
    PP_k1_k1(k)     = P_k1_k1;
    STD_x_cor(k)    = std_x_cor;
    STD_z(k)        = std_z;
end

time        = toc;

% calculate state estimation error (in real life this is unknown!)
EstErr_x    = XX_k1_k1-X_k;

% calculate measurement estimation error (possible in real life)
EstErr_z    = ZZ_pred-Z_k;

fprintf('IEKF state estimation error RMS = %d, completed run with %d samples in %2.2f seconds.\n', sqrt(mse(EstErr_x)), N, time);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plotID  = 1001;
figure(plotID);
set(plotID, 'Position', [1 425 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(X_k, 'b');
plot(Z_k, 'k');
title('True state and measurement');
legend('True state', 'Measurement', 'Location', 'northeast');
if (printfigs == 1)
    name       = "TrueStateMeasurement";
    savefname  = strcat(figpath,"fig_",mfilename, "_", name);
    print(plotID, '-dpng', '-r300', savefname);
end

plotID = 1002;
figure(plotID);
set(plotID, 'Position', [1 0 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(X_k, 'b');
plot(XX_k1_k1, 'r');
title('True state and estimated state');
legend('True state', 'Estimated state', 'Location', 'northeast');
if (printfigs == 1)
    name       = "TrueStateEstimatedState";
    savefname  = strcat(figpath,"fig_",mfilename, "_", name);
    print(plotID, '-dpng', '-r300', savefname);
end

plotID = 1003;
figure(plotID);
set(plotID, 'Position', [600 425 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
grid on;
plot(X_k, 'b');
plot(XX_k1_k1, 'r');
plot(Z_k, 'k');
title('True state, estimated state and measurement');
legend('True state', 'Estimated state', 'Measurement', 'Location', 'northeast');
if (printfigs == 1)
    name       = "TrueStateEstimatedStateMeasurement";
    savefname  = strcat(figpath,"fig_",mfilename, "_", name);
    print(plotID, '-dpng', '-r300', savefname);
end

% pause;
% 
% plotID = 2001;
% figure(plotID);
% set(plotID, 'Position', [600 425 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
% hold on;
% grid on;
% plot(EstErr_x, 'b');
% plot(STD_x_cor, 'r');
% plot(-STD_x_cor, 'g');
% axis([0, N, min(EstErr_x), max(EstErr_x)]);
% legend('Estimation error', 'Upper error STD', 'Lower error STD', 'Location', 'northeast');
% title('State estimation error with STD');
% if (printfigs == 1)
%     name       = "StateEstimationError";
%     savefname  = strcat(figpath,"fig_",mfilename, "_", name);
%     print(plotID, '-dpng', '-r300', savefname);
% end
% 
% plotID = 2002;
% figure(plotID);
% set(plotID, 'Position', [600 0 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
% hold on;
% grid on;
% plot(EstErr_x, 'b');
% plot(STD_x_cor, 'r');
% plot(-STD_x_cor, 'g');
% axis([0 50 min(EstErr_x) max(EstErr_x)]);
% title('State estimation error with STD (zoomed in)');
% legend('Estimation error', 'Upper error STD', 'Lower error STD', 'Location', 'northeast');
% if (printfigs == 1)
%     name       = "StateEstimationErrorZoom";
%     savefname  = strcat(figpath,"fig_",mfilename, "_", name);
%     print(plotID, '-dpng', '-r300', savefname);
% end
% 
% pause;
% 
% plotID = 3001;
% figure(plotID);
% set(plotID, 'Position', [600 425 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
% hold on;
% grid on;
% plot(EstErr_z, 'b');
% plot(STD_z, 'r');
% plot(-STD_z, 'g');
% axis([0, N, min(EstErr_z), max(EstErr_z)]);
% legend('Measurement estimation error', 'Upper error STD', 'Lower error STD', 'Location', 'northeast');
% title('Measurement estimation error with STD');
% if (printfigs == 1)
%     name       = "MeasurementEstimationError";
%     savefname  = strcat(figpath,"fig_",mfilename, "_", name);
%     print(plotID, '-dpng', '-r300', savefname);
% end
% 
% plotID = 3002;
% figure(plotID);
% set(plotID, 'Position', [600 0 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
% hold on;
% grid on;
% plot(EstErr_z, 'b');
% plot(STD_z, 'r');
% plot(-STD_z, 'g');
% axis([0 50 min(EstErr_z) max(EstErr_z)]);
% title('Measurement estimation error with STD (zoomed in)');
% legend('Measurement estimation error', 'Upper error STD', 'Lower error STD', 'Location', 'northeast');
% if (printfigs == 1)
%     name       = "MeasurementEstimationErrorZoom";
%     savefname  = strcat(figpath,"fig_",mfilename, "_", name);
%     print(plotID, '-dpng', '-r300', savefname);
% end
% 
% pause;
% 
% plotID = 4001;
% figure(plotID);
% set(plotID, 'Position', [100 420 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
% hold on;
% grid on;
% plot(IEKFitcount, 'b');
% ylim([0, max(IEKFitcount)]);
% title('IEKF iterations at each sample');
% if (printfigs == 1)
%     name       = "NumberOfIterations";
%     savefname  = strcat(figpath,"fig_",mfilename, "_", name);
%     print(plotID, '-dpng', '-r300', savefname);
% end
