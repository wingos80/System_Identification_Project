
dataname = 'F16traindata_CMabV_2023';
% valdataname = 'F16validationdata_2023';
% measurement dataset
load(dataname, 'Cm', 'Z_k', 'U_k')
% % special validation dataset
% load(valdataname, 'Cm_val', 'alpha_val', 'beta_val')
Cm = Cm';
Z_k = Z_k';
U_k = U_k';

% measurements Z_k = Z(t) + v(t)
alpha_m = Z_k(1,:); % measured angle of attack
beta_m = Z_k(2,:);  % measured angle of sideslip
Vtot = Z_k(3,:);    % measured velocity

% input to Kalman filter
Au = U_k(1,:); % perfect accelerometer du/dt data
Av = U_k(2,:); % perfect accelerometer dv/dt data
Aw = U_k(3,:); % perfect accelerometer dw/dt data


