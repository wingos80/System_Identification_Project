%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% xdot = kf_calcFx(x) Calculates the system dynamics equation f(x,u,t) 
%   
%   Author: C.C. de Visser, Delft University of Technology, 2013
%   email: c.c.devisser@tudelft.nl
%   Version: 1.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function xdot = kf_calcFx(t, x, u)

    n = size(x, 1);
    xdot = zeros(n, 1);
    
    % system dynamics go here!
    xdot(1) = u(1);
    xdot(2) = u(2);
    xdot(3) = u(3);
    xdot(4) = 0;
end