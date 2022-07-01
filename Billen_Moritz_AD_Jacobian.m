function [J,f_val] = Billen_Moritz_AD_Jacobian(f, x)
% BILLEN_MORITZ_AD_JACOBIAN Calculates the jacobian of f with respect to x
%   Jacobian is calculated for scalar function handle f by use of automatic
%   differentiation. x is overloaded with derivative variables, then the
%   calculation of f is performed. The result will contain the function
%   value and derivative.

    % Overload x
    xAD = Billen_Moritz_AD(x);
    
    % Compute the function values and derivatives
    fAD = f(xAD);

    % Harvest the output
    f_val = getvalue(fAD);
    J = getderivs(fAD);
end