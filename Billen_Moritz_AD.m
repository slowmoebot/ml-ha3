classdef Billen_Moritz_AD
% BILLEN_MORITZ_AD Implements simple forward AD for specific functions.
% This class is a simplified version of 
% https://github.com/martinResearch/MatlabAutoDiff. Forward Automatic 
% Differentiation (AD) is implemented by overloading variables with a
% derivative value. Overloaded operators with this datatype then compute
% the outcome of the operator, and use the chain rule to compute the
% derivative at the same time.

    properties
        values
        derivatives
    end

    methods
        
        % Initialisation of the variable
        function x = Billen_Moritz_AD(values,derivatives)
            % If only values are given, derivatives are set to one.
            if nargin==1
                x.values = values;
                x.derivatives = eye(numel(values));
            
            % If derivatives are also given store them
            else
                x.values = values;
                x.derivatives = derivatives;
            end
        end

        % Get function for derivatives
        function Jac = getderivs(x)
            Jac = x.derivatives;
        end

        % Get functuon for value
        function val = getvalue(x)
            val = x.values;
        end

        % Set function for derivatives (unused)
        function x = setdervis(x, derivatives)
            x.derivatives = derivatives;
        end

        % Matrix multiplication with at least one variable of AD class
        function z = mtimes(x,y)
            if isa(x,"Billen_Moritz_AD")
                if isa(y,"Billen_Moritz_AD")
                    % Implementation of product rule
                    z_values = x.values * y.values;
                    z_derivatives = x.values * y.derivatives + y.values * x.derivatives;
                else
                    % Multiplication with a constant
                    z_values = x.values * y;
                    z_derivatives = x.derivatives * y;
                end
                % Create new variable
                z = Billen_Moritz_AD(z_values,z_derivatives);
            else
                if isa(y,"Billen_Moritz_AD")
                    % Multiplication with a constant
                    z_values = x * y.values;
                    z_derivatives = x * y.derivatives;
                    z = Billen_Moritz_AD(z_values,z_derivatives);
                else
                    % This end will not be reached
                    z = x * y;
                end
            end
        end

        % Derivative of an exponential function
        function x = exp(x)
            x.derivatives = exp(x.values) * x.derivatives;
            x.values = exp(x.values);
        end

        % Derivative of addition
        function z = plus(x, y)
            if isa(x,"Billen_Moritz_AD")
                if isa(y,"Billen_Moritz_AD")
                    % Addition of derivatives
                    z_values = x.values + y.values;
                    z_derivatives = x.derivatives + y.derivatives;
                    z = Billen_Moritz_AD(z_values, z_derivatives);
                else
                    % Adding a constant does not change the derivative
                    z = Billen_Moritz_AD(x.values + y, x.derivatives);
                end
            else
                if isa(y,"Billen_Moritz_AD")
                    % Adding a constant does not change the derivative
                    z = Billen_Moritz_AD(y.values + x, y.derivatives);
                else
                    % This branch will not be reached
                    z = x + y;
                end                
            end
        end

        % Derivative of a variable raised to a power
        function x = mpower(x,p)
            % Clasic polynomial derivative
            x.derivatives = p * x.values.^(p-1) * x.derivatives;
            x.values = x.values.^p;
        end

        % Derivative of x-y (same as addition)
        function z = minus(x, y)
            if isa(x,"Billen_Moritz_AD")
                if isa(y,"Billen_Moritz_AD")
                    z_values = x.values - y.values;
                    z_derivatives = x.derivatives - y.derivatives;
                    z = Billen_Moritz_AD(z_values, z_derivatives);
                else
                    z = Billen_Moritz_AD(x.values - y, x.derivatives);
                end
            else
                if isa(y,"Billen_Moritz_AD")
                    z = Billen_Moritz_AD(x - y.values, y.derivatives);
                else
                    z = x -y;
                end                
            end
        end

        % Derivative of the square root
        function x = sqrt(x)
            x.derivatives = 0.5 ./ sqrt(x.values) * x.derivatives;
            x.values = sqrt(x.values);
        end

        % Derivative of division
        function z = mrdivide(x,y)
            if isa(x,"Billen_Moritz_AD")
                if isa(y,"Billen_Moritz_AD")
                    % Both variables have derivatives
                    z_values = x.values./y.values;
                    z_derivatives = 1./y.values*x.derivatives - x.values/(y.values.^2) * y.derivatives;
                    z = Billen_Moritz_AD(z_values,z_derivatives);
                else
                    % Only x has derivative
                    z = 1/y * x;
                end
            else
                if isa(y,"Billen_Moritz_AD")
                    % Only y has derivative
                    z_values = x/y.values;
                    z_derivatives = (-x) * 1/(y.values.^2) * y.derivatives;
                    z = Billen_Moritz_AD(z_values,z_derivatives);
                else
                    z = x/y;
                end
            end
        end

        function x = uminus(x)
            % Unitary minus just inverts the sign
            x.values = -x.values;
            x.derivatives = -x.derivatives;
        end

        % Indexing of variables
        function x = subsref(x,s)
            x.values = x.values(s.subs{:});
            x.derivatives = x.derivatives(s.subs{:},:);
        end

    end

end