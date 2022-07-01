% EX1: Program in MATLAB your own neural network that can solve the XOR problem.

%% Setup
clc
clear
rng(2)


%% Data generation

% Create data set at with XOR corners 
D_init = [0 0; 0 1; 1 0; 1 1];
v_init = [  0;   1;   1;   0];

% Repeat data set and add noise
n_reps=15;
noise = 0.05;
D_init = repmat(D_init,n_reps,1)+noise*randn(n_reps*4,2);
v_init = repmat(v_init,n_reps,1);

% Split data set into train and validate
[train,validate] = Billen_Moritz_splitTrainValidate(D_init,v_init,0.8);

%% Neural Network definition

% Set learning rate and number of epochs
learning_rate = 0.1;
n_epochs = 500;

% Initialize weights
n_weights = 9;
weights = randn(n_weights, 1);

% Create function handles
f1 = @(x, w) 1 / (1 + exp(-w(1) *  x(   1) - w(2) *  x(   2) - w(3)));
f2 = @(x, w) 1 / (1 + exp(-w(4) *  x(   1) - w(5) *  x(   2) - w(6)));
f3 = @(x, w) 1 / (1 + exp(-w(7) * f1(x, w) - w(8) * f2(x, w) - w(9)));
L  = @(x, w, z) (z - f3(x, w))^2;

%% Live Plotting Setup

% Create Figure
fig = figure(1);
clf(1)

% Subplot 1
sub1 = subplot(1,3,1);
grid on
box on
axis(sub1,[0,n_epochs,0,0.5])
% Animated line for errors
l1 = animatedline(sub1);
l1.LineWidth = 2;
l1.Color = "red";
xlabel("Epoch","Interpreter","latex","FontSize",12)
ylabel("Average Error","Interpreter","latex","FontSize",12)
sub1.TickLabelInterpreter = "latex";

% Subplot 2
sub2 = subplot(1,3,2);
grid on
box on
axis(sub2,[0,n_epochs,-10,10])

colors = [0 0.4470 0.7410;0.8500 0.3250 0.0980;0.9290 0.6940 0.1250];
lines = ["--","-.",":"];
labels = [];

% Set up one animated line per weight
for i_line = 1:n_weights
    l2(i_line) = animatedline(sub2);
    l2(i_line).Color = colors(fix((i_line-1)/3+1),:);
    l2(i_line).LineStyle = lines(mod(i_line-1,3)+1);
    l2(i_line).LineWidth = 2;

    % Create legend labels
    if mod(i_line-1,3)<2
        labels = [labels, sprintf("Node %d: $w_%d$",fix((i_line-1)/3+1),mod(i_line-1,3)+1)];
    else
        labels = [labels, sprintf("Node %d: $b$",fix((i_line-1)/3+1))];
    end
end
% Legend commented out due to occlusion
%legend(labels,"Interpreter","latex","Location","northwest","FontSize",6)
xlabel("Epoch","Interpreter","latex","FontSize",12)
ylabel("Value of $w_i$","Interpreter","latex","FontSize",12)
sub2.TickLabelInterpreter = "latex";

% Subplot 3
sub3 = subplot(1,3,3);
grid on 
box on
hold on

% Calculate dataset bounds
x_bounds = [min(D_init(:,1));max(D_init(:,1))];
y_bounds = [min(D_init(:,2));max(D_init(:,2))];

% set axis bounds
axis(sub3,[x_bounds;y_bounds])
c_s = ["red";"blue"];

% Scatter data points
mask = train.idx==1;
scatter(train.data( mask,1),train.data( mask,2),50,"red","Marker","x")
scatter(train.data(~mask,1),train.data(~mask,2),50,"blue","Marker","x")

% Calculate initial seperation line equations (2 points from min to max in
% x)
y1 = -(weights(1)*x_bounds+weights(3))./weights(2);
y2 = -(weights(4)*x_bounds+weights(6))./weights(5);
y3 = -(weights(7)*x_bounds+weights(9))./weights(8);

% Plot seperation lines
plot(x_bounds,y1,"Color",colors(1,:),"LineWidth",2);
plot(x_bounds,y2,"Color",colors(2,:),"LineWidth",2);
plot(x_bounds,y3,"Color",colors(3,:),"LineWidth",2);

% Set up data source
sub3.Children(1).YDataSource="y3";
sub3.Children(2).YDataSource="y2";
sub3.Children(3).YDataSource="y1";
sub3.TickLabelInterpreter = "latex";
xlabel("Feature 1","FontSize",12,"Interpreter","latex")
ylabel("Feature 2","FontSize",12,"Interpreter","latex")


%% Train Neural Network

% Shorthands for data and labels of training set
D=train.data;
v=train.idx;

n_samples = length(v);

% Setup weight, error and jacobian arrays
J=zeros(n_samples,n_weights);
e=zeros(n_samples,n_epochs);
ws= zeros(n_epochs+1,n_weights);
ws(1,:) = weights;

% Iterate over epochs
for epoch = 1:n_epochs
    % Iterate over all samples
    for i = 1:n_samples
        x_i =  D(i,:);
        z_i = v(i);

        % Setup loss function handle
        L_i = @(w) L(x_i,w,z_i);

        % Get jacobian for all weights from AD implementation
        [J_i, e_i] = Billen_Moritz_AD_Jacobian(L_i, weights);

        % Write into data sets
        J(i,:)=J_i;
        e(i, epoch)=e_i;
    end

    % Update weights by Newton method
    weights = weights + learning_rate * sum(J)';
    ws(epoch+1,:) = weights;

    % Calculate mean error
    curr_err = mean(e(:,epoch));
    
    % Plot updates
    y1 = -(weights(1)*x_bounds+weights(3))./weights(2);
    y2 = -(weights(4)*x_bounds+weights(6))./weights(5);
    y3 = -(weights(7)*x_bounds+weights(9))./weights(8);
    refreshdata(sub3.Children(1:3))
    for i_line = 1:n_weights
        addpoints(l2(i_line),epoch,weights(i_line))
    end
    addpoints(l1,epoch,curr_err)
    drawnow

    % Console updates
    clc
    fprintf("%d/%d Iterations complete. Current Error %f \n",epoch,n_epochs,curr_err)
end

%% Evaluation

% Optimal solution function handle 
w_opt = 100 * [-2;-2;1;2;2;-3;-2;-2;1];
f_opt = @(x) f3(x, w_opt);

n_validate = length(validate.idx);

f_final = @(x) f3(x, weights);

% Compute final labels
v_model = zeros(n_validate,1);
v_opt = zeros(n_validate,1);
for i = 1:n_validate
    v_opt(i) = f_opt(validate.data(i,:));
    v_model(i) = f_final(validate.data(i,:));
end

% Print final accuracy
accuracy=nnz(round(v_model)==validate.idx)/n_validate;
fprintf("Final accuracy on the validation data set is: %f\n",accuracy)

%% Export Figure

width = 18;
height = 6;
name = "ex1";
set(fig, 'PaperPositionMode', 'Auto', ...
    'PaperUnits', 'centimeters', 'PaperSize', [width, height], ...
    'Units', 'centimeters', 'Position', [0, 0, width, height]);

% Save figure
print(fig, sprintf("figs/%s.pdf", name), '-dpdf', '-r0', '-fillpage');