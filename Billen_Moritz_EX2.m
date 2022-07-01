% EX2 Program in MATLAB your own SOM (Self Organizing Map) implementation.
% Provide an example to demonstrate its performance.

%% Setup
clc
clear
rng(2)

%% Data Generation
n_points = [50, 50, 50, 50];

means = [-5, -5; 5, -5; 5, 5; -5,5];
cov(1).mat = eye(2);
cov(2).mat = eye(2);
cov(3).mat = eye(2);
cov(4).mat = eye(2);

[data, labels] = Billen_Moritz_genGaussianDataset(n_points,means,"cov",cov);

% Scramble data
new_idx = randperm(length(labels));
data = data(new_idx,:);
labels = labels(new_idx,:);

%% SOM

% Train Self Organizing Map
[out,weights] = Billen_Moritz_SOM(data,"n_grid",9,"sigm_init",3,"sigm_decay",10);

%% Compute U-Matrix

n_edge = size(weights,1);

umat = zeros(2*n_edge-1,2*n_edge-1);

% Get all distances in vertival direction
for i = 2*(1:(n_edge-1))
    for j = 1:2:(2*n_edge-1) 
        umat(i,j)=norm(squeeze(weights(i/2,(j+1)/2,:))-squeeze(weights(i/2+1,(j+1)/2,:)));
    end
end

% Get all distances in horizontal direction
for i = 1:2:(2*n_edge-1)
    for j = 2*(1:(n_edge-1)) 
        umat(i,j)=norm(squeeze(weights((i+1)/2,j/2,:))-squeeze(weights((i+1)/2,j/2+1,:)));
    end
end

offsets = [-1, 1, 0, 0; 0, 0,-1, 1];

% Average all remaining entries from neighboring cells
for i = 1:(2*n_edge-1)
    for j = 1:(2*n_edge-1)
        if mod(i,2)==mod(j,2)
            sum = 0;
            n_sum = 0;
            for offset = offsets
                % If index runs out of range (boundaries) do nothing
                try
                    sum = sum + umat(i+offset(1),j+offset(2));
                    n_sum = n_sum+1;
                catch
                end
            end
            umat(i,j) = sum/n_sum;
        end
    end
end

x = linspace(1,n_edge,2*n_edge-1);
y = linspace(1,n_edge,2*n_edge-1);
[X,Y] = meshgrid(x,y);

umat_max = max(umat,[],"all");
umat = umat/umat_max*255;

%% Plotting

fig = figure(1);
clf(1)
box on


ax = subplot(1,2,1);
image([1 n_edge],[1 n_edge],umat)
cb = colorbar();
cb.Limits = [0 255];
cb.Ticks = linspace(0,255,11);
cb.TickLabels =linspace(0,1,11);
cb.TickLabelInterpreter = "latex";

xticks(1:n_edge)
yticks(1:n_edge)
ax.TickLabelInterpreter = "latex";
ax.FontSize = 12;
xlabel("X Weights","Interpreter","latex","FontSize",12)
ylabel("Y Weights","Interpreter","latex","FontSize",12)
grid on



ax = subplot(1,2,2);
hold on 
grid on
box on
scatter(data(:,1),data(:,2),[],labels,"Marker","+")
plot(weights(:,:,1) ,weights(:,:,2) ,"LineWidth",2);
plot(weights(:,:,1)',weights(:,:,2)',"LineWidth",2);
xlabel("Feature 1","Interpreter","latex","FontSize",12)
ylabel("Feature 2","Interpreter","latex","FontSize",12)
ax.TickLabelInterpreter="latex";

%% Export figure

width = 18;
height = 6;
name = "ex2";
set(fig, 'PaperPositionMode', 'Auto', ...
    'PaperUnits', 'centimeters', 'PaperSize', [width, height], ...
    'Units', 'centimeters', 'Position', [0, 0, width, height]);

% Save figure
print(fig, sprintf("figs/%s.pdf", name), '-dpdf', '-r0', '-fillpage');
