% EX3 Create a simple convolutional neural network for classification using
% MATLAB’s Deep Learning ToolBox.

%% Setup
clc
clear
rng(1)

%% Load data set

% Load data set from example files
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

labelCount = countEachLabel(imds);

% Perform train validate split
numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

%% Setup layers

% Set up layers according to Matlab tutorial
layers1 = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same',"WeightsInitializer","he")
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same',"WeightsInitializer","he")
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same',"WeightsInitializer","he")
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

% Set up layers according to blog post on Medium
layers2 =[
  imageInputLayer([28 28 1])

  convolution2dLayer(3,32,"Stride",1,"Padding","same")
  batchNormalizationLayer
  reluLayer

  maxPooling2dLayer(2,"Stride",2)

  convolution2dLayer(3,64,"Stride",1,"Padding","same")
  batchNormalizationLayer
  reluLayer

  maxPooling2dLayer(2,"Stride",2)

  fullyConnectedLayer(128)
  fullyConnectedLayer(10)
  softmaxLayer

  classificationLayer

];

% Set up layers according to Python tutorial
layers3 =[
    imageInputLayer([28 28 1])

    convolution2dLayer(3,32,"Stride",1,"Padding","same","WeightsInitializer","he")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,"Stride",2)

    fullyConnectedLayer(100,"WeightsInitializer","he")
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer

    classificationLayer
];

% Set training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Training

% Train CNNs and save training info
[net1, trainInfo1] = trainNetwork(imdsTrain,layers1,options);
[net2, trainInfo2] = trainNetwork(imdsTrain,layers2,options);
[net3, trainInfo3] = trainNetwork(imdsTrain,layers3,options);

%% Plotting

fig=figure(1);
clf(1)
box on

% Subplot 2
ax = subplot(1, 2, 2);
hold on

% Generate and plot layerGraphs
ls = {layers1,layers2,layers3};
for i = 1:3
    lg = layerGraph(ls{i});
    plot(lg);
    ax(1).Children(1).XData = ax(1).Children(1).XData * i;
    ax(1).Children(1).NodeFontSize = 6;
end

xlim([0.9 4])
xticks(1:3)
xticklabels(["Model A","Model B","Model C"])
yticklabels([])
ax.TickLabelInterpreter = "latex";

% Subplot 1
ax = subplot(1,2,1);
hold on
box on
grid on

% Accuracy plot for each CNN
colors = [0 0.4470 0.7410;0.8500 0.3250 0.0980;0.9290 0.6940 0.1250];
infos=[trainInfo1,trainInfo2,trainInfo3];
for i = 1:3
    trainInfo=infos(i);
    mask = ~isnan(trainInfo.ValidationAccuracy);
    x=1:length(trainInfo.ValidationAccuracy);
    x = x(mask);
    plot(trainInfo.TrainingAccuracy,"Color",[colors(i,:),0.5],"LineStyle","-","LineWidth",2)
    plot(x,trainInfo.ValidationAccuracy(mask),"-o","Color",colors(i,:),"LineWidth",2)
end

xlabel("Iterations","Interpreter","latex","FontSize",12)
ylabel("Accuracy [\%]","Interpreter","latex","FontSize",12)
legend(["","Model A","","Model B","","Model C"],"Interpreter","latex","FontSize",12,"Location","southeast")
ax.TickLabelInterpreter = "latex";

%% Export plot

width = 18;
height = 9;
name = "ex3";
set(fig, 'PaperPositionMode', 'Auto', ...
    'PaperUnits', 'centimeters', 'PaperSize', [width, height], ...
    'Units', 'centimeters', 'Position', [0, 0, width, height]);

% Save figure
print(fig, sprintf("figs/%s.pdf", name), '-dpdf', '-r0', '-fillpage');

