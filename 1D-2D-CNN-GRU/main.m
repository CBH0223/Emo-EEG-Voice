clc;
clear;

AllData2D = readmatrix('2D-data.xlsx');

folders = unique(AllData2D(:, 1));

height = 29; 
width = 6; 


numFolders = numel(folders);
numSamples = numFolders; 
dataGroupedByFolder = cell(numSamples, 1);


for i = 1:numFolders
    folderData = AllData2D(AllData2D(:, 1) == folders(i), 2:end); 
    numEntries = size(folderData, 1);
    

    if numEntries ~= height
        error('error', folders(i), numEntries, height);
    end

    reshapedData = reshape(folderData', width, height)';

    dataGroupedByFolder{i} = reshapedData;
end


AllData1D = xlsread('1D-data.xlsx'); 


trainRatio = 0.7; 
testRatio = 0.3;


numSamples1D = size(AllData1D, 2); 
idx1D = randperm(numSamples1D); 
numTrain1D = floor(trainRatio * numSamples1D);
numTest1D = numSamples1D - numTrain1D;

trainIdx1D = idx1D(1:numTrain1D);
testIdx1D = idx1D(numTrain1D+1:end);

TrainData1D = AllData1D(:, trainIdx1D);
TestData1D = AllData1D(:, testIdx1D);


numSamples2D = numSamples; 
idx2D = randperm(numSamples2D); 
numTrain2D = floor(trainRatio * numSamples2D);
numTest2D = numSamples2D - numTrain2D;

trainIdx2D = idx2D(1:numTrain2D);
testIdx2D = idx2D(numTrain2D+1:end);

TrainData2D = dataGroupedByFolder(trainIdx2D);
TestData2D = dataGroupedByFolder(testIdx2D);


TrainData2DLabels = cellfun(@(x) x(end), TrainData2D);
TrainData2DFeatures = cellfun(@(x) x(:, 1:end), TrainData2D, 'UniformOutput', false); 

TestData2DLabels = cellfun(@(x) x(end), TestData2D); 
TestData2DFeatures = cellfun(@(x) x(:, 1:end), TestData2D, 'UniformOutput', false); 


TrainFeature1D = TrainData1D(1:12, :); 
TrainLabel1D   = categorical(TrainData1D(13, :));

TestFeature1D = TestData1D(1:12, :); 
TestLabel1D   = categorical(TestData1D(13, :)); 


Featuredata1D = reshape(TrainFeature1D, 1, 12, 1, numTrain1D);   
Featuredata1D_test = reshape(TestFeature1D, 1, 12, 1, numTest1D);

NumSample1D = numTrain1D;
for i = 1:NumSample1D
    SequenceSamples1D{i, 1} = Featuredata1D(:,:,1,i);
end
for i = 1:numel(TestLabel1D)
    SequenceSamples1D_test{i, 1} = Featuredata1D_test(:,:,1,i);
end


SequenceTrain1D = arrayDatastore(SequenceSamples1D, "ReadSize", 1, "OutputType", "same");
LabelTrain1D = arrayDatastore(TrainLabel1D', "ReadSize", 1, "OutputType", "cell");

SequenceTest1D = arrayDatastore(SequenceSamples1D_test, "ReadSize", 1, "OutputType", "same");
LabelTest1D = arrayDatastore(TestLabel1D', "ReadSize", 1, "OutputType", "cell");


SequenceTrain2D = arrayDatastore(TrainData2DFeatures, "ReadSize", 1, "OutputType", "same");
LabelTrain2D = arrayDatastore(TrainData2DLabels, "ReadSize", 1, "OutputType", "cell");

SequenceTest2D = arrayDatastore(TestData2DFeatures, "ReadSize", 1, "OutputType", "same");
LabelTest2D = arrayDatastore(TestData2DLabels, "ReadSize", 1, "OutputType", "cell");


TrainDataStore = combine(SequenceTrain1D, SequenceTrain2D, LabelTrain1D);
TestDataStore = combine(SequenceTest1D, SequenceTest2D, LabelTest1D);


options = trainingOptions('adam', ...
    'MiniBatchSize', 15, ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 70, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');


lgraph = layerGraph();


tempLayers1D = [
    imageInputLayer([1 12 1], "Name", "data_1D", "Normalization", "zscore")
    flattenLayer("Name", "flatten_1D")
    fullyConnectedLayer(16, "Name", "fc1D")
    reluLayer("Name", "relu_1D")];
lgraph = addLayers(lgraph, tempLayers1D);


tempLayers2D = [
    imageInputLayer([height width 1], "Name", "data_2D", "Normalization", "zscore")
    convolution2dLayer(1, 8, "Name", "conv1_2D", "Padding", "same")
    batchNormalizationLayer("Name", "bn_conv1_2D")
    reluLayer("Name", "relu1_2D")
    maxPooling2dLayer(2, "Name", "pool1_2D")
    fullyConnectedLayer(16, "Name", "fc2D")
    flattenLayer("Name", "flatten_2D")];
lgraph = addLayers(lgraph, tempLayers2D);


tempLayersMerge = [
    concatenationLayer(1, 2, "Name", "concat")
    fullyConnectedLayer(2, "Name", "fc_final")  
    softmaxLayer("Name", "softmax")
    classificationLayer("Name", "classification")];
lgraph = addLayers(lgraph, tempLayersMerge);


lgraph = connectLayers(lgraph, "relu_1D", "concat/in1");


lgraph = connectLayers(lgraph, "flatten_2D", "concat/in2");

analyzeNetwork(lgraph);


[netTransfer, info] = trainNetwork(TrainDataStore, lgraph, options);


T_smi1 = classify(netTransfer, TestDataStore);


accuracy = sum(T_smi1 == categorical(TestData2DLabels)) / numel(TestData2DLabels) * 100;

accuracy
