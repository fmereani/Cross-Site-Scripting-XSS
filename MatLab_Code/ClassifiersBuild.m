%% Classifiers Training

Dataset = XSSTraining; % Training dataset name
Si = size(Dataset);		% Dataset Size


ScTBL = Dataset(:,1:Si(1,2)-1);  % Inputs
ScResponseVarName = Dataset(:,Si(1,2));  % Output
rng('default');

SVML = fitcsvm(ScTBL,ScResponseVarName,'KernelFunction','linear', 'BoxConstraint', 0.5);   % Support Vector Machine- Linear kernel Classifier
SVMP = fitcsvm(ScTBL,ScResponseVarName,'KernelFunction','polynomial', 'OutlierFraction', 0.73); % Support Vector Machine- Polynomial kernel Classifier
KNN = fitcknn (ScTBL,ScResponseVarName, 'NumNeighbors', 1);  %k-NN Classifier
RF = TreeBagger(70,ScTBL,ScResponseVarName, 'OOBPrediction','On','Method','classification');   % Random Forest Classifier

TBL2 = ScTBL';  % Inputs table for NN
ResponseVarName2 = ScResponseVarName';  % Output for NN
net = patternnet(10, 'trainbr');  % Creating the net
NN = train(net,TBL2,ResponseVarName2);  % Training NN classifier
clear TBL2 ResponseVarName2;    % Remove unnecessary files