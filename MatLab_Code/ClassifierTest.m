DatasetTest = XSSTesting;  % Dataset name
TBL = DatasetTest(:,1:Si(1,2)-1);  % Inputs
ResponseVarName = DatasetTest(:,Si(1,2)); % Output

tic; % Starting Time Counter

%% Section of testing classifiers, Remove comment from the selected classifier.

% label = predict(SVML,TBL);
% label = predict(SVMP,TBL);
% label = predict(KNN,TBL);

% Section of testing Random Forest Classifier.

% Label = predict(RF,TBL);
% Label = str2num(cell2mat(Label));

% Section of testing NN Classifier.

XTest = TBL'; 
Label = NN(XTest); 
output = Label'; 
performance = perform(NN,XTest,Label);
output1 = round(output); %abs for intger number
Label = output1;

PerformanceTime = toc; % Ending Time Counter

disp('Performance Results');
CM = confusionmat(ResponseVarName,Label);
disp(CM)
accuracy = (CM(1,1)+CM(2,2))/(CM(1,1)+CM(1,2)+CM(2,1)+CM(2,2));
disp('Accuracy')
disp(accuracy*100)
disp('Precision')
dr = CM(1,1)/(CM(1,1)+CM(1,2));
disp(dr*100)
Sensitivity = CM(1,1)/(CM(1,1)+CM(2,1));
Specificity = CM(2,2)/(CM(2,2)+CM(1,2));
disp('Sensitivity')
disp(Sensitivity*100)
disp('Specificity')
disp(Specificity*100)
disp('Timing');
disp(PerformanceTime);
