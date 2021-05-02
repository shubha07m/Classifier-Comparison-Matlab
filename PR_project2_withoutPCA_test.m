%% Loading feature and target data set
load P; 
load T;

%% Dividing feature and target data set to train and test
[trainP,valP,testP,trainInd,valInd,testInd] = dividerand(P,0.7,0,0.3);
[trainT,valT,testTarget] = divideind(T,trainInd,valInd,testInd);


%% Defining class one and two and defining test sample
C1=trainP(:,(find(trainT==-1))); % Defining the benign sample as class one or C1
C2=trainP(:,(find(trainT==1))); % Defining the cancerous sample as class two or C2
C1Test=testP(:,(find(testTarget==-1))); % Performing the same for test data set 
C2Test=testP(:,(find(testTarget==1))); % Performing the same for test data se


%% Creating various types of discriminant for classification

DISCR_lin=fitcdiscr([C1';C2'],[-ones(size(C1,2),1);ones(size(C2,2),1)],'DiscrimType','linear');

DISCR_diaglin=fitcdiscr([C1';C2'],[-ones(size(C1,2),1);ones(size(C2,2),1)],'DiscrimType','diagLinear');

DISCR_quad=fitcdiscr([C1';C2'],[-ones(size(C1,2),1);ones(size(C2,2),1)],'DiscrimType','quadratic');

DISCR_diagquad=fitcdiscr([C1';C2'],[-ones(size(C1,2),1);ones(size(C2,2),1)],'DiscrimType','diagQuadratic');


%% Predictions performance of discriminant on test data
%% Making predictions using various types of discriminant


[TSTpred_lin,TSTpred_lin_score, ~]  = predict(DISCR_lin,[C1Test';C2Test']);

[TSTpred_diaglin, TSTpred_diaglin_score, ~] = predict(DISCR_diaglin,[C1Test';C2Test']);

[TSTpred_quad, TSTpred_quad_score, ~] = predict(DISCR_quad,[C1Test';C2Test']);

[TSTpred_diagquad,TSTpred_diagquad_score, ~] = predict(DISCR_diagquad,[C1Test';C2Test']);


%% Creating confusion matrix using various types of discriminant
confusionchart(confusionmat(TSTpred_lin,[-ones(size(C1Test,2),1);ones(size(C2Test,2),1)]))

confusionchart(confusionmat(TSTpred_diaglin,[-ones(size(C1Test,2),1);ones(size(C2Test,2),1)]))

confusionchart(confusionmat(TSTpred_quad,[-ones(size(C1Test,2),1);ones(size(C2Test,2),1)]))

confusionchart(confusionmat(TSTpred_diagquad,[-ones(size(C1Test,2),1);ones(size(C2Test,2),1)]))

%% Creating ROC using various types of discriminant separately

% linear discriminant
[X,Y,~,AUC_lin] = perfcurve([zeros(103,1);ones(68,1)]',TSTpred_lin_score(:,2)',1);
plot(X,Y,'b','LineWidth',3)
xlabel('False positive rate'); ylabel('True positive rate');
title(['test ROC for linear discriminant - with AUC ' num2str(AUC_lin)])

% diagLinear discriminant
[X,Y,~,AUC_diaglin] = perfcurve([zeros(103,1);ones(68,1)]',TSTpred_diaglin_score(:,2)',1);
plot(X,Y,'r','LineWidth',3)
xlabel('False positive rate'); ylabel('True positive rate');
title(['test ROC for diagLinear discriminant - with AUC ' num2str(AUC_diaglin)])

% quadratic discriminant
[X,Y,~,AUC_quad] = perfcurve([zeros(103,1);ones(68,1)]',TSTpred_quad_score(:,2)',1);
plot(X,Y,'g','LineWidth',3)
xlabel('False positive rate'); ylabel('True positive rate');
title(['test ROC for quadratic discriminant - with AUC ' num2str(AUC_quad)])

%diagQuadratic discriminant
[X,Y,~,AUC_diagquad] = perfcurve([zeros(103,1);ones(68,1)]',TSTpred_diagquad_score(:,2)',1);
plot(X,Y,'k','LineWidth',3)
xlabel('False positive rate'); ylabel('True positive rate');
title(['test ROC for diagQuadratic discriminant - with AUC ' num2str(AUC_diagquad)])



%% Creating and comparing ROC using various types of discriminants all together

[X,Y,~,AUC_lin] = perfcurve([zeros(103,1);ones(68,1)]',TSTpred_lin_score(:,2)',1);
plot(X,Y,'b','LineWidth',3)
hold on;
[X,Y,~,AUC_diaglin] = perfcurve([zeros(103,1);ones(68,1)]',TSTpred_diaglin_score(:,2)',1);
plot(X,Y,'r','LineWidth',3)
hold on;
[X,Y,~,AUC_quad] = perfcurve([zeros(103,1);ones(68,1)]',TSTpred_quad_score(:,2)',1);
plot(X,Y,'g','LineWidth',3)
hold on;
[X,Y,~,AUC_diagquad] = perfcurve([zeros(103,1);ones(68,1)]',TSTpred_diagquad_score(:,2)',1);
plot(X,Y,'k','LineWidth',3)
legend(['linear: ' num2str(AUC_lin)],['diagLinear: ' num2str(AUC_diaglin)],['quadratic: ' num2str(AUC_quad)],['diagQuadratic: ' num2str(AUC_diagquad)])
xlabel('False positive rate'); ylabel('True positive rate');
