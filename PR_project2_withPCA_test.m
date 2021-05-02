%% Loading feature and target data set
load P; 
load T;

%% Dividing feature and target data set to trainand test
[trainP,valP,testP,trainInd,valInd,testInd] = dividerand(P,0.7,0,0.3);
[trainT,valT,testTarget] = divideind(T,trainInd,valInd,testInd);


%% Defining class one and two and defining test sample
C1=trainP(:,(find(trainT==-1))); % Defining the benign sample as class one or C1
C2=trainP(:,(find(trainT==1))); % Defining the cancerous sample as class two or C2
C1Test=testP(:,(find(testTarget==-1))); % Performing the same for test data set 
C2Test=testP(:,(find(testTarget==1))); % Performing the same for test data se


%% Using PCA for explaining variance and feature reduction
[coeff, score_train, latent, ~ ,explained,mu] = pca(trainP');
figure,plot(cumsum(explained),'r','LineWidth',2),title('Scree plot')
xlabel('Eigen vectors'); ylabel('% of explained variance');

%% Using PCA for prediction wth various discriminant

[~, score_test, ~, ~ ,~,~] = pca(testP');
trnPCA = score_train(:,1:15);
tstPCA = score_test(:,1:15);

DISCR_linpca = fitcdiscr(trnPCA, trainT, 'DiscrimType','linear');
DISCR_diaglinpca = fitcdiscr(trnPCA, trainT, 'DiscrimType','diagLinear');
DISCR_quadpca = fitcdiscr(trnPCA, trainT, 'DiscrimType','quadratic');
DISCR_diagquadpca = fitcdiscr(trnPCA, trainT, 'DiscrimType','diagQuadratic');


%% For PCA with linear discriminant
[TSTpred_pca,TSTpred_pca_score, ~]  = predict(DISCR_linpca,tstPCA);
confusionchart(confusionmat(TSTpred_pca,testTarget))
plotroc([(testTarget+1)/2],TSTpred_pca_score(:,2)')
[X,Y,~,AUC_linpca] = perfcurve([(testTarget+1)/2],TSTpred_pca_score(:,2)',1);
plot(X,Y,'c','LineWidth',3)
xlabel('False positive rate'); ylabel('True positive rate');
title(['test ROC for Linear discriminant using PCA - with AUC ' num2str(AUC_linpca)])


%% For PCA with diaglinear discriminant
[TSTpred_pca,TSTpred_pca_score, ~]  = predict(DISCR_diaglinpca,tstPCA);
confusionchart(confusionmat(TSTpred_pca,testTarget))
plotroc([(testTarget+1)/2],TSTpred_pca_score(:,2)')
[X,Y,~,AUC_diaglinpca] = perfcurve([(testTarget+1)/2],TSTpred_pca_score(:,2)',1);
plot(X,Y,'g','LineWidth',3)
xlabel('False positive rate'); ylabel('True positive rate');
title(['test ROC for DiagLinear discriminant using PCA - with AUC ' num2str(AUC_diaglinpca)])


%% For PCA with quadratic discriminant
[TSTpred_pca,TSTpred_pca_score, ~]  = predict(DISCR_quadpca,tstPCA);
confusionchart(confusionmat(TSTpred_pca,testTarget))
plotroc([(testTarget+1)/2],TSTpred_pca_score(:,2)')
[X,Y,~,AUC_quadpca] = perfcurve([(testTarget+1)/2],TSTpred_pca_score(:,2)',1);
plot(X,Y,'m','LineWidth',3)
xlabel('False positive rate'); ylabel('True positive rate');
title(['test ROC for Quadratic discriminant using PCA - with AUC ' num2str(AUC_quadpca)])


%% For PCA with Diagquadratic discriminant
[TSTpred_pca,TSTpred_pca_score, ~]  = predict(DISCR_diagquadpca,tstPCA);
confusionchart(confusionmat(TSTpred_pca,testTarget))
plotroc([(testTarget+1)/2],TSTpred_pca_score(:,2)')
[X,Y,~,AUC_diagquadpca] = perfcurve([(testTarget+1)/2],TSTpred_pca_score(:,2)',1);
plot(X,Y,'b','LineWidth',3)
xlabel('False positive rate'); ylabel('True positive rate');
title(['test ROC for DiagQuadratic discriminant using PCA - with AUC ' num2str(AUC_diagquadpca)])
