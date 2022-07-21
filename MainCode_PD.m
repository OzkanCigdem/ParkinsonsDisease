clc
clear all
close all
warning off;
%% Load Data
loaded_data = load('D:\TEZson\basic models\40\GM_WM\TIV\WORKSPACE_GM_WM_F_TIV');
MainData          = loaded_data.new_A_GM_WM_masked;
MainData = double(MainData(:,1:size(MainData,2)-1));  % to remove PD and HC labels (added at the end of matrix) from the loaded data
%%
[n ,m]=size(MainData); % n is the no of rows, m is no of columns. n=80, m=121x145x121
MainLable = ones(n,1); % assign 1 to all labels
MainLable((n/2)+1:n) = -1; %assign -1 to all HCs
%%
test_label = [ones(4,1); zeros(4,1)];
yfit = zeros(8,10); %output labels
Obtained_Labels = zeros(80,1); %obtained labels
K  = 10  ; %10-fold CV
Foldaccuracy= zeros(1,K); 
bestacctrainmain=zeros(1,K);
cvFolds =  zeros(n,1);
%%
for z = 1 :K : n        % divide data into K groups
    for w = 1 : K       %assign data to the z_th group
        cvFolds(z) = w;
        z = z+1;
        w = w+1 ;
    end
end
cvFolds = cvFolds( 1:n ,:);
%%
p_prime = 1;
nf= n;
numF = size(MainData,2);
% Select a feature selection method from the list
% listFS = {'ILFS','InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs','fschisquare','lda'};
listFS = {'relieff','laplacian','ILFS','mrmr','llcfs','cfs'};
% [methodID] = readInput( listFS );
% selection_method = listFS{methodID}; % Selected
selection_method = 'ILFS'
counter = 0;
%%
for  FoldNumber =1:10                        %# for each fold  for FoldNumber = 1 : K
    testIdx = (cvFolds == FoldNumber);       %# get indices of test instances
    trainIdx = ~testIdx;                     %# get indices training instances
    Label=MainLable(trainIdx);
    Label(Label==-1)=2;
    DataSD=MainData(trainIdx,:);   
    
%%  feature Selection on training data
switch lower(selection_method)
    case 'ilfs'
        % Infinite Latent Feature Selection - ICCV 2017
        [ranking, weights, subset] = ILFS(DataSD, Label , 4, 0 );
    case 'mrmr'
         ranking = mRMR(DataSD, Label, numF);
    case 'relieff'
        [ranking, w] = reliefF( DataSD, Label, 15);
%     case 'mutinffs'
%         [ ranking , w] = mutInfFS( DataSD, Label, numF );
%     case 'fsv'
%         [ ranking , w] = fsvFS( DataSD, Label, numF );
    case 'laplacian'
        W = dist(DataSD');
        W = -W./max(max(W)); % it's a similarity
        [lscores] = LaplacianScore(DataSD, W);
        [junk, ranking] = sort(-lscores);
    case 'mcfs'  
        options = [];
        options.k = 5; %For unsupervised feature selection, you should tune this parameter k, the default k is 5.
        options.nUseEigenfunction = 4;  %You should tune this parameter.
        [FeaIndex,~] = MCFS_p(DataSD,numF,options);
        ranking = FeaIndex{1};
    case 'rfe'
        ranking = spider_wrapper(DataSD,Label,numF,lower(selection_method));
    case 'l0'
        ranking = spider_wrapper(DataSD,Label,numF,lower(selection_method));
    case 'fisher'
        ranking = spider_wrapper(DataSD,Label,numF,lower(selection_method));
    case 'inffs'
        alpha = 0.5;    % default, it should be cross-validated.
        sup = 1;        % Supervised or Not
        [ranking, w] = infFS( DataSD , Label, alpha , sup , 0 );    
    case 'ecfs'
        alpha = 0.5; % default, it should be cross-validated.
        ranking = ECFS( DataSD, Label, alpha )  ;
    case 'udfs'
        nClass = 2;
        ranking = UDFS(DataSD , nClass ); 
    case 'cfs'
        ranking = cfs(DataSD);     
    case 'llcfs'   
        ranking = llcfs( DataSD );
   case 'fschisquare'   
        ranking = fsChiSquare( DataSD,Label );
         ranking= ranking.fList;
    case 'lda'   
        [DataSD, W, lambda] = LDA(DataSD,Label);
         ranking =lambda';
    otherwise
        disp('Unknown method.')
end   
ranked_feat(:,FoldNumber) = ranking';
%%
 for vv = 1 : m
    Datafisher = DataSD(:,ranking(1:vv));
    J(vv)=FisherCriterion(Datafisher ,Label);   %% Stopping criteria
 end
[ maxJ , k_f]=max(J); %Get the top ranked selected features info.
%%
%%%Use a linear support vector machine classifier        
NoofFeat(FoldNumber) =k_f; % The top ranked selected features
DataTrain = DataSD(:,ranking(1:k_f));
predictors =DataTrain;
response = Label;
% Train a classifier
classification = fitcsvm(predictors,response,'KernelFunction','gaussian', 'KernelScale','auto');
Datatest=MainData(testIdx,:); 
Datatest = Datatest(:,ranking(1:k_f));
% Prediction
[yfit ,scores] = predict(classification,Datatest);
%%
indx = ~isnan(scores(:,2));
hoObs = find(indx);% Holdout observation numbers
scores_prob(testIdx,:) = [hoObs, scores(indx,2)];
%%
yfit(yfit==2)=0;
Obtained_Labels(testIdx,:)=yfit;
[x1,y1,~,auc1] = perfcurve(MainLable(testIdx),scores(:,2),-1);
AUC_total(FoldNumber) = 100*auc1;
%%
test_rate(FoldNumber) = (100*sum(yfit(1:n/(2*K),1))/(n/(2*K)) +(100-100*sum(yfit((n/(2*K)+1):(n/K),1))/(n/(2*K))))/2;
TP =sum(yfit(1:n/(2*K),1)); %true positive
FN = n/(2*K)-TP;             %false negative
FP=sum(yfit((n/(2*K)+1):(n/K),1));  % false positive
TN =n/(2*K)-FP;              %true negative
acc(FoldNumber) = 100*((TP + TN )/(TP +TN +FN +FP));  %% Accuracy
Sens(FoldNumber) = 100*(TP / (TP +FN)); %% Sensitivity
Spec(FoldNumber) =100* (TN / (TN +FP)); %% Specificity
counter = counter+1;
end
test_score(p_prime) = sum(test_rate)/counter;
total_Sens(p_prime) = sum(Sens)/counter;
total_Spec(p_prime)= sum(Spec)/counter;
total_AUC(p_prime) = sum(AUC_total)/counter;
%%
Results_acc_sen_spec= [total_AUC;test_score;total_Sens;total_Spec]  
