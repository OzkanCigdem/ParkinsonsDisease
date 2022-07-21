function [J]=FisherCriterion(Data ,Label)
minLable= min(Label);
maxLable= max(Label);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_sample_1=sum((Label==maxLable));
num_sample_2= sum(Label==minLable);
S1= (Label==maxLable);
S2= (Label==minLable);
Group1 = Data(S1,:);
Group2 = Data(S2,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class1_mean = mean(Group1);
class2_mean = mean(Group2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
within_class1=0;
within_class2=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Find S_W and w_0
for i=1:num_sample_1
    within_class1 =within_class1 + ((Group1(i,:))'-class1_mean')*((Group1(i,:))'-class1_mean')';
end
for i=1:num_sample_2
    within_class2 =within_class2 + (Group2(i,:)'-class2_mean')*(Group2(i,:)'-class2_mean')';
end
S_W=within_class1+within_class2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w_0= (inv(S_W))*(class1_mean'-class2_mean');
w_0=w_0/norm(w_0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S_B = (class1_mean-class2_mean)'*(class1_mean-class2_mean);
J = (w_0'*S_B*w_0)/(w_0'*S_W*w_0);
end

