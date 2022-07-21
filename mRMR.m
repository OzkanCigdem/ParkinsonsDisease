function [fea, score] = mRMR(X_train, Y_train, K)
bdisp=0;

nd = size(X_train,2);
nc = size(X_train,1);

t1=cputime;
for i=1:nd
   t(i) = mutualinfo(X_train(:,i), Y_train);
end

[tmp, idxs] = sort(-t);
fea_base = idxs(1:K);

fea(1) = idxs(1);
KMAX = min(1000,nd); %500
idxleft = idxs(2:KMAX);

k=1;
for k=2:K
   t1=cputime;
   ncand = length(idxleft);
   curlastfea = length(fea);
   for i=1:ncand
      t_mi(i) = mutualinfo(X_train(:,idxleft(i)), Y_train); 
      mi_array(idxleft(i),curlastfea) = getmultimi(X_train(:,fea(curlastfea)), X_train(:,idxleft(i)));
      c_mi(i) = mean(mi_array(idxleft(i), :)); 
   end
   [score(k), fea(k)] = max(t_mi(1:ncand) - c_mi(1:ncand));
   tmpidx = fea(k); fea(k) = idxleft(tmpidx); idxleft(tmpidx) = [];

end

return;
%===================================== 
function c = getmultimi(da, dt) 
for i=1:size(da,2)
   c(i) = mutualinfo(da(:,i), dt);
end
    
