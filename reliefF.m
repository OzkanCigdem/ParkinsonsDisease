function [RANKED, WEIGHT] = reliefF( X, Y, K )
fprintf('\n+ Feature selection method: Relief-F \n');
[RANKED,WEIGHT] = relieff(X,Y,K);

