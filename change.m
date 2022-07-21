filename='results.csv';
xlswrite(filename,AGE_TIV_SEX);
NewFileName='deneme.csv';
Data = fileread(filename);
Data = strrep(Data, '/', '.');
FID = fopen(NewFileName, 'w');
fwrite(FID, Data, 'char');
fclose(FID);