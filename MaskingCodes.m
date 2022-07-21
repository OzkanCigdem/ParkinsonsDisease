clc
clear all;
close all;
warning off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PD1_HG0=[ones(40,1);zeros(40,1)];  %40PD-->1 and 40HC-->0 are labelled
i=1; 
j=1;
n=1;
data_address = 'D:\TEZson\DATA\adres.xlsx'; % Directory of the excel file 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Directory of the mask that we obtained from SPM-->Results-->Save-->allclusters(binary)
mask_address = 'D:\Thesis\Papers\CBM2018\Reference_Papers\Babu\All_HC-PD_t-contrast.nii';
contrast = "t" ; %  t or f
TYPE = 1; % %% GM(1) WM(2) or CSF(3)
% Histogram_BIN = 256;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mask_bin          = load_nii(mask_address);
Mask_original     = Mask_bin.img;
Mask_original     = im2double(Mask_original);
Mask_original_binary_vector = Mask_original(:);  %matrix to vector
% The mask is binary and the nonzero values is set to 1 (it is 0,387 instead of 1)   
for n=1:size(Mask_original_binary_vector,1)
   if Mask_original_binary_vector(n)>0
        Mask_original_binary_vector(n)=[1];%set nonzero values to one
   else                          
        Mask_original_binary_vector(n)=[0];  %set zero values to zero
   end
   n=n+1;
end
xlswrite(data_address,TYPE,2,'B3');
[txt, raw] = xlsread(data_address,2, 'G8:G87');                                

    for i=1:size(PD1_HG0,1)  
%     counts          = 0;  
      adres           = raw(i,1);
      data            = load_nii(adres{1,1});
      type1           = im2double(data.img);
      type1           = type1(:);
      original(i,:)   = type1;
      masked    	  = type1.* Mask_original_binary_vector;
      Masked_red(i,:) = masked(masked ~= 0); %reducing non-mask elements. taking only the mask ones
     
%       [counts,binLocations]    = hist(type1,Histogram_BIN); %%%%%%
%       Hist_Original (i,:)      = counts;
%       Hpdf_Original (i,:)      = Hist_Original (i,:)/sum(Hist_Original(i,:));      
%       [counts1,binLocations1]  = hist(Masked_red(i,:),Histogram_BIN); %%%%%%
%       Hist_masked_red(i,:)     = counts1;
%       Hpdf_masked_red(i,:)     = Hist_masked_red(i,:)/sum(Hist_masked_red(i,:)); 
        i=i+1;
    end
       switch TYPE %% GM(1) WM(2) or CSF(3)
        case 1
            switch contrast % f-contrast(1) ort-contrast(2)
                case "f"
        A_GM_F_masked         = [Masked_red,PD1_HG0];
%         A_GM_F_Hpdf_Masked    = [Hpdf_masked_red,PD1_HG0];    
%         A_GM_F_Hpdf_original  = [Hpdf_Original,PD1_HG0]; 
%         A_GM_Original_data  = [original,PD1_HG0];
%         A_GM_F_Hist_masked    = [Hist_masked_red,PD1_HG0];
                case "t"
        A_GM_T_masked         = [Masked_red,PD1_HG0];
%         A_GM_T_Hpdf_Masked    = [Hpdf_masked_red,PD1_HG0];    
%         A_GM_T_Hpdf_original  = [Hpdf_Original,PD1_HG0]; 
%         A_GM_Original_data  = [original,PD1_HG0];
%         A_GM_T_Hist_masked    = [Hist_masked_red,PD1_HG0];
            end
        case 2
            switch contrast % f-contrast(1) ort-contrast(2)
                case "f"
        A_WM_F_masked         = [Masked_red,PD1_HG0];
%         A_WM_F_Hpdf_Masked    = [Hpdf_masked_red,PD1_HG0];    
%         A_WM_F_Hpdf_original  = [Hpdf_Original,PD1_HG0]; 
%         A_WM_Original_data  = [original,PD1_HG0];
%         A_WM_F_Hist_masked    = [Hist_masked_red,PD1_HG0];        
                case "t"
        A_WM_T_masked         = [Masked_red,PD1_HG0];
%         A_WM_T_Hpdf_Masked    = [Hpdf_masked_red,PD1_HG0];    
%         A_WM_T_Hpdf_original  = [Hpdf_Original,PD1_HG0]; 
%         A_WM_Original_data  = [original,PD1_HG0];
%         A_WM_T_Hist_masked    = [Hist_masked_red,PD1_HG0];    
            end
        case 3
            switch contrast % f-contrast(1) ort-contrast(2)
                case "f"
        A_CSF_F_masked         = [Masked_red,PD1_HG0];
%         A_CSF_F_Hpdf_Masked    = [Hpdf_masked_red,PD1_HG0];    
%         A_CSF_F_Hpdf_original  = [Hpdf_Original,PD1_HG0]; 
%         A_CSF_Original_data  = [original,PD1_HG0];
%         A_CSF_F_Hist_masked    = [Hist_masked_red,PD1_HG0];      
                case "t"
        A_CSF_T_masked         = [Masked_red,PD1_HG0];
%         A_CSF_T_Hpdf_Masked    = [Hpdf_masked_red,PD1_HG0];    
%         A_CSF_T_Hpdf_original  = [Hpdf_Original,PD1_HG0]; 
%         A_CSF_Original_data  = [original,PD1_HG0];
%         A_CSF_T_Hist_masked    = [Hist_masked_red,PD1_HG0];        
            end
       end
    
          



