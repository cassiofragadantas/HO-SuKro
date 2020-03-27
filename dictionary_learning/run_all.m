% Calls the DL_image_denoise_3D_input or DL_HSI_denoise_input function 
% for a given set of parameters.
%
% DL_image_denoise_3D_input(imnum, exp_type, algo_type, sigma, mc_it,samples_training,blocksize,blocksize_m)
% Exemple of call:
% DL_image_denoise_3D_input(2,2,2,20,5,[2000,40000],[6,6,3],[12,12,6])

data_type = 1;              % 1: Color image, 2: Hyperspectral Image

imnum = [1];                % 1 to 5: one given image. 6: all 5 images.
exp_type = 1;               %1. Single-r3un (a few minutes)  2. Complete   (a few hours)
algo_type = [2];              % 1. SuKro  2. HO-SuKro  3. K-SVD  4. ODCT
iternum = 2;               % Nb. iterations of Dict.update + Sparse coding. Default is 20.
sigma = [50]; %round([1650 928 522 293 165]*1.2); %[5 3 2]*128;               % Noise standard deviation (on pixel values)
mc_it = 1;                  % Number of noise realization iterations
samples_training = 40000;   %[1000 2000 5000 10000 20000]; % Number of training samples
n = [6, 6, 3];                % 3D-data dimensions 
m = [12, 12, 6];              % nb. atoms of each subdictionary D1, D2 and D3 respectively

gain = 1.02; %1:00.01:1.10; % Calibrated gains:
                            % Color image: gain_KSVD=TODO, gain_SuKro=TODO
                            % HSI: gain_KSVD=1.06, gain_SuKro=1.02

for k_algo_type = algo_type
    for k_sigma = sigma
        for k_imnum = imnum
            for k_gain = gain
                if data_type == 1
                    DL_image_denoise_3D_input(k_imnum, exp_type, k_algo_type, iternum, k_sigma, mc_it,samples_training,n,m,k_gain);
                else
                    DL_HSI_denoise_input(k_imnum, exp_type, k_algo_type, iternum, k_sigma, mc_it,samples_training,n,m,k_gain);
                end
            end
        end
    end
end