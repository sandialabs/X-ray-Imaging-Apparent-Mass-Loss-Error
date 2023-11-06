%%S03_PROCESS_EXPERIMENTAL.m
% Script that extracts parameters from the experimentally imaged objects.
% Output: 'output/experimental_params.mat'.
%
% Processing pipeline:
%   --  s01_create_synthetic.m
%   --  s02_process_synthetic.m
%   --> s03_process_experimental.m
%   --  s04_create_model.m
%
% Authors: Naveed Rahman, Benjamin R. Halls
% Diagnostic Science and Engineering (01512), Sandia National Laboratory
%
% Copyright 2022 National Technology & Engineering Solutions of
% Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
% the U.S. Government retains certain rights in this software.

clc, clear all

% Transmission to EPL function for stainless steel specifically for this setup.
% Potentially changes with different X-ray sources!
load('TtoEPL.mat');

% Load normalized stainless steel fragment X-ray radiographs.
load('fragments_norm.mat');

% 10-90% pixel rise. Experimentally calculated from an image of a thin edge.
rise_static = 20;

% Pixel size (mm).
px_size = 0.0522;

% Stainless steel density (g/mm^3).
density = 0.0080;

% True mass (as measured on scale) for each of the frags in mg.
mass_true = 1000*[0.0366 , 0.0525 , 0.0818 , 0.0283 , 0.0933];

% Process all of the fragment images.
fragments = process_images( ...
                           fragments_norm, ...
                           TtoEPL, ...
                           density, ...
                           rise_static, ...
                           px_size, ...
                           mass_true ...
                           );

% Collect all parameters at the different blurs.
exp_params = fragments{1}.params;
X = gather_inputs( ...
                  fragments, ...
                  exp_params, ...
                  length(fragments)...
                  );

y = cellfun(@(x) x.mass_ratio, fragments);
y_calc = cellfun(@(x) x.mass_calc, fragments);
Y = {y, y_calc};

% Save output.
if ~exist([pwd, '/output'], 'dir')
    mkdir([pwd, '/output']);
end

save([pwd, '/output/experimental_params.mat'], 'X', 'Y', 'exp_params', ...
     'px_size', 'mass_true', 'density', 'rise_static', '-v7.3');


%% SUB-FUNCTIONS
function bin = imBin(obj, method, TtoEPL)
    obj(isnan(obj)) = 0;
    obj(isinf(obj)) = 0;

    mlth = 0.5;

    % Retrieve a mask of the object and some background.
    % The complement to this gives just the surrounding background.
    switch method
        % Binarize EPL object.
        case 'epl'
            bin = imfilter(obj, fspecial('average', round(150*0.3)));
            bin = obj > TtoEPL(0.98);
            % Erode and then dilate the mask to remove any random lines/pixels
            bin = imdilate(imerode(bin, true(2)), true(2));
        % Binarize attenuation object.
        case 'atten'
            lth = mlth * max(obj, [], 'all');
            bin = obj > lth;
    end
end


function CoM = centerOfMass(A)
% MATLAB Central: 363181-center-of-mass-and-total-mass-of-a-matrix
    tot_mass = sum(A(:));
    [ii,jj] = ndgrid(1:size(A,1),1:size(A,2));
    R = sum(ii(:).*A(:))/tot_mass;
    C = sum(jj(:).*A(:))/tot_mass;
    CoM = [R, C];
end


function obj = process_images( ...
                              obj_norm, ...
                              TtoEPL, ...
                              density, ...
                              rise_static, ...
                              px_size, ...
                              mass_true ...
                              );
    % Loop through each of the objects.
    for i = 1:length(obj_norm)
        norm = obj_norm{i};

        % Convert transmission to EPL (mm).
        epl = (TtoEPL.a .* (norm .^ TtoEPL.b)) + TtoEPL.c;
        epl(isnan(epl)) = 0;
        epl(isinf(epl)) = 0;

        % Binarize the EPL image.
        bin = imBin(epl, 'epl', TtoEPL);

        % Retrieve background (foreground = object).
        objMask = ~bin;
        bg_epl_med = median(nonzeros(epl .* objMask), 'omitnan');
        bg_epl_std = std(nonzeros(epl .* objMask), 'omitnan');

        % Background correction.
        epl = epl - bg_epl_med;

        % Mask out background.
        epl = epl .* bin;

        % Calculate attenuation object.
        atten = 1 - norm;
        atten(atten == 1) = 0;
        attenBin = imBin(atten, 'atten', TtoEPL);

        % Define which representation to use for the model.
        params_obj = atten .* attenBin;
        params_obj(isnan(params_obj)) = 0;
        params_obj(isinf(params_obj)) = 0;

        % Calculate center of mass (CoM) from the attenuation image.
        CoM = centerOfMass(params_obj);

        % Pad array to a 512x512 window and move CoM to the middle.
        params_obj = padarray(params_obj, [512-301, 512-301], 0, 'post');
        params_obj = imtranslate(params_obj, 256-CoM);
        edgeMask = zeros(512,512);
        edgeMask(256-120:256+120, 256-120:256+120) = 1;
        params_obj = params_obj .* edgeMask;
        bin_obj = params_obj > 0;

        % Signal parameters.
        % Attenuation mean.
        avg = mean(nonzeros(params_obj), 'all');

        % Skewness in Fourier domain.
        obj_fft = fft2(params_obj);
        mag = abs(fftshift(obj_fft));
        mag_peak = max(mag, [], 'all');
        fft_skew = skewness(mag, 0, 'all');

        % Signal/shape parameters.
        % Entropy
        entr = entropy(nonzeros(double(params_obj)));

        % Region properties based on the binarized object.
        props = regionprops(bin_obj, 'perimeter', 'area', 'EquivDiameter');
        area = props.Area;
        perim = props.Perimeter;

        % Shape parameters.
        % Equivalent diameter.
        eDiam = props.EquivDiameter;

        % Compactness.
        compactness = area/(perim^2);

        % Shape/extrinsic parameters.
        % Relative blur.
        blur_rel = rise_static / (area/perim);

        % Extrinsic parameters.
        % 10-90% pixel rise.
        px_rise = rise_static;

        % Total (equivalent) path length (mm)
        tpl = sum(epl, 'all', 'omitnan');

        % Volume (mm^3)
        vol = tpl * (px_size)^2;

        % Extrinsic/signal parameters.
        % Calculated mass (mg).
        mass_calc = vol * density * 1000;

        % Mass loss (mg).
        mass_loss = mass_true(i) - mass_calc;

        % Mass ratio (-).
        mass_ratio = mass_true(i) / mass_calc;

        % Collect all image parameters for the model.
        obj{i}.epl = epl;
        obj{i}.mass_calc = mass_calc;
        obj{i}.mass_loss = mass_loss;
        obj{i}.mass_ratio = mass_ratio;
        obj{i}.avg = avg;
        obj{i}.fft_skew = fft_skew;
        obj{i}.entropy = entr;
        obj{i}.equivDiam = eDiam;
        obj{i}.compactness = compactness;
        obj{i}.blur_rel = blur_rel;
        obj{i}.px_rise = px_rise;
        obj{i}.params = {'avg', 'fft_skew', 'entropy', 'equivDiam', ...
                         'compactness', 'blur_rel', 'px_rise', 'mass_calc'};
    end
end


function X = gather_inputs(obj, param_list, numPts)
    % Initialize the inputs matrix.
    X = zeros(length(param_list), numPts);

    % Build up inputs matrix.
    for i = 1:numPts
        X(:,i) = cellfun(@(j) getfield(obj{i}, j), param_list);
    end

    % Transpose to match dimensions of synthetic data.
    X = X';
end
