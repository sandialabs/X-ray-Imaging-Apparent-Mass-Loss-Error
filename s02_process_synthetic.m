%%S02_PROCESS_SYNTHETIC.m
% Script that extracts parameters from the synthetic objects for the model.
% Note: Processing time is ~1 hour for 1000 objects!
% Output: 'output/synthetic_params.mat' (~40 MB total size for 1000 objects).
%
% Processing pipeline:
%   --  s01_create_synthetic.m
%   --> s02_process_synthetic.m
%   --  s03_process_experimental.m
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

% Load synthetic objects.
load([pwd, '/output/synthetic_database.mat']);

% Number of objects.
numObj = length(cone);

% Set synthetic object image size.
width = 512;
hw = floor(width/2);

% Set blur settings (min/max) from the experimental data.
exp_blur = [4, 6, 19, 28];

% Number of blur settings to model.
blur_steps = 10;

% Calculate the synthetic 10-90% rise (px).
[psf, rise_syn] = calc_1090(width, hw, exp_blur, blur_steps);

% Extract structure fields from the cell array.
cone0 = cellfun(@(x) x.proj0, cone, 'UniformOutput', false);
cone0_noise = cellfun(@(x) x.proj0_noise, cone, 'UniformOutput', false);
cone90 = cellfun(@(x) x.proj90, cone, 'UniformOutput', false);
cone90_noise = cellfun(@(x) x.proj90_noise, cone, 'UniformOutput', false);

cyln0 = cellfun(@(x) x.proj0, cyln, 'UniformOutput', false);
cyln0_noise = cellfun(@(x) x.proj0_noise, cyln, 'UniformOutput', false);
cyln90 = cellfun(@(x) x.proj90, cyln, 'UniformOutput', false);
cyln90_noise = cellfun(@(x) x.proj90_noise, cyln, 'UniformOutput', false);

% Calculate model parameters.
tic
params_cone0 = process_obj(cone0, cone0_noise, TtoEPL, psf, rise_syn);
params_cone90 = process_obj(cone90, cone90_noise, TtoEPL, psf, rise_syn);
params_cyln0 = process_obj(cyln0, cyln0_noise, TtoEPL, psf, rise_syn);
params_cyln90 = process_obj(cyln90, cyln90_noise, TtoEPL, psf, rise_syn);
elapsedTime = toc;

% Output processing time.
disp(sprintf('Processing time for %d objects: %d s', numObj, elapsedTime));

% Parameter list for the synthetic objects.
param_list = params_cone0.param_list;

% Gather parameters from all views.
[X_cone, Y_cone, blur_cone] = gather_inputs( ...
                                            param_list, ...
                                            params_cone0, ...
                                            params_cone90 ...
                                            );
[X_cyln, Y_cyln, blur_cyln] = gather_inputs( ...
                                            param_list, ...
                                            params_cyln0, ...
                                            params_cyln90 ...
                                            );

% Collate all lists.
x = [X_cone; X_cyln];
y = [Y_cone{1}; Y_cyln{1}];
y_calc = [Y_cone{2}; Y_cyln{2}];
y_true = [Y_cone{3}; Y_cyln{3}];
blurlevel = [blur_cone; blur_cyln];

% Create indices for tracking the objects.
% Total number of data points.
numPts = length(y);

% Total number of objects and views.
numTypes = 4;

% Bounds for each of the objects and views (cone0, cone90, cyln0, cyln90).
ind_bounds = linspace(0, numPts, numTypes+1);

% Cell array of all of the indices for each object/view combination.
ind_all = arrayfun( ...
                   @(x,y) (x+1):y, ...
                   ind_bounds(1:numTypes), ...
                   ind_bounds(2:(numTypes+1)), ...
                   'UniformOutput', false ...
                   );

% Create a randomized permutation of integers for indexing.
ind_rand = randperm(numPts);

% Assign each of the object/view pairings to a random location.
x_rand = x(ind_rand, :);
y_rand = y(ind_rand);
y_calc_rand = y_calc(ind_rand);
y_true_rand = y_true(ind_rand);

% Create data point marker size/shape/color. Not utilized in this particular
% workflow, but can be used to track each of the individual synthetic objects
% if desired.
marker_color = cell(numPts, 1);
marker_shape = cell(numPts, 1);
marker_size = zeros(numPts, 1, 'single');
cmap = parula;
colors = cmap(1:numTypes, :);               % colors based on shape & view
shapes = {'^', '^', 'square', 'square'};    % cone: ^, cylinder: square
sizes = linspace(1, 5, length(psf));        % marker size based on blur level

for i = 1:numTypes
    marker_color(ind_rand(ind_all{i})) = repmat( ...
                                                num2cell(colors(i, :), 2), ...
                                                length(ind_all{i}), ...
                                                1 ...
                                                );
    marker_shape(ind_rand(ind_all{i})) = repmat( ...
                                                shapes(i), ...
                                                length(ind_all{i}), ...
                                                1 ...
                                                );
    marker_size(ind_rand(ind_all{i})) = blurlevel(ind_all{i});
end

% Pack outputs.
indices = {ind_rand, ind_all};
X = x_rand;
Y = {y_rand, y_calc_rand, y_true_rand};
marker_styles = {marker_color, marker_shape, marker_size};

% Save output.
if ~exist([pwd, '/output'], 'dir')
    mkdir([pwd, '/output']);
end

save([pwd, '/output/synthetic_params.mat'], 'indices', 'X', 'Y', ...
     'param_list', 'marker_styles', 'rise_syn', '-v7.3');


%% SUB-FUNCTIONS
function [psf, rise_syn] = calc_1090(width, hw, exp_blur, blur_steps)
    % Knife edge.
    edge = zeros(width, width, 'single');
    edge(:, 1:hw)= ones;

    % Use experimental blur settings for the synthetic model.
    min_blur = 1;
    blur = linspace(min_blur, max(exp_blur)/1.4, blur_steps);

    % Calculate PSF at each blur and find corresponding 10-90% rise.
    psf = arrayfun( ...
                   @(x) fspecial('gaussian', round(5*x), x), ...
                   blur, ...
                   'UniformOutput', false ...
                   );

    % Add in [1] to simulate no blur.
    psf = [{1}, psf];
    line_rise = cellfun( ...
                        @(x) imfilter(edge, x, 'replicate'), ...
                        psf, ...
                        'UniformOutput', false ...
                        );
    line_rise = cellfun( ...
                        @(x) x(hw, :), ...
                        line_rise, ...
                        'UniformOutput', false ...
                        );

    % Calculate number of pixels for the 10-90% rise.
    rise_syn = single(cellfun(@(x) sum(x >= 0.1 & x <= 0.9), line_rise));
end


function bin = imBin(img, method, px_rise)
    img(isnan(img)) = 0;
    img(isinf(img)) = 0;

    mlth = 0.05;

    % Retrieve a mask of the object and some background.
    % The complement to this gives just the surrounding background.
    switch method
        % Binarize EPL object.
        case 'epl'
            bin = imfilter( ...
                           img, ...
                           fspecial( ...
                                    'average', ...
                                    round(double(px_rise+2)*0.3) ...
                                    ) ...
                           );
            bin = bin > 0.01;
            bin = imdilate(imerode(bin, true(3)), true(3));
        % Binarize attenuation object.
        case 'atten'
            lth = mlth * max(img, [], 'all');
            bin = imdilate(imerode(img > lth, true(3)), true(3));
    end
end


function CoM = centerOfMass(A)
% MATLAB Central: 363181-center-of-mass-and-total-mass-of-a-matrix.
    tot_mass = sum(A(:));
    [ii,jj] = ndgrid(1:size(A,1),1:size(A,2));
    R = sum(ii(:).*A(:))/tot_mass;
    C = sum(jj(:).*A(:))/tot_mass;
    CoM = [R, C];
end


function params = process_obj(obj, obj_noise, TtoEPL, psf, rise_syn)
    % Total number of objects.
    numObj = length(obj);

    % Convert all of the transmission objects to path length (EPL) objects.
    epl = cellfun( ...
                  @(x) reshape(TtoEPL(x), size(x)), ...
                  obj(1:numObj), ...
                  'UniformOutput', false ...
                  );

    % Convert to array and apply background offset (make sure EPL = 0 @ T = 1).
    epl = cat(3, epl{:});
    epl = epl - TtoEPL(1);

    % Calculate "mass" (sum of all path lengths).
    mass_true = squeeze(sum(epl, [1, 2]));

    % Number of blur steps.
    numBlur = length(psf);

    % Initialize parameters.
    avg = cell(numObj, 1);          % signal: attenuation mean
    fft_skew = cell(numObj, 1);     % signal: skewness in FFT components
    entr = cell(numObj, 1);         % signal/shape: entropy
    eDiam = cell(numObj, 1);        % shape: equivalent diameter
    compactness = cell(numObj, 1);  % shape: compactness
    blur_rel = cell(numObj, 1);     % shape/extrinsic: relative blur
    px_rise = cell(numObj, 1);      % extrinsic: 10-90% pixel rise
    mass_blur = cell(numObj, 1);    % extrinsic/signal: calculated mass

    blur_level = cell(numObj, 1);   % blur level applied to the object (1-10)
    mass_loss = cell(numObj, 1);    % apparent mass loss (absolute)
    mass_ratio = cell(numObj, 1);   % apparent mass loss (ratio)

    % Loop through each of the objects.
    for i = 1:numObj
        % Apply all blur levels from simulated PSF.
        norm_blur = cellfun( ...
                            @(x) imfilter(obj_noise{i}, x, 'replicate'), ...
                            psf, ...
                            'UniformOutput', false ...
                            );

        % Calculate EPL for the blurry objects.
        epl_blur = cellfun( ...
                           @(x) reshape(TtoEPL(x), size(x)), ...
                           norm_blur, ...
                           'UniformOutput', false ...
                           );

        % Binarize blurry objects for background masking.
        bin_blur = arrayfun( ...
                            @(j) imBin(epl_blur{j}, 'epl', rise_syn(j)), ...
                            1:length(psf), ...
                            'UniformOutput', false ...
                            );

        % Retrieve EPL background and apply background offset.
        epl_bg = arrayfun( ...
                          @(j) median(epl_blur{j} .* ~bin_blur{j}, 'all'), ...
                          1:length(psf) ...
                          );
        epl_masked = arrayfun( ...
                              @(j) (epl_blur{j} - epl_bg(j)) .* bin_blur{j}, ...
                              1:length(psf), ...
                              'UniformOutput', false ...
                              );
        epl_masked = cat(3, epl_masked{:});

        % Calculate mass parameters.
        mass_blur{i} = squeeze(sum(epl_masked, [1, 2]));
        mass_loss{i} = mass_true(i) - mass_blur{i};
        mass_ratio{i} = mass_true(i) ./ mass_blur{i};

        % Calculate center of mass (CoM) for the blurred objects.
        CoM = cellfun( ...
                      @(x) centerOfMass(1 - x), ...
                      norm_blur, ...
                      'UniformOutput', false ...
                      );

        % Translate CoM to middle of image.
        norm_blur = arrayfun( ...
                             @(i) imtranslate( ...
                                              norm_blur{i}, ...
                                              256-fliplr(CoM{i}), ...
                                              'FillValues', 1 ...
                                              ), ...
                             1:length(CoM), ...
                             'UniformOutput', 0 ...
                             );

        % Calculate attenuation object (used for parameter calculations).
        atten = cellfun( ...
                        @(x) 1 - x, ...
                        norm_blur, ...
                        'UniformOutput', false ...
                        );

        % Binarize and mask attenuation object.
        bin_obj = arrayfun( ...
                           @(j) imBin(atten{j}, 'atten', rise_syn(j)), ...
                           1:length(psf), ...
                           'UniformOutput', false ...
                           );
        params_obj = arrayfun( ...
                              @(j) atten{j} .* bin_obj{j}, ...
                              1:length(psf), ...
                              'UniformOutput', false ...
                              );

        % Signal parameters.
        % Attenuation mean.
        avg{i} = cellfun(@(x) mean2(nonzeros(x)), params_obj);

        % Skewness in Fourier domain.
        obj_fft = cellfun( ...
                          @(x) fft2(x), ...
                          params_obj, ...
                          'UniformOutput', false ...
                          );
        mag = cellfun( ...
                      @(x) abs(fftshift(x)), ...
                      obj_fft, ...
                      'UniformOutput', false ...
                      );
        mag_peak = cellfun(@(x) max(x, [], 'all'), mag);
        fft_skew{i} = cellfun(@(x) skewness(x, 0, 'all'), mag);

        % Signal/shape parameters.
        % Entropy.
        entr{i} = cellfun(@(x) entropy(nonzeros(double(x))), params_obj);

        % Region properties based on the binarized object.
        props = cellfun( ...
                        @(x) regionprops( ...
                                         x, ...
                                         'perimeter', ...
                                         'area', ...
                                         'EquivDiameter' ...
                                         ), ...
                        bin_obj, ...
                        'UniformOutput', false ...
                        );

        area = cellfun(@(x) x.Area, props);
        perim = cellfun(@(x) x.Perimeter, props);

        % Shape parameters.
        % Equivalent diameter.
        eDiam{i} = cellfun(@(x) x.EquivDiameter, props);

        % Compactness.
        compactness{i} = area ./ (perim.^2);

        % Shape/extrinsic parameters.
        % Relative blur.
        blur_rel{i} = rise_syn ./ (area./perim);

        % Extrinsic parameters.
        % 10-90% pixel rise.
        px_rise{i} = rise_syn;
        blur_level{i} = 1:numBlur;
    end

    % Retrieve true (non-blurry) masses. This is the same for all blur levels.
    params.mass_true = arrayfun( ...
                                @(x) repmat(x, [length(psf), 1]), ...
                                mass_true, ...
                                'UniformOutput', false ...
                                );

    % Collect all parameters into output structure.
    params.mass_calc = mass_blur;
    params.mass_loss = mass_loss;
    params.mass_ratio = mass_ratio;
    params.avg = avg;
    params.fft_skew = fft_skew;
    params.entropy = entr;
    params.compactness = compactness;
    params.blur_rel = blur_rel;
    params.px_rise = px_rise;
    params.blur_level = blur_level;
    params.equivDiam = eDiam;
    params.param_list = {'avg', 'fft_skew', 'entropy', 'equivDiam', ...
                         'compactness', 'blur_rel', 'px_rise', 'mass_calc'};
end


function [X, Y, blur] = gather_inputs( ...
                                      param_list, ...
                                      params_obj0, ...
                                      params_obj90 ...
                                      )
    % Number of data points in each of the views.
    numPts = length(params_obj0.(param_list{1})) * ...
             length(params_obj0.(param_list{1}){1});

    % Initialize the inputs matrix.
    X1 = zeros(numPts, length(param_list));
    X2 = zeros(numPts, length(param_list));

    for i = 1:length(param_list)
        X1(:, i) = reshape(cell2mat(params_obj0.(param_list{i})), [], 1);
        X2(:, i) = reshape(cell2mat(params_obj90.(param_list{i})), [], 1);
    end

    % Collate all parameters.
    X = [X1; X2];

    % Retrieve mass loss ratios.
    y1 = reshape(cell2mat(params_obj0.mass_ratio), [], 1);
    y2 = reshape(cell2mat(params_obj90.mass_ratio), [], 1);
    Y{1} = [y1; y2];

    % Retrieve calculated mass.
    y1 = reshape(cell2mat(params_obj0.mass_calc), [], 1);
    y2 = reshape(cell2mat(params_obj90.mass_calc), [], 1);
    Y{2} = [y1; y2];

    % Retrieve true mass.
    y1 = reshape(cell2mat(params_obj0.mass_true), [], 1);
    y2 = reshape(cell2mat(params_obj90.mass_true), [], 1);
    Y{3} = [y1; y2];

    % Collect all blur levels.
    blur1 = reshape(cell2mat(params_obj0.blur_level), [], 1);
    blur2 = reshape(cell2mat(params_obj90.blur_level), [], 1);
    blur = [blur1; blur2];
end
