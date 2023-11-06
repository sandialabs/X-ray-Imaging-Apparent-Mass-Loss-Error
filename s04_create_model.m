%%S04_CREATE_MODEL
% Script that builds the regression model.
% Note: Results may not perfectly match the paper figures, as the outputs are
% dependent on the initial randomly generated synthetic database. However,
% the overall trends and reduction in errors should still be consistent!
%
% Processing pipeline:
%   --  s01_create_synthetic.m
%   --  s02_process_synthetic.m
%   --  s03_process_experimental.m
%   --> s04_create_model.m
%
% Authors: Naveed Rahman, Benjamin R. Halls
% Diagnostic Science and Engineering (01512), Sandia National Laboratory
%
% Copyright 2022 National Technology & Engineering Solutions of
% Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
% the U.S. Government retains certain rights in this software.

clc, clear all

% Load synthetic results.
synParams = load([pwd, '/output/synthetic_params.mat']);

% Load experimental results.
expParams = load([pwd, '/output/experimental_params.mat']);

% Pixel size from the experimental data.
px_size = expParams.px_size;

% Density of the experimental objects.
density = expParams.density;

%% PRE-PROCESS
% Number of loss ratio intervals.
% This splits up the synthetic database to get a uniform distribution of
% objects for training the model.
numInt = 7;

% Number of objects (for each of the "numInt" loss ratio intervals).
% May need to change if you fiddle with "numObj" in s01_create_synthetic.m.
numModel = 300; % model training (numModel*numInt = total training points)
numVerify = 20; % model verification (numVerify*numInt = total verify points)

[mdl_params, mdl_X, mdl_Y, markers, idxVerify] = preprocess( ...
                                                            synParams, ...
                                                            px_size, ...
                                                            density, ...
                                                            numInt, ...
                                                            numModel, ...
                                                            numVerify ...
                                                            );

%% MODEL TRAINING
% Ridge and lasso hyper-parameters.
lambda = logspace(-5, 3, 1000);
phi = logspace(-5, 3, 1000);

% Train the model.
[mdl_lasso, mdl_relax] = run_regression(mdl_X, mdl_Y, lambda, phi);

% Sort the coefficients for each model.
[~, lasso_sort] = sort(abs(mdl_lasso.B), 'descend');
[~, relax_sort] = sort(abs(mdl_relax.B), 'descend');

% Summary of coefficient strengths.
lasso_coeff_summ = [num2cell(mdl_lasso.B(lasso_sort)), mdl_params(lasso_sort)];
relax_coeff_summ = [num2cell(mdl_relax.B(relax_sort)), mdl_params(relax_sort)];

% Output coefficient strengths prior to variable selection.
disp('Five most significant coefficients prior to variable selection...');
disp(lasso_coeff_summ(1:5, :));

%% MODEL VERIFICATION
% Test model on independent set of synthetic objects.
[verify_Y, verify_RMSE, verify_MBE] = verify_model( ...
                                                   mdl_lasso, ...
                                                   mdl_relax, ...
                                                   mdl_X, ...
                                                   mdl_Y, ...
                                                   markers, ...
                                                   idxVerify ...
                                                   );

% Plot verification results.
figure();
plot(verify_Y{1}, 'linestyle', 'none', 'marker', 'o');
hold on;
plot(verify_Y{2}, 'linestyle', 'none', 'marker', '^');
yline(0, 'linestyle', '--', 'color', 'k', 'linewidth', 1.5);
legend({'Initial', 'Model Corrected', 'Ideal'}, 'Location', 'northwest');
title('Synthetic Model Verification');
xlabel('Synthetic Object');
ylabel('Loss Ratio, Y_L (%)');

%% MODEL APPLICATION
% Experimental fragment processing.
[expResults, expErr] = exp_corr( ...
                                expParams, ...
                                mdl_X, ...
                                mdl_Y, ...
                                mdl_lasso, ...
                                mdl_relax ...
                                );

% Plot experimental results.
init_ratio = 100*(expParams.mass_true - expParams.Y{2}) ./ expParams.mass_true;
corr_ratio = 100*(expParams.mass_true - expResults') ./ expParams.mass_true;
figure();
scatter(expParams.mass_true, init_ratio);
hold on;
scatter(expParams.mass_true, corr_ratio);
yline(0, 'linestyle', '--', 'color', 'k', 'linewidth', 1.5);
legend({'Initial', 'Model Corrected', 'Ideal'});
xlabel('True Mass (mg)');
ylabel('Loss Ratio, Y_L (%)');
title('Experimental Results');


%% SUB-FUNCTIONS
function [params, X, Y, markers, idxVerify] = preprocess( ...
                                                         synParams, ...
                                                         px_size, ...
                                                         density, ...
                                                         numInt, ...
                                                         numModel, ...
                                                         numVerify ...
                                                         )
    % Rename variables into "init" which contains all of the synthetic objects.
    param_list = synParams.param_list;
    x_init = synParams.X;
    y_init = synParams.Y{1};
    y_calc_init = synParams.Y{2};
    y_true_init = synParams.Y{3};
    marker_color_init = synParams.marker_styles{1};
    marker_shape_init = synParams.marker_styles{2};
    marker_size_init = synParams.marker_styles{3};

    % Convert y into volume (mm^3).
    y_true_init = y_true_init * (px_size^2);
    y_calc_init = y_calc_init * (px_size^2);

    % Convert y values of volume (mm^3) to mass (mg).
    y_true_init = y_true_init * density * 1000;
    y_calc_init = y_calc_init * density * 1000;

    % Crop out values based on experimental object mass.
    maxMass = 5000;
    del_ind = y_true_init > maxMass;
    x_init(del_ind, :) = [];
    y_init(del_ind) = [];
    y_calc_init(del_ind) = [];
    y_true_init(del_ind) = [];
    marker_color_init(del_ind) = [];
    marker_shape_init(del_ind) = [];
    marker_size_init(del_ind) = [];

    % Create bounds for model/verification object selection.
    y_loss_ratio = (y_true_init - y_calc_init) ./ y_true_init;
    lo_bnd = linspace(0, 0.30, numInt);
    hi_bnd = linspace(0.05, 0.35, numInt);

    % Retrive objects for model training and verification.
    rng('default');
    rng(2);
    for i = 1:length(lo_bnd)
        ind = find(y_loss_ratio > lo_bnd(i) & y_loss_ratio < hi_bnd(i));
        numInd = nnz(ind);
        equivLoss_ind{i} = ind(randperm(numInd, numModel+numVerify));
        inModel = randperm(numModel+numVerify, numModel);
        inVerify = find(~ismember(1:(numModel+numVerify), inModel));
        idxModel{i} = equivLoss_ind{i}(inModel);
        idxVerify{i} = equivLoss_ind{i}(inVerify);
    end
    idxModel = reshape(cell2mat(idxModel), [], 1);
    idxVerify = reshape(cell2mat(idxVerify), [], 1);

    % Model training.
    x_model = x_init(idxModel, :);
    y_model = y_init(idxModel);
    y_calc_model = y_calc_init(idxModel);
    y_true_model = y_true_init(idxModel);
    marker_color_model = marker_color_init(idxModel, :);
    marker_shape_model = marker_shape_init(idxModel, :);
    marker_size_model = marker_size_init(idxModel, :);

    % Create design matrix from the predictor variables.
    % Linear + interaction effects, no constant.
    Dz = x2fx(x_model, 'interaction');
    Dz(:,1) = [];

    % Number of main effects (x1, x2, x3 ... xn).
    numMain = length(param_list);

    % Retrieve labels for the design matrix to include the interaction effects.
    params = cell(size(Dz, 2), 1);
    mainInd = 1:numMain;
    params(mainInd) = cellfun( ...
                              @(i) sprintf('%s', i), ...
                              param_list, ...
                              'UniformOutput', false ...
                              );

    % Two-way interactions.
    inter2 = nchoosek(1:numMain, 2);

    % Number of two-way effects (x1x2, x1x3, ...).
    numEff = size(inter2, 1);
    effInd = numMain+1:length(params);
    params(effInd) = arrayfun( ...
                              @(i) sprintf( ...
                                           '%s, %s', ...
                                           param_list{inter2(i,:)} ...
                                           ), ...
                              1:numEff, ...
                              'UniformOutput', false ...
                              );

    % Normalize the explanatory variables.
    [Dz, Xc, Xs] = getNorm(Dz);

    % Convert y into mass loss ratio.
    y_model = (y_true_model - y_calc_model) ./ y_true_model;

    % Normalize the response variable.
    [Yz, Yc, Ys] = getNorm(y_model);

    % Pack outputs.
    X = {Dz, Xc, Xs, x_init};
    Y = {Yz, Yc, Ys, y_init, y_calc_init, y_true_init};
    markers = {marker_color_model, marker_shape_model, marker_size_model, ...
               marker_color_init, marker_shape_init, marker_size_init};
end


function [Xz, Xc, Xs] = getNorm(X)
    Xc = mean(X, 1);
    Xs = std(X, 1, 1);
    Xz = (X - Xc) ./ Xs;
end


function [mdl_init, mdl_final] = run_regression(X, Y, lambda_lasso, phi_ridge)
    % Unpack inputs.
    Xz = X{1};
    Xc = X{2};
    Xs = X{3};

    Yz = Y{1};
    Yc = Y{2};
    Ys = Y{3};

    % Create CV Partition object (K-fold = 5).
    cvp = cvpartition(length(Yz), 'KFold', 5);

    % Lasso regression.
    [B_init, B0_init, FitInfo_init] = penalized_regression( ...
                                                           Xz, ...
                                                           Yz, ...
                                                           Yc, ...
                                                           Ys, ...
                                                           cvp, ...
                                                           lambda_lasso, ...
                                                           'lasso' ...
                                                           );

    % Relaxed ridge regression (lasso->ridge).
    [B, B0, FitInfo] = penalized_regression( ...
                                            Xz(:, B_init~=0), ...
                                            Yz, ...
                                            Yc, ...
                                            Ys, ...
                                            cvp, ...
                                            phi_ridge, ...
                                            'ridge' ...
                                            );

    % Pack outputs.
    mdl_init.B = B_init;
    mdl_init.B0 = B0_init;
    mdl_init.FitInfo = FitInfo_init;

    mdl_final.B = B;
    mdl_final.B0 = B0;
    mdl_final.FitInfo = FitInfo;
end


function [B, B0, FitInfo] = penalized_regression( ...
                                                 X, ...
                                                 Y, ...
                                                 Yc, ...
                                                 Ys, ...
                                                 cvp, ...
                                                 lambda, ...
                                                 rType ...
                                                 )
    % Number of folds.
    numTests = cvp.NumTestSets;

    % Build model for each set.
    for i = 1:numTests
        trainXData = X(training(cvp, i), :);
        trainYData = Y(training(cvp, i));
        testXData = X(test(cvp, i), :);
        testYData = Y(test(cvp, i));

        if strcmp(rType, 'lasso')
            [B, FI] = lasso( ...
                            trainXData, ...
                            trainYData, ...
                            'Lambda', lambda, ...
                            'Intercept', false, ...
                            'Standardize', false ...
                            );
            B0 = FI.Intercept;
            y_pred = (B' * testXData') + FI.Intercept';
        elseif strcmp(rType, 'ridge')
            B = cell2mat(arrayfun( ...
                                  @(j) ridge(trainYData, trainXData, j, 0), ...
                                  lambda, ...
                                  'UniformOutput', false ...
                                  ));
            B0 = B(1, :);
            B = B(2:end, :);
            y_pred = (B' * testXData') + B0';
        end

        cvMSE{i} = mean((testYData - y_pred').^2, 1);
    end

    % Calculate cross-validation MSE and SE values.
    cvMSE = cell2mat(cvMSE');
    MSE = mean(cvMSE);
    SE = std(cvMSE) / sqrt(size(cvMSE, 1));

    % Find minimum lambda value and index.
    [minMSE, idxMinMSE] = min(MSE);
    lambdaMinMSE = lambda(idxMinMSE);

    % Retrieve the lambda one SE away.
    minSE = SE(idxMinMSE);
    [~, idx1SE] = min(abs(MSE - (minMSE + minSE)));
    lambda1SE = lambda(idx1SE);

    % Retrieve final cross-validated model parameters.
    if strcmp(rType, 'lasso')
        [B, FI] = lasso( ...
                        X, ...
                        Y, ...
                        'Lambda', lambda1SE, ...
                        'Intercept', false, ...
                        'Standardize', false ...
                        );
        B0 = FI.Intercept;
    elseif strcmp(rType, 'ridge')
        B = ridge(Y, X, lambda1SE, 0);
        B0 = B(1);
        B = B(2:end);
    end

    % Calculate Y_hat for the final fit.
    Y_hat = (X * B) + B0;

    % Calculate R^2.
    R2 = 1 - (sum((Y_hat - Y).^2) / sum((Y - mean(Y)).^2));

    % Collect errors and lambdas into FitInfo.
    FitInfo.MSE = MSE;
    FitInfo.SE = SE;
    FitInfo.lambdaMinMSE = lambdaMinMSE;
    FitInfo.lambda1SE = lambda1SE;
    FitInfo.MSEmin = MSE(idxMinMSE);
    FitInfo.MSE1SE = MSE(idx1SE);
    FitInfo.Yz = Y_hat;
    FitInfo.Y = (Y_hat * Ys) + Yc;
    FitInfo.R2 = R2;
end


function [Y, RMSE, MBE] = verify_model( ...
                                       mdl_lasso, ...
                                       mdl_relax, ...
                                       X, ...
                                       Y, ...
                                       markers, ...
                                       idxVerify ...
                                       )
    % Unpack inputs.
    Xc = X{2};
    Xs = X{3};
    x_init = X{4};
    Yc = Y{2};
    Ys = Y{3};
    y_init = Y{4};
    y_calc_init = Y{5};
    y_true_init = Y{6};
    marker_color_init = markers{4};
    marker_shape_init = markers{5};
    marker_size_init = markers{6};

    % Model verification.
    x_verify = x_init(idxVerify, :);
    Xz_verify = x2fx(x_verify, 'interaction');
    Xz_verify(:,1) = [];
    Dz = (Xz_verify - Xc) ./ Xs;
    marker_color_verify = marker_color_init(idxVerify, :);
    marker_shape_verify = marker_shape_init(idxVerify, :);
    marker_size_verify = marker_size_init(idxVerify, :);
    y_verify = y_init(idxVerify);
    y_calc_verify = y_calc_init(idxVerify);
    y_true_verify = y_true_init(idxVerify);

    % Convert y into mass loss ratio.
    y_verify = (y_true_verify - y_calc_verify) ./ y_true_verify;

    % RMSE and MBE (mean bias error) in the true and initial calculated values.
    RMSE.init = sqrt(goodnessOfFit(y_calc_verify, y_true_verify, 'MSE'));
    MBE.init = mean((y_true_verify - y_calc_verify) ./ y_true_verify);

    % Predict the Yz values using relaxed ridge (lasso->ridge).
    yz = (Dz(:, mdl_lasso.B~=0) * mdl_relax.B) + mdl_relax.B0;
    y = (yz * Ys) + Yc;
    y_mdl_verify = y_calc_verify ./ (1 - y);

    % RMSE and MBE (mean bias error) in the true and model calculated values.
    RMSE.mdl = sqrt(goodnessOfFit(y_mdl_verify, y_true_verify, 'MSE'));
    MBE.mdl = mean((y_true_verify - y_mdl_verify) ./ y_true_verify);

    % Pack outputs.
    Y = {y_verify, (y_true_verify - y_mdl_verify) ./ y_true_verify};
end


function [res, err] = exp_corr(expParams, mdl_X, mdl_Y, mdl_lasso, mdl_relax)
    % Unpack inputs.
    X = expParams.X;
    Y_calc = expParams.Y{2};
    mass_obj = expParams.mass_true;
    Xc = mdl_X{2};
    Xs = mdl_X{3};
    Yc = mdl_Y{2};
    Ys = mdl_Y{3};

    % Build design matrix.
    Dz = x2fx(X, 'interaction');
    Dz(:,1) = [];
    Dz = (Dz - Xc) ./ Xs;

    % Variable selection with lasso.
    rel = mdl_lasso.B ~= 0;
    Dz = Dz(:, rel);

    % Prediction using relaxed ridge.
    Yz = (Dz * mdl_relax.B) + mdl_relax.B0;
    Y = (Yz * Ys) + Yc;

    % Correct the initial calculated mass by the model-fitted mass loss ratio.
    Y_corr = Y_calc' ./ (1 - Y);

    % Calculate errors.
    RMSE.init = sqrt(mean(((mass_obj - Y_calc) ./ mass_obj).^2));
    RMSE.corr = sqrt(mean(((mass_obj - Y_corr') ./ mass_obj).^2));
    bias.init = mean(((mass_obj - Y_calc) ./ mass_obj));
    bias.corr = mean(((mass_obj - Y_corr') ./ mass_obj));

    % Pack outputs.
    res = Y_corr;
    err.RMSE = RMSE;
    err.MBE = bias;
end
