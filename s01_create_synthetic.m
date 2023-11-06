%%S01_CREATE_SYNTHETIC
% Script that creates cones and cylinders for the synthetic database.
% Note: Processing time may take a while (> 2 hours) for 1000 objects!
% Output: 'output/synthetic_data.mat' (~3 GB total size for 1000 objects).
%
% Processing pipeline:
%   --> s01_create_synthetic.m
%   --  s02_process_synthetic.m
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

% Set synthetic object image size (512x512 px).
width = 512;
hw = floor(width/2);
imScale = 1024 / width;

% Range of transmission (I/I0) from experimental data.
T_range = [0.4, 0.98];

% Range of attenuation values from experimental data.
A_range = 1 - T_range;

% Maximum attenuation to model.
maxAtten = max(A_range);

% Pseudo-density value to enforce attenuation limit.
rho = maxAtten / (0.8*hw);

% Number of objects. Changing this value may impact the "numModel" and
% "numVerify" in s04_create_model.m!
numObj = 1000;

% Experimentally measured noise levels.
noise_std = [0.0048, 0.0079];

% Initialize objects.
cone = cell(numObj, 1);
cyln = cell(numObj, 1);

% Retrieve projection data.
tic
for i = 1:numObj
     cone{i} = create_proj(0, width, hw, T_range, rho, noise_std);
     cyln{i} = create_proj(1, width, hw, T_range, rho, noise_std);
end
elapsedTime = toc;

% Output processing time.
disp(sprintf('Processing time for %d objects: %d s', numObj, elapsedTime));

% Save output.
if ~exist([pwd, '/output'], 'dir')
    mkdir([pwd, '/output']);
end

save([pwd, '/output/synthetic_database.mat'], 'cone', 'cyln', '-v7.3');


%% SUB-FUNCTIONS
function output = create_proj( ...
                              shape, ...
                              width, ...
                              hw, ...
                              T_range, ...
                              rho, ...
                              noise_std ...
                              )

    % Set bounds for the modeled radii.
    rad_lo = hw/10;
    rad_hi = hw/2;

    % Set bounds for the modeled half-lengths.
    hl_lo = hw/10;
    hl_hi = hw/1.3;

    % Set initial condition for the while loop.
    proj_min = 0;

    % Create random object; while loop to ensure transmission range is met.
    while proj_min < min(T_range)
        % Retrieve randomized radius and half-lengths.
        seed = rand(1);
        rad = (rad_hi - rad_lo) * seed + rad_lo;
        hl = (hl_hi - hl_lo) * seed + hl_lo;

        % Round to integer values for indexing.
        a = round(hw-hl);
        b = round(hw-hl+(hl*2*shape));
        c = round(hw+hl);

        % Create radius profile.
        r = zeros(width, 1);
        r(a:b) = rad;
        r(b:c) = linspace(rad, 0, c-b+1);

        % Build volume.
        x = 1:width;
        y = 1:width;
        z = 1:width;
        slice = arrayfun(@(k) arrayfun( ...
                                       @(i) (i-hw).^2+(y-hw).^2<r(k)^2, ...
                                       x, 'UniformOutput', false), ...
                                       z, 'UniformOutput', false ...
                                       );
        slice = cellfun(@(x) cell2mat(x'), slice, 'UniformOutput', false);
        vol = cell2mat(slice);
        vol = reshape(vol, [width, width, width]);
        vol = permute(vol, [3, 2, 1]);
        vol = single(vol) * rho;

        % Transmission projections at head on and 90 deg rotation views.
        proj0 = 1 - squeeze(sum(vol, 3));
        proj90 = 1 - squeeze(sum(vol, 1));

        % Create masks.
        mask0 = proj0 ~= 1;
        mask90 = proj90 ~= 1;

        % Add random Gaussian noise.
        seed2 = rand(1);
        noise_sigma = range(noise_std) * seed2 + min(noise_std);
        noise = normrnd(0, noise_sigma, size(proj0));
        proj0_noise = proj0 + noise;
        proj90_noise = proj90 + noise;

        % Retrieve minimum transmission values.
        proj0_min = min(nonzeros(proj0 .* mask0), [], 'all');
        proj90_min = min(nonzeros(proj90 .* mask90), [], 'all');

        % Update the bounds for the while loop.
        proj_min = min([proj0_min, proj90_min]);
    end

    % Populate output structure.
    output.proj0 = proj0;                   % head on view
    output.proj90 = proj90;                 % orthogonal side view
    output.proj0_noise = proj0_noise;       % head on view w/ noise
    output.proj90_noise = proj90_noise;     % orthogonal side view w/ noise
    output.noise_sigma = noise_sigma;       % object noise level
    output.rad = rad;                       % object radius
    output.hl = hl;                         % object half-length
end
