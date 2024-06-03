%% Regression Test for Membrane Simulation Code

% Set up the directory where the membrane object is located and add the directory to MATLAB's function pool
dir_mod = './';
addpath(dir_mod);

% Create 'unit' u using the unit module, and 'membrane' m using the membrane module
u = ComUnit('erg', ComUnit.nm_to_cm(1000), 300, ComUnit.kBT_to_erg(10, 300));
m = ModMembrane(2, 'unit', u);
m.pm.Vdh.V0 = 0.1; % Setting internal force's strength using recommended value
m.pm.k_c = 1;
m_original = m;

% Get internal force
Fi = Finternal(m, 'plot_or_not', false);

% Other parameter recommendations
mu = 100;
targetVolume = 0.6 * sum(Volume(m_original)); % Set your target volume here
targetSurfaceArea = 1.0 * sum(Area(m_original)); % Set your target surface area here
kv = 10; % Set appropriate value for kv
ks = 100; % Set appropriate value for ks
n_iter = 10000;
stds = zeros(n_iter, 1);
epsilon = 1e-3;

% Run simulation
for iter = 1:n_iter
    % Compute bending force (Fb)
    Fb = compute_bending_force(m, epsilon);
    
    % Compute volume force (Fv)
    Fv = compute_volume_force(m, targetVolume, kv, epsilon);
    
    % Compute surface force (Fs)
    Fs = compute_surface_force(m, targetSurfaceArea, ks, epsilon);
    
    % Get adaptive time step
    [dt, Ftotal, l] = varDt(m, Fi, Fb + Fv + Fs, mu);
    m.var.coord = m.var.coord + m.pm.mu * Ftotal * dt;
    
    % Remesh
    [m_update, ~] = RemeshCtrl(m, Fi, [min(Fi.rg), max(Fi.rg)], 'l0', l, 'print_or_not', false);
    m = m_update;
end

% Save results for comparison
save('current_results.mat', 'm', 'stds');

% Load reference results
load('reference_results.mat', 'ref_m', 'ref_stds');

% Compare current results with reference results
assert(isequaln(m.var.coord, ref_m.var.coord), 'Membrane coordinates do not match.');
assert(isequaln(stds, ref_stds), 'Standard deviations do not match.');

% If necessary, plot the results for visual inspection
fig = figure;
subplot(1, 2, 1);
col = rand(m_original.var.n_coord, 3);
plot(m_original, 'f', fig, 'col', col, 'col_min', 0, 'col_max', 1, 'colBar', true);
subplot(1, 2, 2);
col = rand(m.var.n_coord, 3);
plot(m, 'f', fig, 'col', col, 'col_min', 0, 'col_max', 1, 'colBar', true);

disp('Regression test passed.');

%% Helper Functions
function Fb = compute_bending_force(m, epsilon)
    Fb = zeros(size(m.var.coord));
    parfor i = 1:length(m.var.coord)
        for dim = 1:3
            m_update = m;
            r_orig = m_update.var.coord(i, dim);
            m_update.var.coord(i, dim) = r_orig + epsilon;
            Hp = Helfrich(m_update);
            m_update.var.coord(i, dim) = r_orig - epsilon;
            Hm = Helfrich(m_update);
            m_update.var.coord(i, dim) = r_orig;
            Fb(i, dim) = -(sum(Hp) - sum(Hm)) / (2 * epsilon);
        end
    end
end

function Fv = compute_volume_force(m, targetVolume, kv, epsilon)
    Fv = zeros(size(m.var.coord));
    parfor i = 1:length(m.var.coord)
        for dim = 1:3
            m_update = m;
            r_orig = m_update.var.coord(i, dim);
            m_update.var.coord(i, dim) = r_orig + epsilon;
            Ev_before = kv * ((sum(Volume(m_update)) - targetVolume)^2) / targetVolume;
            m_update.var.coord(i, dim) = r_orig - epsilon;
            Ev_after = kv * ((sum(Volume(m_update)) - targetVolume)^2) / targetVolume;
            m_update.var.coord(i, dim) = r_orig;
            Fv(i, dim) = -(Ev_before - Ev_after) / (2 * epsilon);
        end
    end
end

function Fs = compute_surface_force(m, targetSurfaceArea, ks, epsilon)
    Fs = zeros(size(m.var.coord));
    parfor i = 1:length(m.var.coord)
        for dim = 1:3
            m_update = m;
            r_orig = m_update.var.coord(i, dim);
            m_update.var.coord(i, dim) = r_orig + epsilon;
            Es_before = ks * ((sum(Area(m_update)) - targetSurfaceArea)^2) / targetSurfaceArea;
            m_update.var.coord(i, dim) = r_orig - epsilon;
            Es_after = ks * ((sum(Area(m_update)) - targetSurfaceArea)^2) / targetSurfaceArea;
            m_update.var.coord(i, dim) = r_orig;
            Fs(i, dim) = -(Es_before - Es_after) / (2 * epsilon);
        end
    end
end
