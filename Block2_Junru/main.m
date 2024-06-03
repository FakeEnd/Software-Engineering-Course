%% requirements:
% 1. gcc for compiling c codes. Load before starting Matlab:
%    module load gcc/6.3.0
% 2. Use Matlab 2020a
%% Setup the directory where the membrane object is located and add the directory to Matlab's function pool
%dir_mod = '/home2/s171152/codes/matlab/mine/git/memCompCourse/memcompcourse';
dir_mod = './';
addpath(dir_mod);
 
%--------------------------------------------------------------------------
% create 'unit' u using the unit module, and 'membrane' m using the membrane module
u = ComUnit('erg', ComUnit.nm_to_cm(1000), 300, ComUnit.kBT_to_erg(10, 300));
m = ModMembrane(2, 'unit', u);
m.pm.Vdh.V0 = 0.1; % Setting internal force's strength using recommended value
m.pm.k_c=4
m_original = m;

% 1. getting internal force:
Fi = Finternal(m, 'plot_or_not', false);
 
% other parameter recommendations:
mu = 500;
 
% Parameters for volume and surface area forces
targetVolume = 0.6*sum(Volume(m_original)); % Set your target volume here
targetSurfaceArea = 1.0*sum(Area(m_original)); % Set your target surface area here

% print targetVolume and targetSurfaceArea
fprintf('Target Volume: %f, Target Surface Area: %f\n', targetVolume, targetSurfaceArea);

kv = 32; % Set appropriate value for kv
ks = 32; % Set appropriate value for ks
 
n_iter = 3000;
stds = zeros(n_iter, 1);
 
% Perturbation amount for finite difference calculation
epsilon = 1e-2;

% profile on;
% 
% profile clear;
 
for iter = 1:n_iter
    % 2. spatial range for remesh later
    rLim = [min(Fi.rg), max(Fi.rg)];
    % Compute bending force (Fb) using finite difference method
    Fb = zeros(size(m.var.coord));
    Fv = zeros(size(m.var.coord));
    Fs = zeros(size(m.var.coord));
    parfor i = 1:length(m.var.coord)
         for dim = 1:3
             m_update = m;
             r_orig = m_update.var.coord(i, dim);
             % Perturb positively
             m_update.var.coord(i, dim) = r_orig + epsilon;
             Ev_before = kv * ((sum(Volume(m_update)) - targetVolume)^2) / targetVolume;
             Es_before = ks * ((sum(Area(m_update)) - targetSurfaceArea)^2) / targetSurfaceArea;
             Hp = Helfrich(m_update); % Compute Helfrich free energy
             % Perturb negatively
             m_update.var.coord(i, dim) = r_orig - epsilon;
             Ev_after = kv * ((sum(Volume(m_update)) - targetVolume)^2) / targetVolume;
             Es_after = ks * ((sum(Area(m_update)) - targetSurfaceArea)^2) / targetSurfaceArea;
             Hm = Helfrich(m_update); % Compute Helfrich free energy
             % Reset the coordinate
             m_update.var.coord(i, dim) = r_orig;
             % Finite difference approximation of the gradient
             Fb(i, dim) = -(sum(Hp) - sum(Hm)) / (2 * epsilon);
             Fv(i, dim) = -(Ev_before - Ev_after) / (2 * epsilon);
             Fs(i, dim) = -(Es_before - Es_after) / (2 * epsilon);
         end
    end

    if mod(iter, 50) == 0
        fig = figure;
        col = rand(m_original.var.n_coord, 3);
        plot(m_original, 'f', fig, 'col', col, 'col_min', 0, 'col_max', 1, 'colBar', true);
        
        fprintf('Volume: %f, Surface Area: %f\n', sum(Volume(m)), sum(Area(m)));
    end
    
    % 3. getting the adaptive time step
    % [dt, Ftotal, l] = varDt(m, Fi, Fb + Fv + Fs, mu);
    [dt, Ftotal, l] = varDt(m, Fi, Fb+Fv+Fs, mu);
    m.var.coord = m.var.coord + m.pm.mu * Ftotal * dt;
    % 4. remeshing
    [m, ~] = RemeshCtrl(m, Fi, rLim, 'l0', l, 'print_or_not', false);
    % [m, ~] = Remesh(m, Fi, rLim, 'l0', l, 'print_or_not', false);
    % Update membrane coordinates
    % m.var.coord = m.var.coord + m.pm.mu * Ftotal * dt;
    % Optional: calculate standard deviation of some quantity for plotting
    % stds(iter) = some calculation involving m.var.coord
end

% 
% profile viewer;
 
% Plot the results
plot(1:n_iter, stds);
 
% Plot the membrane 'm'
fig = figure;
subplot(1, 2, 1);
col = rand(m_original.var.n_coord, 3);
plot(m_original, 'f', fig, 'col', col, 'col_min', 0, 'col_max', 1, 'colBar', true);
subplot(1, 2, 2);
col = rand(m.var.n_coord, 3);
plot(m, 'f', fig, 'col', col, 'col_min', 0, 'col_max', 1, 'colBar', true);
%--------------------------------------------------------------------------
