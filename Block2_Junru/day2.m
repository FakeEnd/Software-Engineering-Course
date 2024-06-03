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
m_original = m;
% 1. getting internal force:
Fi = Finternal(m, 'plot_or_not', false);
 
% other parameter recommendations:
mu = 100;
 
% Parameters for volume and surface area forces
targetVolume = 0.6; % Set your target volume here
targetSurfaceArea = 1.0; % Set your target surface area here
kv = 1; % Set appropriate value for kv
ks = 1; % Set appropriate value for ks
 
n_iter = 100;
stds = zeros(n_iter, 1);

m.pm.k_c = 1
 
% Perturbation amount for finite difference calculation
epsilon = 1e-3;
 
for iter = 1:n_iter
    % 2. spatial range for remesh later
    rLim = [min(Fi.rg), max(Fi.rg)];
    % Compute bending force (Fb) using finite difference method
    Fb = zeros(size(m.var.coord));
    % for dim = 1:3
    %     r_orig = m.var.coord(:, dim);
    %     % Perturb positively
    %     m.var.coord(:, dim) = r_orig + epsilon;
    %     Hp = Helfrich(m); % Compute Helfrich free energy
    %     % Perturb negatively
    %     m.var.coord(:, dim) = r_orig - epsilon;
    %     Hm = Helfrich(m); % Compute Helfrich free energy
    %     % Reset the coordinate
    %     m.var.coord(:, dim) = r_orig;
    %     % Finite difference approximation of the gradient
    %     Fb(:, dim) = -(Hp - Hm) / (2 * epsilon);
    % end
    for i = 1:length(m.var.coord)
         for dim = 1:3
             r_orig = m.var.coord(i, dim);
             % Perturb positively
             m.var.coord(i, dim) = r_orig + epsilon;
             Hp = Helfrich(m); % Compute Helfrich free energy
             % Perturb negatively
             m.var.coord(i, dim) = r_orig - epsilon;
             Hm = Helfrich(m); % Compute Helfrich free energy
             % Reset the coordinate
             m.var.coord(i, dim) = r_orig;
             % Finite difference approximation of the gradient
             Fb(i, dim) = -(sum(Hp) - sum(Hm)) / (2 * epsilon);
         end
     end
    % Compute volume force (Fv)
    % V = computeVolume(m); % Function to compute the volume enclosed by the membrane
    % Ev = kv * ((V - targetVolume)^2) / targetVolume;
    % Fv = gradient(Ev, m.var.coord); % Compute the gradient of Ev with respect to the coordinates
    Fv = zeros(size(m.var.coord));
    for i = 1:length(m.var.coord)
         for dim = 1:3
             r_orig = m.var.coord(i, dim);
             % Perturb positively
             m.var.coord(i, dim) = r_orig + epsilon;
             Ev_before = kv * ((sum(Volume(m)) - targetVolume)^2) / targetVolume;
             % Perturb negatively
             m.var.coord(i, dim) = r_orig - epsilon;
             Ev_after = kv * ((sum(Volume(m)) - targetVolume)^2) / targetVolume;
             % Reset the coordinate
             m.var.coord(i, dim) = r_orig;
             % Finite difference approximation of the gradient
             Fv(i, dim) = -(Ev_before - Ev_after) / (2 * epsilon);
         end
     end
    % Compute surface force (Fs)
    % S = computeSurfaceArea(m); % Function to compute the surface area of the membrane
    % Es = ks * ((S - targetSurfaceArea)^2) / targetSurfaceArea;
    % Fs = gradient(Es, m.var.coord); % Compute the gradient of Es with respect to the coordinates
    Fs = zeros(size(m.var.coord));
    for i = 1:length(m.var.coord)
         for dim = 1:3
             r_orig = m.var.coord(i, dim);
             % Perturb positively
             m.var.coord(i, dim) = r_orig + epsilon;
             Es_before = kv * ((sum(Area(m)) - targetSurfaceArea)^2) / targetVolume;
             % Perturb negatively
             m.var.coord(i, dim) = r_orig - epsilon;
             Es_after = kv * ((sum(Area(m)) - targetSurfaceArea)^2) / targetVolume;
             % Reset the coordinate
             m.var.coord(i, dim) = r_orig;
             % Finite difference approximation of the gradient
             Fs(i, dim) = -(Es_before - Es_after) / (2 * epsilon);
         end
     end
    % 3. getting the adaptive time step
    % [dt, Ftotal, l] = varDt(m, Fi, Fb + Fv + Fs, mu);
    [dt, Ftotal, l] = varDt(m, Fi, Fb+Fv+Fs, mu);
    % 4. remeshing
    [m, ~] = RemeshCtrl(m, Fi, rLim, 'l0', l, 'print_or_not', false);
    % [m, ~] = Remesh(m, Fi, rLim, 'l0', l, 'print_or_not', false);
    % Update membrane coordinates
    % m.var.coord = m.var.coord + m.pm.mu * Ftotal * dt;
    % Optional: calculate standard deviation of some quantity for plotting
    std(l)
    stds(iter) = std(l);
end
 
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
