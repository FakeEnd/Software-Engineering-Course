%% Setup
dir_mod = './';
addpath(dir_mod);

%% Initialize the membrane object
u = ComUnit('erg', ComUnit.nm_to_cm(1000), 300, ComUnit.kBT_to_erg(10, 300));
m = ModMembrane(2, 'unit', u);

%% Create a figure for plotting
fig = figure;

%% Copy properties for visualization
m_original = ModMembrane(2, 'unit', u);
m_original.var = m.var;  

%% Set percentage of vertices to perturb and select vertices
percentage_to_perturb = 20;  % Percentage of total vertices
num_vertices = size(m.var.coord, 1);
num_perturbed = round(num_vertices * percentage_to_perturb / 100);
perturbed_indices = randperm(num_vertices, num_perturbed);  % Randomly select vertices

%% Apply perturbation in spherical coordinates
perturbation_magnitude = 0.1;  % Smaller magnitude since this is angular

for idx = perturbed_indices
    % Convert to spherical coordinates
    x = m.var.coord(idx, 1);
    y = m.var.coord(idx, 2);
    z = m.var.coord(idx, 3);
    r = sqrt(x^2 + y^2 + z^2);
    theta = acos(z/r);
    phi = atan2(y, x);

    % Apply angular perturbation
    theta_perturbed = theta + perturbation_magnitude * randn();
    phi_perturbed = phi + perturbation_magnitude * randn();

    % Convert back to Cartesian coordinates
    x_new = r * sin(theta_perturbed) * cos(phi_perturbed);
    y_new = r * sin(theta_perturbed) * sin(phi_perturbed);
    z_new = r * cos(theta_perturbed);

    % Update the coordinates of the perturbed vertex
    m.var.coord(idx, :) = [x_new, y_new, z_new];
end
%% Visualization
m_original.pm.k_c = 1
Helfrich_energy_original = Helfrich(m_original);
H_global_perturbed_origianl = sum(Helfrich_energy_original);
mean_Helfrich_original = mean(Helfrich_energy_original);
std_Helfrich_original = std(Helfrich_energy_original);
fprintf('Mean of Helfrich Bending Energy (before perturbation): %f\n', mean_Helfrich_original);
fprintf('Standard Deviation of Helfrich Bending Energy (before perturbation): %f\n', std_Helfrich_original);
fprintf('Global Helfrich Bending Energy (before perturbation): %f\n', H_global_perturbed_origianl);


subplot(1,2,1);
col=rand(m_original.var.n_coord,3);
plot(m_original,'f',fig,'col',col,'col_min',0,'col_max',1,'colBar',true);  
title('Before Perturbation');
scatter3(m_original.var.coord(perturbed_indices, 1), m_original.var.coord(perturbed_indices, 2), m_original.var.coord(perturbed_indices, 3), ...
         10, 'red', 'filled');
m.pm.k_c = 1

Helfrich_energy = Helfrich(m);
H_global_perturbed = sum(Helfrich_energy);
mean_Helfrich = mean(Helfrich_energy);
std_Helfrich= std(Helfrich_energy);
fprintf('Mean of Helfrich Bending Energy (after perturbation): %f\n', mean_Helfrich);
fprintf('Standard Deviation of Helfrich Bending Energy (after perturbation): %f\n', std_Helfrich);
fprintf('Global Helfrich Bending Energy (after perturbation): %f\n', H_global_perturbed);


subplot(1,2,2);
col=rand(m.var.n_coord,3);
plot(m,'f',fig,'col',col,'col_min',0,'col_max',1,'colBar',true);  
hold on;
% Highlight the perturbed vertices
scatter3(m.var.coord(perturbed_indices, 1), m.var.coord(perturbed_indices, 2), m.var.coord(perturbed_indices, 3), ...
         10, 'red', 'filled');
title('After Perturbation');

figure;
subplot(2,1,1);
histogram(Helfrich_energy_original, 30,'Normalization', 'pdf');
title('PDF of Helfrich Bending Energy (before perturbation)');
xlabel('Helfrich Bending Energy');
ylabel('Probability Density');

subplot(2,1,2);
histogram(Helfrich_energy, 30,'Normalization', 'pdf');
title('PDF of Helfrich Bending Energy (after perturbation)');
xlabel('Helfrich Bending Energy');
ylabel('Probability Density');
hold off;
