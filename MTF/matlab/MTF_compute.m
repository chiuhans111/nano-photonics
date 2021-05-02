close all

%% basic coordinates
resolution = 972;
radius = 16.99;

mon_resolution = 128;
mon_radius = 2;

% source
X = linspace(-1, 1, resolution) * radius;
Y = linspace(-1, 1, resolution) * radius;

[X, Y] = meshgrid(X, Y);

% xy cut monitor
X2 = linspace(-1, 1, mon_resolution) * mon_radius;
Y2 = linspace(-1, 1, mon_resolution) * mon_radius;
[X2, Y2] = meshgrid(X2, Y2);

% xz cut monitor
z_start = 1
z_end = 20
X3 = linspace(-1, 1, mon_resolution) * radius;
Y3 = zeros(mon_resolution, mon_resolution);
Z3 = linspace(z_start, z_end, mon_resolution);
[X3, Z3] = meshgrid(X3, Z3);

%% lens spec
focal_length = 11.6;
wavelength = 0.55;
lens_radius = radius;

lens = IdealLens(X, Y, focal_length, wavelength);
lens = lens .* aperture(X, Y, lens_radius);

airy_radius = 1.22 * wavelength / lens_radius / 2 ...
    * sqrt(focal_length^2 + lens_radius^2)

%% visualization
figure
pcolor(X,Y,angle(lens))
shading flat

%% propagation
Z = ones(mon_resolution, mon_resolution) * focal_length;
field = lens;


output_field = propagation_gpu(X, Y, field, wavelength, X2, Y2, Z);
%% visualization
figure
pcolor(X2, Y2, abs(output_field).^2)
viscircles([0 0], airy_radius)
shading interp

%% prop2
xz_cut = propagation_gpu(X, Y, field, wavelength, X3, Y3, Z3);

%% visualization
figure
pcolor(X3, Z3, abs(xz_cut).^2)
shading interp


%% MTF

NA = lens_radius / sqrt(focal_length^2+lens_radius^2);
cutoff = 2 * NA/wavelength;

w = mon_radius * 2;
[fx, fy, MTF] = mtf2d(output_field, w, w);

figure
mesh(fx, fy, abs(MTF))
viscircles([0 0], cutoff)
shading interp


% -----------------------------------------------------------------
% ----------------------------------------------------------------- :D
%
%% metalens

metalens_data = dlmread("Phase_mask(Ideal_Amp_Phase).dat", '', 4);
amp_data = metalens_data(:, 1:2:end);
phase_data = metalens_data(:, 2:2:end);

metalens_field = exp(-j*deg2rad(phase_data)) .* aperture(X, Y, lens_radius);

figure
pcolor(angle(metalens_field));
shading flat

%% propagation
output_field_metalens = propagation_gpu(X, Y, metalens_field, wavelength, X2, Y2, Z);


%% visualization
figure
pcolor(X2, Y2, abs(output_field_metalens))
viscircles([0 0], airy_radius)
shading interp


%% metalens prop2
xz_cut_metalens = propagation_gpu(X, Y, metalens_field, wavelength, X3, Y3, Z3);

%% visualization
figure
pcolor(X3, Z3, abs(xz_cut_metalens))
shading interp


%% MTF
w = mon_radius * 2
[fx, fy, MTF_metalens] = mtf2d(output_field_metalens, w, w);

figure
hold on
s = surf(fx, fy, abs(MTF))
s.FaceColor = 'red'
alpha 0.5

s = surf(fx, fy, abs(MTF_metalens))
s.FaceColor = 'interp'


hold off


viscircles([0 0], cutoff)

%% PSF compare
figure

subplot(221)
pcolor(X2, Y2, abs(output_field))
axis equal
axis tight
shading interp

subplot(222)
pcolor(X2, Y2, abs(output_field_metalens))
axis equal
axis tight
shading interp

%% field section
X4 = linspace(-1, 1, mon_resolution) * mon_radius;
Y4 = zeros(1, mon_resolution);
output_field_section = interp2(X2, Y2, output_field, X4, Y4);
output_field_metalens_section = interp2(X2, Y2, output_field_metalens, X4, Y4);

output_field_section = output_field_section / max(output_field_section)
output_field_metalens_section = output_field_metalens_section / max(output_field_metalens_section)
figure
hold on
plot(X4, abs(output_field_section))
plot(X4, abs(output_field_metalens_section))
hold off

%% MTF section
fft_resolution = mon_resolution;
X4 = linspace(0, 1, fft_resolution) * cutoff;
Y4 = zeros(1, fft_resolution);
MTF_section = interp2(fx, fy, MTF, X4, Y4);
MTF_metalens_section = interp2(fx, fy, MTF_metalens, X4, Y4);

figure
hold on
plot(X4, abs(MTF_section))
plot(X4, abs(MTF_metalens_section))
plot([0 cutoff], [1 0])
hold off

%% MTF spin
fft_resolution = mon_resolution;
X4 = linspace(0, 1, fft_resolution) * mon_radius;
Y4 = zeros(1, fft_resolution);
MTF_section = interp2(fx, fy, MTF, X4, Y4);
MTF_metalens_section = interp2(fx, fy, MTF_metalens, X4, Y4);

