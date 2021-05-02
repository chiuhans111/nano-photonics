close all

%% basic coordinates
resolution = 972;
radius = 17;


% source
X = linspace(-1, 1, resolution) * radius;
Y = linspace(-1, 1, resolution) * radius;
[X, Y] = meshgrid(X, Y);


%% lens spec
focal_length = 12;
wavelength = 0.55;
lens_radius = radius;

distance = sqrt(X.^2 + Y.^2 + focal_length^2) - focal_length;
phase_distance = distance / wavelength;
lens = phase_distance;

%lens = lens .* aperture(X, Y, lens_radius);

pcolor(lens)
shading flat

%%
ssz = 2000
data = string(round(reshape(lens*ssz, [], 1)));

%%

fileID = fopen('ideal.int','w');

fprintf(fileID, "IDEAL LENS\n");
fprintf(fileID, 'GRD %d %d WFR WVL %f SSZ %d\n', ...
    resolution, resolution, wavelength, ssz);

fprintf(fileID, '%s %s %s %s %s %s %s %s \n', data);

fclose(fileID);
fprintf("done");
