close all

%% basic coordinates
resolution = 972;
radius = 17;

metalens_data = dlmread("Phase_mask(Ideal_Amp_Phase).dat", '', 4);
amp_data = metalens_data(:, 1:2:end);
phase_data = metalens_data(:, 2:2:end);

lens = phase_data/360;

pcolor(lens)
shading flat

%%
ssz = 2000
data = string(round(reshape(lens*ssz, [], 1)));

%%

fileID = fopen('metalens.int','w');

fprintf(fileID, "IDEAL LENS\n");
fprintf(fileID, 'GRD %d %d WFR WVL %f SSZ %d\n', ...
    resolution, resolution, wavelength, ssz);

fprintf(fileID, '%s %s %s %s %s %s %s %s \n', data);

fclose(fileID);
fprintf("done");
