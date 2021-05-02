function phase = IdealLens(X, Y, focal_length, wavelength)
    distance = sqrt(X.^2 + Y.^2 + focal_length^2) - focal_length;
    phase_distance = distance / wavelength;
    phase = exp(-2j * pi * phase_distance);
end
