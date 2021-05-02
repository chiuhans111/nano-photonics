function mask = aperture(X, Y, radius)
    r = sqrt(X.^2+Y.^2);
    mask = r<=radius;
end