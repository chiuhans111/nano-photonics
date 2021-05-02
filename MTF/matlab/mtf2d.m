function [fx, fy, MTF] = mtf2d(field, w, h)

    PSF = abs(field).^2;
    PSF = PSF / sum(PSF, 'all')
    % MTF coordinate
    s = size(PSF);
    n = s(2);
    m = s(1);

    fx=(1/w)*(-n/2:n/2-1);
    fy=(1/h)*(-m/2:m/2-1);
    [fx, fy] = meshgrid(fx, fy);

    MTF = fftshift(fft2(fftshift(PSF)));
end
