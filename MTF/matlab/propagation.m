function output_field = propagation(X, Y, field, wavelength, X2, Y2, Z2)
    coord_in = cat(3, X, Y);
    so = size(X2);
    output_field = zeros(so(1), so(2));

    for x = 1:so(2)
        if mod(x, 10) == 0
            fprintf("%.2f%%\n", x/so(2)*100)
        end
        for y = 1:so(1)
            coord_out = reshape([X2(y,x) Y2(y,x)], 1, 1, 2);
            displace = coord_out - coord_in;
            distance = sqrt(sum(displace.^2, 3)+Z2(y,x).^2);
            directivity = abs(Z2(y,x)./distance);
            phase_distance = distance/wavelength;
            phase = exp(2j*pi*phase_distance);
            output_field(y, x) = sum(field .* phase ...
                .* directivity ./ distance, [1 2]);
        end
    end
    fprintf("done\n")
end
