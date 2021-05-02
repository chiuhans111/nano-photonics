function output_field = propagation_huge(X, Y, field, wavelength, X2, Y2, Z2)
    si = size(field);
    so = size(X2);
    input_field = reshape(field, 1, 1, si(1), si(2), 1);

    coord_in = reshape(cat(3, X, Y), 1, 1, si(1), si(2), 2);
    coord_out = reshape(cat(3, X2, Y2), so(1), so(2), 1, 1, 2);
    coord_outZ = reshape(Z2, so(1), so(2), 1, 1);
    displace = coord_out - coord_in;
    distance = sqrt(sum(displace.^2, 5)+coord_outZ.^2);
    phase_distance = distance/wavelength;
    phase = exp(2j*pi*phase_distance);
    output_field = sum(input_field .* phase ./ distance, [3 4]);
    
    fprintf("done\n")
end
