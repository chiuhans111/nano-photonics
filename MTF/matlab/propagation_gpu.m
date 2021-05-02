function output_field = propagation_gpu(X, Y, field, wavelength, X2, Y2, Z2)
    coord_in = gpuArray(cat(3, X, Y));
    coord_out = gpuArray(cat(3, X2, Y2));
    
    Z2 = gpuArray(Z2);
    field = gpuArray(field);
    
    Z22 = Z2.^2;
    so = size(X2);
    
    
    
    output_field = zeros(so(1), so(2));

    for x = 1:so(2)
        if mod(x, 10) == 0
            fprintf("%.2f%%\n", x/so(2)*100)
        end
        for y = 1:so(1)
            displace = reshape(coord_out(y, x, :), 1, 1, 2) - coord_in;
            distance = sqrt(sum(displace.^2, 3) + Z22(y,x));
            directivity = abs(Z2(y,x)./distance);
            phase_distance = distance/wavelength;
            phase = exp(2j*pi*phase_distance);
            output_field(y, x) = gather(sum(field .* phase ...
                .* directivity ./ distance, [1 2]));
        end
    end
    
    fprintf("done\n")
end
