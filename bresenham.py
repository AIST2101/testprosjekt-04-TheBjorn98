def bresenham(x0, y0, x1, y1):


    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    error = dx - dy
    x, y = x0, y0
    line_pixels = [(x0, y0)]

    while True:
        if x == x1 and y == y1:
            break  # We've reached our destination

        double_error = 2 * error
        if double_error > -dy:  # Walk along x-axis
            error -= dy
            x += sx
        if double_error < dx:  # Walk along y-axis
            error += dx
            y += sy

        line_pixels.append((x, y))

    return line_pixels
