def checkerboard_texture(point):
    scale = 2.0
    if (int(point[0] * scale) + int(point[2] * scale)) % 2 == 0:
        return [1.0, 1.0, 1.0]  # White
    else:
        return [0.0, 0.0, 0.0]  # Black


def gradient_texture(point):
    t = (point[1] - 0) / 5
    t = np.clip(t, 0, 1)
    return [1 - t, 1 - t, t]  # Gradient
