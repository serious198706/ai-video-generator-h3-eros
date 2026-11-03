def standard_target(width, height):
    return (1080, 1080) if width == height else ((1920, 1080) if width > height else (1080, 1920))
