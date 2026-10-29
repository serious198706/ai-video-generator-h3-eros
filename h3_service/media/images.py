import warnings
from PIL import Image, ImageOps


def read_image(path):
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(path) as image:
            if image.width * image.height > 40_000_000:
                raise ValueError('image pixel limit exceeded')
            return ImageOps.exif_transpose(image).convert('RGB')


def prepare(first, last, resolution, directory):
    image = read_image(first)
    width, height = (960, 544) if resolution == '540p' else (1344, 768)
    if image.height > image.width:
        width, height = height, width
    paths = []
    for name, source in [('first', image), ('last', read_image(last) if last else None)]:
        path = directory / f'{name}.png' if source else None
        if source:
            ImageOps.fit(source, (width, height), method=Image.Resampling.LANCZOS).save(path)
            source.close()
        paths.append(path)
    return paths[0], paths[1], width, height
