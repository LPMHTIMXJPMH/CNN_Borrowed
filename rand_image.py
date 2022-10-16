import numpy as np;image_size = {"x": 28, "y": 28, "ch": 1}
eight_times = 256
# generating random images
image = np.random.randint(eight_times, size = (image_size['x'] * image_size['y'] * image_size['ch']), dtype = np.uint8)

if image_size['ch'] == 1:
    image = image.reshape(image_size['y'], image_size['x'])
else:
    image = image.reshape(image_size['y'], image_size['x'], image_size['ch'])