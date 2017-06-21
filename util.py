"""
other utilities
"""
from skimage import img_as_float
from imageio import get_reader, get_writer
from numpy import asarray, array
from numpy.linalg import inv

yiq_from_rgb = array([[0.299     ,  0.587     ,  0.114     ],
                         [0.59590059, -0.27455667, -0.32134392],
                         [0.21153661, -0.52273617,  0.31119955]])
rgb_from_yiq = inv(yiq_from_rgb)

def rgb2yiq(img):
    return img_as_float(img).dot(yiq_from_rgb.T)

def yiq2rgb(img):
    return img_as_float(img).dot(rgb_from_yiq.T)

def load_video(filename):
    reader = get_reader(filename)
    orig_vid = []
    for i, im in enumerate(reader):
        orig_vid.append(im)
    return asarray(orig_vid)

def write_video(video, fps, name):
    writer = get_writer(name, fps=fps)
    for i in range(video.shape[0]):
        writer.append_data(video[i])
    writer.close()

