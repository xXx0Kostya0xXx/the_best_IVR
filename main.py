from PIL import Image
from functools import reduce
import numpy as np

ORIGIN = np.array([0, 0, 0, 0])
DEFAULT_OBJ_COLOR = 255
DEFAULT_BG_COLOR = 0
FARAWAY = 1.0e39  # an implausibly huge distance


class Cylinder:
    def __init__(self, position, direction, r, velocity):
        self.position = position
        self.direction = direction
        self.r = r
        self.velocity = velocity

    def intersect(self, ray):
        ray_3 = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, np.delete(ray, 0, axis=1))
        oc = ray_3 - self.position
        card = np.sum(self.direction*ray_3, axis=1)
        caoc = np.sum(self.direction*oc, axis=1)
        a = 1.0 - np.square(card)
        b = np.sum(oc*ray_3, axis=1) - caoc*card
        c = np.sum(oc*oc, axis=1) - np.multiply(caoc, caoc) - np.multiply(self.r, self.r)
        h = b * b - a * c
        return np.where(h < 0.0, -1, (-b - np.sqrt(h))/a)

    def boost_matrix(self):
        beta = np.asarray(self.velocity)
        beta_squared = np.inner(beta, beta)
        if beta_squared >= 1:
            raise ValueError("beta^2 = {} not physically possible".format(beta_squared))
        if beta_squared == 0:
            return np.identity(4)
        gamma = 1 / np.sqrt(1 - beta_squared)
        lambda_00 = np.matrix([[gamma]])
        lambda_0j = -gamma * np.matrix([beta, beta, beta])
        lambda_i0 = lambda_0j.transpose()
        lambda_ij = np.identity(3) + (gamma - 1) * np.outer(beta, beta) / beta_squared
        return np.asarray(np.bmat([[lambda_00, lambda_0j], [lambda_i0, lambda_ij]]))


(w, h) = (400, 300)

x = np.tile(np.linspace(-1, 1, w), h)
y = np.repeat(np.linspace(h/w, -h/w, h), w)
focal_length = np.ones(h * w)
ct = -np.sqrt(x ** 2 + y ** 2 + focal_length ** 2)

rays = np.stack((ct, x, y, focal_length), axis=1)

def raytrace(ray, obj):
    return 0


O = Cylinder(np.asarray([3.,2.,10.]), np.asarray([0.,1.,0.]), 0.2, 0.6)
boost_matrix = O.boost_matrix()
temp_rays = np.apply_along_axis(boost_matrix.dot, 1, rays) - boost_matrix.dot(ORIGIN)
image = Image.fromarray(255*np.clip(O.intersect(temp_rays), 0, 1).reshape((h,w)).astype(np.uint8), mode="L")
image.save("final_image.png")
scene = [O]

image_data = []

for obj in scene:
    boost_matrix = obj.boost_matrix()
    temp_rays = np.apply_along_axis(boost_matrix.dot, 1, rays) - boost_matrix.dot(ORIGIN)
    image_data.append(raytrace(temp_rays, obj))

a = np.asarray(([1.,2.,3.],[1.,2.,3.]))
b = np.asarray(([1.,2.,3.],[1.,2.,3.]))
