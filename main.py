from PIL import Image
import numpy as np

class Cylinder:
    def __init__(self, position, direction, r, velocity):
        self.position = position
        self.direction = direction
        self.r = r
        self.velocity = velocity

    def intersect(self, ray):
        #функция, определяющая по геометрии лучей (начало и направление) и 
        #по геометрии цилиндра (радиус, направление, точка, лежащая на оси цилиндра)
        #пересекает ли луч цилиндр. Возвращает расстояние до ближайшего пересечения, если есть, иначе возвращает -1
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
        #ищет матрицу преобразования Лоренца, взято из http://home.thep.lu.se/~malin/LectureNotesFYTA12_2016/SR6.pdf
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


(w, h) = (400, 300) #задаются ширина и высота снимка (в пикселях)

x = np.tile(np.linspace(-1, 1, w), h)
y = np.repeat(np.linspace(h/w, -h/w, h), w)
focal_length = np.ones(h * w)
ct = -np.sqrt(x ** 2 + y ** 2 + focal_length ** 2)

rays = np.stack((ct, x, y, focal_length), axis=1) #создает список лучей, заданных 4 координатами (ct, x, y, z)

O = Cylinder(np.asarray([3.,2.,10.]), np.asarray([0.,1.,0.]), 0.2, 0.6) #содаем наш цилиндр

boost_matrix = O.boost_matrix()#ищем матрицу преобразования Лоренца для перехода в систему отсчета цилиндра

temp_rays = np.apply_along_axis(boost_matrix.dot, 1, rays) - boost_matrix.dot(np.array([0, 0, 0, 0]))#считаем координаты лучей, перейдя в систему отсчета цилиндра, используя np.apply_along_axis

image = Image.fromarray(255*np.clip(O.intersect(temp_rays), 0, 1).reshape((h,w)).astype(np.uint8), mode="L")#используя O.intersect(temp_rays), ищем пересечения лучей с цилиндром
image.save("final_image.png")
