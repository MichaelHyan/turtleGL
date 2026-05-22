import math
class ray():
    def __init__(self):
        self.sunlight = []
        self.pointlight = []
        self.ray = []

    def add_sunlight(self,vec,brightness=1):
        self.sunlight.append([vec,brightness])

    def add_pointlight(self,pos,brightness):
        self.pointlight.append([pos,brightness])

    def add_ray(self,point,vec,brightness):
        self.ray.append([point,vec,brightness])

    def shade_sunlight(self,vector,brightness,point1,point2,point3):
        v = []
        for i in vector:
            v.append(i*-1)
        cos = self.normalvect(v,point1,point2,point3)
        return cos*brightness

    def shade_pointlight(self,pointlight,brightness,point1,point2,point3):
        point = [(point1[0]+point2[0]+point3[0])/3,
                 (point1[1]+point2[1]+point3[1])/3,
                 (point1[2]+point2[2]+point3[2])/3]
        vector = [pointlight[0]-point[0],
                  pointlight[1]-point[1],
                  pointlight[2]-point[2]]
        factor = brightness/math.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)
        cos = self.normalvect(vector,point1,point2,point3)
        return cos*factor

    def shade_ray(self,raypoint,raylight,brightness,point1,point2,point3):
        vector = [raylight[0]-raypoint[0],
                  raylight[1]-raypoint[1],
                  raylight[2]-raypoint[2]]
        cos = self.normalvect(vector,point1,point2,point3)
        return cos*brightness

    def get_value(self,point1,point2,point3):
        value = -1
        for i in self.sunlight:
            v= self.shade_sunlight(i[0],i[1],point1,point2,point3)
            value += v if v > 0 else 0
        for i in self.pointlight:
            v = self.shade_pointlight(i[0],i[1],point1,point2,point3)
            value += v if v > 0 else 0
        for i in self.ray:
            v= self.shade_ray(i[1],i[0],i[2],point1,point2,point3)
            value += v if v > 0 else 0
        if value > 1:
            value = 1
        if value < -1:
            value = -1
        return value

    def normalvect(self, vector, point1, point2, point3):
        vector1 = (
            point2[0] - point1[0],
            point2[1] - point1[1], 
            point2[2] - point1[2]
        )
        vector2 = (
            point3[0] - point2[0],
            point3[1] - point2[1],
            point3[2] - point2[2]
        )
        cross_product = (
            vector1[1] * vector2[2] - vector1[2] * vector2[1],
            vector1[2] * vector2[0] - vector1[0] * vector2[2],
            vector1[0] * vector2[1] - vector1[1] * vector2[0]
        )
        dot_product = (
            cross_product[0] * vector[0] +
            cross_product[1] * vector[1] +
            cross_product[2] * vector[2]
        )
        
        length_cross = math.sqrt(
            cross_product[0] ** 2 + 
            cross_product[1] ** 2 + 
            cross_product[2] ** 2
        )
        length_vector = math.sqrt(
            vector[0] ** 2 + 
            vector[1] ** 2 + 
            vector[2] ** 2
        )
        if length_cross == 0 or length_vector == 0:
            return 0.0
        cosine_value = dot_product / (length_cross * length_vector)
        return cosine_value
    