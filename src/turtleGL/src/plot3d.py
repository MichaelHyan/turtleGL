import numpy as np
class plot3d():
    def __init__(self):
        self.xlim = [-10,10]
        self.ylim = [-10,10]
        self.step = 1
        self.line = []
        self.face = []
        self.center = [0,0,0]
    
    def generate_face(self,func,c=True):
        x = self.xlim[0]
        y = self.ylim[0]
        if self.xlim[0] < self.xlim[1]:
            xstep = self.step
        else:
            xstep = -self.step
        if self.ylim[0] < self.ylim[1]:
            ystep = self.step
        else:
            ystep = -self.step
        while x < self.xlim[1]:
            while y < self.ylim[1]:
                self.face.append(
                    [[[x,y,func(x,y)],
                      [x+xstep,y,func(x+xstep,y)],
                      [x+xstep,y+ystep,func(x+xstep,y+ystep)],
                      [x,y+ystep,func(x,y+ystep)]],'']
                )
                y += ystep
            y = self.ylim[0]
            x += xstep
        def hex(c):
            c = int(c)
            return "#{:02X}{:02X}{:02X}".format(c,c,c)
        def avg(a,b,c,d):
            return (a+b+c+d)/3
        def liner(zlim,x):
            return (x-zlim[0])/(zlim[1]-zlim[0])
        i = self.face[0]
        zlim = [avg(i[0][0][2],i[0][1][2],i[0][2][2],i[0][3][2]),
                avg(i[0][0][2],i[0][1][2],i[0][2][2],i[0][3][2])]
        for i in self.face:
            a = avg(i[0][0][2],i[0][1][2],i[0][2][2],i[0][3][2])
            if a > zlim[1]:
                zlim[1] = a
            if a < zlim[0]:
                zlim[0] = a
        if c:
            for i in self.face:
                i[1] = hex(255*liner(
                    zlim,
                    avg(i[0][0][2],i[0][1][2],i[0][2][2],i[0][3][2])
                ))
        else:
            for i in self.face:
                i[i] = '#000000'

    def generate_line(self,func,color='#000000'):
        x = self.xlim[0]
        y = self.ylim[0]
        if self.xlim[0] < self.xlim[1]:
            xstep = self.step
        else:
            xstep = -self.step
        if self.ylim[0] < self.ylim[1]:
            ystep = self.step
        else:
            ystep = -self.step
        while x < self.xlim[1]:
            while y < self.ylim[1]:
                self.line.append([
                    [[x,y+ystep,func(x,y+ystep)],
                     [x+xstep,y+ystep,func(x+xstep,y+ystep)]],
                    color
                ])
                self.line.append([
                    [[x+xstep,y,func(x+xstep,y)],
                     [x+xstep,y+ystep,func(x+xstep,y+ystep)]],
                     color
                ])
                y += ystep
            y = self.ylim[0]
            x += xstep
        x = self.xlim[0]
        y = self.ylim[0]
        while x < self.xlim[1]:
            self.line.append([
                [[x,y,func(x,y)],
                 [x+xstep,y,func(x+xstep,y)]],
                 color
            ])
            x += xstep
        x = self.xlim[0]
        while y < self.xlim[1]:
            self.line.append([
                [[x,y,func(x,y)],
                 [x,y+ystep,func(x,y+ystep)]],
                 color
            ])
            y += ystep

    def get_center(self):
        c = [0,0,0]
        count = 0
        for i in self.face:
            for j in i[0]:
                c[0] += j[0]
                c[1] += j[1]
                c[2] += j[2]
                count += 1
        self.center = [x/count for x in c]
        return self.center
    
    def rotate_point(self,rotation_vector,center,point):
        rotation_vector = np.array([rotation_vector[0],
                                    rotation_vector[2],
                                    rotation_vector[1]], dtype=float)
        center = np.array([center[0],
                            center[2],
                            center[1]], dtype=float)
        point = np.array([point[0],
                            point[2],
                            point[1]], dtype=float)
        theta = np.linalg.norm(rotation_vector)
        if theta < 1e-10:
            return point
        axis = rotation_vector / theta
        translated_point = point - center
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        I = np.eye(3)
        R_matrix = I + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
        rotated_translated_point = np.dot(R_matrix, translated_point)        
        rotated_point = rotated_translated_point + center
        return [rotated_point[0],rotated_point[2],rotated_point[1]]

    def rotate(self,rotate_vector,center=[0,0,0]):
        for i in self.face:
            for j in range(len(i[0])):
                i[0][j] = self.rotate_point(rotate_vector,center,i[0][j])
        for i in self.line:
            for j in range(len(i[0])):
                i[0][j] = self.rotate_point(rotate_vector,center,i[0][j])

    def move_point(self,move_vector,point):
        return [point[0]+move_vector[0],point[1]+move_vector[1],point[2]+move_vector[2]]

    def move(self,move_vector):
        for i in self.face:
            for j in range(len(i[0])):
                i[0][j] = self.move_point(move_vector,i[0][j])
        for i in self.line:
            for j in range(len(i[0])):
                i[0][j] = self.move_point(move_vector,i[0][j])

    def scale_point(self,scale_vector,center,point):
        return [scale_vector[0]*(point[0]-center[0])+center[0],
                scale_vector[1]*(point[1]-center[1])+center[1],
                scale_vector[2]*(point[2]-center[2])+center[2]]

    def scale(self,scale_vector,center=[0,0,0]):
        for i in self.face:
            for j in range(len(i[0])):
                i[0][j] = self.scale_point(scale_vector,center,i[0][j])
        for i in self.line:
            for j in range(len(i[0])):
                i[0][j] = self.scale_point(scale_vector,center,i[0][j])

    def sort_line_min(self,camera_pos):#[[[x,x,x],[x,x,x]],'#xxxxxx']
        diatance = []
        for i in self.line:
            dis = (i[0][0][0]-camera_pos[0])**2+(i[0][0][1]-camera_pos[1])**2+(i[0][0][2]-camera_pos[2])**2
            for j in i[0]:
                t_dis = (j[0]-camera_pos[0])**2+(j[0]-camera_pos[1])**2+(j[0]-camera_pos[2])**2
                if t_dis < dis:
                    dis = t_dis
            diatance.append(dis)
        self.line = [x for _, x in sorted(zip(diatance, self.line), reverse=True)]
        return self.line

    def sort_face_min(self,camera_pos):#[[[x,x,x],[x,x,x]],'#xxxxxx']
        diatance = []
        dis = (self.face[0][0][0][0]-camera_pos[0])**2+(self.face[0][0][0][1]-camera_pos[1])**2+(self.face[0][0][0][2]-camera_pos[2])**2
        for i in self.face:
            for j in i[0]:
                t_dis = (j[0]-camera_pos[0])**2+(j[0]-camera_pos[1])**2+(j[0]-camera_pos[2])**2
                if t_dis < dis:
                    dis = t_dis
            diatance.append(dis)
        self.face = [x for _, x in sorted(zip(diatance, self.face), reverse=True)]
        return self.face
    
    def sort_line_avg(self,camera_pos):#[[[x,x,x],[x,x,x]],'#xxxxxx']
        def avg(coordinates):
            if not coordinates:
                return [0, 0, 0]
            sum_x = sum_y = sum_z = 0
            count = len(coordinates)
            for coord in coordinates:
                sum_x += coord[0]
                sum_y += coord[1]
                sum_z += coord[2]
            avg_x = sum_x / count
            avg_y = sum_y / count
            avg_z = sum_z / count
            return [avg_x, avg_y, avg_z]
        diatance = []
        for i in self.line:
            dis = (avg(i[0])[0]-camera_pos[0])**2+(avg(i[0])[1]-camera_pos[1])**2+(avg(i[0])[2]-camera_pos[2])**2
            diatance.append(dis)
        self.line = [x for _, x in sorted(zip(diatance, self.line), reverse=True)]
        return self.line

    def sort_face_avg(self,camera_pos):#[[[x,x,x],[x,x,x]],'#xxxxxx']
        def avg(coordinates):
            if not coordinates:
                return [0, 0, 0]
            sum_x = sum_y = sum_z = 0
            count = len(coordinates)
            for coord in coordinates:
                sum_x += coord[0]
                sum_y += coord[1]
                sum_z += coord[2]
            avg_x = sum_x / count
            avg_y = sum_y / count
            avg_z = sum_z / count
            return [avg_x, avg_y, avg_z]
        diatance = []
        for i in self.face:
            dis = (avg(i[0])[0]-camera_pos[0])**2+(avg(i[0])[1]-camera_pos[1])**2+(avg(i[0])[2]-camera_pos[2])**2
            diatance.append(dis)
        self.face = [x for _, x in sorted(zip(diatance, self.face), reverse=True)]
        return self.face
    
    def sort_line_cabin(self):#[[[x,x,x],[x,x,x]],'#xxxxxx']
        camera_pos = [999999,-999999,999999]
        def avg(coordinates):
            if not coordinates:
                return [0, 0, 0]
            sum_x = sum_y = sum_z = 0
            count = len(coordinates)
            for coord in coordinates:
                sum_x += coord[0]
                sum_y += coord[1]
                sum_z += coord[2]
            avg_x = sum_x / count
            avg_y = sum_y / count
            avg_z = sum_z / count
            return [avg_x, avg_y, avg_z]
        diatance = []
        for i in self.line:
            dis = (avg(i[0])[0]-camera_pos[0])**2+(avg(i[0])[1]-camera_pos[1])**2+(avg(i[0])[2]-camera_pos[2])**2
            diatance.append(dis)
        self.line = [x for _, x in sorted(zip(diatance, self.line), reverse=True)]
        return self.line

    def sort_face_cabin(self):#[[[x,x,x],[x,x,x]],'#xxxxxx']
        camera_pos = [999999,-999999,999999]
        def avg(coordinates):
            if not coordinates:
                return [0, 0, 0]
            sum_x = sum_y = sum_z = 0
            count = len(coordinates)
            for coord in coordinates:
                sum_x += coord[0]
                sum_y += coord[1]
                sum_z += coord[2]
            avg_x = sum_x / count
            avg_y = sum_y / count
            avg_z = sum_z / count
            return [avg_x, avg_y, avg_z]
        diatance = []
        for i in self.face:
            dis = (avg(i[0])[0]-camera_pos[0])**2+(avg(i[0])[1]-camera_pos[1])**2+(avg(i[0])[2]-camera_pos[2])**2
            diatance.append(dis)
        self.face = [x for _, x in sorted(zip(diatance, self.face), reverse=True)]
        return self.face
    
    def sort_line_isometric(self):#[[[x,x,x],[x,x,x]],'#xxxxxx']
        camera_pos = [-999999,-999999,999999]
        def avg(coordinates):
            if not coordinates:
                return [0, 0, 0]
            sum_x = sum_y = sum_z = 0
            count = len(coordinates)
            for coord in coordinates:
                sum_x += coord[0]
                sum_y += coord[1]
                sum_z += coord[2]
            avg_x = sum_x / count
            avg_y = sum_y / count
            avg_z = sum_z / count
            return [avg_x, avg_y, avg_z]
        diatance = []
        for i in self.line:
            dis = (avg(i[0])[0]-camera_pos[0])**2+(avg(i[0])[1]-camera_pos[1])**2+(avg(i[0])[2]-camera_pos[2])**2
            diatance.append(dis)
        self.line = [x for _, x in sorted(zip(diatance, self.line), reverse=True)]
        return self.line

    def triangulation(self):
        temp_face = []
        for i in self.face:
            if len(i[0]) > 3:
                for j in range(len(i[0])-2):
                    temp_face.append([[i[0][0],i[0][j+1],i[0][j+2]],i[1]])
            else:
                temp_face.append(i)
        self.face = temp_face

    def sort_face_isometric(self):#[[[x,x,x],[x,x,x]],'#xxxxxx']
        camera_pos = [-999999,-999999,999999]
        def avg(coordinates):
            if not coordinates:
                return [0, 0, 0]
            sum_x = sum_y = sum_z = 0
            count = len(coordinates)
            for coord in coordinates:
                sum_x += coord[0]
                sum_y += coord[1]
                sum_z += coord[2]
            avg_x = sum_x / count
            avg_y = sum_y / count
            avg_z = sum_z / count
            return [avg_x, avg_y, avg_z]
        diatance = []
        for i in self.face:
            dis = (avg(i[0])[0]-camera_pos[0])**2+(avg(i[0])[1]-camera_pos[1])**2+(avg(i[0])[2]-camera_pos[2])**2
            diatance.append(dis)
        self.face = [x for _, x in sorted(zip(diatance, self.face), reverse=True)]
        return self.face
    
    def sort_all_avg(self,camera_pos):#[[[x,x,x],[x,x,x]],'#xxxxxx']
        def avg(coordinates):
            if not coordinates:
                return [0, 0, 0]
            sum_x = sum_y = sum_z = 0
            count = len(coordinates)
            for coord in coordinates:
                sum_x += coord[0]
                sum_y += coord[1]
                sum_z += coord[2]
            avg_x = sum_x / count
            avg_y = sum_y / count
            avg_z = sum_z / count
            return [avg_x, avg_y, avg_z]
        fl = []
        for i in self.face:
            fl.append(i)
        for i in self.line:
            fl.append(i)
        diatance = []
        for i in fl:
            dis = (avg(i[0])[0]-camera_pos[0])**2+(avg(i[0])[1]-camera_pos[1])**2+(avg(i[0])[2]-camera_pos[2])**2
            diatance.append(dis)
        fl = [x for _, x in sorted(zip(diatance, fl), reverse=True)]
        return fl
    def sort_all_cabin(self):
        camera_pos = [999999,-999999,999999]
        def avg(coordinates):
            if not coordinates:
                return [0, 0, 0]
            sum_x = sum_y = sum_z = 0
            count = len(coordinates)
            for coord in coordinates:
                sum_x += coord[0]
                sum_y += coord[1]
                sum_z += coord[2]
            avg_x = sum_x / count
            avg_y = sum_y / count
            avg_z = sum_z / count
            return [avg_x, avg_y, avg_z]
        fl = []
        for i in self.face:
            fl.append(i)
        for i in self.line:
            fl.append(i)
        diatance = []
        for i in fl:
            dis = (avg(i[0])[0]-camera_pos[0])**2+(avg(i[0])[1]-camera_pos[1])**2+(avg(i[0])[2]-camera_pos[2])**2
            diatance.append(dis)
        fl = [x for _, x in sorted(zip(diatance, fl), reverse=True)]
        return fl
    
    def sort_all_isometric(self):
        camera_pos = [-999999,-999999,999999]
        def avg(coordinates):
            if not coordinates:
                return [0, 0, 0]
            sum_x = sum_y = sum_z = 0
            count = len(coordinates)
            for coord in coordinates:
                sum_x += coord[0]
                sum_y += coord[1]
                sum_z += coord[2]
            avg_x = sum_x / count
            avg_y = sum_y / count
            avg_z = sum_z / count
            return [avg_x, avg_y, avg_z]
        fl = []
        for i in self.face:
            fl.append(i)
        for i in self.line:
            fl.append(i)
        diatance = []
        for i in fl:
            dis = (avg(i[0])[0]-camera_pos[0])**2+(avg(i[0])[1]-camera_pos[1])**2+(avg(i[0])[2]-camera_pos[2])**2
            diatance.append(dis)
        fl = [x for _, x in sorted(zip(diatance, fl), reverse=True)]
        return fl

    def reverse_normvect(self,i):
        self.face[i][0] = self.face[i][0][::-1]

    def normalvect(self,vector,point1,point2,point3):
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
        if dot_product > 0:
            return True
        else:
            return False

def warning():
    str='''
============================================================
|                   代码运行中，请勿触碰。                   |
|                                                          |
|              Code is running, do not touch.              |
|                                                          |
|         Code en cours d'exécution, ne pas toucher.       |
|                                                          |
|              Código en ejecución, no tocar.              |
|                                                          |
|            Код выполняется, не прикасайтесь.             |
|                                                          |
|               الكود قيد التشغيل، لا تلمس.               |
============================================================
'''
    print(str)


if __name__ == '__main__':
    pass