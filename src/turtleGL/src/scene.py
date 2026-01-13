import numpy as np
import random,cv2,os
from PIL import ImageGrab
class scene():
    def __init__(self):
        self.line = []
        self.face = []
        self.center = [0,0,0]
    
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
        self.center = self.rotate_point(rotate_vector,center,self.center)

    def move_point(self,move_vector,point):
        return [point[0]+move_vector[0],point[1]+move_vector[1],point[2]+move_vector[2]]

    def move(self,move_vector):
        for i in self.face:
            for j in range(len(i[0])):
                i[0][j] = self.move_point(move_vector,i[0][j])
        for i in self.line:
            for j in range(len(i[0])):
                i[0][j] = self.move_point(move_vector,i[0][j])
        self.center = self.move_point(move_vector,self.center)

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

    def addline(self,l):#[[x,x,x],[x,x,x],'#xxxxxx']
        self.line.append([[l[0],l[1]],l[2]])

    def addface(self,f):#[[x,x,x],[x,x,x],[x,x,x],'#xxxxxx']
        t_face = []
        for i in range(len(f)-1):
            t_face.append(f[i])
        self.face.append([t_face,f[-1]])

    def import_line(self,path):
        with open(path,'r',encoding='utf-8') as f:
            line = f.read().split('\n')
        for i in line:
            if i != '':
                tempi = i.split(',')
                self.line.append([[[float(tempi[0]),float(tempi[1]),float(tempi[2])],
                                   [float(tempi[3]),float(tempi[4]),float(tempi[5])]],
                                   tempi[6]])
    
    def import_face(self,path):
        with open(path,'r',encoding='utf-8') as f:
            line = f.read().split('\n')
        for i in line:
            if i != '':
                tempi = i.split(',')
                tempj = []
                for j in range(len(tempi)//3):
                    tempj.append([float(tempi[3*j]),float(tempi[3*j+1]),float(tempi[3*j+2])])
                self.face.append([tempj,tempi[-1]])
    
    def export_line(self,path):#[[[x,x,x],[x,x,x]],'#xxxxxx']
        a = ''
        for i in self.line:
            for j in i[0]:
                for k in j:
                    a += str(k) + ','
            a += str(i[1]) + '\n'
        with open(path,'w',encoding='utf-8') as f:
            f.write(a)

    def export_face(self,path):
        a = ''
        for i in self.face:
            for j in i[0]:
                for k in j:
                    a += str(k) + ','
            a += str(i[1]) + '\n'
        with open(path,'w',encoding='utf-8') as f:
            f.write(a)
    
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

    def triangulation(self):
        temp_face = []
        for i in self.face:
            if len(i[0]) > 3:
                for j in range(len(i[0])-2):
                    temp_face.append([[i[0][0],i[0][j+1],i[0][j+2]],i[1]])
            else:
                temp_face.append(i)
        self.face = temp_face

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

    def import_obj(self,filename,scale=1,color=''):
        def random_color():
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            return f'#{r:02x}{g:02x}{b:02x}'
        try:
            vertices = []
            faces = []
            with open(filename, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    if parts[0] == 'v':
                        if len(parts) >= 4:
                            try:
                                x, y, z = map(float, parts[1:4])
                                vertices.append((x, y, z))
                            except ValueError:
                                pass
                    elif parts[0] == 'f':
                        face_vertices = []
                        for part in parts[1:]:
                            vertex_info = part.split('/')[0]
                            
                            try:
                                vertex_index = int(vertex_info)
                                if vertex_index > 0:
                                    adjusted_index = vertex_index - 1
                                elif vertex_index < 0:
                                    adjusted_index = len(vertices) + vertex_index
                                else:
                                    continue
                                if 0 <= adjusted_index < len(vertices):
                                    face_vertices.append(adjusted_index)
                                else:
                                    pass
                            except ValueError:
                                pass                    
                        if len(face_vertices) >= 3:
                            faces.append(face_vertices)
                        else:
                            pass
            self.face = []
            for i, face in enumerate(faces, 1):
                face_temp = []
                for j, vertex_idx in enumerate(face, 1):
                    x, y, z = vertices[vertex_idx]
                    face_temp.append([x*scale,z*scale,y*scale])
                if color == '':
                    self.face.append([face_temp,random_color()])
                else:
                    self.face.append([face_temp,color])
        except FileNotFoundError:
            return [], []
        except Exception as e:
            return [], []
        
    def import_obj_normal(self,filename):
        try:
            vertices = []
            vertex_normals = []
            faces = []
            face_normals = []
            with open(filename, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    if parts[0] == 'v':
                        if len(parts) >= 4:
                            try:
                                x, y, z = map(float, parts[1:4])
                                vertices.append((x, y, z))
                            except ValueError:
                                pass
                    elif parts[0] == 'vn':
                        if len(parts) >= 4:
                            try:
                                x, y, z = map(float, parts[1:4])
                                vertex_normals.append((x, y, z))
                            except ValueError:
                                pass
                    elif parts[0] == 'f':
                        face_vertices = []
                        face_normal_indices = []
                        has_vertex_normals = False
                        
                        for part in parts[1:]:
                            vertex_parts = part.split('/')
                            try:
                                vertex_index = int(vertex_parts[0])
                                if vertex_index > 0:
                                    adjusted_index = vertex_index - 1
                                else:
                                    adjusted_index = len(vertices) + vertex_index
                                
                                if 0 <= adjusted_index < len(vertices):
                                    face_vertices.append(adjusted_index)
                                else:
                                    continue
                            except ValueError:
                                continue
                            if len(vertex_parts) >= 3 and vertex_parts[2]:
                                try:
                                    normal_index = int(vertex_parts[2])
                                    if normal_index > 0:
                                        adjusted_normal_index = normal_index - 1
                                    else:
                                        adjusted_normal_index = len(vertex_normals) + normal_index
                                    
                                    if 0 <= adjusted_normal_index < len(vertex_normals):
                                        face_normal_indices.append(adjusted_normal_index)
                                        has_vertex_normals = True
                                    else:
                                        pass
                                except ValueError:
                                    pass
                        if len(face_vertices) >= 3:
                            faces.append(face_vertices)
                            if has_vertex_normals and len(face_normal_indices) == len(face_vertices):
                                avg_normal = np.array([0.0, 0.0, 0.0])
                                for normal_idx in face_normal_indices:
                                    avg_normal += np.array(vertex_normals[normal_idx])
                                length = np.linalg.norm(avg_normal)
                                if length > 0:
                                    avg_normal = avg_normal / length
                                face_normals.append(tuple(avg_normal))
                            else:
                                pass
            norm = []
            for i, (face, normal) in enumerate(zip(faces, face_normals), 1):
                nx, ny, nz = normal
                norm.append([nx,nz,ny])
            return norm
        except FileNotFoundError:
            return [], [], []
        except Exception as e:
            return [], [], []
    
    def check_obj_norm(self,path):
        norm = self.import_obj_normal(path)
        for i in range(len(self.face)):
            if not self.normalvect(norm[i],self.face[i][0][0],self.face[i][0][1],self.face[i][0][2]):
                self.face[i][0] = self.face[i][0][::-1]

    def add_obj(self,filepath,scale=1,color=''):
        temp = self.face
        self.import_obj(filepath,scale,color)
        self.check_obj_norm(filepath)
        for i in temp:
            self.face.append(i)

    def generate_line(self,color='#000000'):
        self.line = []
        line_temp = []
        for i in self.face:
            for j in range(len(i[0])):
                line_temp.append([[i[0][j%len(i[0])],i[0][(j+1)%len(i[0])]],color])
        for i in line_temp:
            if i not in self.line and [[i[0][1],i[0][0]],i[1]] not in self.line:
                self.line.append(i)