'''--- turtleGL v1.0.3 ---'''
'''---     by Hyan     ---'''
import numpy as np
import turtle
import math,random
class camera():
    def __init__(self):
        self.camera_position = [0, 0, 0]
        self.camera_direction = [0, 0, 1]
        self.camera_rotation = 0
        self.camera_focal = 1
        self.ray = [0,0,-1]
        self.rend = 0
        self.shade_value = 128
        self.pensize = 1
        self.pencolor = '#000000'
        self.type = 1
        turtle.penup()
        turtle.tracer(0)
        turtle.hideturtle()

    def status(self):
        print('=================================')
        print(f'camera position : {self.camera_position}')
        print(f'camera direction : {self.camera_direction}')
        print(f'camera rotation : {self.camera_rotation}')
        print(f'camera focal : {self.camera_focal}')
        print(f'ray direction : {self.ray}')
        print(f'using rend : {'material preview' if self.rend == 0 else 'shade' if self.rend == 1 else 'normal vector preview' if self.rend == 2 else self.rend}')
        print(f'shade value : {self.shade_value}')
        print(f'type : {'focal' if self.type == 1 else 'cabin'}')
        print('=================================')

    def tracer(self, t):
        turtle.tracer(t)

    def to_target(self,t):
        self.camera_direction = [t[0]-self.camera_position[0],t[1]-self.camera_position[1],t[2]-self.camera_position[2]]

    def pointfocal(self, point_3d):
        position = [self.camera_position[0],self.camera_position[2],self.camera_position[1]]
        direction = [self.camera_direction[0],self.camera_direction[2],self.camera_direction[1]]
        point = [point_3d[0],point_3d[2],point_3d[1]]
        cam_pos = np.array(position)
        cam_dir = np.array(direction)
        point = np.array(point)
        cam_dir = cam_dir / np.linalg.norm(cam_dir)
        z_axis = cam_dir
        z_axis = z_axis / np.linalg.norm(z_axis)
        up = np.array([0, 1, 0])
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        view_matrix = np.eye(4)
        view_matrix[0, :3] = x_axis
        view_matrix[1, :3] = y_axis
        view_matrix[2, :3] = z_axis
        view_matrix[0, 3] = -np.dot(x_axis, cam_pos)
        view_matrix[1, 3] = -np.dot(y_axis, cam_pos)
        view_matrix[2, 3] = -np.dot(z_axis, cam_pos)
        point_homo = np.append(point, 1)
        point_cam = view_matrix @ point_homo
        if point_cam[2] <= 0:
            return [0,0]
        u = (self.camera_focal * point_cam[0]) / point_cam[2]
        v = (self.camera_focal * point_cam[1]) / point_cam[2]
        x = u * math.cos(self.camera_rotation) - v * math.sin(self.camera_rotation)
        y = u * math.sin(self.camera_rotation) + v * math.cos(self.camera_rotation)
        return [x,y]

    def pointcabinet(self, point_3d):
        return [point_3d[0]+0.5*point_3d[1]*math.cos(45), point_3d[2]+0.5*point_3d[1]*math.sin(45)]
    
    def draw_axis(self,l):
        if self.type == 1:
            turtle.pensize = self.pensize
            turtle.color('#ff0000')
            turtle.goto(self.pointfocal([0,0,0]))
            turtle.pendown()
            turtle.goto(self.pointfocal([l,0,0]))
            turtle.penup()
            turtle.color('#00ff00')
            turtle.goto(self.pointfocal([0,0,0]))
            turtle.pendown()
            turtle.goto(self.pointfocal([0,l,0]))
            turtle.penup()
            turtle.color('#0000ff')
            turtle.goto(self.pointfocal([0,0,0]))
            turtle.pendown()
            turtle.goto(self.pointfocal([0,0,l]))
            turtle.penup()
        else:
            pass
    
    def done(self):
        turtle.hideturtle()
        turtle.done()

    def drawline(self,l):#l=[[[x,x],[x,x],'#xxxxxx']
        turtle.pensize = self.pensize
        turtle.color(l[1])
        if self.type == 1:
            turtle.goto(self.pointfocal(l[0][0]))
            turtle.pendown()
            turtle.goto(self.pointfocal(l[0][1]))
            turtle.penup()
        else:
            turtle.goto(self.pointcabinet(l[0][0]))
            turtle.pendown()
            turtle.goto(self.pointcabinet(l[0][1]))
            turtle.penup()

    def drawface(self,f):
        turtle.pensize = self.pensize
        turtle.color(f[1])
        if self.type == 1:
            if self.rend == 1:
                if self.normalvect(self.ray,f[0][0],f[0][1],f[0][2]):
                    turtle.color(self.multiply(f[1]))
            elif self.rend == 2:
                if self.normalvect(self.camera_direction,f[0][0],f[0][1],f[0][2]):
                    turtle.color('#FF0000')
                else:
                    turtle.color('#0000FF')
            turtle.goto(self.pointfocal(f[0][0]))
            self.pointfocal(f[0][0])#???
            turtle.begin_fill()
            for i in range(len(f[0])):
                turtle.goto(self.pointfocal(f[0][i]))
            turtle.end_fill()
            turtle.penup()
        else:
            turtle.goto(self.pointcabinet(f[0][0]))
            self.pointcabinet(f[0][0])
            turtle.begin_fill()
            for i in range(len(f[0])):
                turtle.goto(self.pointcabinet(f[0][i]))
            turtle.end_fill()
            turtle.penup()

    def draw_from_scene(self,sce):
        for i in sce:
            if len(i[0]) == 2:
                self.drawline(i)
            else:
                self.drawface(i)

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
        
    def multiply(self,color):
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        def rgb_to_hex(rgb):
            return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
        r, g, b = hex_to_rgb(color)
        new_r = (r * self.shade_value) // 255
        new_g = (g * self.shade_value) // 255
        new_b = (b * self.shade_value) // 255
        return rgb_to_hex((new_r, new_g, new_b))
    
    def delay(self,time):
        turtle.delay(time)
    
    def clear(self):
        turtle.clear()

    def bgcolor(self,color):
        turtle.bgcolor(color)

    def update(self):
        turtle.update()

class scene():
    def __init__(self):
        self.line = []
        self.face = []
    
    def addline(self,l):#[[x,x,x],[x,x,x],'#xxxxxx']
        self.line.append([[l[0],l[1]],l[2]])

    def addface(self,f):#[[x,x,x],[x,x,x],[x,x,x],'#xxxxxx']
        t_face = []
        for i in range(len(f)-1):
            t_face.append(f[i])
        self.face.append(t_face,f[-1])

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

        
if __name__ == '__main__':
    pass