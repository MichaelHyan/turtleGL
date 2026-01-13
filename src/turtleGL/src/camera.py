import numpy as np
import turtle,math,cv2,os
class camera():
    def __init__(self,title = 'turtleGL v1.2.2'):
        self.title = title
        self.camera_position = [0, 0, 0]
        self.camera_direction = [0, 0, 1]
        self.camera_rotation = 0
        self.camera_focal = 1
        self.point_behind_cam_type = 0
        self.ray = [0,0,-1]
        self.rend = 0
        self.shade_value = 128
        self.pensize = 1
        self.pencolor = '#000000'
        self.type = 1
        self.grating_size = [500,400]
        self.image_size = [500,400]
        self.image = []
        turtle.title = title
        turtle.penup()
        turtle.tracer(0)
        turtle.hideturtle()

    def write(self,point,str,move=False,align='left',font=("Arial", 12, "bold")):
        if self.type == 0:
            turtle.goto(self.pointcabinet(point))
        elif self.type == 1:
            turtle.goto(self.pointfocal(point))
        elif self.type == 2:
            turtle.goto(self.pointisometric(point))
        turtle.write(str,move=move,align=align,font=font)

    def create_image(self,bgcolor='#ffffff'):
        color = bgcolor.lstrip('#')
        color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        self.image = np.full((self.image_size[1],self.image_size[0],3),color,dtype=np.uint8)

    def setposition(self,a):
        self.camera_position = a
    
    def setdirection(self,a):
        self.camera_direction = a

    def setfocal(self,a):
        self.camera_focal = a
    
    def settype(self,a):
        a = str(a)
        if a == '1' or a == 'focal':
            self.type = 1
        elif a == '0' or a == 'cabin':
            self.type = 0
        elif a == '2' or a == 'isometric':
            self.type = 2
        else:
            print(f'unknow type: {a}')

    def status(self):
        print('=================================')
        print(f'camera position : {self.camera_position}')
        print(f'camera direction : {self.camera_direction}')
        print(f'camera rotation : {self.camera_rotation}')
        print(f'camera focal : {self.camera_focal}')
        print(f'point behind cam(focal mode): {'do nothing' if self.point_behind_cam_type == 0 else 'reverse uv' if self.point_behind_cam_type == 1 else 'switch to orthografic mode' if self.point_behind_cam_type == 2 else self.point_behind_cam_type}')
        print(f'ray direction : {self.ray}')
        print(f'using rend : {'material preview' if self.rend == 0 else 'shade' if self.rend == 1 else 'normal vector preview' if self.rend == 2 else self.rend}')
        print(f'shade value : {self.shade_value}')
        print(f'type : {'focal' if self.type == 1 else 'cabin' if self.type == 0 else 'isometric' if self.type == 2 else 'orthografic' if self.type == -1 else self.type}')
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
        if point_cam[2] > 0:
            u = (self.camera_focal * point_cam[0]) / point_cam[2]
            v = (self.camera_focal * point_cam[1]) / point_cam[2]
        else:
            if self.point_behind_cam_type == 0:#直接计算
                u = (self.camera_focal * point_cam[0]) / point_cam[2]
                v = (self.camera_focal * point_cam[1]) / point_cam[2]
            elif self.point_behind_cam_type == 1:#uv取反
                u = (-1 * self.camera_focal * point_cam[0]) / point_cam[2]
                v = (-1 * self.camera_focal * point_cam[1]) / point_cam[2]
            elif self.point_behind_cam_type == 2:#改为倍数正交透视
                u = self.camera_focal * point_cam[0]
                v = self.camera_focal * point_cam[1]
        x = u * math.cos(self.camera_rotation) - v * math.sin(self.camera_rotation)
        y = u * math.sin(self.camera_rotation) + v * math.cos(self.camera_rotation)
        return [x,y]
        
    def pointfocal_inverse(self, point_2d):
        x, y = point_2d
        u = x * math.cos(-self.camera_rotation) - y * math.sin(-self.camera_rotation)
        v = x * math.sin(-self.camera_rotation) + y * math.cos(-self.camera_rotation)
        position = [self.camera_position[0], self.camera_position[2], self.camera_position[1]]
        direction = [self.camera_direction[0], self.camera_direction[2], self.camera_direction[1]]
        cam_pos = np.array(position)
        cam_dir = np.array(direction)
        cam_dir = cam_dir / np.linalg.norm(cam_dir)
        z_axis = cam_dir
        up = np.array([0, 1, 0])
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        view_matrix_inv = np.eye(4)
        view_matrix_inv[:3, 0] = x_axis
        view_matrix_inv[:3, 1] = y_axis
        view_matrix_inv[:3, 2] = z_axis
        view_matrix_inv[:3, 3] = cam_pos
        #point_cam = np.array([u * self.camera_focal, v * self.camera_focal, self.camera_focal, 1])
        point_cam = np.array([u, v, self.camera_focal, 1])
        point_world = view_matrix_inv @ point_cam
        return [point_world[0], point_world[2], point_world[1]]

    def pointorthografic(self, point_3d):
        scale = self.camera_focal
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
        u = scale * point_cam[0]
        v = scale * point_cam[1]
        x = u * math.cos(self.camera_rotation) - v * math.sin(self.camera_rotation)
        y = u * math.sin(self.camera_rotation) + v * math.cos(self.camera_rotation)
        return [x,y]
    
    def pointcabinet(self, point_3d):
        return [point_3d[0]+0.5*point_3d[1]*math.cos(45), point_3d[2]+0.5*point_3d[1]*math.sin(45)]

    def pointisometric(self,point_3d):
        v3 = 1.73205080756888
        return [point_3d[0]*v3-point_3d[1]*v3,point_3d[0]*0.5+point_3d[1]*0.5+point_3d[2]*v3]

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
        if self.type == -1:
            turtle.goto(self.pointorthografic(l[0][0]))
            turtle.pendown()
            turtle.goto(self.pointorthografic(l[0][1]))
            turtle.penup()
        elif self.type == 0:
            turtle.goto(self.pointcabinet(l[0][0]))
            turtle.pendown()
            turtle.goto(self.pointcabinet(l[0][1]))
            turtle.penup()
        elif self.type == 2:
            turtle.goto(self.pointisometric(l[0][0]))
            turtle.pendown()
            turtle.goto(self.pointisometric(l[0][1]))
            turtle.penup()
        else:
            pass

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
        elif self.type == -1:
            if self.rend == 1:
                if self.normalvect(self.ray,f[0][0],f[0][1],f[0][2]):
                    turtle.color(self.multiply(f[1]))
            elif self.rend == 2:
                if self.normalvect(self.camera_direction,f[0][0],f[0][1],f[0][2]):
                    turtle.color('#FF0000')
                else:
                    turtle.color('#0000FF')
            turtle.goto(self.pointorthografic(f[0][0]))
            self.pointorthografic(f[0][0])
            turtle.begin_fill()
            for i in range(len(f[0])):
                turtle.goto(self.pointorthografic(f[0][i]))
            turtle.end_fill()
            turtle.penup()
        elif self.type == 0:
            turtle.goto(self.pointcabinet(f[0][0]))
            self.pointcabinet(f[0][0])
            turtle.begin_fill()
            for i in range(len(f[0])):
                turtle.goto(self.pointcabinet(f[0][i]))
            turtle.end_fill()
            turtle.penup()
        elif self.type == 2:
            turtle.goto(self.pointisometric(f[0][0]))
            self.pointisometric(f[0][0])
            turtle.begin_fill()
            for i in range(len(f[0])):
                turtle.goto(self.pointisometric(f[0][i]))
            turtle.end_fill()
            turtle.penup()
        else:
            pass
    
    def draw_from_scene(self,sce):
        for i in sce:
            if len(i[0]) == 2:
                self.drawline(i)
            else:
                self.drawface(i)

    def hex_to_bgr(self,hex_color):
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return (rgb[2], rgb[1], rgb[0])
    
    def drawline_cv2(self,l):#l=[[[x,x],[x,x],'#xxxxxx']
        turtle.pensize = self.pensize
        turtle.color(l[1])
        if self.type == 1:
            point1 = list(map(int,self.pointfocal(l[0][0])))
            point2 = list(map(int,self.pointfocal(l[0][1])))
            point1[0] = point1[0] + self.image_size[0]//2
            point1[1] = -1*point1[1] + self.image_size[1]//2
            point2[0] = point2[0] + self.image_size[0]//2
            point2[1] = -1*point2[1] + self.image_size[1]//2
        if self.type == -1:
            point1 = list(map(int,self.pointorthografic(l[0][0])))
            point2 = list(map(int,self.pointorthografic(l[0][1])))
            point1[0] = point1[0] + self.image_size[0]//2
            point1[1] = -1*point1[1] + self.image_size[1]//2
            point2[0] = point2[0] + self.image_size[0]//2
            point2[1] = -1*point2[1] + self.image_size[1]//2
        elif self.type == 0:
            point1 = list(map(int,self.pointcabinet(l[0][0])))
            point2 = list(map(int,self.pointcabinet(l[0][1])))
            point1[0] = point1[0] + self.image_size[0]//2
            point1[1] = -1*point1[1] + self.image_size[1]//2
            point2[0] = point2[0] + self.image_size[0]//2
            point2[1] = -1*point2[1] + self.image_size[1]//2
        elif self.type == 2:
            point1 = list(map(int,self.pointisometric(l[0][0])))
            point2 = list(map(int,self.pointisometric(l[0][1])))
            point1[0] = point1[0] + self.image_size[0]//2
            point1[1] = -1*point1[1] + self.image_size[1]//2
            point2[0] = point2[0] + self.image_size[0]//2
            point2[1] = -1*point2[1] + self.image_size[1]//2
        else:
            pass
        cv2.line(self.image,
                     point1,
                     point2,
                     self.hex_to_bgr(l[1]),
                     self.pensize)

    def drawface_cv2(self,f):
        color = f[1]
        if self.type == 1:
            if self.rend == 1:
                if self.normalvect(self.ray,f[0][0],f[0][1],f[0][2]):
                    color = self.multiply(f[1])
            elif self.rend == 2:
                if self.normalvect(self.camera_direction,f[0][0],f[0][1],f[0][2]):
                    color = '#FF0000'
                else:
                    color = '#0000FF'
            m = []
            self.pointfocal(f[0][0])#???
            for i in range(len(f[0])):
                m.append(self.pointfocal(f[0][i]))
            m = np.array(m,np.int32)
            m *= [1,-1]
            m += [self.image_size[0]//2,self.image_size[1]//2]
            cv2.fillPoly(self.image,[m],self.hex_to_bgr(color))
        if self.type == -1:
            if self.rend == 1:
                if self.normalvect(self.ray,f[0][0],f[0][1],f[0][2]):
                    color = self.multiply(f[1])
            elif self.rend == 2:
                if self.normalvect(self.camera_direction,f[0][0],f[0][1],f[0][2]):
                    color = '#FF0000'
                else:
                    color = '#0000FF'
            m = []
            self.pointorthografic(f[0][0])#???
            for i in range(len(f[0])):
                m.append(self.pointorthografic(f[0][i]))
            m = np.array(m,np.int32)
            m *= [1,-1]
            m += [self.image_size[0]//2,self.image_size[1]//2]
            cv2.fillPoly(self.image,[m],self.hex_to_bgr(color))
        elif self.type == 0:
            m = []
            self.pointcabinet(f[0][0])
            for i in range(len(f[0])):
                m.append(self.pointcabinet(f[0][i]))
            m = np.array(m,np.int32)
            m *= [1,-1]
            m += [self.image_size[1]//2,self.image_size[0]//2]
            cv2.fillPoly(self.image,[m],self.hex_to_bgr(color))
        elif self.type == 2:
            m = []
            self.pointisometric(f[0][0])
            for i in range(len(f[0])):
                m.append(self.pointisometric(f[0][i]))
            m = np.array(m,np.int32)
            m *= [1,-1]
            m += [self.image_size[1]//2,self.image_size[0]//2]
            cv2.fillPoly(self.image,[m],self.hex_to_bgr(color))
        else:
            pass

    def draw_from_scene_cv2(self,sce):
        for i in sce:
            if len(i[0]) == 2:
                self.drawline_cv2(i)
            else:
                self.drawface_cv2(i)

    def imshow(self):
        cv2.imshow(self.title,self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def imwrite(self,path):
        cv2.imwrite(path,self.image)

    def capture(self,path,i):
        cv2.imwrite(f'{path}/{i:08d}.png',self.image)

    def to_video(self,path,fps=30):
        images = [img for img in os.listdir(path) if img.endswith(".png")]
        if not images:
            return
        first_image = cv2.imread(os.path.join(path, images[0]))
        height, width, layers = first_image.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(f'{path}.mp4', fourcc, fps, (width, height))
        images.sort()
        for image in images:
            image_path = os.path.join(path, image)
            frame = cv2.imread(image_path)
            video.write(frame)
        video.release()
        cv2.destroyAllWindows()
        print('save complete')

    def ray_triangle_intersect(self,point,ray,triangle_vertices):
        epsilon = 1e-6
        v0 = np.array([triangle_vertices[0][0], triangle_vertices[0][2], triangle_vertices[0][1]])
        v1 = np.array([triangle_vertices[1][0], triangle_vertices[1][2], triangle_vertices[1][1]])
        v2 = np.array([triangle_vertices[2][0], triangle_vertices[2][2], triangle_vertices[2][1]])
        edge1 = v1 - v0
        edge2 = v2 - v0
        ray_direction = np.array([ray[0], ray[2], ray[1]])
        ray_origin = np.array([point[0], point[2], point[1]])
        pvec = np.cross(ray_direction, edge2)
        det = np.dot(edge1, pvec)
        if abs(det) < epsilon:
            return False, None, None, None
        inv_det = 1.0 / det
        tvec = ray_origin - v0
        u = np.dot(tvec, pvec) * inv_det
        if u < 0.0 or u > 1.0:
            return False, None, None, None
        qvec = np.cross(tvec, edge1)
        v = np.dot(ray_direction, qvec) * inv_det
        if v < 0.0 or u + v > 1.0:
            return False, None, None, None
        t = np.dot(edge2, qvec) * inv_det
        if t < epsilon:
            return False, None, None, None
        intersection_point = ray_origin + t * ray_direction
        return True, intersection_point[0], intersection_point[2], intersection_point[1]

    def grating(self,face):
        if self.rend == 0:
            for i in range(-1*self.grating_size[0]//2,self.grating_size[0]//2,1):
                for j in range(-1*self.grating_size[1]//2,self.grating_size[1]//2,1):
                    [x,y,z] = self.pointfocal_inverse([i,j])
                    ray = [x-self.camera_position[0],y-self.camera_position[1],z-self.camera_position[2]]
                    for k in face:
                        a,t,u,v = self.ray_triangle_intersect(self.camera_position,ray,k[0])
                        if a:
                            turtle.goto(i,j)
                            turtle.dot(2,k[1])
                            continue
                        else:
                            pass
        elif self.rend == 1:
            for i in range(-1*self.grating_size[0]//2,self.grating_size[0]//2,1):
                for j in range(-1*self.grating_size[1]//2,self.grating_size[1]//2,1):
                    [x,y,z] = self.pointfocal_inverse([i,j])
                    ray = [x-self.camera_position[0],y-self.camera_position[1],z-self.camera_position[2]]
                    for k in face:
                        a,t,u,v = self.ray_triangle_intersect(self.camera_position,ray,k[0])
                        if a:
                            if not self.normalvect(self.ray,k[0][0],k[0][1],k[0][2]):
                                rect_r = [-x for x in self.ray]
                                mark = 0
                                for m in face:
                                    b,o,p,q = self.ray_triangle_intersect([t,u,v],rect_r,m[0])
                                    if b and (m != k):
                                        mark = 1
                                        continue
                                    else:
                                        pass
                                color = self.multiply(k[1]) if mark == 1 else k[1]
                                turtle.goto(i,j)
                                turtle.dot(2,color)
                            else:
                                turtle.goto(i,j)
                                turtle.dot(2,self.multiply(k[1]))
                                continue
                        else:
                            pass
    
    def grating_cv2(self,face):
        if self.rend == 0:
            for i in range(-1*self.image_size[0]//2,self.image_size[0]//2,1):
                for j in range(-1*self.image_size[1]//2,self.image_size[1]//2,1):
                    [x,y,z] = self.pointfocal_inverse([i,j])
                    ray = [x-self.camera_position[0],y-self.camera_position[1],z-self.camera_position[2]]
                    for k in face:
                        a,t,u,v = self.ray_triangle_intersect(self.camera_position,ray,k[0])
                        if a:
                            self.image[-j+self.image_size[1]//2,
                                       i+self.image_size[0]//2] = self.hex_to_bgr(k[1])
                            continue
                        else:
                            pass
        elif self.rend == 1:
            for i in range(-1*self.image_size[0]//2,self.image_size[0]//2,1):
                for j in range(-1*self.image_size[1]//2,self.image_size[1]//2,1):
                    [x,y,z] = self.pointfocal_inverse([i,j])
                    ray = [x-self.camera_position[0],y-self.camera_position[1],z-self.camera_position[2]]
                    for k in face:
                        a,t,u,v = self.ray_triangle_intersect(self.camera_position,ray,k[0])
                        if a:
                            if not self.normalvect(self.ray,k[0][0],k[0][1],k[0][2]):
                                rect_r = [-x for x in self.ray]
                                mark = 0
                                for m in face:
                                    b,o,p,q = self.ray_triangle_intersect([t,u,v],rect_r,m[0])
                                    if b and (m != k):
                                        mark = 1
                                        continue
                                    else:
                                        pass
                                color = self.multiply(k[1]) if mark == 1 else k[1]
                                self.image[-j+self.image_size[1]//2,i+self.image_size[0]//2] = self.hex_to_bgr(color)
                            else:
                                self.image[-j+self.image_size[1]//2,i+self.image_size[0]//2] = self.hex_to_bgr(k[1])
                                continue
                        else:
                            pass

    def show_grating_limit(self,c='#000000'):
        turtle.pencolor(c)
        turtle.goto(self.grating_size[0]//2,self.grating_size[1]//2)
        turtle.pendown()
        turtle.goto(-1*self.grating_size[0]//2,self.grating_size[1]//2)
        turtle.goto(-1*self.grating_size[0]//2,-1*self.grating_size[1]//2)
        turtle.goto(self.grating_size[0]//2,-1*self.grating_size[1]//2)
        turtle.goto(self.grating_size[0]//2,self.grating_size[1]//2)
        turtle.penup()

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
