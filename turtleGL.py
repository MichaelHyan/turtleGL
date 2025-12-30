'''--- turtleGL v1.2.0 ---'''
'''---     by Hyan     ---'''
import numpy as np
import turtle,math,random,cv2,os
import torch
from PIL import ImageGrab
class camera():
    def __init__(self,title = 'turtleGL v1.2.0'):
        self.title = title
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
        self.grating_size = [500,400]
        self.device = None
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
        print(f'ray direction : {self.ray}')
        print(f'using rend : {'material preview' if self.rend == 0 else 'shade' if self.rend == 1 else 'normal vector preview' if self.rend == 2 else self.rend}')
        print(f'shade value : {self.shade_value}')
        print(f'type : {'focal' if self.type == 1 else 'cabin' if self.type == 0 else 'isometric' if self.type == 2 else self.type}')
        print('=================================')

    def tracer(self, t):
        turtle.tracer(t)

    def to_target(self,t):
        self.camera_direction = [t[0]-self.camera_position[0],t[1]-self.camera_position[1],t[2]-self.camera_position[2]]

    def pointfocal(self, point_3d):
        if self.device == None:
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
                pass
                #return [0,0]
            u = (self.camera_focal * point_cam[0]) / point_cam[2]
            v = (self.camera_focal * point_cam[1]) / point_cam[2]
            x = u * math.cos(self.camera_rotation) - v * math.sin(self.camera_rotation)
            y = u * math.sin(self.camera_rotation) + v * math.cos(self.camera_rotation)
            return [x,y]
        else:
            device = self.device
            position = torch.tensor([self.camera_position[0], self.camera_position[2], self.camera_position[1]], 
                                  dtype=torch.float32, device=device)
            direction = torch.tensor([self.camera_direction[0], self.camera_direction[2], self.camera_direction[1]], 
                                    dtype=torch.float32, device=device)
            point = torch.tensor([point_3d[0], point_3d[2], point_3d[1]], 
                               dtype=torch.float32, device=device)
            cam_pos = position
            cam_dir = direction
            cam_dir = cam_dir / torch.norm(cam_dir)
            z_axis = cam_dir
            z_axis = z_axis / torch.norm(z_axis)
            up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
            x_axis = torch.linalg.cross(up, z_axis)
            x_axis = x_axis / torch.norm(x_axis)
            y_axis = torch.linalg.cross(z_axis, x_axis)
            view_matrix = torch.eye(4, dtype=torch.float32, device=device)
            view_matrix[0, :3] = x_axis
            view_matrix[1, :3] = y_axis
            view_matrix[2, :3] = z_axis
            view_matrix[0, 3] = -torch.dot(x_axis, cam_pos)
            view_matrix[1, 3] = -torch.dot(y_axis, cam_pos)
            view_matrix[2, 3] = -torch.dot(z_axis, cam_pos)
            point_homo = torch.cat([point, torch.tensor([1], dtype=torch.float32, device=device)])
            point_cam = torch.mv(view_matrix, point_homo)
            if point_cam[2] <= 0:
                pass
                # return [0,0]
            u = (self.camera_focal * point_cam[0]) / point_cam[2]
            v = (self.camera_focal * point_cam[1]) / point_cam[2]
            cos_r = torch.cos(torch.tensor(self.camera_rotation, dtype=torch.float32, device=device))
            sin_r = torch.sin(torch.tensor(self.camera_rotation, dtype=torch.float32, device=device))
            x = u * cos_r - v * sin_r
            y = u * sin_r + v * cos_r
            return [x.item(), y.item()]

    def pointfocal_inverse(self, point_2d):
        if self.device == None:
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
        else:
            device = self.device
            x, y = point_2d
            x = torch.tensor(x, dtype=torch.float32, device=device)
            y = torch.tensor(y, dtype=torch.float32, device=device)
            cos_r = torch.cos(torch.tensor(-self.camera_rotation, dtype=torch.float32, device=device))
            sin_r = torch.sin(torch.tensor(-self.camera_rotation, dtype=torch.float32, device=device))
            u = x * cos_r - y * sin_r
            v = x * sin_r + y * cos_r
            position = torch.tensor([self.camera_position[0], self.camera_position[2], self.camera_position[1]], 
                                  dtype=torch.float32, device=device)
            direction = torch.tensor([self.camera_direction[0], self.camera_direction[2], self.camera_direction[1]], 
                                    dtype=torch.float32, device=device)
            cam_pos = position
            cam_dir = direction
            cam_dir = cam_dir / torch.norm(cam_dir)
            z_axis = cam_dir
            up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
            x_axis = torch.linalg.cross(up, z_axis)
            x_axis = x_axis / torch.norm(x_axis)
            y_axis = torch.linalg.cross(z_axis, x_axis)
            view_matrix_inv = torch.eye(4, dtype=torch.float32, device=device)
            view_matrix_inv[:3, 0] = x_axis
            view_matrix_inv[:3, 1] = y_axis
            view_matrix_inv[:3, 2] = z_axis
            view_matrix_inv[:3, 3] = cam_pos
            point_cam = torch.tensor([u, v, self.camera_focal, 1], dtype=torch.float32, device=device)
            point_world = torch.mv(view_matrix_inv, point_cam)
            return [point_world[0].item(), point_world[2].item(), point_world[1].item()]


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
        elif self.type == 0:
            point1 = list(map(int,self.pointcabinet(l[0][0])))
            point2 = list(map(int,self.pointcabinet(l[0][1])))
        elif self.type == 2:
            point1 = list(map(int,self.pointisometric(l[0][0])))
            point2 = list(map(int,self.pointisometric(l[0][1])))
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
        if self.device == None:
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
        else:
            device = self.device
            epsilon = 1e-6
            v0 = torch.tensor([float(triangle_vertices[0][0]), float(triangle_vertices[0][2]), float(triangle_vertices[0][1])], 
                            dtype=torch.float32, device=device)
            v1 = torch.tensor([float(triangle_vertices[1][0]), float(triangle_vertices[1][2]), float(triangle_vertices[1][1])], 
                            dtype=torch.float32, device=device)
            v2 = torch.tensor([float(triangle_vertices[2][0]), float(triangle_vertices[2][2]), float(triangle_vertices[2][1])], 
                            dtype=torch.float32, device=device)
            edge1 = v1 - v0
            edge2 = v2 - v0
            ray_direction = torch.tensor([float(ray[0]), float(ray[2]), float(ray[1])], 
                                    dtype=torch.float32, device=device)
            ray_origin = torch.tensor([float(point[0]), float(point[2]), float(point[1])], 
                                    dtype=torch.float32, device=device)
            pvec = torch.cross(ray_direction, edge2)
            det = torch.dot(edge1, pvec)
            if torch.abs(det) < epsilon:
                return False, None, None, None
            inv_det = 1.0 / det
            tvec = ray_origin - v0
            u = torch.dot(tvec, pvec) * inv_det
            if u < 0.0 or u > 1.0:
                return False, None, None, None
            qvec = torch.cross(tvec, edge1)
            v = torch.dot(ray_direction, qvec) * inv_det
            if v < 0.0 or u + v > 1.0:
                return False, None, None, None
            t = torch.dot(edge2, qvec) * inv_det
            if t < epsilon:
                return False, None, None, None
            intersection_point = ray_origin + t * ray_direction
            return True, intersection_point[0].item(), intersection_point[2].item(), intersection_point[1].item()

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
        if self.device == None:
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
        else:
            device = self.device
            vector = torch.tensor(vector, dtype=torch.float32, device=device)
            point1 = torch.tensor(point1, dtype=torch.float32, device=device)
            point2 = torch.tensor(point2, dtype=torch.float32, device=device)
            point3 = torch.tensor(point3, dtype=torch.float32, device=device)
            vector1 = point2 - point1
            vector2 = point3 - point2
            cross_product = torch.cross(vector1, vector2)
            dot_product = torch.dot(cross_product, vector)
            return bool(dot_product > 0)
        
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

class vidtool():
    def __init__(self,name):
        self.name = name
        try:
            os.mkdir(name)
        except:
            pass
    
    def capture(self,i):
        screenshot = ImageGrab.grab()
        screenshot.save(f'{self.name}/{i:08d}.png')

    def to_video(self,path = '',fps=30):
        if path == '':
            path = self.name
        else:
            path = path = f'{path}/{self.name}'
        images = [img for img in os.listdir(path) if img.endswith(".png")]
        if not images:
            return
        first_image = cv2.imread(os.path.join(path, images[0]))
        height, width, layers = first_image.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(f'{self.name}.mp4', fourcc, fps, (width, height))
        images.sort()
        for image in images:
            image_path = os.path.join(path, image)
            frame = cv2.imread(image_path)
            video.write(frame)
        video.release()
        cv2.destroyAllWindows()
        print('save complete')

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