#示例代码 透视模式 光栅模式
import turtleGL
import math
camera = turtleGL.camera('turtleGL grating example')
camera.camera_position = [-101,-121,-150]
camera.to_target([0,0,50])
camera.camera_focal = 500
camera.ray = [-1,1,-1]
camera.type = 1
camera.rend = 1
scene = turtleGL.scene()

path = 'test.obj'
scene.import_obj(path,50,'#66ccff')
scene.check_obj_norm(path)
scene.triangulation() #光栅化前需要三角化模型

camera.grating_size = [400,400] #光栅模式渲染区
camera.bgcolor('#000000') #背景色
camera.camera_position = [-150,-150,150]
camera.to_target([0,0,0])
camera.grating(scene.sort_face_avg(camera.camera_position)) #光栅模式绘制
camera.show_grating_limit('#ffffff') #绘制渲染区边缘
camera.done()