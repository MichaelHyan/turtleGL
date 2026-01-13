#示例代码 函数绘制
import turtleGL
camera = turtleGL.camera('turtleGL plot3D example')
camera.type = 0 #斜二侧模式
scene = turtleGL.plot3d()
scene.xlim = [-100,100] #x定义域
scene.ylim = [-100,100] #y定义域
scene.step = 10 #采样步长
def function(x,y): #定义函数
    return 0.01*(x**2-y**2)
scene.generate_face(function) #面采样
scene.generate_line(function,color="#000000") #边采样
camera.draw_from_scene(scene.sort_all_cabin())
camera.done()