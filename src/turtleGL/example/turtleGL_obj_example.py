#示例代码 透视模式 导入obj模型
import turtleGL
import math
camera = turtleGL.camera('turtleGL obj example')
camera.camera_position = [-101,-121,-150]
camera.to_target([0,0,50])
camera.camera_focal = 500
camera.ray = [1,1,-1]
camera.type = 1
camera.rend = 1
scene = turtleGL.scene()
path = 'test.obj'
scene.import_obj(path,50,'#66ccff')#导入模型，缩放倍率50，洛天依色
scene.check_obj_norm(path)#检查法线
scene.generate_line('#ffffff')#生成边
camera.bgcolor('#000000')#背景色
for i in range(3000):#动画显示
    camera.clear()
    camera.camera_position = [150*math.cos(math.radians(i)),150*math.sin(math.radians(i)),150*math.cos(math.radians(i))]
    camera.to_target([0,0,0])
    camera.draw_from_scene(scene.sort_all_avg(camera.camera_position))
    #scene.sort_face_avg(camera.camera_position)
    #camera.draw_from_scene(scene.face)
    #scene.sort_line_avg(camera.camera_position)
    #camera.draw_from_scene(scene.line)
    camera.update()