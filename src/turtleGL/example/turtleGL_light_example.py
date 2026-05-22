#示例代码 透视模式 导入obj模型
import turtleGL
import math
camera = turtleGL.camera('turtleGL obj example')
camera.camera_position = [-150,-150,150]
camera.to_target([0,0,0])
camera.camera_focal = 500
camera.type = 1
camera.rend = 1
scene = turtleGL.scene()
path = 'test.obj'
scene.import_obj(path,30,'#66ccff')#导入模型，缩放倍率50，洛天依色
scene.check_obj_norm(path)#检查法线
scene.generate_line('#ffffff')#生成边
camera.bgcolor('#000000')#背景色
camera.show_bar = True

ray = turtleGL.ray()
ray.add_pointlight([100,100,100],300)

for i in range(1):
    ray.pointlight[0] = [[150*math.sin(i/50),150*math.cos(i/50),100],300]
    camera.create_image('#000000')
    camera.draw_from_scene_cv2(scene.sort_face_avg(camera.camera_position),ray)
    #camera.imshow()
    camera.capture('light',i)
    print(f' {i}',end='')
camera.to_video('light') #将生成的图片拼接成视频