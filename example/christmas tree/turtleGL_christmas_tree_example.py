#示例代码 透视模式 动画 obj模型导入
import turtleGL
import math,time
camera = turtleGL.camera()
camera.camera_position = [-101,-121,-150]
camera.to_target([0,0,50])
camera.camera_focal = 500
camera.ray = [1,1,-1]
camera.type = 1
camera.rend = 1
scene = turtleGL.scene()

tree_face = [] #面暂存列表
tree_line = [] #边暂存列表

scene.import_obj('tree.obj',50,"#66ff78") #使用scene加载obj，缩放50倍，设置边为绿色
scene.check_obj_norm('tree.obj') #法线修正
scene.generate_line("#00630F") #生成边
for i in scene.face: #保存至暂存列表
    tree_face.append(i)
for i in scene.line:
    tree_line.append(i)

scene.import_obj('tree1.obj',45,"#fff700")
scene.check_obj_norm('tree1.obj')
scene.generate_line("#757C00")
for i in scene.face:
    tree_face.append(i)
for i in scene.line:
    tree_line.append(i)

scene.import_obj('tree2.obj',45,"#ff6666")
scene.check_obj_norm('tree2.obj')
scene.generate_line("#6B0000")
for i in scene.face:
    tree_face.append(i)
for i in scene.line:
    tree_line.append(i)

scene.import_obj('tree3.obj',45,"#4b60ff")
scene.check_obj_norm('tree3.obj')
scene.generate_line("#14006D")
for i in scene.face:
    tree_face.append(i)
for i in scene.line:
    tree_line.append(i)

scene.import_obj('tree4.obj',50,"#ffff00")
scene.check_obj_norm('tree4.obj')
scene.generate_line("#6A7300")
for i in scene.face:
    tree_face.append(i)
for i in scene.line:
    tree_line.append(i)

scene.face = tree_face #暂存列表数据传递到scene数据
scene.line = tree_line

#仅显示
camera.bgcolor('#000000')
for i in range(1200):
    camera.clear()
    camera.camera_position = [150*math.cos(math.radians(i)),150*math.sin(math.radians(i)),150]
    camera.to_target([0,0,50])
    camera.draw_from_scene(scene.sort_all_avg(camera.camera_position))
    camera.update()
    print(i)

#仅生成视频
'''
camera.image_size = [600,700] #设置绘制区尺寸
for i in range(1200):
    camera.create_image('#000000') #创建绘制区空白图像，暂存在camera.image属性中
    camera.camera_position = [150*math.cos(math.radians(i)),150*math.sin(math.radians(i)),150]
    camera.to_target([0,0,50])
    camera.draw_from_scene_cv2(scene.sort_all_avg(camera.camera_position)) #使用cv2绘制，不显示，暂存在camera.image
    camera.capture('tree',i) #输出暂存图像到.\tree\{i:08d}.png
    #camera.imshow() 显示cv2绘制的图像
    #camera.imwrite('tree.png') #输出暂存图像到.\tree.png
    print(i)
camera.to_video('tree') #将生成的图片拼接成视频
'''
