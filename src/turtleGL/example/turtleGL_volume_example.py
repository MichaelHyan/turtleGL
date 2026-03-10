#示例代码 体积生成
import turtleGL
import math
camera = turtleGL.camera()
camera.camera_position = [10,120,10]
camera.to_target([0,0,0])
camera.camera_focal = 500
camera.ray = [1,1,-1]
camera.pensize = 2
camera.type = 1
camera.rend = 1
scene = turtleGL.scene()
path = 'monkey.obj'
scene.import_obj(path,50,'#66ccff')
scene.check_obj_norm(path)
scene.move([0.1,0.1,0.1])
#scene.rotate_edge()#循环边，用于生成不同的三角化划分
scene.triangulation()#体积仅可在三角化之后计算
scene.generate_line('#ff0000')
volume = turtleGL.volume()
volume.sample_distance=5#采样距离
volume.check = True#启用多向检查
volume.allow_edge=True#允许边界交点
#volume.grid_limit=300#分区加速算法界限，在一定三角形数量以上时启用，默认正无穷，仅在凸多面体下正常工作
volume.volume(scene.face)#体积生成

camera.draw_from_scene(scene.sort_line_avg(camera.camera_position))
for i in volume.points:
    camera.dot(i)
camera.done()