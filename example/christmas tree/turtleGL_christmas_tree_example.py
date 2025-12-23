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

tree_face = []
tree_line = []
scene.import_obj('tree.obj',50,"#66ff78")
scene.check_obj_norm('tree.obj')
scene.generate_line("#00630F")
for i in scene.face:
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
scene.check_obj_norm('tree2.obj')
scene.generate_line("#14006D")
for i in scene.face:
    tree_face.append(i)
for i in scene.line:
    tree_line.append(i)
scene.import_obj('tree4.obj',50,"#ffff00")
scene.check_obj_norm('tree2.obj')
scene.generate_line("#6A7300")
for i in scene.face:
    tree_face.append(i)
for i in scene.line:
    tree_line.append(i)

scene.face = tree_face
scene.line = tree_line

#vid = turtleGL.vidtool('tree') #视频方法，慎用
#time.sleep(5)
camera.bgcolor("#000000")

for i in range(1500):
    camera.clear()
    #camera.camera_position = [150*math.cos(math.radians(i)),150*math.sin(math.radians(i)),150*math.cos(math.radians(i))]
    camera.camera_position = [150*math.cos(math.radians(i)),150*math.sin(math.radians(i)),150]
    camera.to_target([0,0,50])
    camera.draw_from_scene(scene.sort_all_avg(camera.camera_position))
    camera.update()
    #vid.capture(i) #视频方法，慎用

#vid.to_video() #视频方法，慎用