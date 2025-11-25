
import turtleGL
import math
camera = turtleGL.camera()
camera.camera_position = [-101,-121,-150]
camera.to_target([0,0,50])
camera.camera_focal = 300
camera.ray = [1,1,-1]
camera.type = 1
camera.rend = 1
scene = turtleGL.scene()
path = 'test.obj'
scene.import_obj(path,50,'#66ccff')
scene.check_obj_norm(path)
for i in range(3000):
    camera.clear()
    camera.camera_position = [150*math.cos(math.radians(i)),150*math.sin(math.radians(i)),150*math.cos(math.radians(i))]
    camera.to_target([0,0,0])
    camera.draw_from_scene(scene.sort_all_avg(camera.camera_position))
    camera.update()