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
scene.triangulation()
scene.generate_line('#ffffff')
camera.grating_size = [400,400]
camera.bgcolor('#000000')
camera.camera_position = [-150,-150,150]
camera.to_target([0,0,0])
camera.grating(scene.sort_face_avg(camera.camera_position))
camera.show_grating_limit('#ffffff')
camera.done()