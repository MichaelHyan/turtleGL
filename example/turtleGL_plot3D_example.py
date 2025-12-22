import turtleGL
camera = turtleGL.camera('turtleGL plot3D example')
camera.type = 0
scene = turtleGL.plot3d()
scene.xlim = [-100,100]
scene.ylim = [-100,100]
scene.step = 10
def function(x,y):
    return 0.01*(x**2-y**2)
scene.generate_face(function)
scene.generate_line(function,color="#000000")
camera.draw_from_scene(scene.sort_all_cabin())
camera.done()