#示例代码 贴图
import turtleGL,cv2,math
camera = turtleGL.camera('turtleGL texture example')
camera.camera_position = [201,201,131]
camera.to_target([0,0,50])
camera.camera_focal = 500
camera.ray = [-1,-1,-1]
camera.type = 1
camera.rend = 1
scene = turtleGL.scene()
'''
scene.face = [[[[50.0, 50.0, 0.0], [-50.0, 50.0, 0.0], [-50.0, -50.0, 0.0], [50.0, -50.0, 0.0]], '#FF0000'], 
              [[[50.0, 50.0, 0.0], [50.0, 50.0, 100.0], [50.0, -50.0, 100.0], [50.0, -50.0, 0.0]], '#0000FF'], 
              [[[-50.0, 50.0, 0.0], [-50.0, -50.0, 0.0], [-50.0, -50.0, 100.0], [-50.0, 50.0, 100.0]], '#FFFF00'], 
              [[[50.0, 50.0, 0.0], [-50.0, 50.0, 0.0], [-50.0, 50.0, 100.0], [50.0, 50.0, 100.0]], '#FF00FF'], 
              [[[50.0, 50.0, 100.0], [-50.0, 50.0, 100.0], [-50.0, -50.0, 100.0], [50.0, -50.0, 100.0]], '#00FF00'],
              [[[-50.0, -50.0, 0.0], [50.0, -50.0, 0.0], [50.0, -50.0, 100.0], [-50.0, -50.0, 100.0]], '#00FFFF']]
'''
scene.tex = [[[[50.0, 50.0, 100.0], [-50.0, 50.0, 100.0], [-50.0, -50.0, 100.0], [50.0, -50.0, 100.0]], 'grass_up.png'],
             [[[-50.0, -50.0, 0.0], [50.0, -50.0, 0.0], [50.0, -50.0, 100.0],[-50.0, -50.0, 100.0]], 'grass_side.png'],
             [[[-50.0, 50.0, 0.0], [-50.0, -50.0, 0.0], [-50.0, -50.0, 100.0], [-50.0, 50.0, 100.0]], 'grass_side.png'],
             [[[50.0, -50.0, 0.0], [50.0, 50.0, 0.0], [50.0, 50.0, 100.0], [50.0, -50.0, 100.0]], 'grass_side.png'],
             [[[50.0, 50.0, 0.0], [-50.0, 50.0, 0.0], [-50.0, 50.0, 100.0], [50.0, 50.0, 100.0]], 'grass_side.png'],
             [[[50.0, 50.0, 0.0], [-50.0, 50.0, 0.0], [-50.0, -50.0, 0.0], [50.0, -50.0, 0.0]], 'grass_bottom.png']]

camera.image_size = [700,700]
camera.create_image('#ffffff')
camera.draw_from_scene_cv2(scene.sort_all_avg(camera.camera_position))

camera.imshow()
#camera.done()
'''
for i in range(0):
    camera.create_image('#ffffff')
    camera.camera_position = [200*math.cos(math.radians(i)),200*math.sin(math.radians(i)),130+math.sin(math.radians(i/5))]
    camera.to_target([0,0,50+math.sin(math.radians(i/5))])
    camera.draw_from_scene_cv2(scene.sort_all_avg(camera.camera_position))
    camera.capture('grass_block',i)
    #camera.imshow()
    print(i)
camera.to_video('grass_block')
'''