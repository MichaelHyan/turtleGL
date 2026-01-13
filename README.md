## turtleGL-3d

### A 3D Drawing Library based on the turtle library

### Data Structure

Implemented using separate camera objects and scene objects.

Scene objects can store data within themselves and can be specified and invoked by camera objects.

Multiple cameras can be used for flexible switching.

#### Camera Attributes

You can directly reassign values to the attributes of the camera object.

```python
camera_position# Camera position [x,y,z], positive up direction is [0,0,1]
camera_direction# Camera facing direction [x,y,z]
camera_rotation# Camera rotation angle, specifically manifested as left/right tilt, using radians
camera_focal# Camera focal length
point_behind_cam_type# Back point handling 0/1/2
ray# Light ray direction, used to determine if a face is facing the light source
rend# Render type 0 Material preview 1 Shadow mode 2 Normal preview
shade_value# Multiply coefficient 0-255
pensize# Pen size, effective only for lines
pencolor# Pen color, effective only for lines
type# Camera type 0 Cabinet projection 1 Perspective projection 2 Isometric projection -1 Orthographic projection
```

In point mapping calculations, points behind the camera cannot be calculated normally; three handling logics are provided:

0: No handling, back points will appear in the opposite direction

1: Flip UV, turning the point back to the normal direction

2: Switch to using orthographic mode, deviation is relatively smaller than the previous two

##### Material Preview Mode

Faces directly display the specified color.

##### Shadow Mode

This project does not currently support stable rasterization algorithms, so light ray calculations do not exist.

After enabling shadow mode, face data will calculate whether it belongs to a lit surface based on the angle between the light ray direction and the normal. If not, the multiply coefficient is used to recalculate the surface color.

##### Normal Mode

Displays normals. When the cosine value between the camera direction and the face normal is greater than 0 (i.e., an obtuse angle), it is determined as the front face and displayed in blue; otherwise, it is displayed in red.

##### Image Export

Since turtle itself does not have image capture functionality, the following methods can be used to store single-frame images.

```python
# Drawing area initialization before screenshot
camera.image_size = [500,400]
camera.create_image('hex background color')# Can be called repeatedly to achieve clear screen effect

# Drawing
camera.draw_from_scene_cv2(scene object data)# Temporarily store current frame image after use

# Export
camera.imwrite('filename')

# Video processing
# Screenshot by sequence number
camera.capture('name', current index)# Store to .\filename\index:08d.png
# Compose video
camera.to_video('name')
```

#### Camera Methods

```python
setposition([x,y,z])# Set camera position, can also directly set camera_position attribute
setdirection([x,y,z])# Set camera direction, can also directly set camera_direction attribute
setfocal(x):# Set camera focal length, can also directly set camera_focal attribute
settype(x):# Set camera type, can also directly set type attribute
status()# Output current camera attributes
tracer(0)# Turn off animation, operate turtle directly
to_target([x,y,z])# Set camera to face target point
pointfocal([x,y,z])# In perspective mode, return coordinates mapped from space to camera
pointcabinet([x,y,z])# In cabinet mode, return coordinates mapped from space to camera
draw_axis(l)# Draw reference coordinate axes, l adjusts size
drawline(linedata)# Input single edge data, draw edge
drawface(facedata)# Input single face coordinates, draw face
draw_from_scene(scenedata)# Input integrated data, draw all
delay()# Delay, same effect as turtle.delay()
clear()# Clear canvas, same effect as turtle.clear()
bgcolor()# Canvas background color, same effect as turtle.bgcolor()
update()# Update canvas, same effect as turtle.update()
done()# Prevent automatic window closing, same effect as turtle.done()
```

#### Scene Attributes

The scene contains line and face attributes, storing edge/face data.

Multiple scene objects can be used.

##### Data Format

```python
[
    [
        [point1],
        [point2],
        ...
        [pointn]
    ],
    'hexadecimal color'
]
```

When the number of points is 2, it will be recognized as an edge.

When using face data, the direction of the normal ray is considered positive, and points are input in a counter-clockwise direction.

The line and face attributes of the scene object are arrays containing the above data; each element is separate line/face data and can be directly called or modified.

#### Scene Methods

```python
addline([[x1,y1,z1],[x2,y2,z2],'hex color'])# Add edge
addface([[x1,y1,z1],[x2,y2,z2],...,[xn,yn,zn],'hex color'])# Add face
export_line(path)# Export line data csv
export_face(path)# Export face data csv
import_line(path)# Import line data
import_face(path)# Import face data
sort_line_avg([camera x, camera y, camera z])# Adjust layer order in perspective or orthographic mode, modify scene object attributes and return adjusted data
sort_face_avg([camera x, camera y, camera z])# Adjust layer order in perspective or orthographic mode, modify scene object attributes and return adjusted data
sort_all_avg([camera x, camera y, camera z])# Return all adjusted data, does not modify scene object attributes
sort_line_cabin()# Adjust layer order in cabinet mode, modify scene object attributes and return adjusted data
sort_face_cabin()# Adjust layer order in cabinet mode, modify scene object attributes and return adjusted data
sort_all_cabin()# Return all adjusted data, does not modify scene object attributes
sort_line_isometric()# Adjust layer order in isometric mode, modify scene object attributes and return adjusted data
sort_face_isometric()# Adjust layer order in isometric mode, modify scene object attributes and return adjusted data
sort_all_isometric()# Return all adjusted data, does not modify scene object attributes
reverse_normvect(i)# Modify the normal direction of the i-th face data
import_obj(path, scale factor, color)# Import obj model, random color if color is empty
check_obj_norm(path)# Correct normal direction according to obj file information
scene.generate_obj_line(color)# Generate edges based on face data
```

##### Adjusting Order

This project does not currently support stable rasterization algorithms, so light ray calculations do not exist.

The front/back order of faces is determined by the layer and rendering order.

##### obj Model

Acts directly on scene.face after import. Only one object can be imported at a time, and the object name must be in English.

#### 3D Function Graphs

Similar to scene object operations, but data generation depends on the target function.

Includes line and face data, drawing range, and sampling step.

##### Function Methods

Assume the function is defined as follows:

```python
def func(x,y):
    # Unknown operations
    return z
```

Set the domain (drawing range):

```python
scene.xlim = [x1,x2] # Indicates sampling from x1 until x2
scene.ylim = [y1,y2] # Indicates sampling from y1 until y2
scene.step = d # Sampling step
```

Generate graph:

```python
scene.generate_face(func)
scene.generate_line(func)
```

Other operations are consistent with scene objects.

##### Operation Flow

1. Instantiate camera object
2. Adjust lens attributes
3. Instantiate scene object
4. Add data to scene object
5. Sort scene object data
6. Use camera object to invoke data
7. Draw

### Test Feature: Rasterization Algorithm

This feature is not yet stable and is for reference only.

```python
scene.triangulation()# Face triangulation, currently raster mode can only handle triangular faces
camera.grating_size = [x,y]# Rendering area size
camera.show_grating_limit()# Display rendering area edges
camera.grating(face)# Calculate using rasterization algorithm
```

When render mode is 1, shadow mode is no longer used; instead, light ray paths are calculated.

### Usage (Example)

#### Camera Object

Set camera attributes.

```python
camera = turtleGL.camera()# Instantiate camera object
camera.camera_position = [-101,-121,131]# Camera position
camera.camera_direction = [1,1,-1]# Camera direction
camera.to_target([0,0,0])# Camera looks at target point
camera.camera_focal = 300# Focal length
camera.ray = [1,1,-1]# Light direction
camera.type = 1# 1 Perspective mode  0 Cabinet mode
camera.rend = 1# 0 Material preview 1 Shadow 2 Normal
camera.status()# View camera attributes
camera.grating_size = [500,400]# Raster mode set rendering area size
camera.image_size = [500,400]# Export image size
camera.image # Current stored image
```

#### Scene Object

Use structured data to store scene information, divided into lines and faces, directly stored in the scene object and can be invoked.

```python
scene = turtleGL.scene()# Instantiate scene
# Custom edge information
scene.line = [[[[50.0, 50.0, 0.0], [-50.0, 50.0, 0.0]], '#000000'], 
              [[[-50.0, 50.0, 0.0], [-50.0, -50.0, 0.0]], '#000000'], 
              [[[-50.0, -50.0, 0.0], [50.0, -50.0, 0.0]], '#000000'], 
              [[[50.0, -50.0, 0.0], [50.0, 50.0, 0.0]], '#000000'], 
              [[[50.0, 50.0, 100.0], [-50.0, 50.0, 100.0]], '#000000'], 
              [[[-50.0, 50.0, 100.0], [-50.0, -50.0, 100.0]], '#000000'], 
              [[[-50.0, -50.0, 100.0], [50.0, -50.0, 100.0]], '#000000'], 
              [[[50.0, -50.0, 100.0], [50.0, 50.0, 100.0]], '#000000'], 
              [[[50.0, 50.0, 0.0], [50.0, 50.0, 100.0]], '#000000'], 
              [[[-50.0, 50.0, 0.0], [-50.0, 50.0, 100.0]], '#000000'], 
              [[[-50.0, -50.0, 0.0], [-50.0, -50.0, 100.0]], '#000000'], 
              [[[50.0, -50.0, 0.0], [50.0, -50.0, 100.0]], '#000000']]
# Custom face information
scene.face = [[[[50.0, 50.0, 0.0], [-50.0, 50.0, 0.0], [-50.0, -50.0, 0.0], [50.0, -50.0, 0.0]], '#FF0000'], 
              [[[50.0, 50.0, 0.0], [50.0, 50.0, 100.0], [50.0, -50.0, 100.0], [50.0, -50.0, 0.0]], '#0000FF'], 
              [[[-50.0, 50.0, 0.0], [-50.0, -50.0, 0.0], [-50.0, -50.0, 100.0], [-50.0, 50.0, 100.0]], '#FFFF00'], 
              [[[50.0, 50.0, 0.0], [-50.0, 50.0, 0.0], [-50.0, 50.0, 100.0], [50.0, 50.0, 100.0]], '#FF00FF'], 
              [[[50.0, 50.0, 100.0], [-50.0, 50.0, 100.0], [-50.0, -50.0, 100.0], [50.0, -50.0, 100.0]], '#00FF00'], 
              [[[-50.0, -50.0, 0.0], [50.0, -50.0, 0.0], [50.0, -50.0, 100.0], [-50.0, -50.0, 100.0]], '#00FFFF']]
```

Organize display order (taking perspective mode as an example)

```python
scene.sort_line_avg(camera.camera_position)# Edges only
scene.sort_face_avg(camera.camera_position)# Faces only
scene.sort_all_avg(camera.camera_position)# Organize all, directly return structured data, does not modify object
```

Draw

```python
camera.draw_from_scene(scene.sort_all_avg(camera.camera_position))# Draw all content
camera.draw_from_scene(scene.line)# Draw edges only
camera.draw_from_scene(scene.face)# Draw faces only
camera.draw_axis(10)# Show axes
camera.done()
```

Import/Export

```python
# Export
scene.export_line('example_line.csv')
scene.export_face('example_face.csv')
# Import
scene.import_line('example_line.csv')
scene.import_face('example_face.csv')
```