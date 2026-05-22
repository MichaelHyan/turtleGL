# 🐢 TurtleGL-3d

**A 3D graphics library based on Python turtle and OpenGL-style rendering techniques.**

An object-oriented, intuitive Python 3D graphics library based on the turtle library.

---

## ✨ Features

- 🎥 **Camera System** — Independent camera object supporting perspective / cabinet / isometric / orthographic projection modes.
- 🎨 **Scene Management** — Structured scene object supporting line, face, and texture data management.
- 📐 **3D Function Plotting** — 3D function image plotting based on `plot3d` object.
- 📦 **Volume Calculation** — Ray method volume calculation supporting multi-directional checks and partitioning acceleration.
- 🖼️ **Texture Mapping** — Homography matrix-based texture mapping.
- 🌗 **Shading & Normal** — Three rendering modes: material preview / shadow mode / normal preview.
- 💡 **Lighting System** — Independent `ray` object supporting sunlight, point light, and directional ray with brightness control.
- 📁 **OBJ Import** — Support for importing `.obj` 3D models with normal correction.
- 🎬 **Image & Video Export** — OpenCV-based image export and video compositing.
- 🔄 **Transform** — Spatial transformations: rotation, translation, scaling, etc.
- 📊 **Depth Sorting** — Depth sorting algorithms for different projection modes.

---

## 📦 Installation

```bash
pip install TurtleGL-3d
```

### Dependencies

- Python >= 3.8
- numpy >= 1.24.4
- opencv-python >= 4.12.0

---

## 🚀 Quick Start

### Basic Example: Drawing a Colorful Cube

```python
import turtleGL

# Instantiate the camera
camera = turtleGL.camera()
camera.camera_position = [100, 100, 100]
camera.to_target([0, 0, 0])       # Face the origin
camera.camera_focal = 300          # Focal length
camera.type = 1                    # Perspective mode
camera.rend = 1                    # Shadow mode

# Instantiate the lighting
ray = turtleGL.ray()
ray.add_sunlight([1, 1, -1])       # Add sunlight direction

# Instantiate the scene
scene = turtleGL.scene()
scene.face = [
    [[[50, 50, 0], [-50, 50, 0], [-50, -50, 0], [50, -50, 0]], '#FF0000'],
    [[[50, 50, 100], [-50, 50, 100], [-50, -50, 100], [50, -50, 100]], '#00FF00'],
    [[[50, 50, 0], [50, 50, 100], [50, -50, 100], [50, -50, 0]], '#0000FF'],
    [[[-50, 50, 0], [-50, -50, 0], [-50, -50, 100], [-50, 50, 100]], '#FFFF00'],
    [[[50, 50, 0], [-50, 50, 0], [-50, 50, 100], [50, 50, 100]], '#FF00FF'],
    [[[-50, -50, 0], [50, -50, 0], [50, -50, 100], [-50, -50, 100]], '#00FFFF'],
]

# Depth sorting and drawing
camera.draw_from_scene(scene.sort_all_avg(camera.camera_position), ray)
camera.done()
```

### Importing an OBJ Model

```python
import turtleGL, math

camera = turtleGL.camera('OBJ Example')
camera.camera_position = [-101, -121, -150]
camera.to_target([0, 0, 50])
camera.camera_focal = 500
camera.type = 1
camera.rend = 1

ray = turtleGL.ray()
ray.add_sunlight([1, 1, -1])       # Add sunlight direction

scene = turtleGL.scene()
scene.import_obj('model.obj', 50, '#66ccff')  # Scale 50x, specify color
scene.check_obj_norm('model.obj')              # Correct normals
scene.generate_line('#ffffff')                  # Generate edges

# Rotation animation
for i in range(360):
    camera.clear()
    camera.camera_position = [150 * math.cos(math.radians(i)),
                               150 * math.sin(math.radians(i)),
                               150]
    camera.to_target([0, 0, 0])
    camera.draw_from_scene(scene.sort_all_avg(camera.camera_position), ray)
    camera.update()
```

### 3D Function Plotting

```python
import turtleGL

camera = turtleGL.camera('Plot3D Example')
camera.type = 0  # Cabinet mode

scene = turtleGL.plot3d()
scene.xlim = [-100, 100]
scene.ylim = [-100, 100]
scene.step = 10

def function(x, y):
    return 0.01 * (x**2 - y**2)

scene.generate_face(function)
scene.generate_line(function, color='#000000')
camera.draw_from_scene(scene.sort_all_cabin())
camera.done()
```

### Volume Calculation

```python
import turtleGL

camera = turtleGL.camera()
camera.camera_position = [10, 120, 10]
camera.to_target([0, 0, 0])
camera.camera_focal = 500
camera.type = 1
camera.rend = 1

ray = turtleGL.ray()
ray.add_sunlight([1, 1, -1])

scene = turtleGL.scene()
scene.import_obj('model.obj', 50, '#66ccff')
scene.check_obj_norm('model.obj')
scene.triangulation()       # Triangulation is required for volume calculation
scene.generate_line('#ff0000')

volume = turtleGL.volume()
volume.sample_distance = 5   # Sampling distance
volume.check = True          # Multi-directional check
volume.allow_edge = True     # Allow edge intersections
volume.volume(scene.face)    # Calculate volume

camera.draw_from_scene(scene.sort_line_avg(camera.camera_position), ray)
for i in volume.points:
    camera.dot(i)
camera.done()
```

### Texture Mapping & Image Export

```python
import turtleGL

camera = turtleGL.camera('Texture Example')
camera.camera_position = [201, 201, 131]
camera.to_target([0, 0, 50])
camera.camera_focal = 500
camera.type = 1
camera.rend = 1

ray = turtleGL.ray()
ray.add_sunlight([1, 1, -1])

scene = turtleGL.scene()
scene.tex = [
    [[[50, 50, 100], [-50, 50, 100], [-50, -50, 100], [50, -50, 100]], 'grass_up.png'],
    [[[50, 50, 0], [-50, 50, 0], [-50, -50, 0], [50, -50, 0]], 'grass_bottom.png'],
]

# Render and export using OpenCV
camera.image_size = [700, 700]
camera.create_image('#ffffff')
camera.draw_from_scene_cv2(scene.sort_all_avg(camera.camera_position), ray)
camera.imshow()       # Display
camera.imwrite('output.png')  # Save image
```

---

## 📖 API Reference

### Camera Object

The camera object handles 3D to 2D projection and rendering.

#### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `camera_position` | `[x,y,z]` | `[0,0,0]` | Camera position |
| `camera_direction` | `[x,y,z]` | `[0,0,1]` | Camera look direction |
| `camera_rotation` | `float` | `0` | Camera rotation angle (radians), causing left/right tilt |
| `camera_focal` | `float` | `1` | Focal length |
| `point_behind_cam_type` | `int` | `0` | Handling method for points behind the camera (0/1/2/3) |
| `point_behind_cam_allow_count` | `int` | `0` | Number of allowed behind-camera points during material rendering |
| `rend` | `int` | `0` | Rendering type: 0 material preview / 1 shadow mode / 2 normal preview |
| `pensize` | `int` | `2` | Pen size (only affects lines) |
| `pencolor` | `str` | `'#000000'` | Pen color (only affects lines) |
| `type` | `int` | `1` | Camera type: 0 cabinet / 1 perspective / 2 isometric / -1 orthographic |
| `grating_size` | `[x,y]` | `[500,400]` | Raster rendering area size |
| `grating_length` | `int` | `1` | Raster sampling step |
| `image_size` | `[x,y]` | `[500,400]` | Export image size |
| `image` | `ndarray` | `[]` | Currently stored OpenCV image |

#### Handling Points Behind the Camera

| Mode | Description |
|------|-------------|
| `0` | No processing; points behind the camera appear in the opposite direction |
| `1` | Flip UV to bring points back to normal orientation |
| `2` | Use orthographic mode with small deviation |
| `3` | Use scaled orthographic perspective |

#### Rendering Modes

| Mode | Description |
|------|-------------|
| `0` — Material Preview | Faces display the specified color directly |
| `1` — Shadow Mode | Shading calculated based on the `ray` lighting object; light intensity is computed from sunlight, point light, and directional ray sources |
| `2` — Normal Mode | Normal direction cosine relative to camera direction > 0 displays blue, otherwise red |

#### Methods

```python
# Basic Setup
setposition([x,y,z])          # Set camera position
setdirection([x,y,z])         # Set camera direction
setfocal(x)                   # Set focal length
settype(x)                    # Set projection type (supports 'focal'/'cabin'/'isometric')
to_target([x,y,z])            # Point camera toward target
status()                      # Print current camera attributes

# Coordinate Mapping
pointfocal([x,y,z])           # Perspective: 3D → 2D mapping, returns [[u,v], bool]
pointcabinet([x,y,z])         # Cabinet: 3D → 2D mapping
pointisometric([x,y,z])       # Isometric: 3D → 2D mapping
pointorthografic([x,y,z])     # Orthographic: 3D → 2D mapping
pointfocal_inverse([u,v])     # Perspective: 2D → 3D inverse mapping

# Drawing (turtle rendering)
dot([x,y,z], color)           # Draw a single point
drawline(linedata)            # Draw an edge
drawface(facedata)            # Draw a face
drawtex(facedata)             # Draw texture
draw_from_scene(scenedata, ray) # Draw integrated data with lighting
draw_axis(l)                  # Draw coordinate axes
write(point, str)             # Write text at a 3D position

# Drawing (OpenCV rendering)
drawline_cv2(linedata)        # Draw edge with OpenCV
drawface_cv2(facedata)        # Draw face with OpenCV
drawtex_cv2(facedata)         # Draw texture with OpenCV
draw_from_scene_cv2(scenedata, ray) # Draw integrated data with OpenCV with lighting

# Image Export
create_image(bgcolor)         # Initialize image (can be called repeatedly to clear)
imshow()                      # Display current image
imwrite(path)                 # Save image to file
capture(path, index)          # Capture frame by index (for video frames)
to_video(path, fps=30)        # Compose video

# Raster Algorithm (experimental)
grating(face)                 # Perform raster calculation
grating_cv2(face)             # Raster with OpenCV
show_grating_limit()          # Show rendering area border

# Utilities
tracer(t)                     # Control animation switch
delay()                       # Delay
clear()                       # Clear canvas
bgcolor(color)                # Set background color
update()                      # Update canvas
done()                        # Prevent window from closing automatically
```

---

### Scene Object

The scene object stores line, face, and texture data, and supports spatial transformations and depth sorting.

#### Data Format

```python
# Edge data
[[[x1,y1,z1], [x2,y2,z2]], '#RRGGBB']

# Face data (enter vertices in counterclockwise order; normal points outward)
[[[x1,y1,z1], [x2,y2,z2], ..., [xn,yn,zn]], '#RRGGBB']

# Texture data (4 vertices, color replaced by image path)
[[[x1,y1,z1], [x2,y2,z2], [x3,y3,z3], [x4,y4,z4]], 'texture.png']
```

#### Attributes

| Attribute | Description |
|-----------|-------------|
| `line` | List of edge data |
| `face` | List of face data |
| `tex` | List of texture data |
| `center` | Center point of the scene |

#### Methods

```python
# Add Data
addline([[x1,y1,z1],[x2,y2,z2],'#color'])   # Add an edge
addface([[x1,y1,z1],...,[xn,yn,zn],'#color'])# Add a face

# Import/Export
import_line(path)              # Import line data (CSV)
import_face(path)              # Import face data (CSV)
export_line(path)              # Export line data (CSV)
export_face(path)              # Export face data (CSV)

# OBJ Model
import_obj(path, scale, color) # Import OBJ model (random color if color is empty)
import_obj_normal(path)        # Import OBJ normal data
check_obj_norm(path)           # Correct face orientation based on normals
add_obj(filepath, scale, color)# Import OBJ and append to existing face data

# Spatial Transformations
rotate(rotate_vector, center)  # Rotate around center (rotation vector in radians)
move(move_vector)              # Translate
scale(scale_vector, center)    # Scale
rotate_edge()                  # Cycle edges (change triangulation partition)

# Depth Sorting — Perspective/Orthographic Modes
sort_line_avg(camera_pos)      # Sort edges (modifies object attribute and returns)
sort_face_avg(camera_pos)      # Sort faces
sort_tex_avg(camera_pos)       # Sort textures
sort_all_avg(camera_pos)       # Sort all (returns data without modifying object)

# Depth Sorting — Cabinet Mode
sort_line_cabin()              # Sort edges
sort_face_cabin()              # Sort faces
sort_tex_cabin()               # Sort textures
sort_all_cabin()               # Sort all

# Depth Sorting — Isometric Mode
sort_line_isometric()          # Sort edges
sort_face_isometric()          # Sort faces
sort_tex_isometric()           # Sort textures
sort_all_isometric()           # Sort all

# Other
reverse_normvect(i)            # Reverse normal direction of the i-th face
generate_line(color)           # Generate edges from face data
triangulation()                # Triangulate faces
get_center()                   # Calculate and return the scene center point
```

---

### Plot3D Object

Object for drawing 3D function graphs, similar to Scene but data generation depends on the target function.

#### Attributes

| Attribute | Default | Description |
|-----------|---------|-------------|
| `xlim` | `[-10, 10]` | X domain |
| `ylim` | `[-10, 10]` | Y domain |
| `step` | `1` | Sampling step |
| `line` | `[]` | Edge data |
| `face` | `[]` | Face data |
| `center` | `[0,0,0]` | Center point |

#### Methods

```python
generate_face(func, color=True)  # Generate face data for the function (auto color by height)
generate_line(func, color)       # Generate edge data for the function
rotate(rotate_vector, center)    # Rotate
move(move_vector)                # Translate
scale(scale_vector, center)      # Scale
# Same depth sorting methods as Scene
```

---

### Volume Object

Volume calculation object using ray casting to determine if a point is inside a closed triangular mesh.

#### Attributes

| Attribute | Default | Description |
|-----------|---------|-------------|
| `points` | `[]` | List of interior points after calculation |
| `sample_distance` | `1` | Sampling grid spacing |
| `grid_limit` | `inf` | Threshold for enabling partitioning acceleration (triangle count) |
| `check` | `True` | Enable multi-directional check |
| `allow_edge` | `True` | Allow counting edge intersections |

#### Methods

```python
volume(scene_face_data)  # Calculate volume, return all interior points
```

> ⚠️ The partitioning acceleration (`grid_limit`) is only recommended for convex polyhedra; irregular shapes may produce erroneous results.

---

### Ray Object

Lighting object that manages light sources and computes shading values for faces.

#### Attributes

| Attribute | Default | Description |
|-----------|---------|-------------|
| `sunlight` | `[]` | List of sunlight sources `[direction, brightness]` |
| `pointlight` | `[]` | List of point light sources `[position, brightness]` |
| `ray` | `[]` | List of directional ray sources `[point, direction, brightness]` |

#### Methods

```python
add_sunlight(vec, brightness=1)    # Add sunlight (directional light), brightness defaults to 1
add_pointlight(pos, brightness)    # Add point light at position with brightness
add_ray(point, vec, brightness)    # Add directional ray from point in direction with brightness
get_value(point1, point2, point3)  # Compute combined shading value for a triangle face (returns -1 to 1)
```

#### Light Types

| Type | Description |
|------|-------------|
| Sunlight | Directional light; shading based on angle between light direction and face normal |
| Point Light | Positional light; shading based on angle and distance attenuation |
| Directional Ray | Light from a point in a direction; shading based on angle between ray and face normal |

---

## 🧪 Experimental: Rasterization

Raster algorithm is experimental and not yet stable.

```python
scene.triangulation()             # Triangulate faces (raster mode only supports triangles)
camera.grating_size = [500, 400]  # Set rendering area size
camera.show_grating_limit()       # Show rendering area border
camera.grating(face, ray)         # Execute raster calculation
```

In rendering mode `rend=1`, raster mode uses the `ray` lighting object to compute light paths and shadows.

---

## 🎬 Image & Video Export

Since turtle itself does not support image capture, the library provides OpenCV-based image/video export:

```python
# Initialization
camera.image_size = [500, 400]
camera.create_image('#ffffff')    # Create image (can be called repeatedly to clear)

# Rendering
camera.draw_from_scene_cv2(scene_data)  # Render with OpenCV

# Export
camera.imwrite('output.png')     # Save image

# Video creation
camera.capture('frames', i)      # Save frames by index → ./frames/00000001.png
camera.to_video('frames')        # Compose video → frames.mp4
```

---

## 📁 Project Structure

```
turtleGL-3d/
├── pyproject.toml
├── setup.py
├── LICENSE
├── README.md
├── README_zh.md
└── src/
    └── turtleGL/
        ├── __init__.py
        ├── src/
        │   ├── camera.py      # Camera object
        │   ├── scene.py       # Scene object
        │   ├── plot3d.py      # 3D function plot object
        │   ├── ray.py         # Lighting object
        │   └── volume.py      # Volume calculation object
        └── example/
```

---

## 📄 License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Homepage**: [https://github.com/MichaelHyan/turtleGL](https://github.com/MichaelHyan/turtleGL)
- **PyPI**: [https://pypi.org/project/TurtleGL-3d/](https://pypi.org/project/TurtleGL-3d/)
