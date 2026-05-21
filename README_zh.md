
# 🐢 TurtleGL-3d

**A 3D graphics library based on Python turtle and OpenGL-style rendering techniques.**

面向对象，直观明了的 Python 3D 绘图库，基于 turtle 库。

---

## ✨ Features

- 🎥 **Camera System** — 独立的摄像头对象，支持透视 / 斜二侧 / 等距 / 正交四种投影模式
- 🎨 **Scene Management** — 结构化的场景对象，支持线、面、贴图数据管理
- 📐 **3D Function Plotting** — 基于 `plot3d` 对象的 3D 函数图像绘制
- 📦 **Volume Calculation** — 射线法体积计算，支持多向检查与分区加速
- 🖼️ **Texture Mapping** — 单应性矩阵（Homography）贴图映射
- 🌗 **Shading & Normal** — 材质预览 / 阴影模式 / 法线预览三种渲染模式
- 📁 **OBJ Import** — 支持 `.obj` 3D 模型导入与法线修正
- 🎬 **Image & Video Export** — 基于 OpenCV 的图像导出与视频合成
- 🔄 **Transform** — 旋转、平移、缩放等空间变换操作
- 📊 **Depth Sorting** — 针对不同投影模式的深度排序算法

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

### 基础示例：绘制一个彩色立方体

```python
import turtleGL

# 实例化摄像头
camera = turtleGL.camera()
camera.camera_position = [100, 100, 100]
camera.to_target([0, 0, 0])       # 面向原点
camera.camera_focal = 300          # 焦距
camera.type = 1                    # 透视模式
camera.rend = 1                    # 阴影模式

# 实例化场景
scene = turtleGL.scene()
scene.face = [
    [[[50, 50, 0], [-50, 50, 0], [-50, -50, 0], [50, -50, 0]], '#FF0000'],
    [[[50, 50, 100], [-50, 50, 100], [-50, -50, 100], [50, -50, 100]], '#00FF00'],
    [[[50, 50, 0], [50, 50, 100], [50, -50, 100], [50, -50, 0]], '#0000FF'],
    [[[-50, 50, 0], [-50, -50, 0], [-50, -50, 100], [-50, 50, 100]], '#FFFF00'],
    [[[50, 50, 0], [-50, 50, 0], [-50, 50, 100], [50, 50, 100]], '#FF00FF'],
    [[[-50, -50, 0], [50, -50, 0], [50, -50, 100], [-50, -50, 100]], '#00FFFF'],
]

# 深度排序并绘制
camera.draw_from_scene(scene.sort_all_avg(camera.camera_position))
camera.done()
```

### 导入 OBJ 模型

```python
import turtleGL, math

camera = turtleGL.camera('OBJ Example')
camera.camera_position = [-101, -121, -150]
camera.to_target([0, 0, 50])
camera.camera_focal = 500
camera.ray = [1, 1, -1]
camera.type = 1
camera.rend = 1

scene = turtleGL.scene()
scene.import_obj('model.obj', 50, '#66ccff')  # 缩放50倍，指定颜色
scene.check_obj_norm('model.obj')              # 修正法线
scene.generate_line('#ffffff')                  # 生成边线

# 旋转动画
for i in range(360):
    camera.clear()
    camera.camera_position = [150 * math.cos(math.radians(i)),
                               150 * math.sin(math.radians(i)),
                               150]
    camera.to_target([0, 0, 0])
    camera.draw_from_scene(scene.sort_all_avg(camera.camera_position))
    camera.update()
```

### 3D 函数图像

```python
import turtleGL

camera = turtleGL.camera('Plot3D Example')
camera.type = 0  # 斜二侧模式

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

### 体积计算

```python
import turtleGL

camera = turtleGL.camera()
camera.camera_position = [10, 120, 10]
camera.to_target([0, 0, 0])
camera.camera_focal = 500
camera.type = 1
camera.rend = 1

scene = turtleGL.scene()
scene.import_obj('model.obj', 50, '#66ccff')
scene.check_obj_norm('model.obj')
scene.triangulation()       # 体积计算需要三角化
scene.generate_line('#ff0000')

volume = turtleGL.volume()
volume.sample_distance = 5   # 采样距离
volume.check = True          # 多向检查
volume.allow_edge = True     # 允许边界交点
volume.volume(scene.face)    # 计算体积

camera.draw_from_scene(scene.sort_line_avg(camera.camera_position))
for i in volume.points:
    camera.dot(i)
camera.done()
```

### 贴图 & 图像导出

```python
import turtleGL

camera = turtleGL.camera('Texture Example')
camera.camera_position = [201, 201, 131]
camera.to_target([0, 0, 50])
camera.camera_focal = 500
camera.type = 1
camera.rend = 1

scene = turtleGL.scene()
scene.tex = [
    [[[50, 50, 100], [-50, 50, 100], [-50, -50, 100], [50, -50, 100]], 'grass_up.png'],
    [[[50, 50, 0], [-50, 50, 0], [-50, -50, 0], [50, -50, 0]], 'grass_bottom.png'],
]

# 使用 OpenCV 渲染并导出
camera.image_size = [700, 700]
camera.create_image('#ffffff')
camera.draw_from_scene_cv2(scene.sort_all_avg(camera.camera_position))
camera.imshow()       # 显示
camera.imwrite('output.png')  # 保存图像
```

---

## 📖 API Reference

### Camera 对象

摄像头对象负责 3D 到 2D 的投影与渲染。

#### 属性

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `camera_position` | `[x,y,z]` | `[0,0,0]` | 摄像头位置 |
| `camera_direction` | `[x,y,z]` | `[0,0,1]` | 摄像头朝向 |
| `camera_rotation` | `float` | `0` | 摄像头旋转角（弧度），体现为左右倾斜 |
| `camera_focal` | `float` | `1` | 焦距 |
| `point_behind_cam_type` | `int` | `0` | 相机背侧点处理方式（0/1/2/3） |
| `point_behind_cam_allow_count` | `int` | `0` | 材质绘制中允许的背侧点数量 |
| `ray` | `[x,y,z]` | `[0,0,-1]` | 光线方向，用于阴影计算 |
| `rend` | `int` | `0` | 渲染类型：0 材质预览 / 1 阴影模式 / 2 法线预览 |
| `shade_value` | `int` | `128` | 正片叠底系数（0-255） |
| `pensize` | `int` | `2` | 画笔大小（仅对线生效） |
| `pencolor` | `str` | `'#000000'` | 画笔颜色（仅对线生效） |
| `type` | `int` | `1` | 相机类型：0 斜二侧 / 1 透视 / 2 等距 / -1 正交 |
| `grating_size` | `[x,y]` | `[500,400]` | 光栅渲染区尺寸 |
| `grating_length` | `int` | `1` | 光栅采样步长 |
| `image_size` | `[x,y]` | `[500,400]` | 导出图像尺寸 |
| `image` | `ndarray` | `[]` | 当前存储的 OpenCV 图像 |

#### 相机背侧点处理

| 模式 | 说明 |
|------|------|
| `0` | 不做处理，背面点会出现在相反方向 |
| `1` | 翻转 UV，将点转回正常方向 |
| `2` | 改用正交模式，偏差较小 |
| `3` | 使用倍数正交透视 |

#### 渲染模式

| 模式 | 说明 |
|------|------|
| `0` — 材质预览 | 面直接显示指定颜色 |
| `1` — 阴影模式 | 根据光线方向和法线夹角计算阴影，背光面使用正片叠底系数 |
| `2` — 法线模式 | 法线方向与摄像头方向余弦 > 0 显示蓝色，否则显示红色 |

#### 方法

```python
# 基础设置
setposition([x,y,z])          # 设置摄像头位置
setdirection([x,y,z])         # 设置摄像头方向
setfocal(x)                   # 设置焦距
settype(x)                    # 设置投影类型（支持 'focal'/'cabin'/'isometric'）
to_target([x,y,z])            # 设置摄像头面向目标点
status()                      # 输出当前摄像头属性

# 坐标映射
pointfocal([x,y,z])           # 透视模式：3D → 2D 映射，返回 [[u,v], bool]
pointcabinet([x,y,z])         # 斜二侧模式：3D → 2D 映射
pointisometric([x,y,z])       # 等距模式：3D → 2D 映射
pointorthografic([x,y,z])     # 正交模式：3D → 2D 映射
pointfocal_inverse([u,v])     # 透视模式：2D → 3D 逆映射

# 绘制（turtle 渲染）
dot([x,y,z], color)           # 绘制单点
drawline(linedata)            # 绘制边
drawface(facedata)            # 绘制面
drawtex(facedata)             # 绘制贴图
draw_from_scene(scenedata)    # 绘制整合数据
draw_axis(l)                  # 绘制坐标轴
write(point, str)             # 在 3D 位置写入文字

# 绘制（OpenCV 渲染）
drawline_cv2(linedata)        # OpenCV 绘制边
drawface_cv2(facedata)        # OpenCV 绘制面
drawtex_cv2(facedata)         # OpenCV 绘制贴图
draw_from_scene_cv2(scenedata)# OpenCV 绘制整合数据

# 图像导出
create_image(bgcolor)         # 初始化图像（可反复调用实现清屏）
imshow()                      # 显示当前图像
imwrite(path)                 # 保存图像到文件
capture(path, index)          # 按序号截图（用于视频帧）
to_video(path, fps=30)        # 合成视频

# 光栅算法（实验性）
grating(face)                 # 光栅算法计算
grating_cv2(face)             # OpenCV 光栅算法
show_grating_limit()          # 显示渲染区边缘

# 工具
tracer(t)                     # 控制动画开关
delay()                       # 延时
clear()                       # 清除画布
bgcolor(color)                # 设置背景色
update()                      # 更新画布
done()                        # 阻止窗口自动关闭
```

---

### Scene 对象

场景对象存储线、面、贴图数据，支持空间变换与深度排序。

#### 数据格式

```python
# 边数据
[[[x1,y1,z1], [x2,y2,z2]], '#RRGGBB']

# 面数据（逆时针方向输入点，法线射出方向为正）
[[[x1,y1,z1], [x2,y2,z2], ..., [xn,yn,zn]], '#RRGGBB']

# 贴图数据（4个顶点，颜色替换为图像路径）
[[[x1,y1,z1], [x2,y2,z2], [x3,y3,z3], [x4,y4,z4]], 'texture.png']
```

#### 属性

| 属性 | 说明 |
|------|------|
| `line` | 边数据列表 |
| `face` | 面数据列表 |
| `tex` | 贴图数据列表 |
| `center` | 场景中心点 |

#### 方法

```python
# 添加数据
addline([[x1,y1,z1],[x2,y2,z2],'#color'])   # 添加边
addface([[x1,y1,z1],...,[xn,yn,zn],'#color'])# 添加面

# 导入/导出
import_line(path)              # 导入线数据 (CSV)
import_face(path)              # 导入面数据 (CSV)
export_line(path)              # 导出线数据 (CSV)
export_face(path)              # 导出面数据 (CSV)

# OBJ 模型
import_obj(path, scale, color) # 导入 OBJ 模型（颜色为空时随机上色）
import_obj_normal(path)        # 导入 OBJ 法线数据
check_obj_norm(path)           # 按法线信息修正面朝向
add_obj(filepath, scale, color)# 导入 OBJ 并追加到现有面数据

# 空间变换
rotate(rotate_vector, center)  # 绕中心旋转（旋转向量，弧度）
move(move_vector)              # 平移
scale(scale_vector, center)    # 缩放
rotate_edge()                  # 循环边（改变三角化划分）

# 深度排序 — 透视/正交模式
sort_line_avg(camera_pos)      # 排序边（修改对象属性并返回）
sort_face_avg(camera_pos)      # 排序面
sort_tex_avg(camera_pos)       # 排序贴图
sort_all_avg(camera_pos)       # 排序全部（返回数据，不修改对象）

# 深度排序 — 斜二侧模式
sort_line_cabin()              # 排序边
sort_face_cabin()              # 排序面
sort_tex_cabin()               # 排序贴图
sort_all_cabin()               # 排序全部

# 深度排序 — 等距模式
sort_line_isometric()          # 排序边
sort_face_isometric()          # 排序面
sort_tex_isometric()           # 排序贴图
sort_all_isometric()           # 排序全部

# 其他
reverse_normvect(i)            # 翻转第 i 个面的法线方向
generate_line(color)           # 根据面数据生成边线
triangulation()                # 面三角化
get_center()                   # 计算并返回场景中心点
```

---

### Plot3D 对象

3D 函数图像绘制对象，与场景对象操作类似，数据生成依赖目标函数。

#### 属性

| 属性 | 默认值 | 说明 |
|------|--------|------|
| `xlim` | `[-10, 10]` | X 轴定义域 |
| `ylim` | `[-10, 10]` | Y 轴定义域 |
| `step` | `1` | 采样步长 |
| `line` | `[]` | 边数据 |
| `face` | `[]` | 面数据 |
| `center` | `[0,0,0]` | 中心点 |

#### 方法

```python
generate_face(func, color=True)  # 生成函数面数据（自动高度着色）
generate_line(func, color)       # 生成函数边数据
rotate(rotate_vector, center)    # 旋转
move(move_vector)                # 平移
scale(scale_vector, center)      # 缩放
# 以及与 Scene 相同的深度排序方法
```

---

### Volume 对象

体积计算对象，使用射线法判断点是否在封闭三角面体内部。

#### 属性

| 属性 | 默认值 | 说明 |
|------|--------|------|
| `points` | `[]` | 计算后的内部点列表 |
| `sample_distance` | `1` | 采样网格间距 |
| `grid_limit` | `inf` | 分区加速算法启用阈值（三角形数量） |
| `check` | `True` | 是否启用多向性检查 |
| `allow_edge` | `True` | 是否允许计入边界交点 |

#### 方法

```python
volume(scene_face_data)  # 计算体积，返回所有内部点
```

> ⚠️ 分区加速算法（`grid_limit`）仅建议在凸多面体下使用，不规则图形可能产生错误结果。

---

## 🧪 Experimental: Rasterization

光栅算法为实验性功能，尚不稳定。

```python
scene.triangulation()             # 面三角化（光栅模式仅支持三角面）
camera.grating_size = [500, 400]  # 设置渲染区尺寸
camera.show_grating_limit()       # 显示渲染区边缘
camera.grating(face)              # 执行光栅计算
```

在渲染模式 `rend=1` 时，光栅模式不再使用阴影计算，而是计算光线路径。

---

## 🎬 Image & Video Export

由于 turtle 本身不支持图像截取，库提供了基于 OpenCV 的图像/视频导出方案：

```python
# 初始化
camera.image_size = [500, 400]
camera.create_image('#ffffff')    # 创建图像（可反复调用实现清屏）

# 渲染
camera.draw_from_scene_cv2(scene_data)  # 使用 OpenCV 渲染

# 导出
camera.imwrite('output.png')     # 保存图像

# 视频制作
camera.capture('frames', i)      # 按序号保存帧 → ./frames/00000001.png
camera.to_video('frames')        # 合成视频 → frames.mp4
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
        │   ├── camera.py      # 摄像头对象
        │   ├── scene.py       # 场景对象
        │   ├── plot3d.py      # 3D 函数图像对象
        │   └── volume.py      # 体积计算对象
        └── example/
```

---

## 📄 License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Homepage**: [https://github.com/MichaelHyan/turtleGL](https://github.com/MichaelHyan/turtleGL)
- **PyPI**: [https://pypi.org/project/TurtleGL-3d/](https://pypi.org/project/TurtleGL-3d/)
