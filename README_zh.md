## turtleGL-3d

### 基于turtle库和赤石技术的3D绘图库

面向对象，直观明了，赤石科技。

### 安装

```
pip install TurtleGL-3d
```

### 数据结构

使用分离的摄像头对象、场景对象实现。

场景对象可将数据存储于自身，可被摄像头对象指定并调用。

可使用多个摄像头，灵活切换。

#### 摄像头属性

可直接为摄像头对象的属性重新赋值。

```python
camera_position#摄像头位置[x,y,z]，正上方向为[0,0,1]
camera_direction#摄像头朝向[x,y,z]
camera_rotation#摄像头旋转角，具体体现为左右倾斜，使用弧度角
camera_focal#摄像头焦距
point_behind_cam_type#背面点处理 0/1/2
ray#光线方向，用于判断面是否面向光源
rend#渲染类型 0 材质预览 1 阴影模式 2 法线预览
shade_value#正片叠底系数 0-255
pensize#画笔大小，只对线生效
pencolor#画笔颜色，只对线生效
type#相机类型 0 斜2侧模式 1 透视模式 2 等距模式 -1 正交模式
```

点映射计算中无法正常计算摄像机背后的点，提供三种处理逻辑：

0: 不做处理，背面点会出现在相反的方向
1: 翻转uv，将点转回正常的方向
2: 改为使用正交模式，偏差相对前两者较小

##### 材质预览模式

面直接显示指定的颜色。

##### 阴影模式

本项目暂不支持稳定的光栅算法，所以不存在光线计算。

启用阴影模式后，面数据将根据光线方向和法线夹角计算是否属于亮面，如否则使用正片叠底系数重计算表面颜色。

##### 法线模式

显示法线，当摄像头方向与面法线余弦值大于0（即为钝角）时判定为正面，显示蓝色，反之显示红色。

##### 图像导出

由于turtle本身不具有图像截取功能，所以可以使用以下方法存储单帧图像。

```python
#截图前绘图区初始化
camera.image_size = [500,400]
camera.create_image('十六进制背景颜色')#可反复调用达到清屏效果

#绘图
camera.draw_from_scene_cv2(场景对象数据)#使用后暂存当前帧的图像

#导出
camera.imwrite('文件名')

#视频处理
#按序号截图
camera.capture('名字',当前序号)#存储到.\文件名\序号:08d.png
#合成视频
camera.to_video('名字')
```

#### 摄像头方法

```python
setposition([x,y,z])#设置摄像头位置，也可直接设置camera_position属性
setdirection([x,y,z])#设置摄像头方向，也可直接设置camera_direction属性
setfocal(x):#设置摄像头焦距，也可直接设置camera_focal属性
settype(x):#设置摄像头类型，也可直接设置type属性
status()#输出当前摄像头属性
tracer(0)#关闭动画，直接操作turtle
to_target([x,y,z])#设置摄像头面向目标点
pointfocal([x,y,z])#透视模式下返回空间内坐标映射至摄像头的坐标
pointcabinet([x,y,z])#斜二侧模式下返回空间内坐标映射至摄像头的坐标
draw_axis(l)#画出基准坐标轴，l调节大小
drawline(linedata)#输入单个边数据，绘制边
drawface(facedata)#输入单个面坐标，绘制面
draw_from_scene(scenedata)#输入整合数据，全部绘制
delay()#延时，与turtle.delay()效果一致
clear()#清除画布，与turtle.clear()效果一致
bgcolor()#画布底色，与turtle.bgcolor()效果一致
update()#更新画布，与turtle.update()效果一致
done()#阻止自动关闭窗口，与turtle.done()效果一致
```

#### 场景属性

场景包含line、face属性，存储边/面数据。

可使用多个场景对象。

##### 数据格式

```bash
[
    [
        [点1],
        [点2],
        ...
        [点n]
    ],
    '十六进制颜色'
]
```

点个数为2时会被识别为边。

使用面数据时以法线射出方向为正，逆时针方向输入点。

场景对象的line、face属性为包含以上数据的数组，每个元素为单独的线/面数据，可直接调用或修改。

#### 场景方法

```python
addline([[x1,y1,z1],[x2,y2,z2],'16进制颜色'])#添加边
addface([[x1,y1,z1],[x2,y2,z2],...,[xn,yn,zn],'16进制颜色'])#添加面
export_line(path)#导出线数据 csv
export_face(path)#导出面数据 csv
import_line(path)#导入线数据
import_face(path)#导入面数据
sort_line_avg([相机坐标x,相机坐标y,相机坐标z])#透视模式或正交模式下调整图层顺序，修改场景对象属性并返回调整后数据
sort_face_avg([相机坐标x,相机坐标y,相机坐标z])#透视模式或正交模式下调整图层顺序，修改场景对象属性并返回调整后数据
sort_all_avg([相机坐标x,相机坐标y,相机坐标z])#返回调整后的所有数据，不修改场景对象属性
sort_line_cabin()#斜二侧模式下调整图层顺序，修改场景对象属性并返回调整后数据
sort_face_cabin()#斜二侧模式下调整图层顺序，修改场景对象属性并返回调整后数据
sort_all_cabin()#返回调整后的所有数据，不修改场景对象属性
sort_line_isometric()#等距模式下调整图层顺序，修改场景对象属性并返回调整后数据
sort_face_isometric()#等距模式下调整图层顺序，修改场景对象属性并返回调整后数据
sort_all_isometric()#返回调整后的所有数据，不修改场景对象属性
reverse_normvect(i)#修改第i个面数据的法向
import_obj(path,缩放系数,颜色)#导入obj模型,颜色为空时随机上色
check_obj_norm(path)#按照obj文件信息修正法向
scene.generate_obj_line(颜色)#根据面数据生成边
```

##### 调整顺序

本项目暂不支持稳定的光栅算法，所以不存在光线计算。

面前后顺序由图层和渲染顺序决定。

##### obj模型

导入后直接作用到scene.face，一次只能导入一个物体，物体名必须为英文。

#### 3D函数图像

和场景对象操作类似，但数据生成依赖目标函数。

包含线、面数据，绘制范围，采样步长。

##### 函数方法

假设使用以下方式定义函数：

```python
def func(x,y):
    #不知道操作了什么
    return z
```

设置定义域(绘制范围)：

```python
scene.xlim = [x1,x2] #表示从x1开始采样直到x2
scene.ylim = [y1,y2] #表示从y1开始采样直到y2
scene.step = d #采样步长
```

生成图像：

```python
scene.generate_face(func)
scene.generate_line(func)
```

其余操作与场景对象一致。

#### 操作流程

1. 实例化摄像头对象
2. 调整镜头属性
3. 实例化场景对象
4. 为场景对象添加数据
5. 场景对象数据排序
6. 使用摄像头对象调用数据
7. 绘制

### 测试功能：光栅算法

该功能尚不稳定，仅参考。

```python
scene.triangulation()#面三角化，目前光栅模式只能处理三角面
camera.grating_size = [x,y]#渲染区尺寸
camera.show_grating_limit()#显示渲染区边缘
camera.grating(face)#使用光栅算法计算
```

在渲染模式为1时不再使用阴影模式，而是计算光线路径。

### 使用方式（示例）

#### 摄像头对象

设置摄像头属性。

```python
camera = turtleGL.camera()#实例化摄像头对象
camera.camera_position = [-101,-121,131]#相机位置
camera.camera_direction = [1,1,-1]#相机方向
camera.to_target([0,0,0])#相机看向目标点
camera.camera_focal = 300#焦距
camera.ray = [1,1,-1]#光照方向
camera.type = 1#1 透视模式  0 斜二侧模式
camera.rend = 1#0 材质预览 1 阴影 2 法线
camera.status()#查看摄像头属性
camera.grating_size = [500,400]#光栅模式 设置渲染区尺寸
camera.image_size = [500,400]#导出图像尺寸
camera.image #当前存储的图像
```

#### 场景对象

使用结构化数据存储场景信息，分为线和面，直接储存在场景对象内，可调用。

```python
scene = turtleGL.scene()#实例化场景
#边信息自定义
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
#面信息自定义
scene.face = [[[[50.0, 50.0, 0.0], [-50.0, 50.0, 0.0], [-50.0, -50.0, 0.0], [50.0, -50.0, 0.0]], '#FF0000'], 
              [[[50.0, 50.0, 0.0], [50.0, 50.0, 100.0], [50.0, -50.0, 100.0], [50.0, -50.0, 0.0]], '#0000FF'], 
              [[[-50.0, 50.0, 0.0], [-50.0, -50.0, 0.0], [-50.0, -50.0, 100.0], [-50.0, 50.0, 100.0]], '#FFFF00'], 
              [[[50.0, 50.0, 0.0], [-50.0, 50.0, 0.0], [-50.0, 50.0, 100.0], [50.0, 50.0, 100.0]], '#FF00FF'], 
              [[[50.0, 50.0, 100.0], [-50.0, 50.0, 100.0], [-50.0, -50.0, 100.0], [50.0, -50.0, 100.0]], '#00FF00'], 
              [[[-50.0, -50.0, 0.0], [50.0, -50.0, 0.0], [50.0, -50.0, 100.0], [-50.0, -50.0, 100.0]], '#00FFFF']]
```

整理显示次序(以透视模式为例)

```python
scene.sort_line_avg(camera.camera_position)#仅边
scene.sort_face_avg(camera.camera_position)#仅面
scene.sort_all_avg(camera.camera_position)#全部整理，直接返回结构化数据，不修改对象
```

绘制
```python
camera.draw_from_scene(scene.sort_all_avg(camera.camera_position))#绘制全部内容
camera.draw_from_scene(scene.line)#绘制 仅边
camera.draw_from_scene(scene.face)#绘制 仅面
camera.draw_axis(10)#显示轴
camera.done()
```

导入/导出
```python
#导出
scene.export_line('example_line.csv')
scene.export_face('example_face.csv')
#导入
scene.import_line('example_line.csv')
scene.import_face('example_face.csv')
```