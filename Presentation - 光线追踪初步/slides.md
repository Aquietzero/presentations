---
# try also 'default' to start simple
theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://source.unsplash.com/collection/94734566/1920x1080
# apply any windi css classes to the current slide
class: 'text-center'
# https://sli.dev/custom/highlighters.html
highlighter: shiki
# show line numbers in code blocks
lineNumbers: true 
# persist drawings in exports and build
drawings:
  persist: false
# page transition
transition: slide-left
# use UnoCSS
css: unocss
---

<style>
.slidev-layout {
  font-family: sans-serif;
}

.slidev-layout h1 {
  color: #005cc5 !important;
  font-weight: bold;
  border-bottom: solid 2px #005cc5;
  width: fit-content;
}
.slidev-layout .my-auto h1 {
  color: black !important;
  border-bottom: none;
  width: auto;
}
.slidev-layout h1 + p {
  opacity: 1 !important;
  padding-top: 20px;
}
.col-left {
}
.col-right {
  padding-left: 25px;
  display: flex;
  justify-content: center;
  flex-direction: column;
}

li code,
p code {
  background-color: #E9F3F7 !important;
  color: #005cc5;
}

strong {
  color: #005cc5 !important;
}

.katex-display {
  margin: 0 auto !important;
}
</style>

# 光线追踪初步

bifnudozhao@tencent.com

---
layout: two-cols
---

# 目录

- 综述
- 数学基础
- 获取光线
  - 渲染模型
  - 相机几何
- 获取光线颜色
  - 光线的传播行为
  - 可击中物体及其材质
- 整体架构

::right::

<div class="w-full h-full flex flex-col items-center justify-center">
  <div>
    <img src="/book-cover.jpg" class="h-80 rounded shadow" />
  </div>
  <p class="text-center">
    Ray Tracing in One Weekend
  </p>
</div>

---
layout: two-cols-header
---

# 综述

::left::

<img src="/exp1.png" class="w-90" style="margin: auto" />
<div style="height: 10px" />
<img src="/exp2.png" class="w-90" style="margin: auto" />

::right::

- 背景
  - 光线追踪是一种计算机图形学技术，用于模拟光在场景中的传播和交互。
  - 它通过追踪光线的路径来生成逼真的图像。

- 使用场景
  - 渲染真实感图像：光线追踪可以生成高度逼真的图像，包括光照、阴影、反射和折射效果。
  - 电影和动画制作：许多电影和动画制作公司使用光线追踪来创建逼真的特效和场景。
  - 游戏开发：光线追踪在游戏开发中越来越受欢迎，用于实现更真实的光照和阴影效果。

---
---

# 综述

<img src="/final-result.png" class="h-100" style="margin: auto" />


---
---

# 综述

**光线追踪**算是一种**离线算法**，因其巨大的计算量，并不适合实时计算（至少在我目前看到的来说）这意味着光线追踪产生的结果是一张**图片**。而不像其他实时算法一样，计算得到的是某个时刻的世界的信息（比如物体位置，运动速度，光线颜色等）。

### 主要思路

- 要得到一张图片，需要确定这张图片每个像素的颜色。
- 光线追踪得名于对光线的追踪，像素的颜色实际上是某一条光线经过各种反射，折射等运动与变换，最终击中这个像素点
- 沿着这条光线反向追踪，通过综合这条光线经过的各种环境与材质参与的颜色混合，得到其最终颜色，而这个颜色就是这个像素点的颜色。


---
layout: two-cols
---

# 综述

```ts
const render = (width: number, height: number) => {
	const image = new Image()
	
	// 逐行循环
	for (let row = 0; row < height; row++) {
		// 逐列循环
		for (let col = 0; col < width; col++) {
			// 当前需要渲染的像素
			const pixel = new Vector2(row, col)
			// 计算从 origin 到该像素世界坐标的光线
			const ray = getRay(pixel)
			// 计算光线的颜色
			const color = getColor(ray)
			// 设置颜色
			image.setColor(pixel, color)
		}
	}
	
	return image
}
```

::right::

从上面的主要流程可以看到，光线追踪的核心逻辑是获取光线的颜色，整个流程最关键只有两步：

1. 获取光线
2. 获取光线的颜色

下面详细阐述如何获取光线，以及如何获取光线的颜色。而整个光线追踪的学问，可以说均包含在这两个步骤当中。

对里面各个环节的细节进行不同的设计和考虑，就得到不同的渲染结果。

---
---

# 获取光线 - 渲染模型


<img src="/render-model.png" class="h-70" style="margin: auto" />


从高层次来看，产物得到的图片，是对世界某一个瞬间的记录，对于静态环境来说，这个瞬间实际上就是光线的集合在一个**平面**上的体现。

对于人眼来说，这个“平面”是我们的眼睛，眼球上满布的视锥细胞将光转化为颜色。对于相机来说，这个“平面”是成像平面，对于旧式相机来说，光的信息留存在了胶片上。

---
layout: two-cols-header
---

# 数学基础

::left::

<img src="/ray.png" class="h-40" />

光线是一条**射线**，射线的参数方程如下

$$
\bold{R}(t) = \bold{P} + t\bold{d}
$$

其中 $\bold{P}$ 是光线的起点，$\bold{d}$ 是光线的方向，$t$ 是参数，$\bold{R}(t)$ 给出了光线在 $t$ 下的坐标。


::right::

要将一个向量变为长度为 $1$，**但方向不变**，这个过程叫做向量的**标准化**。

<img src="/ray-normalization.png" class="w-70" />

向量的标量乘法不改变方向，改变长度，所以用向量除以其长度就能标准化。后面所说的**法向量**，大部分时候都需要进行标准化

$$
\text{normalize}(\bold{R})
= \frac{\bold{R}}{|\bold{R}|}
= \frac{\bold{R}}{\sqrt{\bold{R}_x^2 + \bold{R_y}_x^2}}
$$

---
layout: two-cols-header
---

# 数学基础

::left::

两个向量的**点积**是一个标量，其值为

<img src="/dot-product.png" class="w-50" style="margin: auto" />

$$
\bold{a} \cdot \bold{b} = |\bold{a}||\bold{b}|\cos \theta
$$

其中 $|\bold{a}|\cos \theta$ 部分，实际上是 $\bold{a}$ 在 $\bold{b}$ 上的投影向量。

::right::

<div style="margin-top: -70px" />

<img src="/cross-product.png" class="w-60" style="margin: auto" />

两个向量的**叉乘**是一个向量，它的方向是垂直于两个向量的方向，它的值是

$$
|\bold{a} \cdot \bold{b}| = |\bold{a}||\bold{b}|\sin \theta
$$

所以乘以 $\bold{a}$ 与 $\bold{b}$ 所在平面的法向量 $\bold{n}$，就得到叉乘的结果

$$
\bold{a} \cdot \bold{b} = |\bold{a}||\bold{b}|\sin \theta \bold{n}
$$

---
layout: two-cols-header
---

# 获取光线 - 渲染模型

::left::

回想中学时代的物理，我们看到的物体的颜色，实际上是光射向物体之后反射进入我们眼睛的颜色，简单来说，这个颜色就是由物体的表面性质决定的。

<img src="/render-by-light.png" class="w-90" />

不过实际上情况会复杂很多，因为光经过各种反射会有能量损耗，在不同物体表面反射的时候也有不同的行为。

::right::

但由于我们主流程是渲染一张图片，逐行逐列地确定每个像素的颜色，所以实际上我们是在**追踪这些入射光线**。

<img src="/ray-tracing.png" class="w-90" />

---
layout: two-cols-header
---

# 获取光线 - 相机几何 - 相机变量

如上所述，相机几何处理的问题是计算“相机射向成像平面某一个像素的光线”，由于对于成像平面来说，我们是一个个像素点来反向追踪光线的，所以只要我们知道成像平面左下角（理论上可以选择其他点，选左下角只是一个习惯性约定）的坐标，这样我们通过加上实际像素对应成像平面左下角的偏移，来得到射向该像素的光线。

::left::

由上面的讨论可以得到相机需要记录以下参数。

- `origin` ：相机位置
- `lowerLeftCorner` ：成像平面左下角坐标
- `horizontal` ：成像平面水平向量
- `vertical` ：成像平面竖直向量

有了这些变量，我们可以简单计算这条射向某个像素的光线。为了方便扩展，像素坐标不用整数 $[x, y]$ 来表示，而是**使用水平偏移量百分比 $s$ 与竖直方向偏移百分比 $t$ 来表示**，这样成像平面中心表示为 $[0.5, 0.5]$。下面是获取光线的实现。

::right::

<img src="/screen-coordinate.png" class="h-70" />

---
layout: two-cols-header
---

# 获取光线 - 相机几何 - 相机参数

上面的相机变量是跟相机相关的**位置参数**，而相机还有另一组参数，更像是相机的**固有属性**，比如相机的广角大小，比如相机成像平面的长宽比等，相机变量实际上是需要依赖这些参数计算出来的。

::left::

<img src="/camera-params.png" class="h-70" />

::right::

为了简化模型，将相机焦点与成像平面之间的距离设置为 1，也就是焦距为 1，这个值可以是任意值，对应的计算发生相应变化即可。

由示意图可以计算出成像平面的高度，再乘以成像平面宽高比 `aspectRatio` （我们常说的16:9），则可以得到成像平面的宽度。

---
layout: two-cols-header
---

# 获取光线 - 相机几何 - 相机参数

由于相机是可以任意转向的，所以必须把相机位置，相机朝向统一考虑，才能计算成像平面左下角的坐标。

::left::

<img src="/camera-rotation.png" class="h-80" />

::right::

`lookFrom` 是相机位置，`lookAt` 是照相机向着的物体位置。$w$ 是相机朝向方向的反方向，$vup$ 是竖直向上的方向。可以通过叉乘得到与两个向量垂直的向量，为了计算方便，另 $w, v, u$ 都是单位向量，于是有

$$
\begin{align*}
u &= \text{unitVector}( w \times  vup )\\
v &= \text{unitVector}(w \times u)
\end{align*}
$$

由此得到相机的代码。


---
---

# 获取光线 - 相机几何 - 相机参数

```ts
class Camera {
	origin: Point3;
	lowerLeftCorner: Point3;
	horizontal: Vector3;
	vertical: Vector3;
	
	constructor(lookFrom: Point3, lookAt: Point3, vup: Vector3, vfov: number, aspectRatio: number) {
		const theta = degreesToRadians(vfov);
		const h = tan(theta/2);
		const viewportHeight = 2.0 * h;
		const viewportWidth = aspectRatio * viewportHeight;
		
		const w = unitVector(lookFrom - lookAt);
		const u = unitVector(cross(vup, w));
		const v = cross(w, u);
		
		this.origin = lookFrom;
		this.horizontal = viewportWidth * u;
		this.vertical = viewportHeight * v;
		this.lowerLeftCorner = origin - horizontal/2 - vertical/2 - w;
	}
}
```

---
layout: two-cols-header
---

# 获取光线 - 相机几何 - 发射光线

有了相机几何基础，我们就可以从相机向屏幕的某个像素点发射一条光线。

<div style="height: 30px" />

::left::

<img src="/get-ray.png" class="w-100" />

::right::

```ts
class Camera {
  const getRay(s: number, t: number): Ray => {
    return new Ray(
      this.origin,
      this.lowerLeftCorner
        + s*this.horizontal
        + t*this.vertical	
        - this.origin
    )
  }
}
```


---
---

# 获取光线颜色

要确定像素的颜色，实际上是反向追踪从相机发出的光线，观察其击中的物体（或者无击中）的性质，累积这些性质对光线产生的影响，从而得到光线最终到达成像平面的颜色。

在初中物理中，我们知道太阳光是可以分解为不同的颜色，我们也知道光的三原色，白色的太阳光照射到物体上，有部分光被吸收，部分被反射或者折射，所以在光传播的过程中，每击中一个物体，其能量就会减弱，被反射或者折射的光继续传播，然后击中下一个物体，如此重复，直到击中相机的成像平面。

所以需要获取光线的颜色，有两部分需要计算

- **光线的传播行为**：漫反射，镜面反射，折射，能量衰减…
- **可击中的物体及其材质**：漫反射材质，镜面反射材质，折射材质…

---
layout: two-cols-header
---

# 获取光线颜色 - 光线的传播行为

越真实的物理效果，会使用更多样的光线传播方式，会采用更精确的更符合物理规律的方式计算，比如将光视作光谱，一个连续的波。不过本次分享作为光线追踪的初步，只将光视作一种颜色，一个 $[r, g, b]$ 三元组。

<div style="height: 30px" />

::left::

<img src="/ray-behavior.png" class="w-100" />

::right::

光传播的时候是一个递归过程，从发射点开始，击中物体，然后改变方向继续传播，然后击中物体，最后反向到达光源。我们先定义一个击中记录结构 `HitRecord`，用于记录击中点的性质。

```ts
interface HitRecord {
  p: Point;           // 击中点
  normal: Vector3;    // 击中点的表面法向量
  t: number;          // 光线参数方程的解
  material: Material; // 击中点所在物体的材质
  frontFace: boolean; // 击中点的表面是否向着光来的方向
}
```

---
---

# 获取光线颜色 - 光线的传播行为

整个反射的递归过程可以描述如下

```ts
const rayColor = (ray: Ray, world: Hittable, depth: number) => {
  // 递归达到最大深度限制，就返回黑色
  if (depth <= 0) return new Color(0, 0, 0)
  
  // 计算光线能击中的最近的物体
  const hitRecord: HitRecord = world.hit(ray, 0.001, Infinity)
  
  // 如果有击中，计算散射光线
  if (hitRecord) {
    const { scatterd as Ray, attenuation as Color } = hitRecord.material.scatter(ray, hitRecord)
    // 如果有散射（漫反射，镜面反射，折射）
    if (scattered && attenuation) {
      // 递归调用（不断传播）
      return attenuation * rayColor(scattered, world, depth - 1)
    }
    return Color(0, 0, 0)
  }
  // 没击中任何物体，返回背景色
  return backgroundColor
}
```

---
---

# 获取光线颜色 - 光线的传播行为 - 漫反射

日常中看到的大部分物体都是漫反射物体，这类物体表面是粗糙的，不透明的，我们看到它们的颜色，是由它们反射最多的光来确定的。

<img src="/diffusion.png" class="h-80" />

上图是慢反射示意图。我们看到一个慢反射物体的颜色，是由其反射的最多的光的颜色来决定的。

---
layout: two-cols-header
---

# 获取光线颜色 - 光线的传播行为 - 漫反射


::left::

在漫反射中，为了模拟物体凹凸不平的样子，光线不会严格遵循入射角等于反射角，而是进行一个微小程度的随机。

<img src="/diffusion-model.png" class="h-65" style="margin: auto" />

$$
\bold{r'} = \color{blue}\bold{r} \color{black} + \bold{P} + \color{red}\bold{N} \color{black}+ \color{green}\bold{s}
$$

::right::

图中入射光线是 $\bold{r}$，击中点的法向量是 $\bold{N}$，为了模拟这个微小的随机，在单位圆内随机生成一个小向量 $\bold{s}$，代码实现如下

```ts
const scatter = (r: Ray, hitRecord: HitRecord) => {
	// 漫散射光线
	const scattered =
    r +
    hitRecord.p +
    hitRecord.normal +
    randomVectorInSphere()

	// 物体颜色（红色）
	const color = new Color(255, 0, 0)
	
	return {
		scattered,
		attenuation: color,
	}
}
```

---
---

# 获取光线颜色 - 光线的传播行为 - 漫反射

<img src="/diffusion-result.png" class="h-100" style="margin: auto" />

---
layout: two-cols-header
---

# 获取光线颜色 - 光线的传播行为 - 镜面反射

::left::

在镜面反射里，光线完全遵从入射角等于反射角。

<img src="/reflection.png" class="h-50" style="margin: auto" />

其中 $\bold{s}$ 的方向与击中点的法向量 $\bold{N}$ 方向相同，其值为入射光线 $\bold{r}$ 在 $\bold{N}$ 方向上的投影，通过点乘可以算出。由此得到反射光线的方程为

$$
\bold{r'} = \color{blue}\bold{r} \color{black} + 2 \color{green}\bold{s} = \color{blue}\bold{r} \color{black} + 2 |\color{blue}\bold{r} \color{black} \cdot \color{red}\bold{N} \color{black}| \color{red}\bold{N}
$$

::right::

代码实现如下

```ts
const scatter = (r: Ray, hitRecord: HitRecord) => {
  const N = hitRecord.normal
	// 镜面反射光线
	const scattered = r + 2 * dot(r, N) * N 
	// 物体颜色（红色）
	const color = new Color(255, 0, 0)
	
	return {
		scattered,
		attenuation: color,
	}
}
```


---
---

# 获取光线颜色 - 光线的传播行为 - 镜面反射

<img src="/metal-result.png" class="h-100" style="margin: auto" />

---
layout: two-cols-header
---

# 获取光线颜色 - 光线的传播行为 - 折射

::left::

对于绝缘体，比如水，比如空气来说，会发生折射现象。光线射向这些绝缘体，会直接穿透，并且改变一定角度继续前进，这个改变的角度遵循 Snell 定理。

<img src="/refraction.png" class="h-50" style="margin: auto" />

Snell 定理如下

$$
\eta \cdot \sin \theta = \eta' \cdot \sin \theta '
$$

::right::


为了得到折射光线的方向，我们要计算 $\theta'$

$$
\sin \theta' = \frac{\eta}{\eta'} \cdot \sin \theta
$$

夹角为 $\theta'$，我们可以在法向量平衡，以及法向量垂直两个方向对 $\bold{R}'$ 进行分解，得到

$$
\bold{R}' = \bold{R}'_{\perp} + \bold{R}'_{\shortparallel}
$$

解一下方程可得

$$
\begin{align*}
\bold{R}'_{\perp} &= \frac{\eta}{\eta'}(\bold{R} + \cos \theta \bold{n}) \\
\bold{R}'_\shortparallel &= -\sqrt{1 - |\bold{R}'_\perp|^2 \bold{n}}
\end{align*}
$$

---
layout: two-cols-header
---

# 获取光线颜色 - 光线的传播行为 - 折射

::left::

解一下方程可得

$$
\begin{align*}
\bold{R}'_{\perp} &= \frac{\eta}{\eta'}(\bold{R} + \cos \theta \bold{n}) \\
\bold{R}'_\shortparallel &= -\sqrt{1 - |\bold{R}'_\perp|^2 \bold{n}}
\end{align*}
$$

上式的未知数为 $\cos \theta$，由余弦定理可知 $\bold{a} \cdot \bold{b} = |\bold{a}||\bold{b}| \cos \theta$，如果两个向量都是单位向量，那么 $\bold{a} \cdot \bold{b} = \cos \theta$ 。于是可以将 $\bold{R}'_\perp$ 写为

$$
\bold{R}'_\perp = \frac{\eta}{\eta'}(\bold{R} + (-\bold{R} \cdot \bold{n}) \bold{n})
$$

::right::

下面是代码实现

```tsx
const scatter = (r: Ray, hitRecord: HitRecord) => {
  const N = hitRecord.normal
  const n = N.normalize()
  const uv = r.normalize()
  const cosTheta = dot(-uv, n)
  
	// 折射光线垂直分量
	const rPerp = etaRatio * (uv + cosTheta * n)
	// 折射光线平行分量
	const rParallel = -Math.sqrt(
    1 - rPerp.lengthSquared() * n)
	// 折射光线
	const scattered = rPerp + rParallel
	
	// 折射不衰减，也就是不带上自己的颜色（透明）
	const color = new Color(1, 1, 1)
	
	return { scattered, attenuation: color }
}
```

---
---

# 获取光线颜色 - 光线的传播行为 - 折射

<img src="/refraction-result.png" class="h-100" style="margin: auto" />

---
layout: two-cols-header
---

# 可击中物体及其材质

在上面光的传播一节里，已经给出了如何递归地“回溯”光线的流程，里面涉及到一个比较重要的步骤 `world.hit`，这个步骤是为了找到射出的光是否击中物体，以及击中点的具体信息，而击中点的信息则是上面定义的 `HitRecord` 结构，里面记录了击中点的位置，法向量，击中物的材质等。

::left::

<img src="/hit-or-miss.png" class="h-60" style="margin: auto" />

所以整个步骤的关键在于，如何找到击中的物体，以及计算击中点的位置。

::right::

```ts {4}
interface HitRecord {
  p: Point;           // 击中点
  normal: Vector3;    // 击中点的表面法向量
  t: number;          // 光线参数方程的解
  material: Material; // 击中点所在物体的材质
  frontFace: boolean; // 击中点的表面是否向着光来的方向
}
```

---
layout: two-cols-header
---

# 可击中物体及其材质 - 可击中集合

::left::

```ts {9-18}
class Hittables {
  objects: Hittable[] = [];
  
  hit(ray: Ray, tMin: number, tMax: number) {
    const hitRecord: HitRecord
    const hitAnything: boolean = false
    const closestSoFar = tMax
  
    _.each(this.objects, object => {
      const res = object.hit(ray, tMin, closestSoFar))
    
      // 如果和某个物体发生了碰撞，记录碰撞信息
      if (res.hitAnything) {
        hitAnything = true
        closestSoFar = res.record.t
        hitRecord = res.record
      }
    })
  
    return { hitAnything, hitRecord }
  }
}
```

::right::

从一个简单而抽象的思维考虑这个问题：

- 将世界看成**一组可击中的物体**
- **每个物体都可以单独计算是否被击中**
- 物体也是可以嵌套的

需要注意到的是，每次碰撞检测之后，会记录解 `record.t` ，并且将这个解当做是目前“最大的解”。

<img src="/solutions.png" class="h-50" style="margin: auto" />

---
---

# 可击中物体及其材质 - 可击中物体

在上面的代码里，对于每个物体 `Hittable` 来说，需要自行判断射过来的光线是否击中了自己，也就是 `object.hit` 方法，从这个角度来说，代码是很通用的。可以向世界添加各种各样的物体，只要这些物体提供 `hit` 方法即可。

所以我们可以实现各种可击中物体，比如球体，比如立方体，比如复杂模型。对于这些物体来说，击中方法的实现可能如下

- **球体**：求解射线与球的交点，可以直接解方程获得，但是需要判断解是否在 $[t_\text{min}, t_\text{max}]$ 内
- **立方体**：求解射线与立方体的交点与球体无异
- **复杂模型**：对于复杂模型来说，模型是由一堆三角形组成，判断光线是否与模型相交，等于判断光线是否与模型里面的某个（某些）三角形相交

这实际上是碰撞检测的领域，碰撞检测一般分为两个阶段

- **宏观阶段**：判断粗粒度结构是否相交（判断光线与子空间，或者包围盒是否相交）
- **微观阶段**：判断细粒度是否相交，并且求解（判断光线是否与物体的某个面相交）

---
---

# 可击中物体及其材质 - 可击中物体 - 碰撞检测

<iframe src="http://127.0.0.1:9966/#/examples/primitive-tests/IntersectionOfSegmentAndPlaneExample" class="h-100" style="width: 100%; margin: auto" />



---
layout: two-cols-header
---

# 可击中物体及其材质 - 物体的材质

::left::

```ts
class Hittable {
  material: Material
  
  hit(ray: Ray, tMin: number, tMax: number) {
    const hitRecord: HitRecord
    const hitAnything: boolean = false
  
    // 求解过程
    const t = Solve(this, ray)
  
    // 无击中
    if (t < tMin || t > tMax)
      return { hitAnything: false }
  
    // 击中点
    hitRecord.p = ray.at(t)
    // 计算击中点法向量
    hitRecord.normal = this.getNormal(hitRecord.p)
    // 记录材质信息
    hitRecord.material = this.material
  }
}
```

::right::

材质是物体的属性之一，在物体进行碰撞检测的时候，如果存在击中以及有合理解，那么物体的材质会记录在击中记录里。下面是物体（可击中物体）的定义。

注意到，材质信息实际上是击中之后赋值到击中记录里的，这个材质信息会在计算光的后续光线 `scattered` 的时候用到，具体可以参见“光的传播行为”一节。

---
---

# 整体架构

<img src="/architecture.png" class="h-110" style="margin: auto;" />


---
---

# 整体架构 - 最终效果

<img src="/final-result.png" class="h-100" style="margin: auto" />

---
---

# 整体架构 - 其他课题

- 功能
  - 抗锯齿
  - 失焦模糊
  - 动态模糊
  - 光源
- 性能
  - 更有效的光线追踪
  - 复杂度更低的碰撞检测
  - 尽可能地并行

---
---

<div style="display: flex; width: 100%; height: 100%; justify-content: center; align-items: center">
  <h1 style="font-size: 3em;">Q & A</h1>
</div>