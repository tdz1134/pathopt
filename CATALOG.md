# 算法与场景清单

## 算法

### 插值类 algorithms/interpolation.py

cr_uniform 均匀参数化Catmull-Rom C1
cr_chord 弦长参数化Catmull-Rom C1 防末段曲率爆炸
cr_centripetal 向心参数化CR alpha=0.5 防大间距自交
akima Akima插值 C1 抗过冲
natural_cubic 自然三次样条 C2 曲率连续
pchip PCHIP单调插值 C1 保形无过冲
clothoid 缓和曲线样条 G2 曲率沿弧长线性

### 优化类 algorithms/optimization.py

min_curvature_energy 最小曲率能量 L-BFGS-B
elastic_band 弹性带 路径长度+曲率+偏离惩罚

## 场景

### 基础

gentle_curve 均匀微弯 5点
s_curve S形曲线 6点
dense_hops 均匀密集点 8点

### 压力

sharp_corner 直角急弯 5点
uneven_spacing 间距悬殊前密后疏 5点
noisy_points 带噪声控制点 5点
short_last_segment 前段等距末段极短 5点
hairpin 回头弯U-turn 6点
zigzag 锯齿形 6点

### 退化

two_points 仅两点 2点

## 指标

length 路径总弧长
smoothness 曲率变化率L1范数
max_curvature 最大曲率绝对值
curvature_var 曲率方差
max_deviation 路径偏离控制点最大距离
lat_accel_cost 横向加速度代价

## Tags

basic gentle_curve s_curve dense_hops
smooth gentle_curve s_curve
stress sharp_corner uneven_spacing noisy_points short_last_segment zigzag
corner sharp_corner hairpin
uneven uneven_spacing short_last_segment
extreme hairpin zigzag
noise noisy_points
degenerate two_points
dense dense_hops
