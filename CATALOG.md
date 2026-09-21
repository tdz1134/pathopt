# 算法与场景清单

## 算法

一个算法一个文件，位于 `algorithms/<算法名>.py`，由 `algorithms/__init__.py` 统一导入触发注册。共 29 个。

### Catmull-Rom 系列

cr_uniform 均匀参数化Catmull-Rom C1        -> algorithms/cr_uniform.py
cr_chord 弦长参数化Catmull-Rom C1 防末段曲率爆炸 -> algorithms/cr_chord.py
cr_centripetal 向心参数化CR alpha=0.5 防大间距自交 -> algorithms/cr_centripetal.py

### 样条插值

akima Akima插值 C1 抗过冲                  -> algorithms/akima.py
natural_cubic 自然三次样条 C2 曲率连续      -> algorithms/natural_cubic.py
pchip PCHIP单调插值 C1 保形无过冲          -> algorithms/pchip.py
quintic_hermite 五次Hermite C2 位置+切向+曲率 -> algorithms/quintic_hermite.py
rbf_interp 径向基插值 thin_plate_spline 抗噪 -> algorithms/rbf_interp.py

### B 样条

bspline_uniform_cubic 均匀三次B样条 逼近    -> algorithms/bspline_uniform_cubic.py
bspline_chord_cubic 弦长三次B样条 插值过点  -> algorithms/bspline_chord_cubic.py
bspline_approx splprep最小二乘逼近 平滑因子 -> algorithms/bspline_approx.py
reinsch_smooth_spline Reinsch平滑样条 拟合-光顺权衡 -> algorithms/reinsch_smooth_spline.py
quadratic_bspline_chaikin Chaikin切角=二次B样条 -> algorithms/quadratic_bspline_chaikin.py

### 贝塞尔

bezier_chain 分段三次贝塞尔链 CR切向        -> algorithms/bezier_chain.py
bezier_ls_fit 角点检测分段贝塞尔最小二乘拟合 -> algorithms/bezier_ls_fit.py
bezier_corner_g2 G2连续贝塞尔拐角削角 曲率有界 -> algorithms/bezier_corner_g2.py

### 曲率连续 / 螺旋

clothoid 缓和曲线样条 G2 曲率沿弧长线性     -> algorithms/clothoid.py
polynomial_spiral 多项式螺旋 κ(s)分段三次 G3 Apollo同款 -> algorithms/polynomial_spiral.py

### 多项式 / 滤波 / baseline

polynomial_global 全局5次多项式拟合 龙格反例 -> algorithms/polynomial_global.py
savgol_smooth Savitzky-Golay滤波平滑        -> algorithms/savgol_smooth.py
linear 折线直连 零点参照baseline            -> algorithms/linear.py

### 优化 / 能量最小

min_curvature_energy 最小曲率能量 L-BFGS-B -> algorithms/min_curvature_energy.py
elastic_band 弹性带 路径长度+曲率+偏离惩罚  -> algorithms/elastic_band.py
min_acceleration 最小加速度能量 二阶差分    -> algorithms/min_acceleration.py
min_jerk 最小jerk轨迹 三阶差分 机械臂经典   -> algorithms/min_jerk.py
min_snap 最小snap轨迹 四阶差分 四旋翼经典   -> algorithms/min_snap.py
discrete_bending_qp 弧长加权∫κ²ds IRLS      -> algorithms/discrete_bending_qp.py
laplacian_smooth 拉普拉斯平滑 邻域平均       -> algorithms/laplacian_smooth.py
curve_shortening 曲线缩短流 法向κ演化       -> algorithms/curve_shortening.py

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
