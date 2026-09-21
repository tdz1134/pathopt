"""algorithms 包 —— 一个算法一个文件，导入本子包即触发全部算法注册。

新增算法步骤：
  1. 在 algorithms/ 下新建 <算法名>.py，用 @registry.algorithm("<算法名>") 注册；
  2. 在下方对应类别添加一行 import，并将其加入 _ALGORITHM_MODULES。
"""

# ─── Catmull-Rom 系列 ───
from . import cr_uniform            # noqa: F401
from . import cr_chord              # noqa: F401
from . import cr_centripetal        # noqa: F401

# ─── 样条插值 ───
from . import akima                 # noqa: F401
from . import natural_cubic         # noqa: F401
from . import pchip                 # noqa: F401
from . import quintic_hermite       # noqa: F401
from . import rbf_interp            # noqa: F401

# ─── B 样条 ───
from . import bspline_uniform_cubic   # noqa: F401
from . import bspline_chord_cubic     # noqa: F401
from . import bspline_approx          # noqa: F401
from . import reinsch_smooth_spline   # noqa: F401
from . import quadratic_bspline_chaikin  # noqa: F401

# ─── 贝塞尔 ───
from . import bezier_chain          # noqa: F401
from . import bezier_ls_fit         # noqa: F401
from . import bezier_corner_g2      # noqa: F401

# ─── 曲率连续 / 螺旋 ───
from . import clothoid              # noqa: F401
from . import polynomial_spiral     # noqa: F401

# ─── 多项式 / 滤波 / baseline ───
from . import polynomial_global     # noqa: F401
from . import savgol_smooth         # noqa: F401
from . import linear                # noqa: F401

# ─── 优化 / 能量最小 ───
from . import min_curvature_energy  # noqa: F401
from . import elastic_band          # noqa: F401
from . import min_acceleration      # noqa: F401
from . import min_jerk              # noqa: F401
from . import min_snap              # noqa: F401
from . import discrete_bending_qp   # noqa: F401
from . import laplacian_smooth      # noqa: F401
from . import curve_shortening      # noqa: F401

_ALGORITHM_MODULES = [
    cr_uniform, cr_chord, cr_centripetal,
    akima, natural_cubic, pchip, quintic_hermite, rbf_interp,
    bspline_uniform_cubic, bspline_chord_cubic, bspline_approx,
    reinsch_smooth_spline, quadratic_bspline_chaikin,
    bezier_chain, bezier_ls_fit, bezier_corner_g2,
    clothoid, polynomial_spiral,
    polynomial_global, savgol_smooth, linear,
    min_curvature_energy, elastic_band,
    min_acceleration, min_jerk, min_snap,
    discrete_bending_qp, laplacian_smooth, curve_shortening,
]
