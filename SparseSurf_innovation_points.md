# SparseSurf 创新点梳理与实现指南

> 本文档基于 `/root/autodl-tmp/SparseSurf` 代码库**实测**整理，所有创新点均给出精确的「文件:行号」定位与可直接移植的代码片段。
> 用途：指导另一个项目复刻 SparseSurf 的核心算法改进。
> 版本：2026-09-05（对应代码库已移除 MonoAnchor 模块，保留 baseline 6 阶段 Floater + Stereo 监督）

---

## 0. 整体框架：六阶段 Floater 处理管线

SparseSurf 在基础 3DGS（L1 + DSSIM + Densify）之上，新增了一条**六阶段漂浮面片（floater）处理管线**，从「预防→检测→软惩罚→硬删→补种→调度」全链路抑制 Gaussian Splatting 训练中产生的漂浮面片：

```
Stage 1  Prevention     空间约束 L_space（基于 COLMAP 初始点云 bbox，防止高斯飞出有效区域）
Stage 2  Detection      多信号像素离群检测 + 高斯属性异常 → 累积到 OutlierTracker.EMA
Stage 3  Soft Penalty   Opacity Decay Loss（梯度只走 opacity，软删除漂浮高斯）
Stage 4  Hard Pruning   三条件硬删 + max_prune_ratio 保护 + 渐进强度
Stage 5  Resampling     (A) Stereo 高置信像素反投影补种 / (B) 2D Neighbor Inpainting 空洞补种
Stage 6  Scheduling     Color Down-weighting（按离群度下调颜色 loss 权重）+ 三 loss 权重调度
```

**配套**：Stereo 监督三项（Depth L1 / Normal Prior / Normal Smooth）提供几何锚点，整个 Floater 管线依赖 Stereo prior 作为深度参考。

---

## 1. 创新点 #1：多信号像素级离群检测（Stage 2）

### 1.1 核心思路
不直接用「渲染深度与 stereo 深度的绝对差」，而是用**相对深度误差 + 动态 MAD 阈值 + 颜色-深度冲突**组合出 0~1 的像素离群分，避免绝对尺度导致的误判。

### 1.2 关键公式
```
# 相对深度误差（除以 stereo 深度，消除尺度依赖）
E_depth(p) = M(p) * |D_render(p) - D_stereo(p)| / (D_stereo(p) + 1e-3)

# 动态阈值（MAD 鲁棒估计，每视图自适应）
median = median(E_depth[valid])
mad    = median(|E_depth[valid] - median|)
sigma  = 1.4826 * mad
tau_low  = median + 2.0 * sigma
tau_high = median + 4.0 * sigma

# smoothstep 软阈值 → 0~1 离群分
S_depth(p) = M(p) * smoothstep(tau_low, tau_high, E_depth(p))

# 颜色-深度冲突（过滤「颜色对但深度错」的假阳性）
color_error = mean(|render_rgb - gt_rgb|)
conflict(p) = S_depth(p) * exp(-color_error / 0.1)

# 加权求和（baseline：深度 60% + 冲突 40%）
pixel_score(p) = (0.60 * S_depth(p) + 0.40 * conflict(p)) / 1.0
```

### 1.3 代码定位
- `utils/floater_utils.py:140-154` — `compute_dynamic_threshold()`：MAD 动态阈值
- `utils/floater_utils.py:157-199` — `compute_pixel_outlier_scores()`：双信号像素离群分
- `utils/floater_utils.py:202-248` — `gaussian_attribute_anomaly()`：高斯属性异常（scale 过大 / 各向异性 / opacity 作弊 / SH 高阶异常，4 项加权后 p99 归一化）

### 1.4 移植要点
- **必须用相对误差**（除以 D_stereo），否则大深度场景整体被误判为离群。
- **MAD 动态阈值**比固定阈值鲁棒；每视图独立算，适应不同距离。
- `smoothstep` 代替 hard threshold，保证梯度平滑。
- 高斯属性异常是「空间信号的补充」（占 15%），不单独使用。

---

## 2. 创新点 #2：跨视图跨迭代 EMA 离群追踪（Stage 2→3 桥接）

### 2.1 核心思路
像素离群分 → 投影回每个高斯 → 与高斯属性异常合并 → 用 **β=0.95 的 EMA** 跨迭代长期累积。关键点：**整个 Stage 2 全部 `torch.no_grad()`**，绝不保留 autograd 边，防止 Stage 3 backward 时「Trying to backward through the graph a second time」崩溃。

### 2.2 关键实现
```python
class OutlierTracker:
    def __init__(self, num_gaussians, beta=0.95):
        self.score_ema = torch.zeros(num_gaussians, device=device)  # 每高斯一个长期离群分
        self.obs_count = torch.zeros(num_gaussians, dtype=torch.long)  # 被观测次数
        self.beta = beta

    def update(self, new_scores, new_counts):
        # 关键：detach，确保 EMA 不持有任何 autograd 图引用
        new_scores = new_scores.detach()
        new_counts = new_counts.detach()
        mask = new_counts > 0
        self.score_ema[mask] = self.beta * self.score_ema[mask] + (1 - self.beta) * new_scores[mask]
        self.obs_count[mask] += new_counts[mask].to(torch.long)

    def resize(self, new_N, keep_mask=None):
        # Densify/Prune 后必须同步调整 tracker 大小，否则 shape mismatch
        if keep_mask is not None:
            self.score_ema = self.score_ema[keep_mask]
            self.obs_count = self.obs_count[keep_mask]
        else:
            extra = new_N - self.score_ema.shape[0]
            if extra > 0:
                self.score_ema = torch.cat([self.score_ema, torch.zeros(extra)])
                self.obs_count = torch.cat([self.obs_count, torch.zeros(extra, dtype=torch.long)])
```

### 2.3 代码定位
- `utils/floater_utils.py:349-398` — `OutlierTracker` 类（init / update / resize）
- `utils/floater_utils.py:291-342` — `attribute_scores_to_gaussians()`：像素分→高斯分的反向投影
- `train.py:Stage 2 段` — `outlier_tracker.update(combined_scores, combined_counts)` 调用点

### 2.4 移植要点
- **必须 detach**：这是整个机制能稳定运行的核心，否则每步 backward 都会重入历史 EMA 图。
- **resize 必须在 densify/prune 后立刻调用**，并传 `keep_mask`（prune）或不传（grow）。
- `obs_count >= 3`（min_observations）是 Stage 3/4 的前置条件，防止单视图误判。

---

## 3. 创新点 #3：Soft Opacity Decay Loss（Stage 3，深度差异透明度惩罚）

### 3.1 核心思路
**这是「深度差异→透明度惩罚」的核心机制**。把 EMA 离群分通过 smoothstep 压成 0~1 的离群置信度 φ（常数，无梯度），然后 `L_decay = mean(φ * opacity)`，梯度只流回 opacity：越离群的高斯 opacity 被压得越低（软删除），正常高斯 φ≈0 不受影响。

### 3.2 关键公式
```
# Stage 3 触发条件：iter >= soft_from_iter (默认 3000)
# 离群置信度 φ（no_grad 常数）
valid[i] = 1[obs_count[i] >= 3]
φ[i] = valid[i] * smoothstep(0.3, 0.7, score_ema[i])   # 0=正常, 1=极可疑

# Soft 透明度惩罚
L_decay = (1/N) * Σ φ[i] * opacity[i]

# 合并进总 loss
loss += lambda_decay_sched(iter) * L_decay
# lambda_decay_sched: 3000 iter 前=0，3000~5000 线性升到 0.05 封顶（可被 opt.lambda_decay_max 覆盖）
```

### 3.3 关键设计：梯度流向
- **梯度只对 `opacity` 生效**：`∂L/∂opacity[i] = φ[i] / N`
  - φ=0（正常高斯）→ 梯度=0，完全不惩罚
  - φ=1（漂浮高斯）→ 梯度=1/N，opacity 被压低（渲染时变透明=软删除）
- **φ / score_ema / combined_scores 全是 no_grad 常数**：Stage 2 的累积逻辑、高斯位置、Stereo 深度完全不受 Stage 3 惩罚影响（防止「为降低惩罚而扭曲表面高斯深度」）。

### 3.4 代码定位
- `utils/floater_utils.py:399-419` — `OutlierTracker.opacity_decay_loss()`
- `train.py:Stage 3 段` — `L_decay = outlier_tracker.opacity_decay_loss(opacities)` 及 `loss += lambda_decay_sched * L_decay`
- `utils/floater_utils.py:23-46` — `get_scheduler_lambdas()`：lambda_decay 调度

### 3.5 移植要点
- **φ 必须用 no_grad**，否则梯度会流回 score_ema 导致 Stage 2 与 Stage 3 形成循环依赖。
- 触发时机建议 3000 iter 之后（前期离群证据不足，惩罚会误伤表面）。
- `min_observations=3` 是关键保护，防止单视图噪声误删。

---

## 4. 创新点 #4：带保护的 Hard Pruning（Stage 4）

### 4.1 核心思路
Stage 3 软删不掉的顽固漂浮物，Stage 4 硬删。三个独立触发条件（OR 关系），但有**两层保护**防止一次性删太多：

### 4.2 三条件
```
Cond1: obs_count >= 5  AND  score_ema > eff_tau     （持续高离群）
Cond2: opacity < 0.005                                （极低透明度=已无贡献）
Cond3: distance_to_valid_region > scene_radius * 0.5  （严重空间越界）
prune_mask = Cond1 | Cond2 | Cond3
```

### 4.3 两层保护
```
# 保护 1：渐进强度（刚进入 Stage 4 时阈值从 0.5 线性升到 tau_hard，避免一次性删太多）
prune_intensity = min(1.0, (iter - start_prune_iter) / 2000.0)
eff_tau = tau_hard - (tau_hard - 0.5) * (1 - prune_intensity)

# 保护 2：max_prune_ratio 上限（单次最多删 N * ratio_eff 个，按 badness 排序保留最坏）
total_pruned = sum(prune_mask)
max_allowed = int(N * ratio_eff)   # ratio_eff 裁剪到 [0.005, 0.25]
if total_pruned > max_allowed:
    badness = score_ema + (1 - opacity) + (dist / dist.max())
    prune_mask[badness 排序后超过 max_allowed 的] = False
```

### 4.4 代码定位
- `utils/floater_utils.py:467-560` — `identify_prune_mask()`（三条件 + 两层保护 + badness 排序）

### 4.5 移植要点
- **badness 排序**比随机裁剪更安全：只删最坏的，保留次坏的给下一轮。
- `ratio_eff` 裁剪到 `[0.005, 0.25]`，防止用户传极端值。
- `start_prune_iter` 默认 5000，必须晚于 Stage 3（3000），让软惩罚先过滤一轮。

---

## 5. 创新点 #5：基于 Stereo 先验的表面重采样（Stage 5A）

### 5.1 核心思路
硬删后表面会出现空洞，用 **Stereo 高置信像素反投影到 3D** 补种新高斯。只取 stereo valid mask + confidence > 0.7 的像素，保证补种点落在真实表面上。

### 5.2 关键流程
```
1. 筛选像素：M = valid_mask AND (confidence > 0.7)
2. 降采样：if |M| > 30000 → 随机取 30000 个
3. 反投影：
   xs = (u - Cx) / Fx,  ys = (v - Cy) / Fy
   pts_cam = [xs*d, ys*d, d]
   pts_world = pts_cam @ R.T + T
4. 新高斯初始化：
   xyz = pts_world + N(0, 0.001)
   scale = 0.005, opacity = 0.3
   color = RGB2SH(gt_rgb_at_pixel)
5. cat_tensors_to_optimizer 加入优化器
```

### 5.3 代码定位
- `utils/floater_utils.py:567-620` — `collect_reliable_surface_points()`：反投影
- `utils/floater_utils.py:623-...` — `prepare_new_gaussian_tensors()`：初始化新高斯参数

### 5.4 移植要点
- **confidence 阈值**是关键，太低会把 noise 当表面补种。
- 新高斯 opacity 初始化 0.3（不 1.0），给后续 Stage 3 惩罚留余地。

---

## 6. 创新点 #6：2D Neighbor Inpainting Resample（Stage 5.5，B 类补种）

### 6.1 核心思路
Stage 4 硬删后投影到 2D 形成「空洞」（hole_mask），在 2D 图像域找到空洞周围的「健康像素」（与 stereo 深度一致性好），用 **IDW（反距离加权）** 插值出空洞像素的深度，再反投影补种。这是对 Stage 5A 的补充——5A 只补 stereo 直接覆盖的区域，5.5 补 stereo 没覆盖但邻居可靠的区域。

### 6.2 关键参数（默认值）
| 参数 | 默认值 | 含义 |
|---|---|---|
| `inpaint_window` | 15 | 健康邻居搜索窗口（像素） |
| `inpaint_hole_dilate_radius` | 3 | 空洞膨胀半径（避免边界噪声） |
| `inpaint_min_healthy` | 8 | 至少需要 8 个健康邻居才插值 |
| `inpaint_depth_err_thr` | 0.02 | 健康像素定义：\|D_render - D_stereo\| / D_stereo < 0.02 |
| `inpaint_idw_p` | 2.0 | IDW 幂次 |
| `inpaint_max_points` | 15000 | 单次最多补种点数 |

### 6.3 代码定位
- `utils/point_utils.py` — `inpaint_and_resample()`：空洞检测 + IDW 插值 + 反投影
- `train.py:Stage 5.5 段` — 调用 `inpaint_and_resample()`

### 6.4 移植要点
- 必须先有 Stage 4 的 `hole_mask`（投影 prune_mask 到 2D）。
- `inpaint_depth_err_thr=0.02` 很严格，保证健康邻居真的在表面上。
- 与 Stage 5A 互补，不重复补种。

---

## 7. 创新点 #7：基于离群评分的 Color Down-weighting（Stage 6）

### 7.1 核心思路
漂浮高斯渲染的像素颜色通常是错的，但如果直接用 L1 loss 训练，模型会努力把漂浮区域颜色调对（反而强化漂浮）。解决方案：**按像素离群度下调颜色 loss 权重**，让模型在漂浮区域「不努力」。

### 7.2 关键公式
```
# 把 per-gaussian EMA 离群分投影回 2D 像素（no_grad）
pixel_outlier = outlier_tracker.project_to_pixels(viewpoint_cam, means, H, W)

# 颜色权重：离群越高权重越低，最低 0.1（不完全置零，保留少量信号）
color_weights = clamp(1.0 - gamma * pixel_outlier, min=0.1).detach()

# 加权 L1
Ll1_weighted = (color_weights * per_pixel_l1).mean()

# gamma 调度：3000 iter 前=0，3000~5000 线性升到 0.5
```

### 7.3 代码定位
- `train.py:Stage 6 Color down-weighting 段`（`color_weights = torch.clamp(1.0 - gamma_sched * pixel_outlier, min=0.1).detach()`）
- `utils/floater_utils.py:421-460` — `OutlierTracker.project_to_pixels()`
- `utils/floater_utils.py:23-46` — `get_scheduler_lambdas()` 中 `gamma` 调度

### 7.4 移植要点
- `color_weights.detach()` 必须 detach，否则梯度会流回 pixel_outlier → score_ema。
- `min=0.1` 不完全置零，防止漂浮区域彻底失去监督。

---

## 8. 创新点 #8：Stereo 监督三项（几何锚点）

### 8.1 核心思路
整个 Floater 管线依赖 Stereo depth prior 作为深度参考。Stereo 监督由三项组成：

| Loss | 公式 | 权重 |
|---|---|---|
| **Depth L1** | `mean(|D_render - D_stereo|) × valid_mask` | `lambda_stereo_depth_sup`（默认 0.05） |
| **Normal Prior** | `1 - cos(rend_normal, stereo_depth_normal)` | `lambda_normal_prior` |
| **Normal Smooth** | `loss_depth_smoothness(rend_normal, stereo_normal) + loss_depth_smoothness(surf_normal, stereo_normal)` | `lambda_normal_smooth` |

### 8.2 触发时机
`iter > stereofrom_iterations`（默认 500）后开启。之前 Stereo prior 还没建立，不参与 loss。

### 8.3 代码定位
- `train.py:Stereo Loss 段` — 三项计算与合并
- `utils/foundationstereo_utils.py:141-166` — `predict_disparity()` / `disp2depth()`：FoundationStereo 推理
- `utils/foundationstereo_utils.py:202-228` — `left_right_check()`：左右视差一致性检查生成 valid mask
- `utils/loss_utils.py:81-93` — `loss_depth_smoothness()`：法向/深度梯度平滑

### 8.4 移植要点
- **valid mask 至关重要**：stereo depth 不可靠的区域（遮挡/反光）必须排除，否则引入错误监督。
- Normal Prior / Smooth 是可选增强，Depth L1 是核心。

---

## 9. 创新点 #9：空间约束 L_space（Stage 1，预防漂浮）

### 9.1 核心思路
在训练早期就用 COLMAP 初始点云构造有效区域 bbox，惩罚飞出 bbox 的高斯，从源头减少漂浮产生。

### 9.2 关键公式
```
# 从 COLMAP 点云算 bbox（p5~p95 外扩 margin_ratio=0.4）
bbox_min = p5 - extent * 0.4
bbox_max = p95 + extent * 0.4
allowed_margin = scene_radius * 0.08   # bbox 内 8% 半径不惩罚（软边界）

# 每高斯到有效区域的距离（内部=0，外部=L2）
dist[i] = distance_to_valid_region(means[i])

# 空间约束 loss
L_space = (1/N) * Σ opacity[i] * ReLU(dist[i] - allowed_margin)^2

# 调度：iter 0 起生效，线性升到 0.05 封顶
lambda_space = min(0.05, 0.01 * (iter / 1000))
```

### 9.3 代码定位
- `utils/floater_utils.py:53-105` — `SpatialConstraint` 类
- `utils/floater_utils.py:23-46` — `get_scheduler_lambdas()` 中 `lambda_space` 调度

### 9.4 移植要点
- 用 p5/p95 而非 min/max，抗 COLMAP 离群点。
- `margin_ratio=0.4` 外扩避免边界高斯被误罚。

---

## 10. 总 loss 结构

```
loss_total = L1_RGB_weighted
           + lambda_DSSIM * DSSIM_RGB
           + lambda_space * L_space              # Stage 1（iter 0+）
           + lambda_decay_sched * L_decay        # Stage 3（iter 3000+）
           + lambda_stereo * (Depth_L1 + Normal_Prior + Normal_Smooth)  # iter 500+
           + gamma_sched * (color down-weight 已并入 L1)
```

---

## 11. 移植 Checklist（指导另一个项目）

按以下顺序移植，每步验证后再下一步：

1. **Stereo 监督**（#8）：先接入 stereo depth prior + Depth L1，这是整个 Floater 管线的基础。
2. **空间约束**（#9）：加 L_space，防止高斯乱飞。
3. **OutlierTracker**（#2）：实现 EMA 累积 + resize 钩子（必须在 densify/prune 后调用）。
4. **多信号离群检测**（#1）：像素相对深度误差 + MAD 阈值 + 颜色冲突。
5. **Soft Opacity Decay**（#3）：`L_decay = mean(φ * opacity)`，φ 用 no_grad。
6. **Color Down-weighting**（#7）：pixel_outlier → color_weights。
7. **Hard Prune**（#4）：三条件 + max_prune_ratio 保护。
8. **Stereo Resample**（#5A）：硬删后用 stereo 高置信像素补种。
9. **Inpainting Resample**（#5B）：2D IDW 补种空洞。

---

## 12. 关键参数速查表

| 参数 | 默认值 | 所在阶段 | 含义 |
|---|---|---|---|
| `stereofrom_iterations` | 500 | Stereo | Stereo 监督开启 iter |
| `detect_from_iter` | 1000 | Stage 2 | 离群检测开启 iter |
| `soft_from_iter` | 3000 | Stage 3 | Opacity Decay 开启 iter |
| `hard_from_iter` | 5000 | Stage 4 | Hard Prune 开启 iter |
| `lambda_stereo_depth_sup` | 0.05 | Stereo | Depth L1 权重 |
| `lambda_decay_max` | 2.0 | Stage 3 | L_decay 权重上限 |
| `tau_hard` | 0.7 | Stage 4 | 硬删离群阈值 |
| `max_prune_ratio` | 0.1 | Stage 4 | 单次最大删除比例 |
| `prune_interval` | 500 | Stage 4 | 硬删间隔 |
| `min_obs_soft` | 3 | Stage 3 | 软惩罚最小观测数 |
| `min_obs_hard` | 5 | Stage 4 | 硬删最小观测数 |
| `attr_anom_w` | 0.15 | Stage 2 | 属性异常权重 |
| `inpaint_enabled` | False | Stage 5.5 | Inpainting 补种开关 |
| `resample_max_points` | 30000 | Stage 5A | Stereo 补种上限 |
| `inpaint_max_points` | 15000 | Stage 5.5 | Inpainting 补种上限 |

---

*本文档所有行号基于代码库当前状态（MonoAnchor 模块已移除），移植时请以目标项目的对应代码为准。*
