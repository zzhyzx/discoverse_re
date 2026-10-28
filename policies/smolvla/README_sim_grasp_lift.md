# UR5 抓取抬升数据（30 Hz）

采集入口：`discoverse/examples/tasks_ur5/ur5_single_red_pick_place_task.py`。
保留旧文件名，但命令行默认任务已改为接近、下降、闭合、抬升并保持；不再搬运到绿色区域或释放。
这里仅修改仿真采集，不调用任何实机控制接口。

## 采集

```bash
conda activate discoverse
cd /home/boluo/DISCOVERSE
MUJOCO_GL=egl python discoverse/examples/tasks_ur5/ur5_single_red_pick_place_task.py \
    --auto --data_set_size 3 --max_attempts 10
```

默认采用 `--sampling tabletop`，普通渲染保存到 `data/ur5_grasp_lift_sim_30hz_v1_tabletop`，
添加 `--gaussian` 时保存到 `data/ur5_grasp_lift_gs_30hz_v1_tabletop`。
`--sampling local` 可恢复之前桌面中心附近的小范围采集及不带 `_tabletop` 的目录。
两个视频仍为 `cam_0.mp4`、`cam_1.mp4`。
已有同名样本会移至相邻的 `_replaced` 备份目录；不修改旧放置任务数据。
采集中切换离开 3DGS 会使当前样本失败，以免混入不同渲染方式。

默认成功条件：考虑方块旋转后的**最低点**离桌至少 0.05 m，同时线速度不超过
0.03 m/s、角速度不超过 0.3 rad/s，连续保持至少 1 s。任意一帧不满足即重新计时。
可用 `--success-height 0.05 --hold-seconds 1.0` 调整；这是仿真任务判据，不是实机安全判据。

## 数据约定

- 物理积分 240 Hz，每个动作 8 个积分步；观测、动作、两路视频均 30 Hz。
- 一行保存时刻 t 的实际状态和画面，以及接下来 `[t,t+1/30)` 应用的目标；首帧时间为零。
- 状态：六个实际关节角（弧度）加 `clip(driver_joint/0.8,0,1)`。
- 动作：六个绝对关节目标（弧度）加夹爪指令 `/255`，0 开、1 闭。
- `cam_0` / `primary_view` → `observation.images.d436`（正面）。
- `cam_1` / `wrist_view` → `observation.images.d405`（腕部）。
- 起点从 `policies/smolvla/configs/ur5_real_deploy.json` 的 `start_reference_deg` 读取，
  不对记录值进行角度折返。IK 等价解在关节限位内选择最接近当前状态的分支。
- 附加记录方块底面离桌高度、逐帧成功条件和夹爪碰撞垫几何间隙，便于验收。

**夹爪还没有完成实机物理标定。** 仿真 driver/0.8 与实机 POS/255 仅开合方向一致，
不保证相同数值具有相同毫米开口。元数据明确标注 `physical_opening_equivalence_verified=false`。
后续需实测若干 POS 对应的开口，建立单调映射后再用于严格虚实微调对照；不得仅把 0～1
范围相同当作等价。本次不改实机反馈、不编造标定表。

## 验收

```bash
MUJOCO_GL=egl python scripts/ur5_validate_lift_dataset.py \
    data/ur5_grasp_lift_sim_30hz_v1_tabletop --gripper-sweep \
    --report data/ur5_grasp_lift_sim_30hz_v1_tabletop/validation.json
```

验收逐帧视频时间戳、尺寸、数量、状态动作维度、起点、动作步长和连续保持条件；
可选夹爪扫描只运行仿真。原始样本尚不是 LeRobot 数据集，开始训练前仍需转换、划分训练/
验证集，并与真实数据统一语言指令。当前相机仍是初步位姿及 FOV 近似，不代表精确标定。

## 大范围桌面随机采集

当前桌面边界为 X=[0.19,0.99]、Y=[-0.45,0.45] m。默认 `--edge-margin 0.06`
指方块中心距边缘的余量，已包含旋转后方块的水平半径；`--far-edge-inset 0.12`
额外排除远离机械臂的 +X 一侧。因此候选中心范围为 X=[0.25,0.81]、Y=[-0.39,0.39] m。
方块绕 Z 轴随机转动 ±45°。桌面分成 6×6 个格子，每轮打乱访问顺序并在格内均匀采样，
避免短批次仅集中在中央。每轮覆盖的是候选位置，不保证所有格子都有成功示范。

生成视频前检查接近、预抓取、抓取、抬升的 IK 残差及关节限位。
接近阶段先抬高到桌面上方至少 24 cm，再以不大于 6 cm 的横向路点移动，最后下降到
物块上方；实际轨迹仍由关节位置控制器执行，不能视作严格笛卡尔直线。
碰撞筛选沿关节路径离散检查机械臂/夹爪与桌面、地板及底座台子的穿透。动态执行时继续检查这类接触，只有达到抬升
保持判据才保存有效样本。此检查不是连续碰撞保证，也不是实机安全认证，尚不包含完整
自碰撞、线缆或相机支架模型。

`attempts.jsonl` 记录每次候选网格、位置、朝向、成功/拒绝原因。扩大范围会降低采集通过率，
不能只看最终成功样本数；应检查成功位置分布与各网格拒绝情况。大范围默认目录与旧中心
范围分开保存，旧目录中中断后没有 `obs_action.json` 的样本不应进入训练。

## 保留全部失败样例

```bash
MUJOCO_GL=egl python scripts/ur5_collect_lift_diagnostics.py \
    --attempts 10 --seed 20261020 --output-dir data/ur5_lift_diagnostics_new
```

必须使用新目录。按尝试编号保存，不按成功条数补齐；执行失败保留视频、状态和误差轨迹。
预检查拒绝保留单帧初始视频与明确标记为未执行的候选姿态截图，不跳过碰撞筛选。
所有数据标记为诊断用途，不应直接用于模仿学习训练；结束后自动检查双视频与记录对应。
