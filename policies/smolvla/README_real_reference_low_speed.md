# 真实轨迹参考的低速仿真训练基线
本次方案：保留真实50条的原40训练/10验证划分，按源轨迹参考进行低速物理回放，
重新记录仿真双相机、实际关节/夹爪反馈和最终下发动作。不是慢放真实视频。
源划分：policies/smolvla/checkpoints/real_ur5_20260924_151943/real_training.json。
原数据、旧检查点不覆盖。

## 控制和验收
共享 ur5_low_speed_v1：0.03rad/s、0.05rad/s²、30Hz；跟踪误差<=0.5度才推进源帧，
夹爪闭合有0.2秒判定及接触/稳定反馈等待。源轨迹结束后最多保持5秒。
初始化静止：真实第一帧实际关节角；物块XY/方向从闭合前实际FK推断，Z落桌。
该位置是事后估计，并非独立视觉定位，不应据此声称真实物块定位准确。
成功：轨迹全部执行完、底面离桌>=5cm、双侧夹持接触、运动稳定连续>=1秒。
无强制附着。失败保留完整已有记录与原因；不进入训练。
全部视频逐帧检查640x480、30Hz、帧数和时间戳；关节速度/加速度、动作状态有限性也验收。
共有50次来源，每个来源仅属于一个分区。失败不挪分区、不补重复数据。
至少32训练/8验证成功才自动继续。通过后原划分映射保存在转换目录。
图像是普通MuJoCo渲染；开口映射、接触动力学与真实仍未完全标定。
仿真命令30Hz、实机125Hz；共享约束不等于完全同一控制器。

## 自动流水线
MUJOCO_GL=egl PYTHONPATH=. python scripts/ur5_real_low_speed_pipeline.py
输出：
- data/ur5_real_reference_low_speed_v1：000..049、两路纯仿真视频、obs_action.json、
  source_progress.json、result.json、source_split.json、summary.json、pipeline_status.json、pipeline_logs/
- data/ur5_real_reference_low_speed_v21：验收通过的数据、保留划分和源编号映射。
- policies/smolvla/checkpoints/sim_real_reference_low_speed_v1_smoke：两步训练验收。
- policies/smolvla/checkpoints/sim_real_reference_low_speed_v1：从smolvla_base训练10000步、batch4。
每1000步保存。归一化仅由训练分区统计计算。不是从旧任务微调模型继续。
pipeline_status.json显示collecting/converting/training_smoke/training/complete/blocked。
中断的采集片段移入partial_*保留；完成的result.json不重复执行。
转换失败的半成品目录不会自动覆盖，需先排查。
训练存在可用检查点时支持恢复；不会自动启动实机控制。
