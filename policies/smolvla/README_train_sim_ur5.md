# UR5 仿真抓取抬升训练

所有命令在 DISCOVERSE 根目录、discoverse 环境执行。仅离线训练，不连接真实机械臂。

## 1. 验收与转换

```bash
python scripts/ur5_validate_lift_dataset.py data/ur5_lift_sim_train300_conservative_v1 --report data/ur5_lift_sim_train300_conservative_v1/validation.json
python scripts/convert_sim_ur5_lift.py
```

转换目标 `data/ur5_lift_sim_v21` 必须不存在；已有完整产物不必重复转换。
转换失败时不生成 `conversion/COMPLETE.json`，训练不会读取半成品。保留失败目录诊断，重试请指定新 `--output`。
原始动作不移位、不折返角度，视频逐字节复制。30Hz，状态/动作各7维；cam_0→d436，cam_1→d405。
每条轨迹独立拆分，300条固定划分270训练/30验证，种子42。
每个网格先按10%向下取整分配验证数，再随机补齐余数；小样本网格不保证两集合都有样本。
图像每10帧、每8像素抽样计算各episode统计；状态动作使用全帧。训练只聚合训练episode统计。

## 2. 短训练验收

```bash
python policies/smolvla/train_sim_ur5.py --smoke
```

完成2次更新、有限损失/梯度检查、验证集前向、模型保存及权重重载。

## 3. 正式训练

```bash
python policies/smolvla/train_sim_ur5.py --steps 10000 --batch-size 4
```

从 `policies/smolvla/pretrained/smolvla_base` 开始，不加载真实任务模型。
默认输出 `policies/smolvla/checkpoints/sim_ur5_lift_<时间>/`，每1000步保存。
冻结视觉编码器，训练action expert和state projection；chunk_size=50，n_action_steps=10。
复用真实训练实现；检查点内的 `real_training.json` 是兼容文件名，内容明确标明仿真数据目录和图像来源。
不上传Hub，使用本地基础模型及缓存backbone。

## 4. 续训

```bash
python policies/smolvla/train_sim_ur5.py --resume /绝对路径/checkpoints/005000 --steps 10000
```

resume指向含training_state的检查点目录，不是pretrained_model；steps是总更新数。
恢复优化器、调度器及随机状态，但数据打乱迭代器重新开始。

## 实验边界

任务为夹取、抬升、保持，不包含放置释放。默认训练范围X=[0.31,0.73]、Y=[-0.18,0.18]米。
当前相机为初步匹配，夹爪反馈尚未毫米级物理标定；不将仿真成功视为实机安全或迁移成功。
训练后需独立随机种子的仿真闭环评估；本入口不执行闭环评估或真实微调。
