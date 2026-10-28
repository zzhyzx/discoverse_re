# UR5 低速基线 v1

公共规范：configs/control/ur5_low_speed_v1.json。
本版本没有训练新模型，没有启动实机运动。旧高速入口/数据/检查点保留。

## 固定内容
- 六轴绝对关节目标(rad)，状态为实际关节角+夹爪反馈；动作记录最终下发目标。
- 速度0.03rad/s，加速度0.05rad/s²；数据及动作时钟30Hz。
- 动作段最多30步，至少推进10步且当前目标跟踪到0.5度以内才允许换段。
- 夹爪0.65/0.35阈值、0.2秒持续判定；到位后发出夹爪指令；等待反馈期间暂停推进。
- 不允许释放；任务终点为夹取抬升保持；最长180秒。
- 模型预测有效期、越界、异常停止保留。慢速不能绕过这些保护。

## 统一范围和剩余差异
公共Limiter、GripperDecision、ExecutionProgress及guarded_command在仿真模型评估与部署中复用。
仿真专家采集复用同一限速和夹爪命令核，状态机基于实际到位推进，记录限速后的目标。
专家没有模型预测，因此不适用预测有效期和每段10步规则。
仿真控制更新30Hz、实机servo125Hz；同一算法使用各自dt，尚不是逐tick完全等效。
仿真夹持反馈使用双侧物块接触及运动稳定；实机使用URCap状态。开口与力度尚未物理标定。
仿真推理为同步暂停；评估将实测推理耗时计入观察有效期，但不替代实机实时调度验证。
因此这是一套统一运动约束/动作语义的初始基线，不是已经验证的完整控制器等效模型。

## 采集
conda activate discoverse
cd /home/boluo/DISCOVERSE
MUJOCO_GL=egl python scripts/ur5_collect_low_speed.py --attempts 20 --seed 20261027 --arm-gravity-compensation --output-dir data/ur5_low_speed_v1_attempt20

保留全部失败案例。新入口成功样本training_eligible=true，失败样本diagnostic_only=true。
输出目录必须是新目录。data/ur5_low_speed_v1_smoke是首次诊断验证，不用于训练。
预检拒绝和失败不会自动补足成功数量。

## 转换（至少10条成功样本）
python scripts/convert_sim_ur5_lift.py --source data/ur5_low_speed_v1_attempt20 --output data/ur5_low_speed_v1_lerobot --successful-only --control-profile configs/control/ur5_low_speed_v1.json

转换核对全部所选样本控制配置与哈希，保存到conversion/control_contract.json。
如果成功不足10条，继续采集新数据并整理后再转换；不能用重复样本凑数。

## 训练
python policies/smolvla/train_sim_ur5.py --data data/ur5_low_speed_v1_lerobot --control-profile configs/control/ur5_low_speed_v1.json --smoke
python policies/smolvla/train_sim_ur5.py --data data/ur5_low_speed_v1_lerobot --control-profile configs/control/ur5_low_speed_v1.json --steps 10000 --batch-size 4

--smoke是两次优化更新，必须先通过，再完整训练。新训练记录保存control_contract.json。
当前只完成一条诊断采集，尚未执行上述转换及训练。

## 模型评估
MUJOCO_GL=egl python policies/smolvla/eval_sim_ur5_low_speed.py --checkpoint /绝对路径/pretrained_model --output data/ur5_low_speed_v1_eval --episodes 5 --seconds 180

任务文字必须与该模型训练一致，真实旧模型需用--task覆盖。
新入口固定共享低速逻辑；不要用旧eval_sim_ur5_lift.py默认0.9rad/s评估新基线。

## 部署配置
policies/smolvla/configs/ur5_deploy_low_speed_v1.json 引用公共规范。
这份配置默认仍指向旧真实模型，仅用于入口接通；训练完必须一起更新checkpoint和train_stats。
先只读预测：
python policies/smolvla/deploy_real_ur5.py --config policies/smolvla/configs/ur5_deploy_low_speed_v1.json --duration 20 --no-video

本任务未实机执行，也未改变旧部署配置。真实控制125Hz的实时性与新模型闭环效果需要独立验证。
## 本次验证
- 共享命令核的完整专家采集成功：62.2秒，1867帧/相机，最终一秒最低离桌24.33cm；数据/视频验收通过。
- 5项离线控制测试通过，包括限速、加速度、动作换段、夹爪等待、过期与异常停止。
- 旧真实010000模型入口测试在43帧后以Active prediction expired停止，记录在data/ur5_low_speed_v1_eval_smoke_fixed。这是停止保护的验证，不是新基线训练或成功率结果。
- 尚未批量采集、转换训练或运行实机；需要至少10条合格成功样本才允许转换。
