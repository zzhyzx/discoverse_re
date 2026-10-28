"""Simulation-only SmolVLA entry point. No robot access or Hub uploads.

python policies/smolvla/train_sim_ur5.py --smoke
python policies/smolvla/train_sim_ur5.py --steps 10000 --batch-size 4
"""
from train_real_ur5 import main

if __name__ == '__main__':
    main(simulation=True)
