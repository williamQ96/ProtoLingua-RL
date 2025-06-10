# ProtoLingua-RL: Emergent Language with Backchannel Grounding in Reinforcement Learning

**Author**: William Qiu  
**Organization**: University of Oregon

---

## 🧠 Overview

**ProtoLingua-RL** is a research-focused reinforcement learning framework that simulates emergent communication between agents. The agents must coordinate to solve a color matching task by evolving a symbolic language from scratch. Our primary contribution is the **Backchannel Grounding Loop**, which significantly boosts success rates in multi-agent communication.

This project explores four different training regimes:
- `Baseline`: No regularization or entropy term.
- `Entropy`: Encourages exploration via message entropy.
- `Hybrid Entropy`: Annealed entropy for a balance between exploration and stability.
- `Backchannel Grounding`: A novel human-inspired feedback loop where speakers reinforce or correct listener guesses.

---

## 📈 Key Results

| Method                 | Episodes | Final Success Rate |
|------------------------|----------|---------------------|
| Baseline               | 10k      | 0.35                |
| Entropy                | 10k      | 0.20                |
| Hybrid Entropy         | 10k      | 0.22                |
| Backchannel Grounding  | 10k      | **0.61**            |
| Backchannel Grounding  | 100k     | **0.70**            |

> 🏆 Our method outperforms Lazaridou et al. (2017)’s benchmark (≈0.60) by up to **10%**, and our best model improves on our own hybrid entropy baseline by **~30–40%** relative gain.

---

## 🛠 Directory Structure

    ├── agents/ # Speaker and Listener agent definitions
    ├── envs/ # ColorMatch environment setup
    ├── train/ # Training scripts for all methods
    ├── tests/ # Unit tests for agents and environment
    ├── utils/ # Logging, visualization, and metrics
    ├── logs/ # Output logs and reward histories
    ├── image/ # Plots and diagrams
    ├── config.py # Global debug toggle
    └── README.md # This file

## 🚀 Quick Start

### 1. Install dependencies

```
pip install torch tqdm
```

### 2. Train a model

Train backchannel grounding for 10,000 episodes:
```
python train/train_backchannel_grounding.py 10000
```


Train the baseline model:

```
python train/train_baseline.py 10000
```

All logs are saved to logs/ and success rates are printed at the end of training.

### 3. Evaluation

You can extract success rates from logs using the included evaluation script:

```
python utils/analyze_reward_log.py logs/reward_log_backchannel_grounding_a=005_100000.txt
```

You can also monitor the global log:

```
cat logs/global_training_log.txt
```

### Visualizations

Use plot_rewards.py to visualize training curves:

```
python plot_rewards.py [mode] [log1] [log2] [log3]
```

mode 0 has both raw and smoothed data
mode 1 just has smoothed data.

Log2 and Log3 are optional for you to compare.


### 📚 Reference

If you use or extend this work, please cite:

```
@misc{qiu2025protolingua,
  author = {William Qiu},
  title = {ProtoLingua-RL: Emergent Communication with Backchannel Grounding},
  year = {2025},
  institution = {University of Oregon}
}
```

### 📄 License

```
This project is released under the MIT License.
```