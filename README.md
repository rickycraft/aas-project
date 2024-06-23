# Autonomous and Adaptive System M

This repository is the result of the the project work for the exam.

# Files

- impala -> contains the impala CNN architecture
- nature -> contains the Nature CNN architecture
- keras-procgen.py -> TD(0)
- reinforce-procgen.py -> REINFORCE with baseline
- advantage-keras-procgen.py -> A2C

## How to run

```bash
USE_IMPALA=1 NUM_LEVEL=5 ENV_NAME=chaser GENERATE_GIF=1 python run-keras-procgen.py <weights_file>
```

## How to train

```bash
USE_IMPALA=1 NUM_LEVEL=5 ENV_NAME=chaser DONE_REWARD=d python reinforce-procgen.py lr clip
```
