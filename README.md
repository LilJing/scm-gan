Structural Causal Model GAN

This repository contains code for the upcoming paper submission
tentatively titled:
*Causal Disentanglement for Model-Based Reinforcement Learning*

The `main.py` file contains PyTorch code to build and train an
action-conditional video prediction model, for reinforcement learning
environments.
This model can predict future frames (given any plan of actions).

The available environments are listed as modules in the `envs`
directory.

To run the system with the example Pong environment, run:

```
python main.py --load-from . --env betterpong
```

Note: The constants defined in `models.py` and the hyperparameters
in `main.py` are dependent on the chosen environment.
Some environments will require changes to parameters or model
architecture.
