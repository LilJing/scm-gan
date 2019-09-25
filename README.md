*Counterfactual Regularization for Model-Based Reinforcement Learning*

The `main.py` file contains PyTorch code to build and train an
action-conditional video prediction model, for reinforcement learning
environments.
This model can predict future frames (given any plan of actions).

To run the system with the MiniPacMan environment, run:

```
python main.py --env minipacman
```

To evaluate after training, run:

```
python main.py --env minipacman --load-from . --evaluate
```

Built for Ubuntu 18.04. Requires Python 3.6+ and ffmpeg with libx264.
See requirements.txt for required modules.

For StarCraft2 environments, use the included custom branch of sc2env.
The `install_starcraft2` script will install a development version of
StarCraftII version Base60321. Works with Ubuntu 18.04.

Note: For fast GPU-based rendering on NVidia cards, ensure that
a version of libEGL.so is available in `ldconfig`.
