# tum-adlr-09

This is a motion planning project for a mobile robot (e.g. a waiter robot in a museum exibition with moving and stationary obstacles)


# Source Code

There are some trial code in pytorch_rl folder including a pytorch toturial on RL.
And gym environment sample code

"""
Used Sources:
https://www.gymlibrary.dev/content/environment_creation/
https://github.com/Farama-Foundation/gym-examples/blob/main/gym_examples/envs/grid_world.py
#https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/SAC
"""


# Setup and usage

- all the following lines from the main folder

- creata a virtual environment:
```
conda create -n adlr
```

- install the requirements with:
```
pip install -r requirements.txt
```

- you can try SAC training to see the the training process: 
```
python SAC/training.py
```


- but SAC-X performs better, so the usage will be with:
```
python SAC-X/training.py
python SAC-X/plot_results.py
python SAC-X/make_results_gif.py
python SAC-X/accuracy_calculation.py 
```
