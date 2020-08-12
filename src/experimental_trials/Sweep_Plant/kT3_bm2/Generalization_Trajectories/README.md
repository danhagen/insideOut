
# Trajectories Chosen to Test Generalizability of ANNs That Predict Joint Angles from Non-Collocated Sensory Information

## Notes

kT3 / bm2

## Parameters

 ```py
 params = {
    'Tendon Stiffness Coefficients' : {'Spring Stiffness Coefficient': 16, 'Spring Shape Coefficient': 125},
    'Motor Damping' : 0.00462,
	'stiffnessRange' : [20, 100],
	'frequency' : 1,
	'delay' : 0.3
}
```

## Figures

#### Angle Step / Stiffness Step

<p align="center">
	<img width="500" src="angleStep_stiffStep/gen_traj_plot_01-04.png">
</p>

#### Angle Step / Stiffness Sinusoid

<p align="center">
	<img width="500" src="angleStep_stiffSin/gen_traj_plot_01-04.png">
</p>

#### Angle Sinusoid / Stiffness Step

<p align="center">
	<img width="500" src="angleSin_stiffStep/gen_traj_plot_01-04.png">
</p>

#### Angle Sinusoid / Stiffness Sinusoid

<p align="center">
	<img width="500" src="angleSin_stiffSin/gen_traj_plot_01-04.png">
</p>
