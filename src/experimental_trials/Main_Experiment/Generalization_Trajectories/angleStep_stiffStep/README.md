# README.md for Figures Created 2020/04/22 at 17:15.36

## Notes

Rerunning to show _actual_ trajectory in histograms. Previous versions forced the bins to be between the maximum and minimum joint angles. Stiffness was not forced to be anywhere, so this is good news for the stiffness plots.

## Parameters

 ```py
 params = {
	'Extra Steps' : 5,
	'Step Duration' : 2.0,
	'frequency' : 1,
	'numberOfSteps' : 100,
	'stiffnessRange' : [20, 100],
	'angleRange' : None,
	'delay' : 0.3
}
```

## Figures

<p align="center">
	<img width="500" src="states_02-01.png"></br>
    <small>Figure 1: Sample output for point-to-point joint angle and sinusoidal joint stiffness reference trajectories (9 seconds shown).</small>
</p>
</br>
</br>
<p align="center">
	<img width="1000" src="states_02-02.png"></br>
    <small>Figure 2: Power spectral densities of joint angle (<em>left</em>) and joint stiffness (<em>right</em>). </small>
</p>
</br>
</br>
<p align="center">
	<img width="1000" src="states_02-03.png"></br>
    <small>Figure 3: Histograms and kernel density estimates of joint angle (<em>left</em>) and joint stiffness (<em>right</em>). </small>
</p>
</br>
</br>
<p align="center">
	<img width="1000" src="states_02-05.png"></br>
    <small>Figure 4: Sample plot of states (and their derivatives when appropriate, denoted by <em>dotted</em> lines) and inputs (9 seconds shown).  </small>
</p>

It should be noted that the tendon tension and motor torques are *highly* correlated. The Pearson coefficients are ~0.99936 for both motor-tendon complexes. This would imply that it would be equivalent to replace the **difficult to measure** tendon tensions in our predictor algorithms with the readily available motor torques. In biology, we do have efference copies, but these do not directly correlate to tendon tension as they have to contend with nonlinear force-length, force-velocity relationships. Additionally, this is for a system that has not contact with the world. If the pendulum were to experience a sudden impact, the tendon tensions would change instantly, but the motor torques would deviate. But if the controller was predictive or proportional, then it would be likely that the motor torques would eventually become correlated again.
</br>
</br>
<p align="center">
	<img width="1000" src="trajectory_01.png"></br>
    <small>Figure 9: Two dimensional kernel density estimate plots for joint angle (with respect to vertical) and joint stiffness.  </small>
</p>
</br>

# Appended on 2020/05/01 at 17:39.08 PST.

## Notes

Rerunning with smaller stiffness range.

## Parameters

```py
params = {
	'Extra Steps' : 5,
	'Step Duration' : 2.0,
	'frequency' : 1,
	'numberOfSteps' : 100,
	'stiffnessRange' : [20, 50],
	'angleRange' : None,
	'delay' : 0.3
}
```

## Figures

<p align="center">
	<img width="500" src="lower_stiffness_02-01.png"></br>
    <small>Figure 10: Sample output for point-to-point joint angle and sinusoidal joint stiffness reference trajectories (3 seconds shown).</small>
</p>
</br>
</br>
<p align="center">
	<img width="1000" src="lower_stiffness_02-02.png"></br>
    <small>Figure 11: Power spectral densities of joint angle (<em>left</em>) and joint stiffness (<em>right</em>, with smaller upper bounds). </small>
</p>
</br>
</br>
<p align="center">
	<img width="1000" src="lower_stiffness_02-03.png"></br>
    <small>Figure 12: Histograms and kernel density estimates of joint angle (<em>left</em>) and joint stiffness (<em>right</em>, with smaller upper bounds). </small>
</p>
</br>
</br>
<p align="center">
	<img width="1000" src="lower_stiffness_02-06.png"></br>
    <small>Figure 13: Sample plot of states (and their derivatives when appropriate, denoted by <em>dotted</em> lines) and inputs (3 seconds shown).  </small>
</p>
</br>
</br>
<p align="center">
	<img width="1000" src="lower_stiffness_02-04.png"></br>
    <small>Figure 14: Two dimensional kernel density estimate plots for joint angle (with respect to vertical) and joint stiffness (smaller upper bounds).  </small>
</p>
</br>

# Appended on 2020/05/05 at 18:17.40 PST.

## Notes

It was discovered that large (seemingly single point) step changes existed in the inputs and the positional states. Upon further inspection it was determined that this was caused by asking the feedback linearization algorithm (which utilizes up to the 4<sup>th</sup> derivative of joint angle and the 2<sup>nd</sup> derivative of joint stiffness) to follow a trajectory that was only continuous up to the 1<sup>st</sup> derivative (i.e., it was C<sup>1</sup> differentiable). Therefore, we changed the reference trajectory to be C<sup>4</sup> differentiable to see if it removes these transients. This was done by creating point-to-point trajectories with a modified "minimum jerk" equation to have zero velocity, acceleration, jerk, and snap when arriving and leaving a point. These trajectories were then filtered to recover the more gradual (sub 3 Hz) movements previously simulated. As we can see when comparing Figures 13 & 18, the transients appear to be removed.


## Parameters

```py
params = {
	'Extra Steps' : 5,
	'Step Duration' : 2.0,
	'frequency' : 1,
	'numberOfSteps' : 100,
	'stiffnessRange' : [20, 50],
	'angleRange' : None,
	'delay' : 0.3
}
```

## Figures
<p align="center">
	<img width="500" src="C4_diff_reference_trajectories_01-01.png"></br>
    <small>Figure 15: Sample output for point-to-point joint angle and sinusoidal joint stiffness reference trajectories (3 seconds shown) when using C<sup>4</sup> differentiable reference trajectories.</small>
</p>
</br>
</br>
<p align="center">
	<img width="1000" src="C4_diff_reference_trajectories_01-02.png"></br>
    <small>Figure 16: Power spectral densities of joint angle (<em>left</em>) and joint stiffness (<em>right</em>, with smaller upper bounds) when using C<sup>4</sup> differentiable reference trajectories. </small>
</p>
</br>
</br>
<p align="center">
	<img width="1000" src="C4_diff_reference_trajectories_01-03.png"></br>
    <small>Figure 17: Histograms and kernel density estimates of joint angle (<em>left</em>) and joint stiffness (<em>right</em>, with smaller upper bounds) when using C<sup>4</sup> differentiable reference trajectories. </small>
</p>
</br>
</br>
<p align="center">
	<img width="1000" src="C4_diff_reference_trajectories_01-06.png"></br>
    <small>Figure 18: Sample plot of states (and their derivatives when appropriate, denoted by <em>dotted</em> lines) and inputs (3 seconds shown) when using C<sup>4</sup> differentiable reference trajectories.  </small>
</p>
</br>
</br>
<p align="center">
	<img width="1000" src="C4_diff_reference_trajectories_01-04.png"></br>
    <small>Figure 19: Two dimensional kernel density estimate plots for joint angle (with respect to vertical) and joint stiffness (smaller upper bounds) when using C<sup>4</sup> differentiable reference trajectories.  </small>
</p>
</br>
