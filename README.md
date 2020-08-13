<p align="center">
  <img width="1000" src="https://github.com/danhagen/insideOut/blob/master/SupplementaryFigures/insideOut_comic_graphic.png?raw=true" alt="insideOut: A Bio-Inspired Machine Learning Approach to Estimating Posture in Robots Driven by Compliant Tendons">
</p>

<h1 align="center"><em>insideOut</em></h1>
<h2 align="center">A Bio-Inspired Machine Learning Approach</br>to Estimating Posture in Robots Driven by Compliant Tendons</h2>
<h3 align="center">Daniel A. Hagen, Ali Marjaninejad, Gerald E. Loeb & Francisco J. Valero-Cuevas</h3>

Estimates of limb posture are critical to control robotic systems. This is generally accomplished with angle sensors at individual joints that simplify control but can complicate mechanical design and robustness. Limb posture should be derivable from each joint’s actuator shaft angle but this is problematic for compliant tendon-driven systems where (_i_) motors are not placed at the joints and (_ii_) nonlinear tendon stiffness decouples the relationship between motor and joint angles. Here we propose a novel machine learning algorithm to accurately estimate joint posture during dynamic tasks by limited training of an artificial neural network (ANN) receiving motor angles _and_ tendon tensions, analogous to biological muscle and tendon mechanoreceptors. Simulating an inverted pendulum—antagonistically-driven by motors and nonlinearly-elastic tendons—we compare how accurately ANNs estimate joint angles when trained with different sets of non-collocated sensory information generated via random motor-babbling. Cross-validating with new movements, we find that ANNs trained with motor _and_ tendon tension data predict joint angles more accurately than ANNs trained without tendon tension. Furthermore, these results are robust to changes in network/mechanical hyper-parameters. We conclude that regardless of the tendon properties, actuator behavior, or movement demands, tendon tension information invariably improves joint angle estimates from non-collocated sensory signals.

<h2 align="center">Installation from GitHub</h2>

Please follow the instructions <a href='https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html'>here</a> in order to install MATLAB engine API for python. Once that is done, you can clone into this repository and install the remaining required packages by copy and pasting the following code into the terminal.

```bash
git clone https://github.com/danhagen/insideOut.git && cd insideOut/src
pip install -r requirements.txt
pip install .
```

Please note that you can find help for many of the python functions in this repository by using the command `run <func_name> -h`.

<h2 align="center">The Plant</h2> 

<p align="center">
  <img width="500" src="https://github.com/danhagen/insideOut/blob/master/SupplementaryFigures/Schematic_1DOF2DOA_system.png?raw=true"></br>
  <small>Fig. 1: Schematic of a tendon-driven system with 1 kinematic DOF and 2 degrees of actuation (motors) that pull on tendons with nonlinear elasticity (creating a tension, <em>f<sub>T,i</sub></em>). The motors were assumed to be backdrivable with torques (<em>&tau;<sub>i</sub></em>) as inputs.</small>
</p>

Here we used a physical inverted pendulum that was controlled by two simulated brushed DC motors (i.e., backdriveable) that pulled on tendons with nonlinear (exponential) stiffness. This plant can either be given feedfoward inputs or controlled via a *feedback linearization controller* that takes advantage of the fact that joint stiffness and joint angle can be controlled independently. Simply prescribe trajectories for both output measures and the controller will track it.

The default `run plant.py` command will test the feedback linearization algorithm. Choosing the options `--saveFigures` will save the figures and `--animate` will animate the output.


<h2 align="center">Generating Babbling Data</h2>

In order to generate motor babbling data, we use the class `motor_babbling_1DOF2DOA` which generates low frequency, band-limited white noise signals (&#8804; 10 Hz) for each motor input where the inputs have a high degree of temporal correlation (emulating physiological co-contraction). The default `run motor_babbling_1DOF2DOA.py` will produce plots of random motor babbling and the resulting states of the plant. Figures can be saved as either PNG or PDF (`--savefigs` and `--savefigsPDF`, respectively) in a time-stamped folder. You also have the option to animate the babbling data (`--animate`). 

<p align="center">
  <img width="500" src="https://github.com/danhagen/insideOut/blob/master/SupplementaryFigures/babblingInputs.png?raw=true"></br>
  <small>Fig.2: Sample low frequency, band-limited white noise motor babbling signals.</small>
</p>
<p align="center">
  <img width="500" src="https://github.com/danhagen/insideOut/blob/master/SupplementaryFigures/Plant_States_vs_Time_from_Babbling.png?raw=true"></br>
  <small>Fig. 3: Resulting pendulum angle/angular velocity, motor rotations/angular velocities, and tendon tensions for a motor babbling trial.</small>
</p>

<h2 align="center">Train Articifical Neural Networks</h2>

To build, train, and test these ANNs, use `build_NN_1DOF2DOA` and `test_NN_1DOF2DOA`.

<h2 align="center">Run Multiple Trials and Plot All Data</h2>

To sweep babbling durations, run `run run_multiple_trials_with_different_babbling_durations.py`. The number of hidden-layer nodes (*default* is 15) can be changed with option `-nodes`. You can choose to plot metrics such as mean absolute error (MAE), root mean squared error (RMSE), or standard deviation of the error (STD) by adding the additional arguments `-metrics [METRICS ...]`. 

Conversely, to sweep hidden-layer nodes, run `run run_multiple_trials_with_different_hidden_layer_nodes.py`. The duration of motor babbling (*default* is 15s) can be changed with option `-dur`. You can choose to plot metrics such as mean absolute error (MAE), root mean squared error (RMSE), or standard deviation of the error (STD) by adding the additional arguments `-metrics [METRICS ...]`. 

Jump to: <a href="https://github.com/danhagen/insideOut/blob/master/src/experimental_trials/Main_Experiment/README.md" id="main_ex">Main Experiment Results</a>

Parameter sensitivity can be performed for movement frequency (`run run_frequency_sweep.py`) or for changes to tendon stiffness/motor damping (`run run_plant_parameter_sweep.py`). To observe the effect of assuming very high tendon stiffness use the function `run run_high_stiffness_experiment.py`. These experiments assume 15 seconds of motor babbling and 15 hidden-layer nodes, but those values can be changed for any function by the aforementioned options.

Jump to: <a href="https://github.com/danhagen/insideOut/blob/master/src/experimental_trials/Sweep_Frequency/README.md" id="freq_sweep">Frequency Sweep Results</a>,
<a href="https://github.com/danhagen/insideOut/blob/master/src/experimental_trials/Sweep_Plant/README.md" id="plant_sweep">Plant Parameter Sweep Results</a>, or
<a href="https://github.com/danhagen/insideOut/blob/master/src/experimental_trials/High_Stiffness_Experiment/README.md" id="high_stiff">High Stiffness Experiment Results</a>

<h2 align="center">Animate a Single Trial (All 4 ANNs Over 4 Different Movements)</h2>

To visualize the performance of ANNs and their ability to generalize to other movement tasks, use the function `animate_sample_trials.py`. This will create an animation of how well each ANN did at predicting joint angle and will sweep across 4 different movements (joint angle and stiffness are either sinusoidal or point-to-point). 


