<p align="center">
  <h1><em>insideOut</em></h1>
</p> 

<p align="center">
  <img width="1000" src="https://github.com/danhagen/iO-IROS-2020/blob/master/SupplementaryFigures/insideOut_comic_graphic.png?raw=true" alt="insideOut: A Bio-Inspired Machine Learning Approach to Estimating Posture in Robots Driven by Compliant Tendons">
</p>

## A Bio-Inspired Machine Learning Approach to Estimating Posture in Robots Driven by Compliant Tendons
### Daniel A. Hagen, Ali Marjaninejad, Gerald E. Loeb & Francisco J. Valero-Cuevas

Estimates of limb posture are critical to control robotic systems. This is generally accomplished with angle sensors at individual joints that simplify control but can complicate mechanical design and robustness. Limb posture should be derivable from each joint’s actuator shaft angle but this is problematic for compliant tendon-driven systems where (_i_) motors are not placed at the joints and (_ii_) nonlinear tendon stiffness decouples the relationship between motor and joint angles. Here we propose a novel machine learning algorithm to accurately estimate joint posture during dynamic tasks by limited training of an artificial neural network (ANN) receiving motor angles _and_ tendon tensions, analogous to biological muscle and tendon mechanoreceptors. Simulating an inverted pendulum—antagonistically-driven by motors and nonlinearly-elastic tendons—we compare how accurately ANNs estimate joint angles when trained with different sets of non-collocated sensory information generated via random motor-babbling. Cross-validating with new movements, we find that ANNs trained with motor _and_ tendon tension data predict joint angles more accurately than ANNs trained without tendon tension. Furthermore, these results are robust to changes in network/mechanical hyper-parameters. We conclude that regardless of the tendon properties, actuator behavior, or movement demands, tendon tension information invariably improves joint angle estimates from non-collocated sensory signals.

## Installation from GitHub
Please follow the instructions <a href='https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html'>here</a> in order to install MATLAB engine API for python. Once that is done, you can clone into this repository and install the remaining required packages by copy and pasting the following code into the terminal.

```bash
git clone https://github.com/danhagen/insideOut.git && cd insideOut/src
pip install -r requirements.txt
pip install .
```

Please note that you can find help for many of the python functions in this repository by using the command `run <func_name> -h`.

## The Plant 

<p align="center">
  <img width="500" src="https://github.com/danhagen/iO-IROS-2020/blob/master/SupplementaryFigures/Schematic_1DOF2DOA_system.png?raw=true">
</p>

Here we used a physical inverted pendulum that was controlled by two simulated brushed DC motors (i.e., backdriveable) that pulled on tendons with nonlinear (exponential) stiffness. This plant can either be given feedfoward inputs or controlled via a *feedback linearization controller* that takes advantage of the fact that joint stiffness and joint angle can be controlled independently. Simply prescribe trajectories for both output measures and the controller will track it.

The default `run plant.py` command will test the feedback linearization algorithm. Choosing the options `--saveFigures` will save the figures and `--animate` will animate the output.


## Generating Babbling Data
In order to generate motor babbling data, we use the class `motor_babbling_1DOF2DOA` which generates low frequency, band-limited white noise signals (&#8804; 10 Hz) for each motor input where the inputs have a high degree of temporal correlation (emulating physiological co-contraction). The default `run motor_babbling_1DOF2DOA.py` will produce plots of random motor babbling and the resulting states of the plant. Figures can be saved as either PNG or PDF (`--savefigs` and `--savefigsPDF`, respectively) in a time-stamped folder. You also have the option to animate the babbling data (`--animate`). 

<p align="center">
  <img width="500" src="https://github.com/danhagen/iO-IROS-2020/blob/master/SupplementaryFigures/babblingInputs.png?raw=true">
  <img width="500" src="https://github.com/danhagen/iO-IROS-2020/blob/master/SupplementaryFigures/Plant_States_vs_Time_from_Babbling.png?raw=true">
</p>

## Train Articifical Neural Networks
To build, train, and test these ANNs, use `build_NN_1DOF2DOA` and `test_NN_1DOF2DOA`.

## Run Multiple Trials and Plot All Data
To sweep babbling durations, run `run run_multiple_trials_with_different_babbling_durations.py`. The number of hidden-layer nodes (*default* is 15) can be changed with option `-nodes`. You can choose to plot metrics such as mean absolute error (MAE), root mean squared error (RMSE), or standard deviation of the error (STD) by adding the additional arguments `-metrics [METRICS ...]`. 

Conversely, to sweep hidden-layer nodes, run `run run_multiple_trials_with_different_hidden_layer_nodes.py`. The number of hidden-layer nodes (*default* is 15) can be changed with option `-nodes`. You can choose to plot metrics such as mean absolute error (MAE), root mean squared error (RMSE), or standard deviation of the error (STD) by adding the additional arguments `-metrics [METRICS ...]`. 

Parameter sensitivity can be performed for movement frequency (`run run_frequency_sweep.py`) or for changes to tendon stiffness/motor damping (`run run_plant_parameter_sweep.py`). To observe the effect of assuming very high tendon stiffness use the function `run run_high_stiffness_experiment.py`.

## Animate a Single Trial (All 4 ANNs Over 4 Different Movements)
To visualize the performance of ANNs and their ability to generalize to other movement tasks, use the function `animate_sample_trials.py`. This will create an animation of how well each ANN did at predicting joint angle and will sweep across 4 different movements (joint angle and stiffness are either sinusoidal or point-to-point). 


