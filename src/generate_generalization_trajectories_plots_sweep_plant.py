from run_plant_parameter_sweep import *
import pickle
from pathlib import Path

plantParams["Simulation Duration"] = 15
sweepPlantPath = Path("experimental_trials/Sweep_Plant/")
stiffnessRange=[20,50]
delay = 0.3
movementTypes = [
    "angleSin_stiffSin",
    "angleStep_stiffSin",
    "angleSin_stiffStep",
    "angleStep_stiffStep"
]
prettyMovementTypes = [
    "Sinusoidal Angle \ Sinusoidal Stiffness",
    "Step Angle \ Sinusoidal Stiffness",
    "Sinusoidal Angle \ Step Stiffness",
    "Step Angle \ Step Stiffness"
]
stepDurations = [None,2,2,1]
frequencies = [1,1,1,None]
numberOfSteps = [None,100,100,200]
angleRanges = [
    [3*np.pi/4,5*np.pi/4],
    [np.pi/2,3*np.pi/2],
    [3*np.pi/4,5*np.pi/4],
    [np.pi/2,3*np.pi/2]
]
extraSteps = [-3,5,3,5]
totalTimer = timer()
statusbar=dsb(0,3*3*4,title="Plotting Generalization Trajectories")
for i in range(3):
    plantParams["Spring Stiffness Coefficient"] = \
        tendonStiffnessParams[str(i+1)]["Spring Stiffness Coefficient"]
    plantParams["Spring Shape Coefficient"] = \
        tendonStiffnessParams[str(i+1)]["Spring Shape Coefficient"]
    for j in range(3):
        plantParams["Motor Damping"] = motorDampingParams[str(j+1)]

        plant = plant_pendulum_1DOF2DOF(plantParams)

        trialPath = sweepPlantPath / f"kT{i+1}_bm{j+1}"

        for k in range(4):
            dataPath = trialPath / f"{movementTypes[k]:s}_outputData.mat"
            if (trialPath/"Generalization_Trajectories"/movementTypes[k]/"README.md").exists():
                pass
            else:
                out = sio.loadmat(dataPath)
                X = np.array([
                    out['x1'],
                    out['dx1'],
                    out['x3'],
                    out['dx3'],
                    out['x5'],
                    out['dx5']
                ])[:,0,:]
                U = np.array([
                    out['u1'],
                    out['u2']
                ])[:,0,:]

                actualTrajectory = np.array([
                    X[0,:],
                    np.array(list(map(plant.hs,X.T)))
                ])

                if stepDurations[k] is not None:
                    plant.plot_desired_trajectory_distribution_and_power_spectrum(actualTrajectory,cutoff=delay+3*stepDurations[k])
                else:
                    plant.plot_desired_trajectory_distribution_and_power_spectrum(actualTrajectory)

                TEMPfig1 = plant.plot_states_and_inputs(
                    X,U,
                    inputString=prettyMovementTypes[k],
                    returnFig=True
                )

                if stepDurations[k] is not None:
                    TEMPfig2 = plant.plot_states_and_inputs(
                        X[:,:int(3*stepDurations[k]/plant.dt)],
                        U[:,:int(3*stepDurations[k]/plant.dt)-1],
                        inputString=prettyMovementTypes[k],
                        returnFig=True
                    )
                else:
                    pass

                plant.plot_tendon_tension_deformation_curves(X)

                save_figures(
                    trialPath/"Generalization_Trajectories",
                    "gen_traj_plot",
                    {
                        "Extra Steps" : extraSteps[k],
                        "Step Duration" : stepDurations[k],
                        "Number of Steps" : numberOfSteps[k],
                        "delay" : delay,
                        "Angle Range" : angleRanges[k],
                        "Stiffness Range" : stiffnessRange,
                        "Tendon Stiffness Coefficients" : tendonStiffnessParams[str(i+1)],
                        "Motor Damping" : plantParams["Motor Damping"]
                    },
                    subFolderName=movementTypes[k]+'/',
                    saveAsPDF=True,
                    saveAsMD=True,
                    addNotes=f"Generalization Trajectories for Settings kT{i+1} and bm{j+1}."
                )
                plt.close('all')
            statusbar.update(12*i+4*j+k)

if path.exists("slack_functions.py"):
    message = (
        '\n'
        + '_Generalization Trajectories Plotting Completed!!!_ \n\n'
        + 'Total Run Time: ' + totalTimer.totalRunTimeStr
    )
    HAL.slack_post_message_code_completed(message)
