from test_NN_1DOF2DOA import *
from experimental_trials.High_Stiffness_Experiment.tendonStiffnessParams import *

if path.exists("slack_functions.py"):
    from slack_functions import *
    HAL = code_progress()

if __name__=="__main__":
    ### Delete the parameters that are not used in Babbling Trials
    if "Boundary Friction Weight" in plantParams.keys():
        del plantParams["Boundary Friction Weight"]
    if "Boundary Friction Gain" in plantParams.keys():
        del plantParams["Boundary Friction Gain"]
    if "Quadratic Stiffness Coefficient 1" in plantParams.keys():
        del plantParams["Quadratic Stiffness Coefficient 1"]
    if "Quadratic Stiffness Coefficient 2" in plantParams.keys():
        del plantParams["Quadratic Stiffness Coefficient 2"]

    ### Additional Arguments?
    parser = argparse.ArgumentParser(
        prog = "<filename>",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        generate_new_trajectories.py

        -----------------------------------------------------------------------------

        Generate new trajectories to test ANN generalizability.

        -----------------------------------------------------------------------------'''),
        epilog=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        Written by Daniel A. Hagen (2020/05/12)

        -----------------------------------------------------------------------------'''
        )
    )

    stiffnessRange=[150,650]
    delay = 0.3

    basePath = Path("experimental_trials/High_Stiffness_Experiment")
    trajectoryCount = 0

    plantParams["Spring Stiffness Coefficient"] = \
        tendonStiffnessParams["Spring Stiffness Coefficient"]
    plantParams["Spring Shape Coefficient"] = \
        tendonStiffnessParams["Spring Shape Coefficient"]
    plantParams["Maximum Joint Stiffness"] = stiffnessRange[1]

    # Define new plant
    plant = plant_pendulum_1DOF2DOF(plantParams)

    ### Angle Step, Stiffness Step
    print("Angle Step / Stiffness Step\n")

    filePath = basePath/"angleStep_stiffStep_outputData.mat"

    if not filePath.exists(): # newTrajectory==True:
        stepDuration=1.0
        numberOfSteps=200
        trajectory = plant.generate_desired_trajectory_STEP_STEP(
            stepDuration=stepDuration,
            numberOfSteps=numberOfSteps,
            delay=delay,
            stiffnessRange=stiffnessRange
        )
        x1d = trajectory[0,:]
        sd = trajectory[1,:]
        X_o = plant.return_X_o_given_s_o(x1d[0],sd[0],[0,0])
        try:
            X,U = generate_and_save_sensory_data(
                plant,x1d,sd,X_o,
                savePath=str(filePath),
                returnOutput=True,
                trim=delay+5*stepDuration
            )
            actualTrajectory = np.array([
                X[0,:],
                np.array(list(map(plant.hs,X.T)))
            ])
            plant.plot_desired_trajectory_distribution_and_power_spectrum(actualTrajectory,cutoff=delay+3*stepDuration)
            TEMPfig1 = plant.plot_states_and_inputs(
                X,U,
                inputString="Angle Step / Stiffness Step",
                returnFig=True
            )
            TEMPfig2 = plant.plot_states_and_inputs(
                X[:,:int(3*stepDuration/plant.dt)],
                U[:,:int(3*stepDuration/plant.dt)-1],
                inputString="Angle Step / Stiffness Step",
                returnFig=True
            )
            plant.plot_tendon_tension_deformation_curves(X)
            save_figures(
                "experimental_trials/",
                "high_tendon_stiffness",
                {
                    "Extra Steps" : 5*stepDuration,
                    "stepDuration" : stepDuration,
                    "numberOfSteps" : numberOfSteps,
                    "delay" : delay,
                    "angleRange" : None,
                    "stiffnessRange" : stiffnessRange,
                    "Tendon Stiffness Coefficients" : tendonStiffnessParams
                },
                subFolderName="High_Stiffness_Experiment/Generalization_Trajectories/angleStep_stiffStep/",
                saveAsMD=True,
                addNotes="Add Notes Here."
            )
            plt.close('all')
            trajectoryCount+=1
        except:
            print("Failed Trajectory!")
            pass
    else:
        trajectoryCount+=1

    ### Angle Step, Stiffness Sinusoid
    print("Angle Step / Stiffness Sinusoid\n")

    filePath = basePath/"angleStep_stiffSin_outputData.mat"

    if not filePath.exists(): # newTrajectory==True:
        frequency = 1
        trajectory = plant.generate_desired_trajectory_STEP_SIN(
            frequency=frequency,
            numberOfSteps=100,
            stiffnessRange=stiffnessRange,
            delay=delay
        )
        x1d = trajectory[0,:]
        sd = trajectory[1,:]
        X_o = plant.return_X_o_given_s_o(x1d[0],sd[0],[0,0])
        try:
            generate_and_save_sensory_data(
                plant,x1d,sd,X_o,
                savePath=str(filePath),
                trim=delay+5*(2/frequency)
            )
            X,U = return_in_bounds_states(plant,filePath)
            actualTrajectory = np.array([
                X[0,:],
                np.array(list(map(plant.hs,X.T)))
            ])
            plant.plot_desired_trajectory_distribution_and_power_spectrum(
                actualTrajectory,
                cutoff=delay+3*(2/frequency)
            )
            TEMPfig1 = plant.plot_states_and_inputs(
                X,U,
                inputString="Angle Step / Stiffness Sinusoidal",
                returnFig=True
            )
            TEMPfig2 = plant.plot_states_and_inputs(
                X[:,:int(3*(2/frequency)/plant.dt)],
                U[:,:int(3*(2/frequency)/plant.dt)-1],
                inputString="Angle Step / Stiffness Sinusoidal",
                returnFig=True
            )
            plant.plot_tendon_tension_deformation_curves(X)
            save_figures(
                "experimental_trials/",
                "high_tendon_stiffness",
                {
                    "Extra Steps":5,
                    "Step Duration" : 2/frequency,
                    "frequency" : frequency,
                    "numberOfSteps" : 100,
                    "stiffnessRange" : stiffnessRange,
                    "angleRange" : None,
                    "delay" : delay,
                    "Tendon Stiffness Coefficients" : tendonStiffnessParams
                },
                subFolderName="High_Stiffness_Experiment/Generalization_Trajectories/angleStep_stiffSin/",
                saveAsMD=True,
                addNotes="Add Notes Here."
            )
            plt.close('all')
            trajectoryCount+=1
        except:
            print("Failed Trajectory!")
            pass
    else:
        trajectoryCount+=1

    ### Angle Sinusoid, Stiffness Step
    print("Angle Sinusoid / Stiffness Step\n")

    filePath = basePath/"angleSin_stiffStep_outputData.mat"

    if not filePath.exists(): # newTrajectory==True:
        frequency = 1
        angleRange = [3*np.pi/4,5*np.pi/4] # Quadrant 4
        trajectory = plant.generate_desired_trajectory_SIN_STEP(
            angleRange=angleRange,
            frequency=frequency,
            stiffnessRange=stiffnessRange,
            numberOfSteps=100,
            delay=delay
        )
        x1d = trajectory[0,:]
        sd = trajectory[1,:]
        X_o = plant.return_X_o_given_s_o(x1d[0],sd[0],[0,0])
        try:
            X,U = generate_and_save_sensory_data(
                plant,x1d,sd,X_o,
                savePath=str(filePath),
                returnOutput=True,
                trim=delay+3*(2/frequency)
            )
            actualTrajectory = np.array([
                X[0,:],
                np.array(list(map(plant.hs,X.T)))
            ])
            plant.plot_desired_trajectory_distribution_and_power_spectrum(
                actualTrajectory,
                cutoff=delay+3*(2/frequency)
            )
            TEMPfig1 = plant.plot_states_and_inputs(
                X,U,
                inputString="Angle Sinusoidal / Stiffness Step",
                returnFig=True
            )
            TEMPfig2 = plant.plot_states_and_inputs(
                X[:,:int(3*(2/frequency)/plant.dt)],
                U[:,:int(3*(2/frequency)/plant.dt)-1],
                inputString="Angle Sinusoidal / Stiffness Step",
                returnFig=True
            )
            plant.plot_tendon_tension_deformation_curves(X)
            save_figures(
                "experimental_trials/",
                "high_tendon_stiffness",
                {
                    "Extra Steps" : 3,
                    "Step Duration" : 2/frequency,
                    "angleRange" : angleRange,
                    "frequency" : frequency,
                    "stiffnessRange" : stiffnessRange,
                    "numberOfSteps" : 100,
                    "delay" : delay,
                    "Tendon Stiffness Coefficients" : tendonStiffnessParams
                },
                subFolderName="High_Stiffness_Experiment/Generalization_Trajectories/angleSin_stiffStep/",
                saveAsMD=True,
                addNotes="Add Notes Here."
            )
            plt.close('all')
            trajectoryCount+=1
        except:
            print("Failed Trajectory!")
            pass
    else:
        trajectoryCount+=1

    ### Angle Sinusoid, Stiffness Sinusoid
    print("Angle Sinusoid / Stiffness Sinusoid\n")

    filePath = basePath/"angleSin_stiffSin_outputData.mat"

    if not filePath.exists(): # newTrajectory==True:
        frequency = 1
        angleRange = [3*np.pi/4,5*np.pi/4] # Quadrant 4
        trajectory = plant.generate_desired_trajectory_SIN_SIN(
            stiffnessRange=stiffnessRange,
            frequency=frequency,
            angleRange=angleRange,
            delay=delay
        )
        x1d = trajectory[0,:]
        sd = trajectory[1,:]
        X_o = plant.return_X_o_given_s_o(x1d[0],sd[0],[0,0])
        try:
            X,U = generate_and_save_sensory_data(
                plant,x1d,sd,X_o,
                savePath=str(filePath),
                returnOutput=True,
                trim=delay+(10-3)/frequency # only want the last three periods
            )
            actualTrajectory = np.array([
                X[0,:],
                np.array(list(map(plant.hs,X.T)))
            ])
            plant.plot_desired_trajectory_distribution_and_power_spectrum(actualTrajectory)
            TEMPfig1 = plant.plot_states_and_inputs(
                X,U,
                InputString="Angle Sinusoidal / Stiffness Sinusoidal",
                returnFig=True
            )
            plant.plot_tendon_tension_deformation_curves(X)
            save_figures(
                "experimental_trials/",
                "high_tendon_stiffness",
                {
                    "stiffnessRange" : stiffnessRange,
                    "frequency" : frequency,
                    "angleRange" : angleRange,
                    "delay" : delay,
                    "Tendon Stiffness Coefficients" : tendonStiffnessParams
                },
                subFolderName="High_Stiffness_Experiment/Generalization_Trajectories/angleSin_stiffSin/",
                saveAsMD=True,
                addNotes="Add Notes Here."
            )
            plt.close('all')
            trajectoryCount+=1
        except:
            print("Failed Trajectory!")
            pass
    else:
        trajectoryCount+=1

    if trajectoryCount==4:
        print("All Trajectories Generated! (Plant Sweep)")
        if path.exists("slack_functions.py"):
            HAL.slack_post_message_code_completed("_New Trajectories Finished!_")
    else:
        print(f'Only {trajectoryCount:d} Trials Generated...')
        if path.exists("slack_functions.py"):
            HAL.slack_post_message_code_completed("_Problem Generating New Trajectories..._\n" + f'Only {trajectoryCount:d} Trajectories Generated...')
