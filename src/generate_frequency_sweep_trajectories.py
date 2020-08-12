from test_NN_1DOF2DOA import *
from experimental_trials.Sweep_Plant.tendonStiffnessParams import *
from experimental_trials.Sweep_Plant.motorDampingParams import *

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

    stiffnessRange=[20,50]
    delay = 0.3

    ### FREQUENCY SWEEP

    basePath = Path("experimental_trials/Sweep_Frequency_More_Damped")

    # Return to default values
    plantParams["Spring Stiffness Coefficient"] = \
        tendonStiffnessParams["1"]["Spring Stiffness Coefficient"]
    plantParams["Spring Shape Coefficient"] = \
        tendonStiffnessParams["1"]["Spring Shape Coefficient"]
    plantParams["Motor Damping"] = motorDampingParams["2"]

    frequencies = [0.5,1,2,4]
    # angleRange = [3*np.pi/4,5*np.pi/4] # Quadrant 4
    angleRange = [10*np.pi/18,19*np.pi/18] # TEMP: to test asymmetry
    trajectoryCount = 0
    # Define new plant
    plant = plant_pendulum_1DOF2DOF(plantParams)

    for frequency in frequencies:
        # Target Folder Name
        folderName = (f'{frequency:0.1f}Hz').replace(".","_")
        print(folderName.replace("H"," H")+"\n")

        ### Angle Sinusoid, Stiffness Step
        print("Angle Sinusoid / Stiffness Step\n")

        filePath = basePath/folderName/"angleSin_stiffStep_outputData.mat"

        if not filePath.exists(): # newTrajectory==True:
            trajectory = plant.generate_desired_trajectory_SIN_STEP(
                angleRange=angleRange,
                frequency=frequency,
                stiffnessRange=stiffnessRange,
                numberOfSteps=100,
                delay=delay
            )
            x1d = trajectory[0,:]
            sd = trajectory[1,:]
            X_o = plant.return_X_o_given_s_o(x1d[0],sd[0],[8,-4]) # TEMP: Guess had to be changed for weird bounds
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
                    folderName,
                    {
                        "Extra Steps" : 3,
                        "Step Duration" : 2/frequency,
                        "angleRange" : angleRange,
                        "frequency" : frequency,
                        "stiffnessRange" : stiffnessRange,
                        "numberOfSteps" : 100,
                        "delay" : delay,
                        "Tendon Stiffness Coefficients" : tendonStiffnessParams["1"],
                        "Motor Damping" : plantParams["Motor Damping"]
                    },
                    subFolderName="Sweep_Frequency_More_Damped/"+folderName+"/Generalization_Trajectories/angleSin_stiffStep/",
                    saveAsMD=True,
                    addNotes="Add Notes Here."
                )
                plt.close('all')
                trajectoryCount+=1
                # try:
                #     HAL.slack_post_message_update(
                #         trajectoryCount/numberOfTrajectories,
                #         time.time()-startTime
                #     )
                # except:
                #     pass
            except:
                print("Failed Trajectory")
                pass
        else:
            trajectoryCount+=1

        ### Angle Sinusoid, Stiffness Sinusoid
        print("Angle Sinusoid / Stiffness Sinusoid\n")

        filePath = basePath/folderName/"angleSin_stiffSin_outputData.mat"


        if not filePath.exists(): # newTrajectory==True:
            trajectory = plant.generate_desired_trajectory_SIN_SIN(
                stiffnessRange=stiffnessRange,
                frequency=frequency,
                angleRange=angleRange,
                delay=delay
            )
            x1d = trajectory[0,:]
            sd = trajectory[1,:]
            X_o = plant.return_X_o_given_s_o(x1d[0],sd[0],[8,-4]) #TEMP: changed guess for weird bounds.
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
                    folderName,
                    {
                        "stiffnessRange" : stiffnessRange,
                        "frequency" : frequency,
                        "angleRange" : angleRange,
                        "delay" : delay,
                        "Tendon Stiffness Coefficients" : tendonStiffnessParams["1"],
                        "Motor Damping" : plantParams["Motor Damping"]
                    },
                    subFolderName="Sweep_Frequency_More_Damped/"+folderName+"/Generalization_Trajectories/angleSin_stiffSin/",
                    saveAsMD=True,
                    addNotes="Add Notes Here."
                )
                plt.close('all')
                trajectoryCount+=1
                # try:
                #     HAL.slack_post_message_update(
                #         trajectoryCount/numberOfTrajectories,
                #         time.time()-startTime
                #     )
                # except:
                #     pass
            except:
                print("Failed Trajectory!")
                pass
        else:
            trajectoryCount+=1
    if trajectoryCount==2*len(frequencies):
        print("All Trajectories Generated! (Frequency Sweep)")
    else:
        print(f'Only {trajectoryCount:d} Trials Generated...')

    if path.exists("slack_functions.py"):
        HAL.slack_post_message_code_completed("_New Trajectories Finished!_")
