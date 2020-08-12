from test_NN_1DOF2DOA import *
import matplotlib
from matplotlib.lines import Line2D
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from datetime import datetime
from danpy.useful_functions import timer
from experimental_trials.High_Stiffness_Experiment.tendonStiffnessParams import *
from matplotlib.colors import LinearSegmentedColormap

if path.exists("slack_functions.py"):
    from slack_functions import *
    HAL = code_progress()

def plot_consolidated_data_high_stiffness_experiment(
        plant,
        directory,
        metrics=None,
        includePSD=False,
        jointAngleBounds=None,
        jointStiffnessBounds=None
    ):

    if metrics is None:
        metrics = ["MAE"]
        metricKeys = ["experimentMAE"]
    else:
        assert type(metrics)==list, "metrics must be a list of strings."
        for metric in metrics:
            assert metric in ["RMSE","STD","MAE"], 'Invalid metric! metrics must include "RMSE","STD", or "MAE".'
        metricKeys = ["experiment"+metric for metric in metrics]

    # angle in radians
    if jointAngleBounds is not None:
        assert type(jointAngleBounds)==list and len(jointAngleBounds)==2, "jointAngleBounds must be a list."
        assert jointAngleBounds[1]>jointAngleBounds[0], "jointAngleBounds must be in ascending order."
    else:
        jointAngleBounds = [
            plantParams["Joint Angle Bounds"]["LB"],
            plantParams["Joint Angle Bounds"]["UB"]
        ]

    if jointStiffnessBounds is not None:
        assert type(jointStiffnessBounds)==list and len(jointStiffnessBounds)==2, "jointStiffnessBounds must be a list."
        assert jointStiffnessBounds[1]>jointStiffnessBounds[0], "jointStiffnessBounds must be in ascending order."
    else:
        jointStiffnessBounds = [10,plantParams["Maximum Joint Stiffness"]]

    ### get the testing trial directories
    directory = Path(directory)
    assert directory.is_dir(), "Enter a valid directory."

    folderName = datetime.now().strftime("Results_%Y_%m_%d-01/")
    while (directory/folderName).exists():
        folderName = (
            datetime.now().strftime("Results_%Y_%m_%d-")
            + f'{int(folderName[-3:-1])+1:02d}/'
        )

    trialDirectories = [
        child for child in directory.iterdir()
        if child.is_dir() and child.stem[:4]=="2020"
    ]
    numberOfTrials = len(trialDirectories)

    # Training Data
    totalTrainingData = {
        "all" : {},
        "bio" : {},
        "kinapprox" : {},
        "allmotor" : {}
    }

    for n in range(numberOfTrials):
        trainingDataPath = trialDirectories[n]/'trainingData.pkl'
        with trainingDataPath.open('rb') as handle:
            tempTrainingData = pickle.load(handle)
        if n == 0:
            for group in groupNames:
                totalTrainingData[group]["all_perf"] = [np.array(
                    tempTrainingData[group]["tr"]["perf"]._data
                )]
                totalTrainingData[group]["avg_best_perf"] = (
                    tempTrainingData[group]["tr"]["best_perf"]
                    / numberOfTrials
                )
                totalTrainingData[group]["best_epoch"] = [
                    tempTrainingData[group]["tr"]["best_epoch"]
                ]
        else:
            for group in groupNames:
                totalTrainingData[group]["all_perf"].append(np.array(
                    tempTrainingData[group]["tr"]["perf"]._data
                ))
                totalTrainingData[group]["avg_best_perf"] += (
                    tempTrainingData[group]["tr"]["best_perf"]
                    / numberOfTrials
                )
                totalTrainingData[group]["best_epoch"].append(
                    tempTrainingData[group]["tr"]["best_epoch"]
                )
    # numberOfEpochs = len(totalTrainingData['all']["perf"])-1

    plot_training_performance(
        totalTrainingData,
        10000,
        numberOfTrials
    )

    fig = plot_training_epoch_bar_plots(
        totalTrainingData,
        addTitle="(Plant Parameter Sweep Experiment)",
        returnFig=True
    )

    saveParams = {"Number of Trials" : numberOfTrials}
    save_figures(
        str(directory)+"/",
        "perf_v_epoch",
        saveParams,
        subFolderName=folderName,
        saveAsPDF=True,
        saveAsMD=True,
        addNotes="### Generated from `run_high_stiffness_experiment()`"
    )
    plt.close('all')

    # Experimental Data

    totalOutputData = {}
    totalRadialBins = {}
    totalAverageErrorBins = {}

    xbins = 20
    ybins = 20
    xbin_width = (jointAngleBounds[1]-jointAngleBounds[0])/xbins # in radians
    xbin_edges = np.arange(
        jointAngleBounds[0]-2*xbin_width,
        jointAngleBounds[1]+2*xbin_width+1e-3,
        xbin_width
    )# in radians
    ybin_width = (jointStiffnessBounds[1]-jointStiffnessBounds[0])/ybins
    ybin_edges = np.arange(
        jointStiffnessBounds[0]-2*ybin_width,
        jointStiffnessBounds[1]+2*ybin_width+1e-3,
        ybin_width
    )
    x_indices,y_indices = return_average_error_bin_indices_high_stiffness_experiment(
        xbin_edges,
        ybin_edges,
        plant,
        directory
    )

    radialBins = 12
    thetaRays = (180/np.pi)*np.arange(0,np.pi+1e-3,np.pi/radialBins)

    keys = ["rawError","expectedJointAngle"]
    [keys.append(metricKey) for metricKey in metricKeys]

    for n in range(numberOfTrials):
        experimentalDataPath = trialDirectories[n]/'experimentalData.pkl'
        with experimentalDataPath.open('rb') as handle:
            tempOutputData = pickle.load(handle)
        if n == 0:
            totalOutputData = tempOutputData
            """
                tempOutputData
                    ..<Movement Type>
                        ..<Group Name>
                            expectedJointAngle (in rad.)
                            predictedJointAngle (in rad.)
                            rawError (in rad.)
                            experimentRMSE (in rad.)
                            experimentMAE (in rad.)
                            experimentSTD (in rad.)
            """
            for movement in movementTypes:
                totalRadialBins[movement] = {
                    "bins" : radialBins,
                    "all" : {},
                    "bio" : {},
                    "kinapprox" : {},
                    "allmotor" : {}
                }
                tempRadialBins = return_radial_bins(
                    tempOutputData,
                    movement,
                    bins=radialBins,
                    metrics=metrics
                )
                totalAverageErrorBins[movement] = {
                    "jointAngleBounds" : jointAngleBounds,
                    "jointStiffnessBounds" : jointStiffnessBounds,
                    "xbins" : xbins,
                    "ybins" : ybins,
                    "all" : {},
                    "bio" : {},
                    "kinapprox" : {},
                    "allmotor" : {}
                }
                for key in metricKeys:
                    if key!="experimentSTD":
                        totalAverageErrorBins[movement]["max"+key[10:]] = 0
                        totalAverageErrorBins[movement]["min"+key[10:]] = 100
                        tempAverageErrorBins = return_average_error_bins(
                            tempOutputData,
                            movement,
                            key[10:],
                            x_indices,
                            y_indices
                        )
                        for group in groupNames:
                            if key[10:]=="RMSE":
                                totalAverageErrorBins[movement][group][key[10:]] =  (
                                    tempAverageErrorBins[group]**2
                                    * np.sign(tempAverageErrorBins[group])
                                    / numberOfTrials
                                )
                            elif key[10:]=="MAE":
                                totalAverageErrorBins[movement][group][key[10:]] =  (
                                    tempAverageErrorBins[group]
                                    / numberOfTrials
                                )
                for group in groupNames:
                    for key in metricKeys:
                        totalOutputData[movement][group][key+"_list"] = [
                            totalOutputData[movement][group][key]
                        ]
                        totalOutputData[movement][group][key] = (
                            totalOutputData[movement][group][key]
                            / numberOfTrials
                        )
                        if key!="experimentSTD":
                            totalRadialBins[movement]["max"+key[10:]] = 0
                            totalRadialBins[movement]["min"+key[10:]] = 100
                            for i in range(len(thetaRays)-1):
                                bin_name = f"{thetaRays[i]:0.1f} to {thetaRays[i+1]:0.1f}"
                                if bin_name in tempRadialBins[group]:
                                    if bin_name not in totalRadialBins[movement][group]:
                                        totalRadialBins[movement][group][bin_name] = {}
                                    totalRadialBins[movement][group][bin_name][key[10:]+"_list"] = [
                                        tempRadialBins[group][bin_name][key[10:]]
                                    ]

                    if includePSD==True:
                        freq, PSD = signal.welch(
                            totalOutputData[movement][group]["rawError"],
                            1/plantParams['dt']
                        )
                        totalOutputData[movement][group]["frequencies"] = freq
                        totalOutputData[movement][group]["avg_PSD"] = PSD/numberOfTrials
                    for key in [
                            "rawError",
                            "expectedJointAngle",
                            "predictedJointAngle",
                            "experimentMAE",
                            "experimentRMSE",
                            "experimentSTD"
                        ]:
                        if key not in metricKeys:
                            del(totalOutputData[movement][group][key])
        else:
            for movement in movementTypes:
                tempRadialBins = return_radial_bins(
                    tempOutputData,
                    movement,
                    bins=radialBins,
                    metrics=metrics
                )
                for key in metricKeys:
                    if key!="experimentSTD":
                        tempAverageErrorBins = return_average_error_bins(
                            tempOutputData,
                            movement,
                            key[10:],
                            x_indices,
                            y_indices
                        )
                        for group in groupNames:
                            if key[10:]=="RMSE":
                                totalAverageErrorBins[movement][group][key[10:]] +=  (
                                    tempAverageErrorBins[group]**2
                                    * np.sign(tempAverageErrorBins[group])
                                    / numberOfTrials
                                )
                            elif key[10:]=="MAE":
                                totalAverageErrorBins[movement][group][key[10:]] +=  (
                                    tempAverageErrorBins[group]
                                    / numberOfTrials
                                )
                for group in groupNames:
                    for key in metricKeys:
                        totalOutputData[movement][group][key+"_list"].append(
                            tempOutputData[movement][group][key]
                        )
                        totalOutputData[movement][group][key] += (
                            tempOutputData[movement][group][key]
                            / numberOfTrials
                        )
                        if key!="experimentSTD":
                            for i in range(len(thetaRays)-1):
                                bin_name = f"{thetaRays[i]:0.1f} to {thetaRays[i+1]:0.1f}"
                                if bin_name in tempRadialBins[group]:
                                    totalRadialBins[movement][group][bin_name][key[10:]+"_list"].append(
                                        tempRadialBins[group][bin_name][key[10:]]
                                    )
                    if includePSD==True:
                        _, PSD = signal.welch(
                            tempOutputData[movement][group]["rawError"],
                            1/plantParams['dt']
                        )
                        totalOutputData[movement][group]["avg_PSD"] += PSD/numberOfTrials

        # delete trial directory
        # shutil.rmtree(trialDirectories[n]) # TEMP: keeping trials 2020/05/15

        """
            totalOutputData
                ..<Movement Type>
                    ..<Group Name>
                        [frequencies] (in Hz.)
                        [avg_PSD] (in rad.^2/Hz.)
                        experiment<Metric> (in rad.)
                        experiment<Metric>_list (in rad.)
        """
    for movement in movementTypes:
        for group in groupNames:

            if "MAE" in metrics:
                tempAE = (
                    (-1e5)
                    * totalAverageErrorBins[movement][group]["MAE"]
                    * (totalAverageErrorBins[movement][group]["MAE"]==-1)
                    + totalAverageErrorBins[movement][group]["MAE"]
                    * (totalAverageErrorBins[movement][group]["MAE"]>0)
                )
                totalAverageErrorBins[movement]["minMAE"] = min([
                    tempAE.min(),
                    totalAverageErrorBins[movement]["minMAE"]
                ])
                totalAverageErrorBins[movement]["maxMAE"] = max([
                    totalAverageErrorBins[movement][group]["MAE"].max(),
                    totalAverageErrorBins[movement]["maxMAE"]
                ])

            if "RMSE" in metrics:
                tempAE = totalAverageErrorBins[movement][group]["RMSE"]
                for i in range(xbins):
                    for j in range(ybins):
                        if tempAE[i,j]>0:
                            totalAverageErrorBins[movement][group]["RMSE"][i,j] = np.sqrt(tempAE[i,j])
                tempAE = (
                    (-1e5)
                    * totalAverageErrorBins[movement][group]["RMSE"]
                    * (totalAverageErrorBins[movement][group]["RMSE"]==-1)
                    + totalAverageErrorBins[movement][group]["RMSE"]
                    * (totalAverageErrorBins[movement][group]["RMSE"]>0)
                )
                totalAverageErrorBins[movement]["minRMSE"] = min([
                    tempAE.min(),
                    totalAverageErrorBins[movement]["minRMSE"]
                ])
                totalAverageErrorBins[movement]["maxRMSE"] = max([
                    totalAverageErrorBins[movement][group]["RMSE"].max(),
                    totalAverageErrorBins[movement]["maxRMSE"]
                ])

            for i in range(len(thetaRays)-1):
                bin_name = f"{thetaRays[i]:0.1f} to {thetaRays[i+1]:0.1f}"
                if bin_name in totalRadialBins[movement][group]:
                    if "MAE" in metrics:
                        totalRadialBins[movement][group][bin_name]["MAE"] = np.mean(totalRadialBins[movement][group][bin_name]["MAE_list"])
                        totalRadialBins[movement]["minMAE"] = min([
                            totalRadialBins[movement]["minMAE"],
                            totalRadialBins[movement][group][bin_name]["MAE"]
                        ])
                        totalRadialBins[movement]["maxMAE"] = max([
                            totalRadialBins[movement]["maxMAE"],
                            totalRadialBins[movement][group][bin_name]["MAE"]
                        ])
                    if "RMSE" in metrics:
                        totalRadialBins[movement][group][bin_name]["RMSE"] = np.sqrt(np.mean([
                            el**2
                            for el in totalRadialBins[movement][group][bin_name]["RMSE_list"]
                        ]))
                        totalRadialBins[movement]["minRMSE"] = min([
                            totalRadialBins[movement]["minRMSE"],
                            totalRadialBins[movement][group][bin_name]["RMSE"]
                        ])
                        totalRadialBins[movement]["maxRMSE"] = max([
                            totalRadialBins[movement]["maxRMSE"],
                            totalRadialBins[movement][group][bin_name]["RMSE"]
                        ])

    # plot_all_error_distributions(totalOutputData,returnFigs=True)
    #
    # save_figures(
    #     directory+folderName,
    #     "err_dist",
    #     {},
    #     subFolderName="Error_Distributions/",
    #     saveAsMD=True
    # )
    # plt.close('all')

    if includePSD==True:
        figs = plot_average_error_signal_power_spectrums(totalOutputData,returnFigs=True)

        save_figures(
            str(directory)+"/",
            "err_PSD",
            {},
            figs=figs,
            subFolderName=folderName+"/",
            saveAsMD=True,
            addNotes="Average error signal with power spectrum analysis."
        )
        plt.close('all')

    for metric in metrics:
        fig = plot_bar_plots(
            totalOutputData,
            metric=metric,
            returnFig=True,
            yscale='log'
        )
        figs = [fig]
        tempFigs1 = plot_metric_distributions(
            totalOutputData,
            metric,
            returnFigs=True
        )
        [figs.append(fig) for fig in tempFigs1]
        if metric!="STD":
            tempFigs2 = plot_all_polar_bar_plots(
                None,
                metric,
                totalRadialBins,
                returnFigs=True
            )
            [figs.append(fig) for fig in tempFigs2]

            tempFigs3 = plot_2D_heatmap_of_error_wrt_desired_trajectory(
                None,
                metric,
                plant,
                totalAverageErrorBins,
                returnFigs=True,
                xbins=xbins+4,
                ybins=ybins+4,
                jointAngleBounds=[
                    xbin_edges[0],
                    xbin_edges[-1]
                ],
                jointStiffnessBounds=[
                    ybin_edges[0],
                    ybin_edges[-1]
                ]
            )
            [figs.append(fig) for fig in tempFigs3]

        save_figures(
            str(directory)+"/",
            metric,
            {"metric":metric},
            figs=figs,
            subFolderName=folderName+"/",
            saveAsPDF=True,
            saveAsMD=True,
            addNotes="Figures for metric _" + metric + "_."
        )
        plt.close('all')
    """
    Consolidated to include:
        - AVERAGE metrics (this is not the same as the metric average for all data points, but instead the average metric across trials)
        - metric lists
    """
    consolidatedOutputData = {}
    for movement in movementTypes:
        consolidatedOutputData[movement]={}
        for group in groupNames:
            consolidatedOutputData[movement][group]={}
            for metric in metricKeys:
                consolidatedOutputData[movement][group][metric] = \
                    totalOutputData[movement][group][metric]
                consolidatedOutputData[movement][group][metric+"_list"] = \
                    totalOutputData[movement][group][metric+"_list"]
    fileName = (
        "consolidatedOutputData.pkl"
    )
    with open(directory/folderName/fileName, 'wb') as handle:
        pickle.dump(
            consolidatedOutputData,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL
        )

def return_average_error_bin_indices_high_stiffness_experiment(xbin_edges,ybin_edges,plant,path):
    x_indices = dict(zip(movementTypes,[[]]*4))
    x_indices["xbin_edges"] = xbin_edges
    y_indices = dict(zip(movementTypes,[[]]*4))
    y_indices["ybin_edges"] = ybin_edges

    path=Path(path)
    assert path.exists() and path.is_dir(), "The directory " + str(path) + " does not exist."

    for movement in movementTypes:
        movementOutput = sio.loadmat(path / f"{movement}_outputData.mat")
        xd = movementOutput["x1"][0] # radians
        X = np.array([
            movementOutput["x1"].T,
            movementOutput["dx1"].T,
            movementOutput["x3"].T,
            movementOutput["dx3"].T,
            movementOutput["x5"].T,
            movementOutput["dx5"].T
        ])[:,:,0]
        sd = np.array(list(map(plant.hs,X.T)))
        x_index = np.clip(
            np.digitize(xd,xbin_edges)-1,
            0,
            len(xbin_edges)-2
        )
        y_index = np.clip(
            np.digitize(sd,ybin_edges)-1,
            0,
            len(ybin_edges)-2
        )
        for j in range(len(x_index)):
            if x_index[j]==len(xbin_edges)-1:
                x_index[j]-=1
            if y_index[j]==len(ybin_edges)-1:
                y_index[j]-=1

        x_indices[movement]=x_index
        y_indices[movement]=y_index
    return(x_indices,y_indices)

basePath = "experimental_trials/High_Stiffness_Experiment"

if __name__=='__main__':

    ### Delete the parameters that are not used in Babbling Trials
    if "Boundary Friction Weight" in plantParams.keys():
        del plantParams["Boundary Friction Weight"]
    if "Boundary Friction Gain" in plantParams.keys():
        del plantParams["Boundary Friction Gain"]
    if "Quadratic Stiffness Coefficient 1" in plantParams.keys():
        del plantParams["Quadratic Stiffness Coefficient 1"]
    if "Quadratic Stiffness Coefficient 2" in plantParams.keys():
        del plantParams["Quadratic Stiffness Coefficient 2"]

    # NOTE: These values were chosen because they showed consistently good performance. Additionally, we effectively removed the upper limit for epochs to allow for convergence. We know that these parameters are a good choice because we see small standard deviations in the performance across trials and we noticed that, while fewer nodes may produce similar results, the average number of epochs was consistent. Similarly, we could argue that 10 seconds of babbling would be sufficient to get the desired performance for 15 internal nodes (instead of the default 15 seconds). But we again see that the number of epochs is consistent (meaning we would not be adding much more computational time). Therefore, the choice of 15 seconds was made in order to (1) minimize the standard deviation of the performance and (2) to avoid "poor babbling" trials that happen with some regularity for trials under 10 seconds.

    ### ANN parameters
    ANNParams = {
        "Number of Nodes" : 15,
        "Number of Epochs" : 10000,
        "Number of Trials" : 50,
    }

    ### plant parameters
    plantParams["Simulation Duration"] = 15

    ### Additional Arguments?
    parser = argparse.ArgumentParser(
        prog = "<filename>",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        run_high_stiffness_experiment.py

        -----------------------------------------------------------------------------

        HIGH HIGH STIFFNESS... *gulp*

        -----------------------------------------------------------------------------'''),
        epilog=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        Written by Daniel A. Hagen (2020/05/14)

        -----------------------------------------------------------------------------'''
        )
    )
    parser.add_argument(
        '-epochs',
        type=int,
        help='Number of epochs for each network to train. Default is given by ANNParams.',
        default=ANNParams["Number of Epochs"]
    )
    parser.add_argument(
        '-nodes',
        type=int,
        help='Number of Nodes for each network to train (single hidden layer). Default is given by ANNParams.',
        default=ANNParams["Number of Nodes"]
    )
    parser.add_argument(
        '-trials',
        type=int,
        default=1,
        help='Number of trials to run. Default is 1.'
    )
    parser.add_argument(
        '-metrics',
        type=str,
        nargs="+",
        default=['MAE'],
        help="Metrics to be compared. Should be either MAE, RMSE, or STD. Default is MAE."
    )
    parser.add_argument(
        '-babType',
        type=str,
        default='continuous',
        help="Type of motor babbling. Can be either 'continuous' or 'step'."
    )
    parser.add_argument(
        '--sampleBabbling',
        action="store_true",
        help='Run and plot a sample babbling trial.'
    )
    args = parser.parse_args()
    if type(args.metrics)==str:
        metrics = [args.metrics]
    else:
        metrics = args.metrics
    for metric in metrics:
        assert metric in ["RMSE","MAE","STD"], "Invalid metric! Must be either 'RMSE', 'MAE', or 'STD'"

    assert args.babType in ['continuous','step'], "babType must be either 'continuous' (default) or 'step'."
    babblingParams['Babbling Type'] = args.babType
    ANNParams["Number of Nodes"] = args.nodes
    ANNParams["Number of Epochs"] = args.epochs
    ANNParams["Number of Trials"] = args.trials
    numberOfTrials = args.trials

    groupNames = [
        "all",
        "bio",
        "kinapprox",
        "allmotor"
    ]

    movementTypes = [
        "angleSin_stiffSin",
        "angleStep_stiffSin",
        "angleSin_stiffStep",
        "angleStep_stiffStep"
    ]

    trialTimer = timer()
    totalTimer = timer()
    plantParams["Spring Stiffness Coefficient"] = \
        tendonStiffnessParams["Spring Stiffness Coefficient"]
    plantParams["Spring Shape Coefficient"] = \
        tendonStiffnessParams["Spring Shape Coefficient"]
    stiffnessRange = [150,650]
    plantParams["Maximum Joint Stiffness"] = stiffnessRange[1]

    # Define new plant
    plant = plant_pendulum_1DOF2DOF(plantParams)

    if args.sampleBabbling==True:
        ANN = neural_network(ANNParams,babblingParams,plantParams)
        ANN.plant.dt=ANN.plant.dt/50
        tempTime = ANN.plant.time
        ANN.plant.time=np.arange(0,ANN.plant.time[-1]+ANN.plant.dt,ANN.plant.dt)
        # Generate babbling data
        babblingTrial = motor_babbling_1DOF2DOA(
            ANN.plant,
            ANN.totalParams
        )
        babblingOutput = babblingTrial.run_babbling_trial(
            np.pi,
            saveFigures=False,
            saveAsPDF=False,
            returnData=True,
            saveData=False,
            saveParams=False
        )

        ANN.plant.dt=ANN.plant.dt*50
        ANN.plant.time=tempTime
        babblingTrial.babblingSignals=babblingTrial.babblingSignals[::50,:]
        Time = babblingOutput["time"][::50]
        X = babblingOutput["X"][:,::50]
        U = babblingOutput["U"][:,::50]

        # self.plant.plot_states(X,Return=True)
        babblingTrial.plant.plot_states_and_inputs(X,U,returnFig=True)

        babblingTrial.plant.plot_output_power_spectrum_and_distribution(X,returnFigs=True)

        babblingTrial.plant.plot_tendon_tension_deformation_curves(X)

        babblingTrial.plot_signals_power_spectrum_and_amplitude_distribution()

        save_figures(
            "experimental_trials/High_Stiffness_Experiment/",
            "babbling_sample",
            ANN.totalParams,
            subFolderName="sample_babbling_plots/",
            returnPath=False,
            saveAsPDF=False,
            saveAsMD=True
        )
        plt.close('all')
    else:
        # Target Folder Name
        for fileNameBase in [
                "angleSin_stiffSin_",
                "angleStep_stiffSin_",
                "angleSin_stiffStep_",
                "angleStep_stiffStep_"
            ]:
            assert path.exists(basePath+'/'+fileNameBase+"outputData.mat"), "No experimental data to test in " + basePath+"/. Please run FBL to generate data."

        trialTimer.reset()
        for n in range(numberOfTrials):
            print(f"Running Trial {str(n+1)}/{str(numberOfTrials)}")

            ANN = neural_network(ANNParams,babblingParams,plantParams)
            experimentalData = ANN.run_experimental_trial(
                upsample=True,
                savePath = basePath+'/'
            )
            """
                experimentalData
                    ..<Group Name>
                        ..<Movement Type>
                            ..expectedJointAngle (in rad.)
                            ..predictedJointAngle (in rad.)
                            ..rawError (in rad.)
                            ..experimentRMSE (in rad.)
                            ..experimentMAE (in rad.)
                            ..experimentSTD (in rad.)
            """

            # SAVE EXPERIMENTAL DATA TO TRIAL FOLDER
            formattedData = {}
            for movement in movementTypes:
                formattedData[movement] = {}
                for group in groupNames:
                    formattedData[movement][group] = {}
                    for key in experimentalData[group][movement]:
                        if type(experimentalData[group][movement][key])==float:
                            formattedData[movement][group][key] = \
                                experimentalData[group][movement][key]
                        else:
                            formattedData[movement][group][key] = np.array(
                                experimentalData[group][movement][key]._data
                            )
            experimentalData = formattedData
            """
                experimentalData
                    ..<Movement Type>
                        ..<Group Name>
                            ..expectedJointAngle (in rad.)
                            ..predictedJointAngle (in rad.)
                            ..rawError (in rad.)
                            ..experimentRMSE (in rad.)
                            ..experimentMAE (in rad.)
                            ..experimentSTD (in rad.)
            """
            with open(path.join(ANN.trialPath,'experimentalData.pkl'), 'wb') as handle:
                pickle.dump(experimentalData, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # ### Plot experimental data
            # figs = plot_experimental_data(experimentalData,returnFigs=True)

            for metric in metrics:
                ### bar plots
                fig = plot_bar_plots(experimentalData,metric=metric,yscale='log')

                ### radial average error versus positions
                figs = [fig]
                # newFigs=plot_all_polar_bar_plots(experimentalData,metric)
                # [figs.append(fig) for fig in newFigs]

                ### save figs
                save_figures(
                    ANN.trialPath,
                    metric,
                    {"metric":metric},
                    figs=figs,
                    subFolderName=metric+"/",
                    saveAsMD=True,
                    addNotes="### Experiment to test effects of very large stiffness values.\n\n"
                )
                plt.close('all')
            print('\a')
            trialTimer.end_trial()

        print("Consolidating Data for High Stiffness Experiment")
        plot_consolidated_data_high_stiffness_experiment(
            plant,
            basePath+'/',
            metrics=args.metrics
        )
        totalTimer.end_trial()

        print(f'All trials completed! Total Experiment Run Time: {totalTimer.totalRunTimeStr}')

        if path.exists("slack_functions.py"):
            message = (
                '\n'
                + '_High Stiffness Experiment Completed!!!_ \n\n'
                + 'Note that this was using frequencies of 1 Hz for all desired movements.\n\n'
                + 'Total Run Time: ' + totalTimer.totalRunTimeStr + '\n\n'
                + '```params = {\n'
                + '\t"Number of Trials" : ' + str(args.trials) + ',\n'
                + '\t"Number of Epochs" : ' + str(args.epochs) + ',\n'
                + '\t"Number of Nodes" : ' + str(args.nodes) + ',\n'
                + '\t"Babbling Duration" : ' + str(plantParams['Simulation Duration']) + ',\n'
                + '\t"Babbling Type" : "continuous"\n'
                + '}```'
            )
            HAL.add_report_to_github_io(basePath+'/'+"README.md")
            HAL.slack_post_message_code_completed(message)
