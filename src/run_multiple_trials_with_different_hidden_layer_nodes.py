from test_NN_1DOF2DOA import *
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from datetime import datetime
from danpy.useful_functions import timer

if path.exists("slack_functions.py"):
    from slack_functions import *
    HAL = code_progress()

basePath = "experimental_trials/"
for fileNameBase in [
        "angleSin_stiffSin_",
        "angleStep_stiffSin_",
        "angleSin_stiffStep_",
        "angleStep_stiffStep_"
    ]:
    assert path.exists(basePath+fileNameBase+"outputData.mat"), "No experimental data to test. Please run FBL to generate data."

if __name__=='__main__':
    ### ANN parameters
    ANNParams = {
        "Number of Epochs" : 50,
        "Number of Trials" : 50,
    }

    ### Additional Arguments?
    parser = argparse.ArgumentParser(
        prog = "<filename>",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        run_multiple_trials_with_different_babbling_durations.py

        -----------------------------------------------------------------------------

        Runs multiple ANNs for different babbling durations to find the average best performance across 4 different movement tasks.

        -----------------------------------------------------------------------------'''),
        epilog=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        Written by Daniel A. Hagen (2020/01/29)

        -----------------------------------------------------------------------------'''
        )
    )
    parser.add_argument(
        '-dur',
        type=float,
        help='Number of seconds (float) for the simulation to run. Default is given by plantParams.',
        default=plantParams["Simulation Duration"]
    )
    parser.add_argument(
        '-epochs',
        type=int,
        help='Number of epochs for each network to train. Default is given by ANNParams.',
        default=ANNParams["Number of Epochs"]
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
        default=['RMSE'],
        help="Metrics to be compared. Should be either MAE, RMSE, or STD. Default is MAE."
    )
    parser.add_argument(
        '-babType',
        type=str,
        default='continuous',
        help="Type of motor babbling. Can be either 'continuous' or 'step'."
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
    plantParams["Simulation Duration"] = args.dur
    ANNParams["Number of Epochs"] = args.epochs

    # numberOfNodesList = list(np.arange(5,50+1,5))
    numberOfNodesList = list(np.arange(3,20+1,2))
    progressReportNodes = [
        numberOfNodesList[
            np.abs(
                np.array(numberOfNodesList)
                - el*(numberOfNodesList[-1]-numberOfNodesList[0])
                + numberOfNodesList[0]
            ).argmin() + 1
        ]
        for el in [0.2,0.4,0.6,0.8]
    ] # closest node number for 20% progress intervals
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
    averagePercentageSpentAwayFromEdges = []
    plotMetrics=False
    for nodeNumber in numberOfNodesList:
        trialTimer.reset()
        print("Number of Nodes: " + str(nodeNumber))
        for i in range(numberOfTrials):
            print("Running Trial " + str(i+1) + "/" + str(numberOfTrials))

            ANNParams["Number of Nodes"] = int(nodeNumber) # returned to original value.

            ANN = neural_network(ANNParams,babblingParams,plantParams)
            experimentalData, babblingOutput = ANN.run_experimental_trial(
                returnBabblingData=True,
                plotTrial=False
            )
            # experimentalData = ANN.run_experimental_trial()
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

            if plotMetrics==True:
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
                        "perf_vs_bab_dur",
                        {"metric":metric},
                        figs=figs,
                        subFolderName=metric+"/",
                        saveAsMD=True,
                        addNotes="###For a single trial, with " + str(nodeNumber) + " hidden layer nodes.\n\n"
                    )
                    plt.close('all')
            trialTimer.end_trial()
        if nodeNumber in progressReportNodes:
            try:
                progress = (np.where(
                    np.array(numberOfNodesList)==nodeNumber
                )[0][0]+1) / len(numberOfNodesList)
                HAL.slack_post_message_update(
                    progress,
                    totalTimer.totalRunTime
                )
            except:
                pass

        print("Consolidating Data from Trials where ANNs have " + str(nodeNumber) + " hidden layer nodes...")
        plot_consolidated_data_number_of_nodes_experiment(nodeNumber,metrics=args.metrics)

        totalTimer.end_trial(0)
        print(f'Run Time for {numberOfTrials} trials for {nodeNumber} hidden layer nodes: {totalTimer.trialRunTimeStr}')

    print(f'All trials completed! Total Experiment Run Time: {totalTimer.totalRunTimeStr}')

    pathName = (
        'experimental_trials/'
    )
    folderName = (
        'All_Consolidated_Trials_Nodes_Experiment_'
        + '{:03d}'.format(int(args.trials))
        + '_Trials_' + babblingParams["Babbling Type"].capitalize()
        + '_Babbling_'
        + '{:03d}'.format(int(args.dur)) + 's'
        + '/'
    )

    print("Plotting all data!")
    for metric in metrics:
        plot_number_of_nodes_vs_average_performance(metric)
        plot_number_of_nodes_vs_average_performance(metric,includeSTD=True)
        plot_number_of_nodes_vs_average_performance(metric,yscale='log')
        plot_number_of_nodes_vs_performance_STD(metric)
        plot_number_of_nodes_vs_performance_STD(metric,yscale='log')
        save_figures(
            pathName,
            "perf_v_node_no_"+metric,
            {"Number of Trials":args.trials,"Babbling Durations":args.dur,"Number of Nodes List":numberOfNodesList,"metrics":args.metrics,"Babbling Type":args.babType},
            subFolderName=folderName,
            saveAsPDF=True,
            saveAsMD=True,
            addNotes="Seeing how the number of nodes affects performance *on the same babbling data* (15.0s at 1kHz). The seed for the motor babbling has been fixed to 0 such that the neural networks will have the same information. Each iteration, the weights of the ANN will be randomized (no longer using MATLAB's initialization function `initnw`, which optimizes the initial weights for better performance). This will hopefully elucidate the affect that number of hidden nodes has on the performance of this algorithm. \n### Next Step\n Run with *random* babbling trials to see how this performs on average when weights and babbling are random."
        )
        plt.close('all')

    trialDirectories = [
        child for child in Path(pathName).iterdir()
        if child.is_dir() and child.stem[:12]=="Consolidated"
    ]
    for folder in trialDirectories:
        shutil.copytree(str(folder), pathName+folderName+"/"+folder.name)
        shutil.rmtree(str(folder))

    totalTimer.end_trial()

    if path.exists("slack_functions.py"):
        message = (
            '\n'
            + '_Test Trial Finished!!!_ \n\n'
            + 'Note that this was using frequencies of 1 Hz for all desired movements.\n\n'
            + 'Total Run Time: ' + totalTimer.totalRunTimeStr + '\n\n'
            + '```params = {\n'
            + '\t"Number of Trials" : ' + str(args.trials) + ',\n'
            + '\t"Babbling Duration" : ' + str(args.dur) + ', # in seconds\n'
            + '\t"Babbling Type" : "' + args.babType + '"\n'
            + '}```'
        )
        HAL.add_report_to_github_io(pathName+folderName+"README.md")
        HAL.slack_post_message_code_completed(message)
    # runTime = time.time()-totalStartTime
    # seconds = runTime % (24 * 3600)
    # hour = seconds // 3600
    # seconds %= 3600
    # minutes = seconds // 60
    # seconds %= 60
    # runTime = "%d:%02d:%02d" % (hour, minutes, seconds)
    #
    # if path.exists("slack_functions.py"):
    #     from slack_functions import *
    #     message = (
    #         '\n'
    #         + 'Total Run Time: ' + runTime + '\n\n'
    #         + '```params = {\n'
    #         + '\t"Number of Trials" : ' + str(args.trials) + ',\n'
    #         + '\t"Babbling Duration" : ' + str(args.dur) + ', # in seconds\n'
    #         + '\t"Babbling Type" : "' + args.babType + '"\n'
    #         + '}```'
    #     )
    #     progress_report_to_slack(
    #         __file__,
    #         message
    #     )
    #
    # print("Total Run Time: " + runTime)
