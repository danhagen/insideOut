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
        "Number of Nodes" : 15,
        "Number of Epochs" : 10000,
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
    ANNParams["Number of Nodes"] = args.nodes
    ANNParams["Number of Epochs"] = args.epochs

    # babblingDurations = list(np.arange(30,360+1,15))
    # babblingDurations = list(np.arange(1,15+1,2))
    # [babblingDurations.append(el) for el in [20,25,30]]
    babblingDurations = [7.5] + list(np.arange(10,30+1e-3,5))
    # babblingDurations.insert(0,1)

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

    plant = plant_pendulum_1DOF2DOF(plantParams)
    trialTimer = timer()
    totalTimer = timer()
    for dur in babblingDurations:
        # startTime = time.time()
        # trialStartTime = startTime
        trialTimer.reset()
        print(f"Babbling Duration: {str(dur)}s")
        for i in range(numberOfTrials):
            print(f"Running Trial {str(i+1)}/{str(numberOfTrials)}")

            plantParams["Simulation Duration"] = int(dur) # returned to original value.

            ANN = neural_network(ANNParams,babblingParams,plantParams)
            experimentalData = ANN.run_experimental_trial()
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
                    "perf_vs_bab_dur",
                    {"metric":metric},
                    figs=figs,
                    subFolderName=metric+"/",
                    saveAsMD=True,
                    addNotes="### For a single trial, with babbling duration of " + str(dur) + " seconds.\n\n"
                )
                plt.close('all')
            print('\a')
            trialTimer.end_trial()

        print("Consolidating Data from " + str(dur) + "s Babbling Trials...")
        plot_consolidated_data_babbling_duration_experiment(dur,plant,metrics=args.metrics)
        try:
            percentDone = (np.where(dur==babblingDurations)[0][0]+1)/len(babblingDurations)
            HAL.slack_post_message_update(
                percentDone,
                time.time()-totalStartTime
            )
        except:
            pass

        totalTimer.end_trial(0)
        print(f'Run Time for {numberOfTrials} trials of {dur}s motor babbling: {totalTimer.trialRunTimeStr}')
        HAL.slack_post_message_update(
            (np.where(dur==np.array(babblingDurations))[0][0]+1)/len(babblingDurations),
            totalTimer.totalRunTime
        )

    print(f'All trials completed! Total Experiment Run Time: {totalTimer.totalRunTimeStr}')

    pathName = (
        'experimental_trials/'
    )
    folderName = (
        'All_Consolidated_Trials_'
        + '{:03d}'.format(int(args.trials)) + '_Trials_'
        + babblingParams["Babbling Type"].capitalize() + '_Babbling_'
        + '{:03d}'.format(int(args.nodes)) + "_Nodes"
    )
    print("Plotting all data!")
    for metric in metrics:
        plot_babbling_duration_vs_average_performance(metric)
        plot_babbling_duration_vs_average_performance(metric,includeSTD=True)
        plot_babbling_duration_vs_average_performance(metric,yscale='log')
        plot_babbling_duration_vs_performance_STD(metric)
        plot_babbling_duration_vs_performance_STD(metric,yscale='log')
        save_figures(
            pathName,
            "perf_v_bab_dur_"+metric,
            {"Number of Trials":args.trials,"Babbling Durations":babblingDurations,"Number of Nodes":args.nodes,"metrics":args.metrics,"Babbling Type":args.babType,"Number of Epochs":args.epochs},
            subFolderName=folderName,
            saveAsPDF=True,
            saveAsMD=True,
            addNotes="Consolidated trials for babbling duration sweep"
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
            + '\t"Number of Nodes" : ' + str(args.nodes) + ',\n'
            + '\t"Babbling Type" : "continuous"\n'
            + '}```'
        )
        HAL.add_report_to_github_io(pathName+folderName+"/README.md")
        HAL.slack_post_message_code_completed(message)
