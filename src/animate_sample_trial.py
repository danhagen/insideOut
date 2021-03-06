from motor_babbling_1DOF2DOA import *
from build_NN_1DOF2DOA import *
from save_params import *
import os
import matlab.engine
import argparse
import textwrap
from danpy.sb import get_terminal_width
from os import path,listdir
from matplotlib.patches import Wedge
import scipy.io as sio
import pickle
from PIL import Image
import time
import shutil
from animate_ANN import *
from animate_babbling import *
# plt.rc('text', usetex=True)
colors = [
    "#2A3179", # all
    "#F4793B", # bio
    "#8DBDE6", # kinapprox
    "#A95AA1" # allmotor
]

def generate_and_save_sensory_data(plant,x1d,sd,savePath=None):
    X1d = np.zeros((5,len(plant.time)))
    X1d[0,:] = LP_filt(100,x1d)
    X1d[1,:] = np.gradient(X1d[0,:],plant.dt)
    X1d[2,:] = np.gradient(X1d[1,:],plant.dt)
    X1d[3,:] = np.gradient(X1d[2,:],plant.dt)
    X1d[4,:] = np.gradient(X1d[3,:],plant.dt)

    Sd = np.zeros((3,len(plant.time)))
    Sd[0,:] = LP_filt(100,sd)
    Sd[1,:] = np.gradient(Sd[0,:],plant.dt)
    Sd[2,:] = np.gradient(Sd[1,:],plant.dt)

    X,U,Y,_ = plant.forward_simulation_FL(X_o,X1d,Sd)

    additionalDict = {
        "X1d" : x1d,
        "Sd" : sd,
        "Y" : Y,
        "U" : U,
        "X" : X
    }

    plant.save_data(X,U,additionalDict=additionalDict,filePath=savePath)

def plot_experimental_data(experimentalData,returnFigs=True):
    # Sin Angle/Sin Stiffness
    fig1, (ax1a,ax1b) = plt.subplots(2,1,figsize=(8,6),sharex=True)
    plt.suptitle("Sinusoidal Angle / Sinusoidal Stiffness")

    # Step Angle/Sin Stiffness
    fig2, (ax2a,ax2b) = plt.subplots(2,1,figsize=(8,6),sharex=True)
    plt.suptitle("Step Angle / Sinusoidal Stiffness")

    # Sin Angle/Step Stiffness
    fig3, (ax3a,ax3b) = plt.subplots(2,1,figsize=(8,6),sharex=True)
    plt.suptitle("Sinusoidal Angle / Step Stiffness")

    # Step Angle/Step Stiffness
    fig4, (ax4a,ax4b) = plt.subplots(2,1,figsize=(8,6),sharex=True)
    plt.suptitle("Step Angle / Step Stiffness")

    figs = [fig1,fig2,fig3,fig4]
    top_axs = [ax1a,ax2a,ax3a,ax4a]
    bot_axs = [ax1b,ax2b,ax3b,ax4b]
    subkeys = list(experimentalData['all'].keys())

    for i in range(4):
        top_axs[i].set_ylabel("Joint Angle (deg.)")
        top_axs[i].spines["right"].set_visible(False)
        top_axs[i].spines["top"].set_visible(False)
        top_axs[i].plot(
            plant.time,
            (
                180/np.pi
                *np.array(
                    experimentalData['all'][subkeys[i]]["expectedJointAngle"]
                ).T
            ),
            c='0.70',
            lw=2
        )
        bot_axs[i].set_xlabel("Time (s)")
        bot_axs[i].set_ylabel("Joint Angle Error (deg.)")
        bot_axs[i].spines["right"].set_visible(False)
        bot_axs[i].spines["top"].set_visible(False)

        for key in experimentalData.keys():
            index = np.where(
                key==np.array(list(experimentalData.keys()))
            )[0][0]
            top_axs[i].plot(
                plant.time,
                (
                    180/np.pi
                    * np.array(
                        experimentalData[key][subkeys[i]]["predictedJointAngle"]
                    ).T
                ),
                c=colors[index]
            )
            bot_axs[i].plot(
                plant.time,
                (
                    180/np.pi
                    * np.array(
                        experimentalData[key][subkeys[i]]["rawError"]
                    ).T
                ),
                c=colors[index]
            )

        legendList = list(experimentalData.keys())
        legendList.insert(0,'Desired')
        ax1a.legend(legendList,loc="upper right")
        ax2a.legend(legendList,loc="upper right")
        ax3a.legend(legendList,loc="upper right")
        ax4a.legend(legendList,loc="upper right")

    if returnFigs==True:
        return(figs)
    else:
        plt.show()

def plot_training_performance(
        trainingData,
        numberOfEpochs,
        numberOfTrials,
        returnFig=True
    ):

    epochArray = np.arange(0,numberOfEpochs+1,1)

    fig = plt.figure(figsize=(8,6))
    plt.yscale("log")
    plt.title("Average Performance vs. Epoch\n" + "(" + str(numberOfTrials) + " Trials)")
    ax = plt.gca()
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Performance (RMSE in deg.)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xticks(list(np.linspace(0,numberOfEpochs,6)))
    ax.set_xticklabels([int(el) for el in ax.get_xticks()])

    for i in range(len(trainingData.keys())):
        key = list(trainingData.keys())[i]
        ax.plot(
            epochArray,
            180*np.sqrt(trainingData[key]["perf"])/np.pi,
            c=colors[i],
            lw=2
        )
    ax.legend(list(trainingData.keys()),loc='upper right')

    if returnFig==True:
        return(fig)
    else:
        plt.show()

def plot_bar_plots(outputData,metric="MAE",returnFig=True):
    assert type(metric)==str and metric in ["MAE","STD","RMSE"], "metric must be either 'MAE' (default), 'RMSE' or 'STD'."
    if metric == "MAE":
        baseTitle = "Bar Plots of MAE by Movement Type"
        ylabel = "Mean Absolute Error (deg.)"
        valueKey = "experimentMAE"
    elif metric == 'STD':
        baseTitle = "Bar Plots of Error Std Dev by Movement Type"
        ylabel = "Error Standard Deviation (deg.)"
        valueKey = "experimentSTD"
    elif metric == 'RMSE':
        baseTitle = "Bar Plots of RMSE by Movement Type"
        ylabel = "Root Mean Squared Error (deg.)"
        valueKey = "experimentRMSE"

    labels = [
        "Sinusoidal Angle \n Sinusoidal Stiffness",
        "Step Angle \n Sinusoidal Stiffness",
        "Sinusoidal Angle \n Step Stiffness",
        "Step Angle \n Step Stiffness"
    ]

    allValue = [
        (180/np.pi)*outputData[key]["all"][valueKey]
        for key in outputData.keys()
    ]
    bioValue = [
        (180/np.pi)*outputData[key]["bio"][valueKey]
        for key in outputData.keys()
    ]
    kinapproxValue = [
        (180/np.pi)*outputData[key]["kinapprox"][valueKey]
        for key in outputData.keys()
    ]
    allmotorValue = [
        (180/np.pi)*outputData[key]["allmotor"][valueKey]
        for key in outputData.keys()
    ]

    xticks = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(12,5))
    rects1 = ax.bar(
        xticks - 3*width/2, allValue, width,
        label="all", color=colors[0]
    )
    rects2 = ax.bar(
        xticks - width/2, bioValue, width,
        label="bio", color=colors[1]
    )
    rects3 = ax.bar(
        xticks + width/2, kinapproxValue, width,
        label="kinapprox", color=colors[2]
    )
    rects4 = ax.bar(
        xticks + 3*width/2, allmotorValue, width,
        label="allmotor", color=colors[3]
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(baseTitle)
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if returnFig==True:
        return(fig)
    else:
        plt.show()

def return_radial_bins(errorArrays,jointAngleArrays,bins=12):
    thetaRays = np.arange(0,np.pi+1e-3,np.pi/bins)
    groups = ["all","bio","kinapprox","allmotor"]
    radial_bins={
        "bins" : bins,
        "maxMAE" : 0,
        "maxRMSE" : 0,
        "maxSTD" : 0,
        "all" : {},
        "bio" : {},
        "kinapprox" : {},
        "allmotor" : {}
    }
    for i in range(len(groups)):
        tempJointAngle = (jointAngleArrays[i,:]-np.pi/2).flatten()
        for j in range(len(thetaRays)-1):
            bin_name = (
                '{:0.1f}'.format(180*thetaRays[j]/np.pi)
                + " to "
                + '{:0.1f}'.format(180*thetaRays[j+1]/np.pi)
            )
            radial_bins[groups[i]][bin_name] = {}
            indices = np.array(
                np.where(
                    np.logical_and(
                        tempJointAngle<thetaRays[j+1],
                        tempJointAngle>=thetaRays[j]
                    )
                )
            )
            radial_bins[groups[i]][bin_name]["abs errors"] = np.array([
                (180/np.pi)*abs(errorArrays[i,k])
                for k in indices
            ]) # in degrees
            radial_bins[groups[i]][bin_name]["errors"] = np.array([
                (180/np.pi)*errorArrays[i,k]
                for k in indices
            ]) # in degrees

            ### Mean absolute error
            radial_bins[groups[i]][bin_name]["MAE"] = \
                radial_bins[groups[i]][bin_name]["abs errors"].mean() # in degrees
            radial_bins["maxMAE"] = max([
                radial_bins["maxMAE"],
                radial_bins[groups[i]][bin_name]["MAE"]
            ]) # in degrees

            ### Root mean squared error
            radial_bins[groups[i]][bin_name]["RMSE"] = np.sqrt(
                (radial_bins[groups[i]][bin_name]["errors"]**2).mean()
            ) # in degrees
            radial_bins["maxRMSE"] = max([
                radial_bins["maxRMSE"],
                radial_bins[groups[i]][bin_name]["RMSE"]
            ]) # in degrees
            radial_bins[groups[i]][bin_name]["abs error std"] = \
                radial_bins[groups[i]][bin_name]["abs errors"].std() # in degrees
            radial_bins[groups[i]][bin_name]["min abs error"] = \
                radial_bins[groups[i]][bin_name]["errors"].min() # in degrees
            radial_bins[groups[i]][bin_name]["max abs error"] = \
                radial_bins[groups[i]][bin_name]["errors"].max() # in degrees

            radial_bins[groups[i]][bin_name]["avg error"] = \
                radial_bins[groups[i]][bin_name]["errors"].mean() # in degrees
            radial_bins[groups[i]][bin_name]["STD"] = \
                radial_bins[groups[i]][bin_name]["errors"].std() # in degrees
            radial_bins["maxSTD"] = max([
                radial_bins["maxSTD"],
                radial_bins[groups[i]][bin_name]["STD"]
            ]) # in degrees
    return(radial_bins)

def plot_polar_bar_plots(
        radial_bins,
        metric="MAE",
        addTitle=None,
        returnFig=False
    ):

    assert type(metric)==str and metric in ["MAE","STD","RMSE"], "metric must be either 'MAE' (default) or 'STD'."
    if metric == "MAE":
        baseTitle = "Polar Bar Plots of MAE vs. Joint Angle"
        xLabel = "Log MAE (in deg.)"
        maxValue = np.log10(radial_bins["maxMAE"])+2
        offset = 2
    elif metric == 'STD':
        baseTitle = "Polar Bar Plots of Error Std Dev vs. Joint Angle"
        xLabel = "Log Error Std Dev (in deg.)"
        maxValue = np.log10(radial_bins["maxSTD"])+2
        offset = 2
    elif metric == 'RMSE':
        baseTitle = "Polar Bar Plots of RMSE vs. Joint Angle"
        xLabel = "Log RMSE (in deg.)"
        maxValue = np.log10(radial_bins["maxRMSE"])+2
        offset = 2

    # assert maxValue<3.3, "Bounds not configured for values this large. Please check values again and determine if bounds need to be changed."

    if addTitle is None:
        title = baseTitle
    else:
        assert type(addTitle)==str, "title must be a string."
        title = baseTitle + "\n" + addTitle

    subTitles = [
        "All\nAvailable\nStates",
        "The\nBio-Inspired\nSet",
        "Motor Position\nand\nVelocity Only",
        "All\nMotor\nStates"
    ]
    fig = plt.figure(figsize=(20,12))
    plt.suptitle(title,fontsize=14)
    ax1=plt.subplot(221)
    ax2=plt.subplot(222)
    ax3=plt.subplot(223)
    ax3.set_xlabel(xLabel,ha="center")
    ax3.xaxis.set_label_coords(0.8, -0.1)
    ax4=plt.subplot(224)
    axs=[ax1,ax2,ax3,ax4]

    slices = radial_bins['bins']
    thetaRays = np.arange(0,np.pi+1e-3,np.pi/slices)
    groups = ["all","bio","kinapprox","allmotor"]
    for i in range(4):
        for j in range(len(thetaRays)-1):
            bin_name = (
                '{:0.1f}'.format(180*thetaRays[j]/np.pi)
                + " to "
                + '{:0.1f}'.format(180*thetaRays[j+1]/np.pi)
            )
            if j%2==0:
                axs[i].add_patch(
                    Wedge(
                        (0,0), 3.3,
                        (180/np.pi)*thetaRays[j],
                        (180/np.pi)*thetaRays[j+1],
                        color = "0.85"
                    )
                )
            axs[i].add_patch(
                Wedge(
                    (0,0),
                    np.log10(radial_bins[groups[i]][bin_name][metric])+offset,
                    (180/np.pi)*thetaRays[j],
                    (180/np.pi)*thetaRays[j+1],
                    color = colors[i],
                    alpha=0.65
                )
            )
        axs[i].set_aspect('equal')
        axs[i].set_ylim([0,3.3])
        axs[i].set_xlim([-3.3,3.3])
        xticks = np.arange(0,3+1e-3,1)
        xticks = np.concatenate([
            -np.array(list(reversed(xticks[1:]))),
            xticks[1:]
        ])
        axs[i].set_xticks(xticks)
        axs[i].set_xticklabels([
            r"$10^{1}$",r"$10^{0}$",r"$10^{-1}$",
            r"$10^{-1}$",r"$10^{0}$",r"$10^{1}$"
        ])
        axs[i].add_patch(Wedge((0,0),1,0,360,color ='w'))
        xticksMinor = np.concatenate(
            [np.linspace(10**(i),10**(i+1),10)[1:-1] for i in range(-1,1)]
        )
        xticksMinor = np.concatenate(
            [-np.array(list(reversed(xticksMinor))),xticksMinor]
        )
        xticksMinor = [np.sign(el)*(np.log10(abs(el))+2) for el in xticksMinor]
        axs[i].set_xticks(xticksMinor,minor=True)

        yticks = list(np.arange(0,3+1e-3,1))
        axs[i].set_yticks(yticks[1:])
        axs[i].set_yticklabels(["" for tick in axs[i].get_yticks()])
        yticksMinor = np.concatenate(
            [np.linspace(10**(i),10**(i+1),10)[1:-1] for i in range(-1,1)]
        )
        yticksMinor = [np.sign(el)*(np.log10(abs(el))+2) for el in yticksMinor]
        axs[i].set_yticks(yticksMinor,minor=True)

        radii = list(axs[i].get_yticks())
        theta = np.linspace(0,np.pi,201)
        for radius in radii:
            axs[i].plot(
                [radius*np.cos(el) for el in theta],
                [radius*np.sin(el) for el in theta],
                "k",
                lw=0.5
            )
        axs[i].plot([1,3.3],[0,0],'k',linewidth=1.5)# double lw because of ylim
        axs[i].plot([-3.3,-1],[0,0],'k',linewidth=1.5)# double lw because of ylim
        axs[i].plot([0,0],[1,3.3],'k',linewidth=0.5)

        axs[i].text(
            0,0.25,
            subTitles[i],
            color=colors[i],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16
        )
        axs[i].spines['bottom'].set_position('zero')
        axs[i].spines['left'].set_position('zero')
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        # maxError = 4.1
        # axs[i].set_aspect('equal')
        # axs[i].set_ylim([0,maxError])
        # axs[i].set_xlim([-maxError,maxError])
        # xticks = list(np.arange(-np.floor(maxError),np.floor(maxError)+1e-3,1))
        # axs[i].set_xticks(xticks)
        # axs[i].set_xticklabels([r"$10^{2}$",r"$10^{1}$",r"$10^{0}$",r"$10^{-1}$","0",r"$10^{-1}$",r"$10^{0}$",r"$10^{1}$",r"$10^{2}$"])
        # xticksMinor = np.concatenate(
        #     [np.linspace(10**(i),10**(i+1),10)[1:-1] for i in range(-1,2)]
        # )
        # xticksMinor = np.concatenate(
        #     [
        #         -np.array(list(reversed(xticksMinor))),
        #         xticksMinor
        #     ]
        # )
        # xticksMinor = [np.sign(el)*(np.log10(abs(el))+2) for el in xticksMinor]
        # axs[i].set_xticks(xticksMinor,minor=True)
        #
        # yticks = [0,1,2,3,4]
        # axs[i].set_yticks(yticks)
        # axs[i].set_yticklabels(["" for tick in axs[i].get_yticks()])
        # yticksMinor = np.concatenate(
        #     [np.linspace(10**(i),10**(i+1),10)[1:-1] for i in range(-1,2)]
        # )
        # yticksMinor = [np.sign(el)*(np.log10(abs(el))+2) for el in yticksMinor]
        # axs[i].set_yticks(yticksMinor,minor=True)
        #
        # radii = list(axs[i].get_yticks())
        # theta = np.linspace(0,np.pi,201)
        # for radius in radii:
        #     axs[i].plot(
        #         [radius*np.cos(el) for el in theta],
        #         [radius*np.sin(el) for el in theta],
        #         "k",
        #         lw=0.5
        #     )
        #
        # for ray in thetaRays:
        #     axs[i].plot(
        #         [0,maxError*np.cos(ray)],
        #         [0,maxError*np.sin(ray)],
        #         'k',
        #         lw=0.5
        #     )
        # axs[i].spines['bottom'].set_position('zero')
        # axs[i].spines['left'].set_position('zero')
        # axs[i].spines['top'].set_visible(False)
        # axs[i].spines['right'].set_visible(False)
    if returnFig==True:
        return(fig)

def plot_polar_bar_plots_together(
        radial_bins,
        metric="MAE",
        addTitle=None,
        returnFig=False
    ):

    assert type(metric)==str and metric in ["MAE","STD","RMSE"], "metric must be either 'MAE' (default), 'RMSE' or 'STD'."
    if metric == "MAE":
        title = "MAE\n vs.\n Joint Angle"
        xLabel = "MAE (in deg.)"
        maxValue = np.log10(radial_bins["maxMAE"])+2
    elif metric == 'STD':
        title = "Error Std Dev\n vs.\n Joint Angle"
        xLabel = "Error Std Dev (in deg.)"
        maxValue = np.log10(radial_bins["maxSTD"])+2
    elif metric == 'RMSE':
        title = "RMSE\n vs.\n Joint Angle"
        xLabel = "Log RMSE (in deg.)"
        maxValue = np.log10(radial_bins["maxRMSE"])+2
        offset = 2

    # assert maxValue<3.3, "Bounds not configured for values this large. Please check values again and determine if bounds need to be changed."

    basePath = path.dirname(__file__)
    filePath = path.abspath(path.join(basePath, "..", "SupplementaryFigures", "Schematic_1DOF2DOA_system.png"))
    im = Image.open(filePath)
    height = im.size[1]
    width = im.size[0]
    aspectRatio = width/height

    fig = plt.figure(figsize=(10,8))
    if addTitle is not None:
        assert type(addTitle)==str, "title must be a string."
        plt.title(addTitle,fontsize=16,y=-0.35)


    newHeight = int(np.ceil(0.15*fig.bbox.ymax)+10)
    size = int(newHeight*aspectRatio),newHeight
    im.thumbnail(size, Image.ANTIALIAS)
    fig.figimage(im, fig.bbox.xmax/2 - im.size[0]/2.2, 0.95*im.size[1],zorder=10)
    ax = plt.gca()

    slices = radial_bins['bins']
    thetaRays = np.arange(0,np.pi+1e-3,np.pi/slices)
    sectorWidth = np.pi/slices/5
    thetaRays_SplitInFourths = []
    for j in range(len(thetaRays)-1):
        midAngle = (thetaRays[j+1]+thetaRays[j])/2
        thetaRays_SplitInFourths.append(
            [(midAngle + i*sectorWidth) for i in [-2,-1,0,1]]
        )
    thetaRays_SplitInFourths = np.concatenate(thetaRays_SplitInFourths)
    groups = ["all","bio","kinapprox","allmotor"]

    for j in range(len(thetaRays)-1):
        bin_name = (
            '{:0.1f}'.format(180*thetaRays[j]/np.pi)
            + " to "
            + '{:0.1f}'.format(180*thetaRays[j+1]/np.pi)
        )
        if j%2==0:
            ax.add_patch(
                Wedge(
                    (0,0), 3.3,
                    (180/np.pi)*thetaRays[j],
                    (180/np.pi)*thetaRays[j+1],
                    color = "0.85"
                )
            )
        for i in range(len(groups)):
            ax.add_patch(
                Wedge(
                    (0,0),
                    np.log10(radial_bins[groups[i]][bin_name][metric])+2,
                    (180/np.pi)*thetaRays_SplitInFourths[4*j+i],
                    (180/np.pi)*(thetaRays_SplitInFourths[4*j+i]+sectorWidth),
                    color = colors[i],
                    alpha=0.65
                )
            )

    ax.set_aspect('equal')
    ax.set_ylim([0,3.3])
    ax.set_xlim([-3.3,3.3])
    xticks = np.arange(0,3+1e-3,1)
    xticks = np.concatenate([-np.array(list(reversed(xticks[1:]))),xticks[1:]])
    ax.set_xticks(xticks)
    ax.set_xticklabels([
        r"$10^{1}$",r"$10^{0}$",r"$10^{-1}$",
        r"$10^{-1}$",r"$10^{0}$",r"$10^{1}$"
    ])
    ax.add_patch(Wedge((0,0),1,0,360,color ='w'))
    xticksMinor = np.concatenate(
        [np.linspace(10**(i),10**(i+1),10)[1:-1] for i in range(-1,1)]
    )
    xticksMinor = np.concatenate(
        [-np.array(list(reversed(xticksMinor))),xticksMinor]
    )
    xticksMinor = [np.sign(el)*(np.log10(abs(el))+2) for el in xticksMinor]
    ax.set_xticks(xticksMinor,minor=True)

    yticks = list(np.arange(0,3+1e-3,1))
    ax.set_yticks(yticks[1:])
    ax.set_yticklabels(["" for tick in ax.get_yticks()])
    yticksMinor = np.concatenate(
        [np.linspace(10**(i),10**(i+1),10)[1:-1] for i in range(-1,1)]
    )
    yticksMinor = [np.sign(el)*(np.log10(abs(el))+2) for el in yticksMinor]
    ax.set_yticks(yticksMinor,minor=True)

    radii = list(ax.get_yticks())
    theta = np.linspace(0,np.pi,201)
    for radius in radii:
        ax.plot(
            [radius*np.cos(el) for el in theta],
            [radius*np.sin(el) for el in theta],
            "k",
            lw=0.5
        )
    ax.plot([1,3.3],[0,0],'k',linewidth=1.5)# double lw because of ylim
    ax.plot([-3.3,-1],[0,0],'k',linewidth=1.5)# double lw because of ylim
    ax.plot([0,0],[1,3.3],'k',linewidth=0.5)

    props = dict(
        boxstyle='round',
        facecolor='w',
        edgecolor='0.70'
    )
    ax.text(
        -0.9*3,3,
        title,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16,
        bbox=props
    )
    ax.text(
        2,-0.35,
        xLabel,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=12
    )
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if returnFig==True:
        return(fig)

def plot_all_polar_bar_plots(outputData,metric,returnFigs=True):
    assert type(metric)==str and metric in ["MAE","STD","RMSE"], "metric must be either 'MAE' (default), 'RMSE' or 'STD'."

    ### radial average error versus positions
    labels = [
        "(Sinusoidal Angle / Sinusoidal Stiffness)",
        "(Step Angle / Sinusoidal Stiffness)",
        "(Sinusoidal Angle / Step Stiffness)",
        "(Step Angle / Step Stiffness)"
    ]
    groups = ["all","bio","kinapprox","allmotor"]

    figs = []
    for key in outputData.keys():
        index = np.where(
            key==np.array(list(outputData.keys()))
        )[0][0]
        jointAngleArrays = np.concatenate(
            [
                outputData[key][subkey]['expectedJointAngle'].flatten()[np.newaxis,:]
                for subkey in groups
            ],
            axis=0
        ) # in radians
        errorArrays = np.concatenate(
            [
                outputData[key][subkey]['rawError'].flatten()[np.newaxis,:]
                for subkey in groups
            ],
            axis=0
        ) # in radians

        radial_bins = return_radial_bins(errorArrays,jointAngleArrays,bins=15)
        tempFig = plot_polar_bar_plots(
            radial_bins,
            metric=metric,
            addTitle=labels[index],
            returnFig=True
        )
        figs.append(tempFig)

        tempFig = plot_polar_bar_plots_together(
            radial_bins,
            metric=metric,
            addTitle=labels[index],
            returnFig=True
        )
        figs.append(tempFig)

    if returnFigs==True:
        return(figs)
    else:
        plt.show()

def plot_all_error_distributions(outputData,returnFigs=True):
    labels = [
        "(Sinusoidal Angle / Sinusoidal Stiffness)",
        "(Step Angle / Sinusoidal Stiffness)",
        "(Sinusoidal Angle / Step Stiffness)",
        "(Step Angle / Step Stiffness)"
    ]
    groups = ["all","bio","kinapprox","allmotor"]

    figs = []
    for key in outputData.keys():
        index = np.where(
            key==np.array(list(outputData.keys()))
        )[0][0]
        tempFig, axs = plt.subplots(2,2,figsize=(10,10))
        plt.suptitle(labels[index],fontsize=16)
        for i in range(len(groups)):
            data = 180*outputData[key][groups[i]]['rawError'].flatten()/np.pi
            axs[int(i/2)][i%2].hist(
                data,
                weights=np.ones(len(data)) / len(data),
                bins=60,
                color=colors[i]
            )
            axs[int(i/2)][i%2].set_yticklabels(["{:.1f}%".format(100*el) for el in axs[int(i/2)][i%2].get_yticks()])
            axs[int(i/2)][i%2].set_title(
                groups[i],
                fontsize=14,
                color=colors[i]
            )
            axs[int(i/2)][i%2].spines['top'].set_visible(False)
            axs[int(i/2)][i%2].spines['right'].set_visible(False)
        figs.append(tempFig)

    if returnFigs==True:
        return(figs)

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

    ### ANN parameters
    ANNParams = {
        "Number of Nodes" : 15,
        "Number of Epochs" : 50,
        "Number of Trials" : 1,
    }

    ### Additional Arguments?
    parser = argparse.ArgumentParser(
        prog = "<filename>",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        animate_sample_trial.py

        -----------------------------------------------------------------------------

        Build ANN for 1 DOF, 2 DOA tendon-driven system with nonlinear tendon
        elasticity in order to predict joint angle from different "sensory"
        states (like tendon tension or motor angle) and animate results.

        -----------------------------------------------------------------------------'''),
        epilog=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        Written by Daniel A. Hagen (2020/01/29)

        -----------------------------------------------------------------------------'''
        )
    )
    parser.add_argument(
        '-dt',
        type=float,
        help='Time step for the simulation (float). Default is given by plantParams.py',
        default=plantParams["dt"]
    )
    parser.add_argument(
        '-dur',
        type=float,
        help='Duration of the simulation (float). Default is given by plantParams.py',
        default=plantParams["Simulation Duration"]
    )
    parser.add_argument(
        '-downsamplingFactor',
        type=int,
        help='Down Sampling Factor for animation. Default is 10.',
        default=10
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
        '-metrics',
        type=str,
        nargs="+",
        default='MAE',
        help="Metrics to be compared. Should be either MAE, RMSE, or STD. Default is MAE."
    )
    args = parser.parse_args()
    if type(args.metrics)==str:
        metrics = [args.metrics]
    else:
        metrics = args.metrics
    for metric in metrics:
        assert metric in ["RMSE","MAE","STD"], "Invalid metric! Must be either 'RMSE', 'MAE', or 'STD'"

    downsamplingFactor = args.downsamplingFactor
    startTime = time.time()
    trialStartTime = startTime
    movementTypes = [
        "Angle Sinusoid / Stiffness Sinusoid",
        "Angle Step / Stiffness Sinusoid",
        "Angle Sinusoid / Stiffness Step",
        "Angle Step / Stiffness Step"
    ]
    movementTypePrefixes = [
        "angleSin_stiffSin_",
        "angleStep_stiffSin_",
        "angleSin_stiffStep_",
        "angleStep_stiffStep_"
    ]
    groups = [
        "all",
        "bio",
        "kinapprox",
        "allmotor"
    ]
    startTime = time.time()

    plantParams["dt"] = args.dt
    ANNParams["Number of Epochs"] = args.epochs
    ANNParams["Number of Nodes"] = args.nodes

    ### Generate plant
    tempSimulationDuration = 600
    plantParams["Simulation Duration"] = tempSimulationDuration
    plant = plant_pendulum_1DOF2DOF(plantParams)
    X_o = plant.return_X_o(np.pi,[0,0])
    passProbability = 0.0005

    allDone = False
    count = 0
    while allDone==False:
        ### Generate Testing DATA (Angle Step, Stiffness Step)
        basePath = "experimental_trials/"

        print("Angle Step / Stiffness Step")
        filePath = (basePath + "angleStep_stiffStep_outputData.mat")
        if path.exists(filePath):
            print("ALREADY COMPLETED!!! (DELETE TO RUN AGAIN...)")
        else:
            [x1d,sd] = plant.generate_desired_trajectory_STEPS(
                passProbability,'both'
            )
            try:
                generate_and_save_sensory_data(plant,x1d,sd,savePath=filePath)
            except:
                pass

        ### Generate Testing DATA (Angle Step, Stiffness Sinusoid)

        print("Angle Step / Stiffness Sinusoid")
        filePath = (basePath + "angleStep_stiffSin_outputData.mat")
        if path.exists(filePath):
            print("ALREADY COMPLETED!!! (DELETE TO RUN AGAIN...)")
        else:
            x1d = plant.generate_desired_trajectory_STEPS(passProbability,'angle')
            sd = plant.generate_desired_trajectory_SINUSOIDAL('stiffness')
            try:
                generate_and_save_sensory_data(plant,x1d,sd,savePath=filePath)
            except:
                pass

        ### Generate Testing DATA (Angle Sinusoid, Stiffness Step)

        print("Angle Sinusoid / Stiffness Step")
        filePath = (basePath + "angleSin_stiffStep_outputData.mat")
        if path.exists(filePath):
            print("ALREADY COMPLETED!!! (DELETE TO RUN AGAIN...)")
        else:
            x1d = plant.generate_desired_trajectory_SINUSOIDAL('angle')
            sd = plant.generate_desired_trajectory_STEPS(
                passProbability,'stiffness'
            )
            try:
                generate_and_save_sensory_data(plant,x1d,sd,savePath=filePath)
            except:
                pass

        ### Generate Testing DATA (Angle Sinusoid, Stiffness Sinusoid)

        print("Angle Step / Stiffness Step")
        filePath = (basePath + "angleSin_stiffSin_outputData.mat")
        if path.exists(filePath):
            print("ALREADY COMPLETED!!! (DELETE TO RUN AGAIN...)")
        else:
            x1d = plant.generate_desired_trajectory_SINUSOIDAL('angle')
            sd = plant.generate_desired_trajectory_SINUSOIDAL('stiffness')
            try:
                generate_and_save_sensory_data(plant,x1d,sd,savePath=filePath)
            except:
                pass

        if np.all([
            path.exists(basePath+el+"outputData.mat")
            for el in [
                "angleStep_stiffStep_",
                "angleStep_stiffSin_",
                "angleSin_stiffStep_",
                "angleSin_stiffSin_"
            ]
        ]):
            allDone = True
        else:
            count+=1
            assert count<10, "Too many unsuccessful trials, please check code and run again."

    ### Generate babbling data and SAVE ALL FIGURES AND DATA IN SPECIFIC FOLDER
    plantParams["Simulation Duration"] = args.dur # returned to original value.

    ANN = neural_network(ANNParams,babblingParams,plantParams)
    experimentalData,babblingData = ANN.run_experimental_trial(
        returnBabblingData=True
    )

    os.makedirs(os.path.join(ANN.trialPath,"animations"))
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(
        fps=1000/downsamplingFactor,
        metadata=dict(artist='Daniel A. Hagen'),
        bitrate=1800
    )
    if args.dur<=10:
        Time = babblingData['time']
        X = babblingData['X']
        U = babblingData['U']
    else:
        startIndex = int(15/plantParams["dt"])
        endIndex = int(25/plantParams["dt"])+1
        Time = babblingData['time'][startIndex:endIndex]
        X = babblingData['X'][:,startIndex:endIndex]
        U = babblingData['U'][:,startIndex:endIndex]

    ani = animate_pendulum_babbling(
        Time,X,U,
        downsamplingFactor,
        **plantParams
    )
    ani.start(downsamplingFactor)
    ani.anim.save(
        (
            ANN.trialPath
            + "animations/"
            + 'babbling.mp4'
        ),
        writer=writer
    )
    ani.anim.event_source.stop()
    del(ani)
    plt.close()

    # ### Plot experimental data
    # figs = plot_experimental_data(experimentalData,returnFigs=True)

    # SAVE EXPERIMENTAL DATA TO TRIAL FOLDER
    formattedData = {
        "all" : {},
        "bio" : {},
        "kinapprox" : {},
        "allmotor" : {}
    }
    formattedData = {}
    for key in experimentalData["all"]:
        formattedData[key] = {}
        for subkey in experimentalData:
            formattedData[key][subkey] = {}
            for subsubkey in experimentalData[subkey][key]:
                if type(experimentalData[subkey][key][subsubkey])==float:
                    formattedData[key][subkey][subsubkey] = \
                        experimentalData[subkey][key][subsubkey]
                else:
                    formattedData[key][subkey][subsubkey] = np.array(
                        experimentalData[subkey][key][subsubkey]._data
                    )
    experimentalData = formattedData
    with open(path.join(ANN.trialPath,'experimentalData.pkl'), 'wb') as handle:
        pickle.dump(experimentalData, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for metric in metrics:
        ### bar plots
        fig = plot_bar_plots(experimentalData,metric=metric)

        ### radial average error versus positions
        figs = [fig]
        newFigs=plot_all_polar_bar_plots(experimentalData,metric)
        [figs.append(fig) for fig in newFigs]

        ### save figs
        save_figures(
            ANN.trialPath,
            babblingParams["Babbling Type"],
            {**plantParams,**babblingParams,**ANNParams},
            figs=figs,
            subFolderName=metric+"/",
            saveAsMD=True,
            addNotes="Simulation and animation of a sample trial. Will include simulation of *motor babbling*, followed by simulations of ANNs, and finally their *generalization* to different movements."
        )
        plt.close('all')

    startIndex = int(3/plantParams["dt"])
    endIndex = int(13/plantParams["dt"])+1
    for i in range(4):
        print("Testing ANN on: " + movementTypes[i])
        filePath = basePath + movementTypePrefixes[i]
        out = sio.loadmat(filePath+"outputData.mat")
        desiredOutput = np.array([out['X1d'][0,:],out['Sd'][0,:]])

        Time = out['Time'][0,startIndex:endIndex]
        X = np.array([
            out['x1'][0,startIndex:endIndex],
            out['dx1'][0,startIndex:endIndex],
            out['x3'][0,startIndex:endIndex],
            out['dx3'][0,startIndex:endIndex],
            out['x5'][0,startIndex:endIndex],
            out['dx5'][0,startIndex:endIndex]
        ])
        U = np.array([
            out['u1'][0,startIndex:endIndex],
            out['u2'][0,startIndex:endIndex]
        ])
        os.makedirs(
            os.path.join(
                os.path.join(ANN.trialPath,"animations"),
                movementTypePrefixes[i][:-1]
            )
        )
        maximumError = 180/np.pi*max([
            max(
                X[0,:]
                - experimentalData[movementTypePrefixes[i][:-1]][group]['predictedJointAngle'][startIndex:endIndex]
            )
            for group in groups
        ])
        minimumError = 180/np.pi*min([
            min(
                X[0,:]
                - experimentalData[movementTypePrefixes[i][:-1]][group]['predictedJointAngle'][startIndex:endIndex]
            )
            for group in groups
        ])
        errorBounds = [minimumError,maximumError]
        for j in range(4):
            desiredAngle = experimentalData[movementTypePrefixes[i][:-1]][groups[j]]['expectedJointAngle'][startIndex:endIndex]
            predictedAngle = experimentalData[movementTypePrefixes[i][:-1]][groups[j]]['predictedJointAngle'][startIndex:endIndex]

            ani = animate_pendulum_ANN(
                Time,X,U,
                predictedAngle,
                downsamplingFactor,
                movementTypes[i],
                groups[j],
                errorBounds,
                **plantParams
            )
            ani.start(downsamplingFactor)
            ani.anim.save(
                (
                    ANN.trialPath
                    + "animations/"
                    + movementTypePrefixes[i][:-1] + "/"
                    + groups[j]+'.mp4'
                ),
                writer=writer
            )
            ani.anim.event_source.stop()
            del(ani)
            plt.close()

    print('\a')
    runTime = time.time()-startTime
    seconds = runTime % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    runTime = "%d:%02d:%02d" % (hour, minutes, seconds)
    print('Run Time: ' + runTime + "\n")
