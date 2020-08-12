from motor_babbling_1DOF2DOA import *
from build_NN_1DOF2DOA import *
from save_params import *
import os
import matlab.engine
import argparse
import textwrap
from danpy.sb import get_terminal_width
from os import path,listdir
from matplotlib.patches import Wedge,Polygon
import scipy.io as sio
import pickle
from PIL import Image
import time
import shutil
import copy
import itertools
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

if path.exists("slack_functions.py"):
    from slack_functions import *
    HAL = code_progress()

groupNames = ["all","bio","kinapprox","allmotor"]
colors = [
    "#2A3179", # all
    "#F4793B", # bio
    "#8DBDE6", # kinapprox
    "#A95AA1" # allmotor
]
movementTypes = [
    "angleSin_stiffSin",
    "angleStep_stiffSin",
    "angleSin_stiffStep",
    "angleStep_stiffStep"
]
labels = [
    "(Sinusoidal Angle / Sinusoidal Stiffness)",
    "(Step Angle / Sinusoidal Stiffness)",
    "(Sinusoidal Angle / Step Stiffness)",
    "(Step Angle / Step Stiffness)"
]

def return_in_bounds_states(plant,filePath=None):
    if filePath is None:
        filePath = Path("experimental_trials/angleStep_stiffSin_outputData.mat")
    else:
        filePath = Path(filePath)
    outputData = sio.loadmat(filePath)
    x1 = outputData["x1"]
    indices = list(
        np.where(
            np.logical_and(
                x1[0,:]>=plant.jointAngleBounds["LB"],
                x1[0,:]<=plant.jointAngleBounds["UB"]
            )
        )[0]
    )

    newOutputData = {}
    for key in outputData:
        if key not in ['__header__','__version__','__globals__','Time']:
            if "u" in key and np.shape(outputData[key])[1] in indices:
                newOutputData[key] = outputData[key][0,indices[:-1]]
            else:
                newOutputData[key] = outputData[key][0,indices]

    sio.savemat(filePath,newOutputData)

    X = np.array([
        newOutputData['x1'],
        newOutputData['dx1'],
        newOutputData['x3'],
        newOutputData['dx3'],
        newOutputData['x5'],
        newOutputData['dx5']
    ])
    U = np.array([
        newOutputData['u1'],
        newOutputData['u2']
    ])
    return(X,U)

def plot_babbling_duration_vs_average_performance(
        metric,
        directory=None,
        yscale="linear",
        includeSTD=False
    ):
    prettyGroupNames = [
        "All Available States",
        "The Bio-Inspired Set",
        "Motor Position and Velocity Only",
        "All Motor States"
    ]

    ### input arguments

    assert metric in ["RMSE","STD","MAE"], 'Invalid metric! metrics must include "RMSE","STD", or "MAE".'

    if directory is None:
        directory = Path("experimental_trials/")
    else:
        directory = Path(directory)
        assert directory.exists(), "Enter a valid directory."

    assert yscale in ["linear", "log"], "yscale should either be 'linear' (default) or 'log'."

    assert type(includeSTD)==bool, "includeSTD must be either true or false (default)."

    ### get the testing trial directories

    trialDirectories = [
        child for child in directory.iterdir()
        if child.is_dir() and child.stem[:12]=="Consolidated"
    ]
    babblingDurations = np.array([
        int(trialPath.stem[-8:-2]) for trialPath in trialDirectories
    ])/1000 # in sec

    totalPerformanceData={}
    for movement in movementTypes:
        totalPerformanceData[movement] = {}
        for group in groupNames:
            totalPerformanceData[movement][group] = {}
            totalPerformanceData[movement][group]['values'] = []
            totalPerformanceData[movement][group]['STDs'] = []

    for n in range(len(trialDirectories)):
        consolidatedOutputPath = (
            trialDirectories[n] / 'consolidatedOutputData.pkl'
        )
        with consolidatedOutputPath.open('rb') as handle:
            tempOutputData = pickle.load(handle)
            """
                tempOutputData
                    ..<Movement Type>
                        ..<Group Name>
                            ..<Metric> (in rad.)
                            ..<Metric List> (in rad.)
            """
        for movement in movementTypes:
            for group in groupNames:
                totalPerformanceData[movement][group]['values'].append(
                    (180/np.pi)
                    * tempOutputData[movement][group]["experiment"+metric]
                ) # (in deg.)
                totalPerformanceData[movement][group]['STDs'].append(
                    (180/np.pi)*np.std(tempOutputData[movement][group]["experiment"+metric+"_list"])
                ) # (in deg.)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,10),sharex=True)
    plt.suptitle("Performance ("+metric+") vs. Babbling Duration",fontsize=24)
    axs = [ax1,ax2,ax3,ax4]
    if yscale=="linear":
        minValue=100
        maxValue=0
        if includeSTD==True:
            for movement in movementTypes:
                for group in groupNames:
                    maxValue = max([
                        maxValue,
                        max(
                            np.array(totalPerformanceData[movement][group]['values'])
                            + np.array(totalPerformanceData[movement][group]['STDs']))
                    ])
                    minValue = min([
                        minValue,
                        min(
                            np.array(totalPerformanceData[movement][group]['values'])
                            - np.array(totalPerformanceData[movement][group]['STDs']))
                    ])
        else:
            minValue=0
            maxValue=0
            for movement in movementTypes:
                for group in groupNames:
                    maxValue = max([
                        maxValue,
                        max(totalPerformanceData[movement][group]['values'])
                    ])
    else:
        minValue=100
        maxValue=0
        if includeSTD==True:
            for movement in movementTypes:
                for group in groupNames:
                    maxValue = max([
                        maxValue,
                        max(
                            np.array(totalPerformanceData[movement][group]['values'])
                            * np.exp(
                                np.array(totalPerformanceData[movement][group]['STDs'])
                                / np.array(totalPerformanceData[movement][group]['values'])
                            )
                        )
                    ])
                    minValue = min([
                        minValue,
                        min(
                            np.array(totalPerformanceData[movement][group]['values'])
                            * np.exp(
                                -np.array(totalPerformanceData[movement][group]['STDs'])
                                / np.array(totalPerformanceData[movement][group]['values'])
                            )
                        )
                    ])
        else:
            for movement in movementTypes:
                for group in groupNames:
                    minValue = min([
                        minValue,
                        min(totalPerformanceData[movement][group]['values'])
                    ])
                    maxValue = max([
                        maxValue,
                        max(totalPerformanceData[movement][group]['values'])
                    ])
        minValue=10**(np.floor(np.log10(minValue)))
        maxValue=10**(np.ceil(np.log10(maxValue)))
    for i in range(len(movementTypes)):
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
        if includeSTD==True and yscale=='linear':
            axs[i].spines["bottom"].set_position("zero")
        axs[i].set_ylim(minValue,maxValue)
        axs[i].set_yscale(yscale)
        axs[i].set_title(labels[i])
        for j in range(len(groupNames)):
            tempDict = \
                totalPerformanceData[movementTypes[i]][groupNames[j]].copy()
            axs[i].plot(babblingDurations,tempDict['values'],c=colors[j])
            if includeSTD==True and yscale=='linear':
                axs[i].fill_between(
                    babblingDurations,
                    (np.array(tempDict['values']) + np.array(tempDict['STDs'])),
                    (np.array(tempDict['values']) - np.array(tempDict['STDs'])),
                    color=colors[j],
                    alpha='0.5'
                )
            elif includeSTD==True and yscale=='log':
                axs[i].fill_between(
                    babblingDurations,
                    (np.array(tempDict['values'])
                        * np.exp(
                            np.array(tempDict['STDs'])
                            / np.array(tempDict['values'])
                        )
                    ),
                    (np.array(tempDict['values'])
                        * np.exp(
                            -np.array(tempDict['STDs'])
                            / np.array(tempDict['values'])
                        )
                    ),
                    color=colors[j],
                    alpha='0.5'
                )
        if i==2:
            # axs[i].legend(prettyGroupNames,loc='upper right')
            axs[i].legend(prettyGroupNames,bbox_to_anchor=(-0.125, -0.25, 2.4, .142), ncol=4, mode="expand", loc=3, borderaxespad=0,fontsize=12)
            axs[i].set_ylabel("Avg. Performance ("+metric+" in deg.)")
            axs[i].set_xlabel("Babbling Duration (sec.)")
        else:
            plt.setp(axs[i].get_xticklabels(), visible=False)

def plot_babbling_duration_vs_performance_STD(
        metric,
        directory=None,
        yscale="linear"
    ):
    prettyGroupNames = [
        "All Available States",
        "The Bio-Inspired Set",
        "Motor Position and Velocity Only",
        "All Motor States"
    ]

    ### input arguments

    assert metric in ["RMSE","STD","MAE"], 'Invalid metric! metrics must include "RMSE","STD", or "MAE".'

    if directory is None:
        directory = Path("experimental_trials/")
    else:
        directory = Path(directory)
        assert directory.exists(), "Enter a valid directory."

    assert yscale in ["linear", "log"], "yscale should either be 'linear' (default) or 'log'."

    ### get the testing trial directories

    trialDirectories = [
        child for child in directory.iterdir()
        if child.is_dir() and child.stem[:12]=="Consolidated"
    ]
    babblingDurations = np.array([
        int(trialPath.stem[-8:-2]) for trialPath in trialDirectories
    ])/1000 # in sec

    totalPerformanceData={}
    for movement in movementTypes:
        totalPerformanceData[movement] = {}
        for group in groupNames:
            totalPerformanceData[movement][group] = {}
            totalPerformanceData[movement][group]['STDs'] = []

    for n in range(len(trialDirectories)):
        consolidatedOutputPath = (
            trialDirectories[n] / 'consolidatedOutputData.pkl'
        )
        with consolidatedOutputPath.open('rb') as handle:
            tempOutputData = pickle.load(handle)
            """
                tempOutputData
                    ..<Movement Type>
                        ..<Group Name>
                            ..<Metric> (in rad.)
                            ..<Metric List> (in rad.)
            """
        for movement in movementTypes:
            for group in groupNames:
                totalPerformanceData[movement][group]['STDs'].append(
                    (180/np.pi)*np.std(tempOutputData[movement][group]["experiment"+metric+"_list"])
                ) # (in deg.)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,10),sharex=True)
    plt.suptitle("Performance (" + metric + ") STDs vs. Babbling Durations",fontsize=24)
    axs = [ax1,ax2,ax3,ax4]
    if yscale=="linear":
        minValue=0
        maxValue=0
        for movement in movementTypes:
            for group in groupNames:
                maxValue = max([
                    maxValue,
                    max(np.array(totalPerformanceData[movement][group]['STDs']))
                ])
    else:
        minValue=100
        maxValue=0
        for movement in movementTypes:
            for group in groupNames:
                minValue = min([
                    minValue,
                    min(totalPerformanceData[movement][group]['STDs'])
                ])
                maxValue = max([
                    maxValue,
                    max(totalPerformanceData[movement][group]['STDs'])
                ])
        minValue=10**(np.floor(np.log10(minValue)))
        maxValue=10**(np.ceil(np.log10(maxValue)))
    for i in range(len(movementTypes)):
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
        axs[i].set_ylim(minValue,maxValue)
        axs[i].set_yscale(yscale)
        axs[i].set_title(labels[i])
        for j in range(len(groupNames)):
            tempDict = \
                totalPerformanceData[movementTypes[i]][groupNames[j]].copy()
            axs[i].plot(babblingDurations,tempDict['STDs'],c=colors[j])

        if i==2:
            # axs[i].legend(prettyGroupNames,loc='upper right')
            axs[i].legend(prettyGroupNames,bbox_to_anchor=(-0.125, -0.25, 2.4, .142), ncol=4, mode="expand", loc=3, borderaxespad=0,fontsize=12)
            axs[i].set_ylabel("Performance STDs ("+metric+" in deg.)")
            axs[i].set_xlabel("Babbling Duration (sec.)")
        else:
            plt.setp(axs[i].get_xticklabels(), visible=False)

def plot_number_of_nodes_vs_average_performance(
        metric,
        directory=None,
        yscale="linear",
        includeSTD=False
    ):
    prettyGroupNames = [
        "All Available States",
        "The Bio-Inspired Set",
        "Motor Position and Velocity Only",
        "All Motor States"
    ]
    ### input arguments

    assert metric in ["RMSE","STD","MAE"], 'Invalid metric! metrics must include "RMSE","STD", or "MAE".'

    assert yscale in ["linear", "log"], "yscale should either be 'linear' (default) or 'log'."

    assert type(includeSTD)==bool, "includeSTD must be either true or false (default)."

    if directory is None:
        directory = Path("experimental_trials/")
    else:
        directory = Path(directory)
        assert directory.exists(), "Enter a valid directory."

    ### get the testing trial directories

    trialDirectories = [
        child for child in directory.iterdir()
        if child.is_dir() and child.stem[:12]=="Consolidated" and child.stem[-2:]!="ms"
    ] # TEMP
    numberOfNodesList = np.array([
        int(trialPath.stem[-9:-6]) for trialPath in trialDirectories
    ])

    totalPerformanceData = {}
    for movement in movementTypes:
        totalPerformanceData[movement] = {}
        for group in groupNames:
            totalPerformanceData[movement][group] = {}
            totalPerformanceData[movement][group]['values'] = []
            totalPerformanceData[movement][group]['STDs'] = []

    for n in range(len(trialDirectories)):
        consolidatedOutputPath = (
            trialDirectories[n] / 'consolidatedOutputData.pkl'
        )
        with consolidatedOutputPath.open('rb') as handle:
            tempOutputData = pickle.load(handle)
            """
                tempOutputData
                    ..<Movement Type>
                        ..<Group Name>
                            ..<Metric> (in rad.)
                            ..<Metric List> (in rad.)
            """

        for movement in movementTypes:
            for group in groupNames:
                totalPerformanceData[movement][group]['values'].append(
                    (180/np.pi)
                    * tempOutputData[movement][group]["experiment"+metric]
                ) # (in deg.)
                totalPerformanceData[movement][group]['STDs'].append(
                    (180/np.pi)
                    * np.std(tempOutputData[movement][group]["experiment"+metric+"_list"])
                )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,10))
    plt.suptitle("Performance ("+metric+") vs. Number of Hidden Nodes",fontsize=24)
    axs = [ax1,ax2,ax3,ax4]
    if yscale=="linear":
        minValue=100
        maxValue=0
        if includeSTD==True:
            for movement in movementTypes:
                for group in groupNames:
                    maxValue = max([
                        maxValue,
                        max(
                            np.array(totalPerformanceData[movement][group]['values'])
                            + np.array(totalPerformanceData[movement][group]['STDs']))
                    ])
                    minValue = min([
                        minValue,
                        min(
                            np.array(totalPerformanceData[movement][group]['values'])
                            - np.array(totalPerformanceData[movement][group]['STDs']))
                    ])
        else:
            minValue=0
            maxValue=0
            for movement in movementTypes:
                for group in groupNames:
                    maxValue = max([
                        maxValue,
                        max(totalPerformanceData[movement][group]['values'])
                    ])
    else:
        minValue=100
        maxValue=0
        if includeSTD==True:
            for movement in movementTypes:
                for group in groupNames:
                    maxValue = max([
                        maxValue,
                        max(
                            np.array(totalPerformanceData[movement][group]['values'])
                            * np.exp(
                                np.array(totalPerformanceData[movement][group]['STDs'])
                                / np.array(totalPerformanceData[movement][group]['values'])
                            )
                        )
                    ])
                    minValue = min([
                        minValue,
                        min(
                            np.array(totalPerformanceData[movement][group]['values'])
                            * np.exp(
                                -np.array(totalPerformanceData[movement][group]['STDs'])
                                / np.array(totalPerformanceData[movement][group]['values'])
                            )
                        )
                    ])
        else:
            for movement in movementTypes:
                for group in groupNames:
                    minValue = min([
                        minValue,
                        min(totalPerformanceData[movement][group]['values'])
                    ])
                    maxValue = max([
                        maxValue,
                        max(totalPerformanceData[movement][group]['values'])
                    ])
        minValue=10**(np.floor(np.log10(minValue)))
        maxValue=10**(np.ceil(np.log10(maxValue)))

    for i in range(len(movementTypes)):
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
        if includeSTD==True and yscale=='linear':
            axs[i].spines["bottom"].set_position("zero")
        axs[i].set_ylim(minValue,maxValue)
        axs[i].set_yscale(yscale)
        axs[i].set_title(labels[i])
        for j in range(len(groupNames)):
            tempDict = \
                totalPerformanceData[movementTypes[i]][groupNames[j]].copy()
            axs[i].plot(numberOfNodesList,tempDict['values'],c=colors[j])
            if includeSTD==True and yscale=='linear':
                axs[i].fill_between(
                    numberOfNodesList,
                    (np.array(tempDict['values']) + np.array(tempDict['STDs'])),
                    (np.array(tempDict['values']) - np.array(tempDict['STDs'])),
                    color=colors[j],
                    alpha='0.5'
                )
            elif includeSTD==True and yscale=='log':
                axs[i].fill_between(
                    numberOfNodesList,
                    (np.array(tempDict['values'])
                        * np.exp(
                            np.array(tempDict['STDs'])
                            / np.array(tempDict['values'])
                        )
                    ),
                    (np.array(tempDict['values'])
                        * np.exp(
                            -np.array(tempDict['STDs'])
                            / np.array(tempDict['values'])
                        )
                    ),
                    color=colors[j],
                    alpha='0.5'
                )
        if i==2:
            # axs[i].legend(prettyGroupNames,loc='upper right')
            axs[i].legend(prettyGroupNames,bbox_to_anchor=(-0.125, -0.25, 2.4, .142), ncol=4, mode="expand", loc=3, borderaxespad=0,fontsize=12)
            axs[i].set_xlabel("Number of Nodes in Hidden Layer")
            axs[i].set_ylabel("Avg. Performance ("+metric+" in deg.)")
        else:
            plt.setp(axs[i].get_xticklabels(), visible=False)

def plot_number_of_nodes_vs_performance_STD(
        metric,
        directory=None,
        yscale="linear"
    ):
    prettyGroupNames = [
        "All Available States",
        "The Bio-Inspired Set",
        "Motor Position and Velocity Only",
        "All Motor States"
    ]

    ### input arguments

    assert metric in ["RMSE","STD","MAE"], 'Invalid metric! metrics must include "RMSE","STD", or "MAE".'

    if directory is None:
        directory = Path("experimental_trials/")
    else:
        directory = Path(directory)
        assert directory.exists(), "Enter a valid directory."

    assert yscale in ["linear", "log"], "yscale should either be 'linear' (default) or 'log'."

    ### get the testing trial directories

    trialDirectories = [
        child for child in directory.iterdir()
        if child.is_dir() and child.stem[:12]=="Consolidated" and child.stem[-2:]!="ms"
    ] # TEMP
    numberOfNodesList = np.array([
        int(trialPath.stem[-9:-6]) for trialPath in trialDirectories
    ])

    totalPerformanceData={}
    for movement in movementTypes:
        totalPerformanceData[movement] = {}
        for group in groupNames:
            totalPerformanceData[movement][group] = {}
            totalPerformanceData[movement][group]['STDs'] = []

    for n in range(len(trialDirectories)):
        consolidatedOutputPath = (
            trialDirectories[n] / 'consolidatedOutputData.pkl'
        )
        with consolidatedOutputPath.open('rb') as handle:
            tempOutputData = pickle.load(handle)
            """
                tempOutputData
                    ..<Movement Type>
                        ..<Group Name>
                            ..<Metric> (in rad.)
                            ..<Metric List> (in rad.)
            """
        for movement in movementTypes:
            for group in groupNames:
                totalPerformanceData[movement][group]['STDs'].append(
                    (180/np.pi)*np.std(tempOutputData[movement][group]["experiment"+metric+"_list"])
                ) # (in deg.)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,10),sharex=True)
    plt.suptitle("Performance (" + metric + ") STDs vs. Number of Hidden Nodes",fontsize=24)
    axs = [ax1,ax2,ax3,ax4]
    if yscale=="linear":
        minValue=0
        maxValue=0
        for movement in movementTypes:
            for group in groupNames:
                maxValue = max([
                    maxValue,
                    max(np.array(totalPerformanceData[movement][group]['STDs']))
                ])
    else:
        minValue=100
        maxValue=0
        for movement in movementTypes:
            for group in groupNames:
                minValue = min([
                    minValue,
                    min(totalPerformanceData[movement][group]['STDs'])
                ])
                maxValue = max([
                    maxValue,
                    max(totalPerformanceData[movement][group]['STDs'])
                ])
        minValue=10**(np.floor(np.log10(minValue)))
        maxValue=10**(np.ceil(np.log10(maxValue)))
    for i in range(len(movementTypes)):
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
        axs[i].set_ylim(minValue,maxValue)
        axs[i].set_yscale(yscale)
        axs[i].set_title(labels[i])
        for j in range(len(groupNames)):
            tempDict = \
                totalPerformanceData[movementTypes[i]][groupNames[j]].copy()
            axs[i].plot(numberOfNodesList,tempDict['STDs'],c=colors[j])

        if i==2:
            # axs[i].legend(prettyGroupNames,loc='upper right')
            axs[i].legend(prettyGroupNames,bbox_to_anchor=(-0.125, -0.25, 2.4, .142), ncol=4, mode="expand", loc=3, borderaxespad=0,fontsize=12)
            axs[i].set_ylabel("Performance STDs ("+metric+" in deg.)")
            axs[i].set_xlabel("Number of Nodes in Hidden Layer")
        else:
            plt.setp(axs[i].get_xticklabels(), visible=False)

def generate_and_save_sensory_data(
        plant,x1d,sd,X_o,
        savePath=None,
        returnOutput=False,
        trim=None,
        downsample=None
    ):

    if downsample is not None:
        is_number(downsample,"downsample")
    else:
        downsample=1

    if trim is not None:
        is_number(trim,"trim",notes="Should be a positive number in seconds.")
        trim = int(trim/(plant.dt*downsample))
    else:
        trim = 0

    X1d = np.zeros((5,len(x1d)))
    X1d[0,:] = x1d
    X1d[1,:] = np.gradient(X1d[0,:],plant.dt)
    X1d[2,:] = np.gradient(X1d[1,:],plant.dt)
    X1d[3,:] = np.gradient(X1d[2,:],plant.dt)
    X1d[4,:] = np.gradient(X1d[3,:],plant.dt)

    Sd = np.zeros((3,len(sd)))
    Sd[0,:] = sd
    Sd[1,:] = np.gradient(Sd[0,:],plant.dt)
    Sd[2,:] = np.gradient(Sd[1,:],plant.dt)

    X,U,_,_ = plant.forward_simulation_FL(X_o,X1d,Sd)

    X,U = X[:,::downsample],U[:,::downsample]

    additionalDict = {
        "X1d" : x1d[::downsample][trim:],
        "Sd" : sd[::downsample][trim:]
    }

    plant.save_data(
        X[:,trim:],U[:,trim:],
        additionalDict=additionalDict,
        filePath=savePath
    )

    if returnOutput is True:
        return(X[:,trim:],U[:,trim:])

def plot_experimental_data(experimentalData,dt,returnFigs=True):
    # TODO: change this to allow for variable timeArray lengths and to get rid of the redundant figure definitions.
    # Sin Angle/Sin Stiffness
    fig1, (ax1a,ax1b) = plt.subplots(2,1,figsize=(8,6),sharex=True)
    plt.suptitle(labels[0][1:-1])

    # Step Angle/Sin Stiffness
    fig2, (ax2a,ax2b) = plt.subplots(2,1,figsize=(8,6),sharex=True)
    plt.suptitle(labels[1][1:-1])

    # Sin Angle/Step Stiffness
    fig3, (ax3a,ax3b) = plt.subplots(2,1,figsize=(8,6),sharex=True)
    plt.suptitle(labels[2][1:-1])

    # Step Angle/Step Stiffness
    fig4, (ax4a,ax4b) = plt.subplots(2,1,figsize=(8,6),sharex=True)
    plt.suptitle(labels[3][1:-1])

    figs = [fig1,fig2,fig3,fig4]
    top_axs = [ax1a,ax2a,ax3a,ax4a]
    bot_axs = [ax1b,ax2b,ax3b,ax4b]

    legendList = copy.copy(groupNames)
    legendList.insert(0,'Desired')
    for i in range(len(movementTypes)):
        timeArray = dt*np.array(list(range(
            len(experimentalData[movementTypes[i]]['all']['rawError'])
        )))
        fig, (top_ax,bot_ax) = plt.subplots(2,1,figsize=(8,6),sharex=True)
        top_ax.set_ylabel("Joint Angle (deg.)")
        top_ax.spines["right"].set_visible(False)
        top_ax.spines["top"].set_visible(False)
        top_ax.legend(legendList,loc="upper right")
        plt.setp(top_ax.get_xticklabels(), visible=False)
        top_ax.plot(
            timeArray,
            (
                (180/np.pi)
                * np.array(
                    experimentalData[movementTypes[i]]['all']["expectedJointAngle"]
                ).T
            ),
            c='0.70',
            lw=2
        ) # in deg.
        bot_ax.set_xlabel("Time (s)")
        bot_ax.set_ylabel("Joint Angle Error (deg.)")
        bot_ax.spines["right"].set_visible(False)
        bot_ax.spines["top"].set_visible(False)

        for j in range(len(groupNames)):
            top_ax.plot(
                timeArray,
                (
                    180/np.pi
                    * np.array(
                        experimentalData[movementTypes[i]][groupNames[j]]["predictedJointAngle"]
                    ).T
                ),
                c=colors[j]
            ) # in deg.
            bot_ax.plot(
                timeArray,
                (
                    180/np.pi
                    * np.array(
                        experimentalData[movementTypes[i]][groupNames[j]]["rawError"]
                    ).T
                ),
                c=colors[j]
            ) # in deg.

    if returnFigs==True:
        return(figs)
    else:
        plt.show()

def plot_average_error_signal_power_spectrums(
        totalOutputData,
        returnFigs=True
    ):
    baseTitle = "Avg. Error Signal Power Spectrum"

    prettyGroupNames = [
        "All Available States",
        "The Bio-Inspired Set",
        "Motor Position and Velocity Only",
        "All Motor States"
    ]

    figList=[]
    for key in totalOutputData:
        i = np.where(
            key==np.array(list(totalOutputData.keys()))
        )[0][0]

        fig = plt.figure(figsize=(7,5))
        plt.suptitle(baseTitle+"\n"+labels[i], fontsize=14)
        ax = plt.gca()
        ax.set_xlabel("Frequency (Hz)", ha="center")
        ax.set_ylabel(r"PSD (rad$^2$/Hz)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for subkey in totalOutputData[key]: # groupNames
            j = np.where(
                subkey==np.array(list(totalOutputData[key].keys()))
            )[0][0]
            ax.semilogx(
                totalOutputData[key][subkey]['frequencies'],
                totalOutputData[key][subkey]['avg_PSD'],
                c=colors[j]
            )

        ax.legend(prettyGroupNames,loc='upper right')
        figList.append(fig)

    if returnFigs==True:
        return(figList)

def plot_training_performance(
        trainingData,
        numberOfEpochs,
        numberOfTrials,
        returnFig=True
    ):
    prettyGroupNames = [
        "All Available States",
        "The Bio-Inspired Set",
        "Motor Position and Velocity Only",
        "All Motor States"
    ]

    epochArray = np.arange(0,numberOfEpochs+1,1)

    fig = plt.figure(figsize=(8,6))
    plt.yscale("log")
    plt.title("Performance vs. Epoch\n" + "(" + str(numberOfTrials) + " Trials)")
    fig.subplots_adjust(bottom=0.2,top=0.9)
    ax = plt.gca()
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Performance (RMSE in deg.)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xticks(list(np.linspace(0,numberOfEpochs,6)))
    ax.set_xticklabels([int(el) for el in ax.get_xticks()])

    lines=[]
    for i in range(len(groupNames)):
        for j in range(len(trainingData[groupNames[i]]["all_perf"])):
            trialMSE = trainingData[groupNames[i]]["all_perf"][j]
            line, = ax.plot(
                epochArray[:len(trialMSE)],
                180*np.sqrt(trialMSE)/np.pi,
                c=colors[i],
                alpha=0.65,
                lw=2,
                label=prettyGroupNames[i]
            )
            if j==0:
                lines.append(line)
    ax.legend(lines,prettyGroupNames,
        bbox_to_anchor=(0.5,-0.125),
        ncol=2,
        loc='upper center',
        fontsize=12
    )
    # ax.legend(groupNames,loc='upper right')

    if returnFig==True:
        return(fig)
    else:
        plt.show()

def plot_training_epoch_bar_plots(trainingData,addTitle=None,returnFig=True):
    if addTitle is not None:
        assert type(addTitle)==str, "addTitle must be a string."
        addTitle="\n"+addTitle
    else:
        addTitle=''

    prettyGroupNames = [
        "All\n Available\n States",
        "The\n Bio-Inspired\n Set",
        "Motor Position\n and\n Velocity Only",
        "All\n Motor\n States"
    ]

    allMean = np.mean(trainingData["all"]["best_epoch"])
    allSTD = np.std(trainingData["all"]["best_epoch"])

    bioMean = np.mean(trainingData["bio"]["best_epoch"])
    bioSTD = np.std(trainingData["bio"]["best_epoch"])

    kinapproxMean = np.mean(trainingData["kinapprox"]["best_epoch"])
    kinapproxSTD = np.std(trainingData["kinapprox"]["best_epoch"])

    allmotorMean = np.mean(trainingData["allmotor"]["best_epoch"])
    allmotorSTD = np.std(trainingData["allmotor"]["best_epoch"])

    means = [allMean,bioMean,kinapproxMean,allmotorMean]
    stds = [allSTD,bioSTD,kinapproxSTD,allmotorSTD]
    xticks = np.arange(len(prettyGroupNames))
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(12,7))
    fig.subplots_adjust(bottom=0.1)
    for i in range(len(prettyGroupNames)):
        ax.bar(xticks[i], means[i],
            yerr=stds[i],
            align='center',
            alpha=0.5,
            color=colors[i],
            capsize=10
        )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Number of Epochs")
    ax.set_title("Average Number of Epochs to Reach Convergence"+addTitle,fontsize=24)
    ax.set_xticks(xticks)
    ax.set_xticklabels(prettyGroupNames)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if returnFig==True:
        return(fig)
    else:
        plt.show()

def plot_bar_plots(experimentalData,metric="MAE",returnFig=True,yscale='linear'):
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
    assert yscale in ['linear','log'], "yscale must be either 'linear' (default) or 'log'."

    formattedLabels = [label[1:-1].replace("/","\n") for label in labels]

    allValue = [
        (180/np.pi)*experimentalData[movement]["all"][valueKey]
        for movement in movementTypes
    ] # in deg.
    bioValue = [
        (180/np.pi)*experimentalData[movement]["bio"][valueKey]
        for movement in movementTypes
    ] # in deg.
    kinapproxValue = [
        (180/np.pi)*experimentalData[movement]["kinapprox"][valueKey]
        for movement in movementTypes
    ] # in deg.
    allmotorValue = [
        (180/np.pi)*experimentalData[movement]["allmotor"][valueKey]
        for movement in movementTypes
    ] # in deg.

    xticks = np.arange(len(formattedLabels))  # the label locations
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
    if yscale=='log':
        ax.set_yscale('log')
    ax.set_title(baseTitle)
    ax.set_xticks(xticks)
    ax.set_xticklabels(formattedLabels)
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if returnFig==True:
        return(fig)
    else:
        plt.show()

def return_radial_bins(experimentalData,movement,bins=12,metrics=["MAE"]):
    # TODO: change to dict to ensure that the errors match up with the group.
    assert type(metrics)==list, "metrics must be a list."
    assert movement in ["angleSin_stiffSin","angleSin_stiffStep","angleStep_stiffSin","angleStep_stiffStep"], 'movement must be either "angleSin_stiffSin","angleSin_stiffStep","angleStep_stiffSin", or "angleStep_stiffStep".'

    assert np.all([type(el)==str for el in metrics]), "metrics must be a list of strings."
    assert np.all([el in ["RMSE","MAE","STD"] for el in metrics]), "Invalid metric entered."

    thetaRays = (180/np.pi)*np.arange(0,np.pi+1e-3,np.pi/bins)
    radial_bins={
        "bins" : bins,
        "all" : {},
        "bio" : {},
        "kinapprox" : {},
        "allmotor" : {}
    }
    for metric in metrics:
        radial_bins["max"+metric]=0
        radial_bins["min"+metric]=100

    for group in groupNames:
        expectedJointAngle = (180/np.pi)*(
            experimentalData[movement][group]['expectedJointAngle'] - np.pi/2
        ) # in degrees
        rawError = (180/np.pi)*experimentalData[movement][group]['rawError'] # in degrees
        for j in range(len(thetaRays)-1):
            bin_name = f"{thetaRays[j]:0.1f} to {thetaRays[j+1]:0.1f}"
            indices = np.array(
                np.where(
                    np.logical_and(
                        expectedJointAngle<thetaRays[j+1],
                        expectedJointAngle>=thetaRays[j]
                    )
                )
            )
            if indices.size>0:
                radial_bins[group][bin_name] = {}
                radial_bins[group][bin_name]["errors"] = rawError[indices] # in degrees
                radial_bins[group][bin_name]["abs errors"] = abs(
                    radial_bins[group][bin_name]["errors"]
                ) # in degrees

                ### Mean absolute error
                if "MAE" in metrics:
                    radial_bins[group][bin_name]["MAE"] = \
                        radial_bins[group][bin_name]["abs errors"].mean() # in degrees
                    radial_bins["minMAE"] = min([
                        radial_bins["minMAE"],
                        radial_bins[group][bin_name]["MAE"]
                    ]) # in degrees
                    radial_bins["maxMAE"] = max([
                        radial_bins["maxMAE"],
                        radial_bins[group][bin_name]["MAE"]
                    ]) # in degrees


                ### Root mean squared error
                if "RMSE" in metrics:
                    radial_bins[group][bin_name]["RMSE"] = np.sqrt(
                        (radial_bins[group][bin_name]["errors"]**2).mean()
                    ) # in degrees
                    radial_bins["minRMSE"] = min([
                        radial_bins["minRMSE"],
                        radial_bins[group][bin_name]["RMSE"]
                    ]) # in degrees
                    radial_bins["maxRMSE"] = max([
                        radial_bins["maxRMSE"],
                        radial_bins[group][bin_name]["RMSE"]
                    ]) # in degrees
                # radial_bins[group][bin_name]["abs error std"] = \
                #     radial_bins[group][bin_name]["abs errors"].std() # in degrees
                # radial_bins[group][bin_name]["min abs error"] = \
                #     radial_bins[group][bin_name]["errors"].min() # in degrees
                # radial_bins[group][bin_name]["max abs error"] = \
                #     radial_bins[group][bin_name]["errors"].max() # in degrees
                #
                # radial_bins[group][bin_name]["avg error"] = \
                #     radial_bins[group][bin_name]["errors"].mean() # in degrees
                if "STD" in metrics:
                    radial_bins[group][bin_name]["STD"] = \
                        radial_bins[group][bin_name]["errors"].std() # in degrees
                    radial_bins["minSTD"] = min([
                        radial_bins["minSTD"],
                        radial_bins[group][bin_name]["STD"]
                    ]) # in degrees
                    radial_bins["maxSTD"] = max([
                        radial_bins["maxSTD"],
                        radial_bins[group][bin_name]["STD"]
                    ]) # in degrees
            # else:
            #     radial_bins[group][bin_name]["errors"] = None
            #     radial_bins[group][bin_name]["abs errors"] = None
            #     radial_bins[group][bin_name]["MAE"] = None
            #     radial_bins[group][bin_name]["RMSE"] = None
            #     radial_bins[group][bin_name]["abs error std"] = None
            #     radial_bins[group][bin_name]["min abs error"] = None
            #     radial_bins[group][bin_name]["max abs error"] = None
            #     radial_bins[group][bin_name]["avg error"] = None
            #     radial_bins[group][bin_name]["STD"] = None
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
        # maxValue = np.log10(radial_bins["maxMAE"])+2
        offset = 1 - np.floor(np.log10(radial_bins['minMAE']))
        maxValue = np.log10(radial_bins['maxMAE'])+offset
    elif metric == 'STD':
        baseTitle = "Polar Bar Plots of Error Std Dev vs. Joint Angle"
        xLabel = "Log Error Std Dev (in deg.)"
        offset = 1 - np.floor(np.log10(radial_bins['minSTD']))
        maxValue = np.log10(radial_bins['maxSTD'])+offset
    elif metric == 'RMSE':
        baseTitle = "Polar Bar Plots of RMSE vs. Joint Angle"
        xLabel = "Log RMSE (in deg.)"
        offset = 1 - np.floor(np.log10(radial_bins['minRMSE']))
        maxValue = np.log10(radial_bins['maxRMSE'])+offset

    if maxValue%1<=np.log10(2):
        maxValue = np.floor(maxValue)

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
    thetaRays = (180/np.pi)*np.arange(0,np.pi+1e-3,np.pi/slices)
    for i in range(4):
        for j in range(len(thetaRays)-1):
            bin_name = f"{thetaRays[j]:0.1f} to {thetaRays[j+1]:0.1f}"
            if j%2==0:
                axs[i].add_patch(
                    Wedge(
                        (0,0), np.ceil(maxValue)+np.log10(2),
                        thetaRays[j],
                        thetaRays[j+1],
                        color = "0.85"
                    )
                )
            # if radial_bins[groupNames[i]][bin_name][metric] is not None:
            if bin_name in radial_bins[groupNames[i]]:
                axs[i].add_patch(
                    Wedge(
                        (0,0),
                        np.log10(radial_bins[groupNames[i]][bin_name][metric])+offset,
                        thetaRays[j],
                        thetaRays[j+1],
                        color = colors[i],
                        alpha=0.65
                    )
                )
            else:
                axs[i].add_patch(
                    Wedge(
                        (0,0),
                        np.ceil(maxValue)+np.log10(2),
                        thetaRays[j],
                        thetaRays[j+1],
                        color = "k",
                        alpha=0.65
                    )
                )
        axs[i].set_aspect('equal')
        axs[i].set_ylim([0,np.ceil(maxValue)+np.log10(2)])
        axs[i].set_xlim([
            -(np.ceil(maxValue)+np.log10(2)),
            np.ceil(maxValue)+np.log10(2)
        ])
        xticks = np.arange(1,np.ceil(maxValue)+1e-3,1)
        xticks = np.concatenate([
            -np.array(list(reversed(xticks))),
            xticks
        ])
        axs[i].set_xticks(xticks)
        xTickLabels = [r"$10^{%d}$" % (abs(el)-offset) for el in xticks]
        axs[i].set_xticklabels(xTickLabels)
        axs[i].add_patch(Wedge((0,0),1,0,360,color ='w'))

        xticksMinor = np.concatenate([
            np.linspace(10**(k),10**(k+1),10)[1:-1]
            for k in range(int(1-offset),int(np.ceil(maxValue)-offset))
        ])
        xticksMinor = np.concatenate(
            [-np.array(list(reversed(xticksMinor))),xticksMinor]
        )
        xticksMinor = [np.sign(el)*(np.log10(abs(el))+offset) for el in xticksMinor]
        axs[i].set_xticks(xticksMinor,minor=True)

        yticks = list(np.arange(1,np.ceil(maxValue)+1e-3,1))
                # yticks = list(np.arange(0,np.floor(maxValue)+1e-3,1))
        axs[i].set_yticks(yticks)
        axs[i].set_yticklabels(["" for tick in axs[i].get_yticks()])
        yticksMinor = np.concatenate([
            np.linspace(10**(k),10**(k+1),10)[1:-1]
            for k in range(int(1-offset),int(np.ceil(maxValue)-offset))
        ])
        yticksMinor = [np.sign(el)*(np.log10(abs(el))+offset) for el in yticksMinor]
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
        axs[i].plot([1,np.ceil(maxValue)+np.log10(2)],[0,0],'k',linewidth=1.5)# double lw because of ylim
        axs[i].plot([-(np.ceil(maxValue)+np.log10(2)),-1],[0,0],'k',linewidth=1.5)# double lw because of ylim
        axs[i].plot([0,0],[1,np.ceil(maxValue)+np.log10(2)],'k',linewidth=0.5)

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
        baseTitle = "Polar Bar Plots of MAE vs. Joint Angle"
        title = "MAE\n vs.\n Joint Angle"
        xLabel = "Log MAE (in deg.)"
        # maxValue = np.log10(radial_bins["maxMAE"])+2
        offset = 1 - np.floor(np.log10(radial_bins['minMAE']))
        maxValue = np.log10(radial_bins['maxMAE'])+offset
    elif metric == 'STD':
        baseTitle = "Polar Bar Plots of Error Std Dev vs. Joint Angle"
        title = "Error Std Dev\n vs.\n Joint Angle"
        xLabel = "Log Error Std Dev (in deg.)"
        offset = 1 - np.floor(np.log10(radial_bins['minSTD']))
        maxValue = np.log10(radial_bins['maxSTD'])+offset
    elif metric == 'RMSE':
        baseTitle = "Polar Bar Plots of RMSE vs. Joint Angle"
        title = "RMSE\n vs.\n Joint Angle"
        xLabel = "Log RMSE (in deg.)"
        offset = 1 - np.floor(np.log10(radial_bins['minRMSE']))
        maxValue = np.log10(radial_bins['maxRMSE'])+offset

    if maxValue%1<=np.log10(2):
        maxValue = np.floor(maxValue)

    try:
        import __main__ as main
        basePath = path.dirname(main.__file__)
        filePath = path.abspath(path.join(basePath, "..", "SupplementaryFigures", "Schematic_1DOF2DOA_system.png"))
    except:
        filePath = Path(r"C:/Users/hagen/Documents/Github/insideOut/SupplementaryFigures/Schematic_1DOF2DOA_system.png")
    im = Image.open(filePath)
    height = im.size[1]
    width = im.size[0]
    aspectRatio = width/height

    fig = plt.figure(figsize=(10,8))
    if addTitle is not None:
        assert type(addTitle)==str, "title must be a string."
    else:
        addTitle=""
    plt.title(baseTitle+"\n"+addTitle,fontsize=16,y=-0.35)

    newHeight = int(np.ceil(0.15*fig.bbox.ymax)+10)
    size = int(newHeight*aspectRatio),newHeight
    im.thumbnail(size, Image.ANTIALIAS)
    fig.figimage(im, fig.bbox.xmax/2 - im.size[0]/2.2, 0.95*im.size[1],zorder=10)
    ax = plt.gca()

    slices = radial_bins['bins']
    thetaRays = (180/np.pi)*np.arange(0,np.pi+1e-3,np.pi/slices) # in degrees
    sectorWidth = (180/np.pi)*(np.pi/slices)/5 # in degrees
    thetaRays_SplitInFourths = [] # in degrees
    for j in range(len(thetaRays)-1):
        midAngle = (thetaRays[j+1]+thetaRays[j])/2 # in degrees
        thetaRays_SplitInFourths.append(
            [(midAngle + i*sectorWidth) for i in [-2,-1,0,1]]
        ) # in degrees
    thetaRays_SplitInFourths = np.concatenate(thetaRays_SplitInFourths)

    for j in range(len(thetaRays)-1):
        count=0
        bin_name = f"{thetaRays[j]:0.1f} to {thetaRays[j+1]:0.1f}"
        if j%2==0:
            ax.add_patch(
                Wedge(
                    (0,0), np.ceil(maxValue)+np.log10(2),
                    thetaRays[j],
                    thetaRays[j+1],
                    color = "0.85"
                )
            )
        for i in range(len(groupNames)):
            # if radial_bins[groupNames[i]][bin_name][metric] is not None:
            if bin_name in radial_bins[groupNames[i]]:
                ax.add_patch(
                    Wedge(
                        (0,0),
                        np.log10(radial_bins[groupNames[i]][bin_name][metric])+offset,
                        thetaRays_SplitInFourths[4*j+i],
                        thetaRays_SplitInFourths[4*j+i]+sectorWidth,
                        color = colors[i],
                        alpha=0.65
                    )
                )
            else:
                count+=1
                if count==len(groupNames):
                    ax.add_patch(
                        Wedge(
                            (0,0),
                            np.ceil(maxValue)+np.log10(2),
                            thetaRays[j],
                            thetaRays[j+1],
                            color = "k",
                            alpha=0.65
                        )
                    )

    ax.set_aspect('equal')
    ax.set_ylim([0,np.ceil(maxValue)+np.log10(2)])
    ax.set_xlim([
        -(np.ceil(maxValue)+np.log10(2)),
        np.ceil(maxValue)+np.log10(2)
    ])
    xticks = np.arange(1,np.ceil(maxValue)+1e-3,1)
    xticks = np.concatenate([
        -np.array(list(reversed(xticks))),
        xticks
    ])
    ax.set_xticks(xticks)
    xTickLabels = [r"$10^{%d}$" % (abs(el)-offset) for el in xticks]
    ax.set_xticklabels(xTickLabels)
    ax.add_patch(Wedge((0,0),1,0,360,color ='w'))

    xticksMinor = np.concatenate([
        np.linspace(10**(k),10**(k+1),10)[1:-1]
        for k in range(int(1-offset),int(np.ceil(maxValue)-offset))
    ])
    xticksMinor = np.concatenate(
        [-np.array(list(reversed(xticksMinor))),xticksMinor]
    )
    xticksMinor = [np.sign(el)*(np.log10(abs(el))+offset) for el in xticksMinor]
    ax.set_xticks(xticksMinor,minor=True)

    yticks = list(np.arange(1,np.ceil(maxValue)+1e-3,1))
            # yticks = list(np.arange(0,np.floor(maxValue)+1e-3,1))
    ax.set_yticks(yticks)
    ax.set_yticklabels(["" for tick in ax.get_yticks()])
    yticksMinor = np.concatenate([
        np.linspace(10**(k),10**(k+1),10)[1:-1]
        for k in range(int(1-offset),int(np.ceil(maxValue)-offset))
    ])
    yticksMinor = [np.sign(el)*(np.log10(abs(el))+offset) for el in yticksMinor]
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
    ax.plot([1,np.ceil(maxValue)+np.log10(2)],[0,0],'k',linewidth=1.5)# double lw because of ylim
    ax.plot([-(np.ceil(maxValue)+np.log10(2)),-1],[0,0],'k',linewidth=1.5)# double lw because of ylim
    ax.plot([0,0],[1,np.ceil(maxValue)+np.log10(2)],'k',linewidth=0.5)

    props = dict(
        boxstyle='round',
        facecolor='w',
        edgecolor='0.70'
    )
    ax.text(
        -0.9*np.ceil(maxValue),np.ceil(maxValue),
        title,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16,
        bbox=props
    )
    ax.text(
        (np.ceil(maxValue)-1)/2+1,-0.35,
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

def plot_all_polar_bar_plots(
        experimentalData,
        metric,
        totalRadialBins=None,
        returnFigs=True
        ):
    assert type(metric)==str and metric in ["MAE","STD","RMSE"], "metric must be either 'MAE' (default), 'RMSE' or 'STD'."

    ### radial average error versus positions

    if totalRadialBins is None:
        figs = []
        for i in range(len(movementTypes)):
            movement = movementTypes[i]
            radial_bins = return_radial_bins(experimentalData,movement,bins=12,metrics=[metric])
            tempFig = plot_polar_bar_plots(
                radial_bins,
                metric=metric,
                addTitle=labels[i],
                returnFig=True
            )
            figs.append(tempFig)

            tempFig = plot_polar_bar_plots_together(
                radial_bins,
                metric=metric,
                addTitle=labels[i],
                returnFig=True
            )
            figs.append(tempFig)
    else:
        figs=[]
        for i in range(len(movementTypes)):
            movement = movementTypes[i]
            tempFig = plot_polar_bar_plots(
                totalRadialBins[movement],
                metric=metric,
                addTitle=labels[i],
                returnFig=True
            )
            figs.append(tempFig)

            tempFig = plot_polar_bar_plots_together(
                totalRadialBins[movement],
                metric=metric,
                addTitle=labels[i],
                returnFig=True
            )
            figs.append(tempFig)

    if returnFigs==True:
        return(figs)
    else:
        plt.show()

def plot_all_error_distributions(outputData,returnFigs=True):

    figs = []
    for i in range(len(movementTypes)):
        tempFig, axs = plt.subplots(2,2,figsize=(10,10))
        plt.suptitle(labels[i],fontsize=16)
        for j in range(len(groupNames)):
            data = 180*outputData[movementTypes[i]][groupNames[j]]['rawError'].flatten()/np.pi
            axs[int(j/2)][j%2].hist(
                data,
                weights=np.ones(len(data)) / len(data),
                bins=60,
                color=colors[j]
            )
            axs[int(j/2)][j%2].set_yticklabels(["{:.1f}%".format(100*el) for el in axs[int(j/2)][j%2].get_yticks()])
            axs[int(j/2)][j%2].set_title(
                groupNames[j],
                fontsize=14,
                color=colors[j]
            )
            axs[int(j/2)][j%2].spines['top'].set_visible(False)
            axs[int(j/2)][j%2].spines['right'].set_visible(False)
        figs.append(tempFig)

    if returnFigs==True:
        return(figs)

def plot_metric_distributions(outputData,metric,returnFigs=True):
    assert metric in ["MAE","RMSE","STD"], "metric must be either 'MAE', 'RMSE', or 'STD'."
    prettyGroupNames = [
        "All\n Available\n States",
        "The\n Bio-Inspired\n Set",
        "Motor Position\n and\n Velocity Only",
        "All\n Motor\n States"
    ]

    """
        You should create a 3,2 plot with the first row being just the overlapping KSE plots (hist=False), then you should do a normed hist for each of the following subplots for each group.
    """
    figs = []
    numberOfTrials = len(
        outputData[movementTypes[0]][groupNames[0]]['experiment'+metric+"_list"]
    )
    for i in range(len(movementTypes)):
        tempFig = plt.figure(figsize=(10,10))
        gs = tempFig.add_gridspec(3,2)
        ax1 = tempFig.add_axes((0.1,0.7,0.8,0.25)) #All KDE plots
        ax1.set_xlabel(metric + " (in deg.)")
        ax1.set_ylabel("Kernel Density Estimation")
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2 = tempFig.add_axes((0.1,0.3625,0.35,0.275)) # All
        ax3 = tempFig.add_axes((0.55,0.3625,0.35,0.275)) # Bio
        ax4 = tempFig.add_axes((0.1,0.05,0.35,0.275)) # Kin Approx
        ax4.set_xlabel(metric + " (in deg.)")
        ax4.set_ylabel("Percentage of Trials (N="+str(numberOfTrials)+")")
        ax5 = tempFig.add_axes((0.55,0.05,0.35,0.275)) # All Motor
        axs = [ax2,ax3,ax4,ax5]
        plt.suptitle(labels[i],fontsize=16)
        for j in range(len(groupNames)):
            data = np.array(
                outputData[movementTypes[i]][groupNames[j]]['experiment'+metric+"_list"]
            )*180/np.pi
            sns.distplot(
                data,
                hist=False,
                color=colors[j],
                ax=ax1
            )
            sns.distplot(
                data,
                hist=True,
                kde=False,
                color=colors[j],
                hist_kws={
                    'weights': np.ones(len(data))/len(data)
                },
                ax=axs[j]
            )
            axs[j].set_yticklabels(["{:.1f}%".format(100*el) for el in axs[j].get_yticks()])
            axs[j].text(
                1.5*np.average(axs[j].get_xlim()),
                0.75*axs[j].get_ylim()[1],
                prettyGroupNames[j],
                fontsize=14,
                color=colors[j],
                horizontalalignment='center',
                verticalalignment='center',
                bbox=dict(
                    facecolor='white',
                    edgecolor=colors[j],
                    boxstyle='round,pad=0.5',
                    alpha=0.75
                )
            )
            # axs[j].set_title(
            #     prettyGroupNames[j],
            #     fontsize=14,
            #     color=colors[j],
            #     y=0.95
            # )
            axs[j].spines['top'].set_visible(False)
            axs[j].spines['right'].set_visible(False)
        figs.append(tempFig)

    if returnFigs==True:
        return(figs)

def plot_consolidated_data_babbling_duration_experiment(
        babblingDuration,
        plant,
        directory=None,
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
    if directory==None:
        directory = Path("experimental_trials/")
    else:
        directory = Path(directory)
        assert directory.is_dir(), "Enter a valid directory."

    folderName = (
        'Consolidated_Trials_'
        + '{:06d}'.format(int(babblingDuration*1000))
        + 'ms/'
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
        addTitle=f"({babblingDuration:0.1f} sec Babbling)",
        returnFig=True
    )

    saveParams = {
        "Babbling Duration" : babblingDuration,
        "Number of Trials" : numberOfTrials
    }
    save_figures(
        str(directory)+"/",
        "perf_v_epoch",
        saveParams,
        subFolderName=folderName,
        saveAsPDF=True,
        saveAsMD=True,
        addNotes="### Generated from `plot_consolidated_data_babbling_duration_experiment()`"
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
    x_indices,y_indices = return_average_error_bin_indices(
        xbin_edges,
        ybin_edges,
        plant
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
        shutil.rmtree(trialDirectories[n])
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
            returnFig=True
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

def plot_consolidated_data_number_of_nodes_experiment(numberOfNodes,directory=None,metrics=None,includePSD=False):
    if metrics is None:
        metrics = ["MAE"]
        metricKeys = ["experimentMAE"]
    else:
        assert type(metrics)==list, "metrics must be a list of strings."
        for metric in metrics:
            assert metric in ["RMSE","STD","MAE"], 'Invalid metric! metrics must include "RMSE","STD", or "MAE".'
        metricKeys = ["experiment"+metric for metric in metrics]

    ### get the testing trial directories
    if directory==None:
        directory = Path("experimental_trials/")
    else:
        directory = Path(directory)
        assert directory.is_dir(), "Enter a valid directory."

    folderName = (
        'Consolidated_Trials_'
        + '{:03d}'.format(int(numberOfNodes))
        + '_Nodes/'
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
                # totalTrainingData[key]["perf"] = np.array(
                #     tempTrainingData[key]["tr"]["perf"]._data
                # ) / numberOfTrials
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
                # totalTrainingData[key]["perf"] += np.array(
                #     tempTrainingData[key]["tr"]["perf"]._data
                # ) / numberOfTrials
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
        addTitle=f"({numberOfNodes:d} Hidden Nodes)",
        returnFig=True
    )

    saveParams = {
        "Babbling Duration" : plantParams["Simulation Duration"],
        "Number of Trials" : numberOfTrials,
        "Number of Nodes" : numberOfNodes
    }
    save_figures(
        directory,
        "perf_v_epoch",
        saveParams,
        subFolderName=folderName,
        saveAsPDF=True,
        saveAsMD=True,
        addNotes="###Plotting Training Performances for Node Number Experiment"
    )
    plt.close('all')

    # Experimental Data

    totalOutputData = {}

    keys = [
        "rawError",
        "expectedJointAngle"
    ]
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
                for group in groupNames:
                    for key in metricKeys:
                        totalOutputData[movement][group][key+"_list"] = [
                            totalOutputData[movement][group][key]
                        ]
                        totalOutputData[movement][group][key] = (
                            totalOutputData[movement][group][key]
                            / numberOfTrials
                        )
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
                for group in groupNames:
                    for key in metricKeys:
                        totalOutputData[movement][group][key+"_list"].append(
                            tempOutputData[movement][group][key]
                        )
                        totalOutputData[movement][group][key] += (
                            tempOutputData[movement][group][key]
                            / numberOfTrials
                        )
                    if includePSD==True:
                        _, PSD = signal.welch(
                            tempOutputData[movement][group]["rawError"],
                            1/plantParams['dt']
                        )
                        totalOutputData[movement][group]["avg_PSD"] += PSD/numberOfTrials
        # else:
        #     for movement in movementTypes:
        #         for group in groupNames:
        #             for key in keys:
        #                 if key not in metricKeys:
        #                     totalOutputData[movement][group][key] = \
        #                         np.concatenate([
        #                             totalOutputData[movement][group][key],
        #                             tempOutputData[movement][group][key]
        #                         ],
        #                         axis=0)
        #                 else:
        #                     totalOutputData[movement][group][key+"_list"].append(
        #                         tempOutputData[movement][group][key]
        #                     )
        #                     totalOutputData[movement][group][key] += (
        #                         tempOutputData[movement][group][key]
        #                         / numberOfTrials
        #                    )

        # delete trial directory
        shutil.rmtree(trialDirectories[n])
        """
            totalOutputData
                ..<Movement Type>
                    ..<Group Name>
                        [frequencies] (in Hz.)
                        [avg_PSD] (in rad.^2/Hz.)
                        experiment<Metric> (in rad.)
                        experiment<Metric>_list (in rad.)
        """

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
            directory+folderName,
            "err_PSD",
            {},
            figs=figs,
            subFolderName="error_PSD/",
            saveAsMD=True,
            addNotes="###Average error signal with power spectrum analysis."
        )
        plt.close('all')

    for metric in metrics:
        fig = plot_bar_plots(
            totalOutputData,
            metric=metric,
            returnFig=True
        )
        figs = plot_metric_distributions(
            totalOutputData,
            metric,
            returnFigs=True
        )
        # figs = plot_all_polar_bar_plots(
        #     totalOutputData,
        #     metric=metric,
        #     returnFigs=True
        # )
        # figs.insert(0,fig)
        figs.insert(0,fig)

        save_figures(
            directory/folderName,
            metric,
            {"metric":metric},
            figs=figs,
            subFolderName=metric+"/",
            saveAsPDF=True,
            saveAsMD=True
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

def return_average_error_bin_indices(xbin_edges,ybin_edges,plant):
    x_indices = dict(zip(movementTypes,[[]]*4))
    x_indices["xbin_edges"] = xbin_edges
    y_indices = dict(zip(movementTypes,[[]]*4))
    y_indices["ybin_edges"] = ybin_edges

    for movement in movementTypes:
        movementOutput = sio.loadmat(
            "experimental_trials/"
            + movement
            + "_outputData.mat"
        )
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

# def return_average_error_bins(experimentalData,movement,metric,xbins=12,ybins=9,jointAngleBounds=None,jointStiffnessBounds=None):
#     """
#         averageErrorBins
#             ...<jointAngleBounds>
#             ...<jointStiffnessBounds>
#             ...<xbins>
#             ...<ybins>
#             ...<group>
#                 ...(xbins,ybins) array of average error values
#     """
#     assert movement in ["angleSin_stiffSin","angleSin_stiffStep","angleStep_stiffSin","angleStep_stiffStep"], 'movement must be either "angleSin_stiffSin","angleSin_stiffStep","angleStep_stiffSin", or "angleStep_stiffStep".'
#
#     assert type(metric)==str and metric in ["MAE","RMSE"], "metric must be either 'MAE' (default) or 'RMSE'."
#     if metric == "MAE":
#         add_error_value = lambda err: abs(err)
#         calc_average = lambda err_total,n : err_total/n
#     elif metric == 'RMSE':
#         add_error_value = lambda err: err**2
#         calc_average = lambda err_total,n : np.sqrt(err_total/n)
#
#     # is_number(xbins,"xbins",default=12,notes="Should be an int.")
#     # is_number(ybins,"ybins",default=9,notes="Should be an int.")
#
#     # if jointAngleBounds is not None:
#     #     assert type(jointAngleBounds)==list and len(jointAngleBounds)==2, "jointAngleBounds must be a list."
#     #     assert jointAngleBounds[1]>jointAngleBounds[0], "jointAngleBounds must be in ascending order."
#     # else:
#     #     jointAngleBounds = [
#     #         plantParams["Joint Angle Bounds"]["LB"],
#     #         plantParams["Joint Angle Bounds"]["UB"]
#     #     ]
#     # jointAngleBounds = [(el-np.pi)*180/np.pi for el in jointAngleBounds] # convert to degrees and center around vertical
#     #
#     # if jointStiffnessBounds is not None:
#     #     assert type(jointStiffnessBounds)==list and len(jointStiffnessBounds)==2, "jointStiffnessBounds must be a list."
#     #     assert jointStiffnessBounds[1]>jointStiffnessBounds[0], "jointStiffnessBounds must be in ascending order."
#     # else:
#     #     jointStiffnessBounds = [10,plantParams["Maximum Joint Stiffness"]]
#     #
#     # xBinSize = (jointAngleBounds[1]-jointAngleBounds[0])/xbins
#     # yBinSize = (jointStiffnessBounds[1]-jointStiffnessBounds[0])/ybins
#     # xbin_edges = np.arange(
#     #     jointAngleBounds[0],
#     #     jointAngleBounds[1]+1e-3,
#     #     xBinSize
#     # )# in degrees
#     # ybin_edges = np.arange(
#     #     jointStiffnessBounds[0],
#     #     jointStiffnessBounds[1]+1e-3,
#     #     yBinSize
#     # )
#     #
#     # movementOutput = sio.loadmat(
#     #     "experimental_trials/"
#     #     + movement
#     #     + "_outputData.mat"
#     # )
#     # xd = (movementOutput["X1d"][0,:]-np.pi)*180/np.pi # (in deg, centered at vertical)
#     # sd = movementOutput["Sd"][0,:]
#
#     averageError = {
#         "jointAngleBounds":jointAngleBounds,
#         "jointStiffnessBounds":jointStiffnessBounds,
#         "xbins":xbins,
#         "ybins":ybins,
#         "all":{},
#         "bio":{},
#         "kinapprox":{},
#         "allmotor":{}
#     }
#     #dict(zip(groupNames,[{}]*4)) TODO: move dict definitions
#     for i in range(4):
#         group = groupNames[i]
#         error = experimentalData[movement][group]["rawError"]*180/np.pi # in degrees
#         averageError[group] = np.zeros((len(xbin_edges)-1,len(ybin_edges)-1)) # in degrees
#         counter = np.zeros(np.shape(averageError[group]))
#
#         x_index = np.digitize(xd,xbin_edges)-1
#         y_index = np.digitize(sd,ybin_edges)-1
#         for j in range(len(x_index)):
#             if x_index[j]==averageError[group].shape[0]:
#                 x_index[j]-=1
#             if y_index[j]==averageError[group].shape[1]:
#                 y_index[j]-=1
#             averageError[group][x_index[j],y_index[j]]+=add_error_value(error[j])
#             counter[x_index[j],y_index[j]]+=1
#
#         for j in range(averageError[group].shape[0]):
#             for k in range(averageError[group].shape[1]):
#                 if counter[j,k]!=0:
#                     averageError[group][j,k] = calc_average(
#                         averageError[group][j,k],
#                         counter[j,k]
#                     )
#                 else:
#                     averageError[group][j,k]=-1 # NOTE: negative values will be ignored since all errors must be positive
#     return(averageError)
def return_average_error_bins(experimentalData,movement,metric,x_indices,y_indices):
    """
        averageErrorBins
            ...<jointAngleBounds>
            ...<jointStiffnessBounds>
            ...<xbins>
            ...<ybins>
            ...<group>
                ...(xbins,ybins) array of average error values
    """
    assert movement in ["angleSin_stiffSin","angleSin_stiffStep","angleStep_stiffSin","angleStep_stiffStep"], 'movement must be either "angleSin_stiffSin","angleSin_stiffStep","angleStep_stiffSin", or "angleStep_stiffStep".'

    assert type(metric)==str and metric in ["MAE","RMSE"], "metric must be either 'MAE' (default) or 'RMSE'."
    if metric == "MAE":
        add_error_value = lambda err: abs(err)
        calc_average = lambda err_total,n : err_total/n
    elif metric == 'RMSE':
        add_error_value = lambda err: err**2
        calc_average = lambda err_total,n : np.sqrt(err_total/n)

    # is_number(xbins,"xbins",default=12,notes="Should be an int.")
    # is_number(ybins,"ybins",default=9,notes="Should be an int.")

    # if jointAngleBounds is not None:
    #     assert type(jointAngleBounds)==list and len(jointAngleBounds)==2, "jointAngleBounds must be a list."
    #     assert jointAngleBounds[1]>jointAngleBounds[0], "jointAngleBounds must be in ascending order."
    # else:
    #     jointAngleBounds = [
    #         plantParams["Joint Angle Bounds"]["LB"],
    #         plantParams["Joint Angle Bounds"]["UB"]
    #     ]
    # jointAngleBounds = [(el-np.pi)*180/np.pi for el in jointAngleBounds] # convert to degrees and center around vertical
    #
    # if jointStiffnessBounds is not None:
    #     assert type(jointStiffnessBounds)==list and len(jointStiffnessBounds)==2, "jointStiffnessBounds must be a list."
    #     assert jointStiffnessBounds[1]>jointStiffnessBounds[0], "jointStiffnessBounds must be in ascending order."
    # else:
    #     jointStiffnessBounds = [10,plantParams["Maximum Joint Stiffness"]]
    #
    # xBinSize = (jointAngleBounds[1]-jointAngleBounds[0])/xbins
    # yBinSize = (jointStiffnessBounds[1]-jointStiffnessBounds[0])/ybins
    # xbin_edges = np.arange(
    #     jointAngleBounds[0],
    #     jointAngleBounds[1]+1e-3,
    #     xBinSize
    # )# in degrees
    # ybin_edges = np.arange(
    #     jointStiffnessBounds[0],
    #     jointStiffnessBounds[1]+1e-3,
    #     yBinSize
    # )
    #
    # movementOutput = sio.loadmat(
    #     "experimental_trials/"
    #     + movement
    #     + "_outputData.mat"
    # )
    # xd = (movementOutput["X1d"][0,:]-np.pi)*180/np.pi # (in deg, centered at vertical)
    # sd = movementOutput["Sd"][0,:]

    averageError = dict(zip(groupNames,[{}]*4))
    #dict(zip(groupNames,[{}]*4)) TODO: move dict definitions
    xbin_edges = x_indices["xbin_edges"]
    ybin_edges = y_indices["ybin_edges"]

    for i in range(4):
        group = groupNames[i]
        error = experimentalData[movement][group]["rawError"]*180/np.pi # in degrees
        averageError[group] = np.zeros((len(xbin_edges)-1,len(ybin_edges)-1)) # in degrees
        counter = np.zeros(np.shape(averageError[group]))

        for j in range(len(x_indices[movement])):
            xi,yi = x_indices[movement][j],y_indices[movement][j]
            averageError[group][xi,yi]+=add_error_value(error[j])
            counter[xi,yi]+=1

        for j in range(averageError[group].shape[0]):
            for k in range(averageError[group].shape[1]):
                if counter[j,k]!=0:
                    averageError[group][j,k] = calc_average(
                        averageError[group][j,k],
                        counter[j,k]
                    )
                else:
                    averageError[group][j,k]=-1 # NOTE: negative values will be ignored since all errors must be positive
    return(averageError)

def plot_2D_heatmap_of_error_wrt_desired_trajectory(outputData,metric,plant,averageErrorBins=None,returnFigs=True,addTitle=None,xbins=12,ybins=9,jointAngleBounds=None,jointStiffnessBounds=None,returnAverageError=False,colorScale="linear"):
    if metric == "MAE":
        baseTitle = "MAE vs. Joint Angle vs. Joint Stiffness"
        colorbarLabel = "Log MAE (in deg.)"
    elif metric == 'RMSE':
        baseTitle = "RMSE vs. Joint Angle vs. Joint Stiffness"
        colorbarLabel = "Log RMSE (in deg.)"

    prettyGroupNames = [
        "All Available States",
        "The Bio-Inspired Set",
        "Motor Position and Velocity Only",
        "All Motor States"
    ]

    assert type(returnAverageError)==bool,'returnAverageError must be either true or false (default).'

    if jointAngleBounds is not None:
        assert type(jointAngleBounds)==list and len(jointAngleBounds)==2, "jointAngleBounds must be a list."
        assert jointAngleBounds[1]>jointAngleBounds[0], "jointAngleBounds must be in ascending order."
    else:
        jointAngleBounds = [np.pi/2,3*np.pi/2]

    if jointStiffnessBounds is not None:
        assert type(jointStiffnessBounds)==list and len(jointStiffnessBounds)==2, "jointStiffnessBounds must be a list."
        assert jointStiffnessBounds[1]>jointStiffnessBounds[0], "jointStiffnessBounds must be in ascending order."
    else:
        jointStiffnessBounds = [10,100]

    assert colorScale in ["linear", "log"], "colorScale should either be 'linear' (default) or 'log'."

    labelDict = dict(zip(movementTypes,labels))

    jointAngleBounds_inDegrees = [(el-np.pi)*180/np.pi for el in jointAngleBounds] # in degrees centered at vertical

    xbin_edges = np.arange(
        jointAngleBounds_inDegrees[0],
        jointAngleBounds_inDegrees[1]+1e-3,
        (jointAngleBounds_inDegrees[1]-jointAngleBounds_inDegrees[0])/xbins
    )# in degrees
    ybin_edges = np.arange(
        jointStiffnessBounds[0],
        jointStiffnessBounds[1]+1e-3,
        (jointStiffnessBounds[1]-jointStiffnessBounds[0])/ybins
    )

    if addTitle is not None:
        assert type(addTitle)==str, "addTitle must be a string."
        title = baseTitle + "\n" + addTitle
    else:
        title = baseTitle
    figs = []

    averageErrorTotal = dict(zip(movementTypes,[{}]*4))
    if averageErrorBins is not None:
        x_indices,y_indices = return_average_error_bin_indices(
            xbin_edges,ybin_edges,plant
        )
    maxError = 0
    minError = 10
    if averageErrorBins is None:
        averageErrorDict = {}
    for movement in movementTypes:
        if averageErrorBins is not None:
            averageError = {}
            for group in groupNames:
                averageError[group] = \
                    averageErrorBins[movement][group][metric]
            maxError = max([maxError,averageErrorBins[movement]["max"+metric]])
            if colorScale=='linear':
                minError=0
            else: #colorScale=='log'
                # minError = 10
                for i in range(np.shape(averageError[group])[0]):
                    for j in range(np.shape(averageError[group])[1]):
                        if averageError[group][i,j]>0:
                            minError = min([minError,averageError[group][i,j]])
                # minError = 10**(np.floor(np.log10(minError)))
        else:
            averageError = return_average_error_bins(
                outputData,movement,metric,
                x_indices,y_indices
            )
            averageErrorDict[movement] = averageError
            maxError = max([
                maxError,
                max([averageError[group].max() for group in groupNames])
            ])
            if colorScale=='linear':
                minError=0
            else: #colorScale=='log'
                minError = min([
                    minError,
                    min([averageError[group].min() for group in groupNames])
                ])

    for movement in movementTypes:
        if averageErrorBins is not None:
            averageError = {}
            for group in groupNames:
                averageError[group] = \
                    averageErrorBins[movement][group][metric]
        else:
            averageError = averageErrorDict[movement]

        if returnAverageError==True:
            averageErrorTotal[movement]=averageError

        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(
            2,2,
            figsize=(12,12),
            sharex=True,
            sharey=True
        )
        figs.append(fig)
        plt.suptitle(title + "\n" + labelDict[movement],fontsize=14)
        axs=[ax1,ax2,ax3,ax4]

        if colorScale=="linear":
            norm = matplotlib.colors.Normalize(vmin=minError, vmax=0.3)
        else: # colorScale=="log"
            norm = matplotlib.colors.LogNorm(vmin=minError, vmax=maxError)

        for i in range(4):
            group = groupNames[i]
            RGB_color = tuple(int(colors[i][1:][j:j+2],16)/256 for j in (0, 2, 4))
            # cmap = LinearSegmentedColormap.from_list(
            #     'my_cm',
            #     [RGB_color,(1,1,1)],
            #     N=100
            # )
            cmap = LinearSegmentedColormap.from_list(
                'my_cm',
                [(0,0,0),(1,1,1)],
                N=100
            )
            for j in range(averageError[group].shape[0]):
                for k in range(averageError[group].shape[1]):
                    vertices = np.array([
                        [xbin_edges[j],ybin_edges[k]],
                        [xbin_edges[j+1],ybin_edges[k]],
                        [xbin_edges[j+1],ybin_edges[k+1]],
                        [xbin_edges[j],ybin_edges[k+1]]
                    ])
                    if averageError[group][j,k]>=0:
                        tempPolygon = Polygon(
                            vertices,
                            facecolor=cmap(norm(averageError[group][j,k])),
                            edgecolor=None,
                            closed=True,
                            zorder=-1
                        )
                        axs[i].add_patch(tempPolygon)
                    else:
                        tempPolygon = Polygon(
                            vertices,
                            color=colors[i],
                            alpha=0.65,
                            closed=True,
                            lw=0.5,
                            zorder=-2
                        )
                        axs[i].add_patch(tempPolygon)

            axs[i].set_title(prettyGroupNames[i],color=colors[i])
            axs[i].set_xlim([xbin_edges[0],xbin_edges[-1]])
            # axs[i].set_xticks(xbin_edges,minor=True)
            # axs[i].set_xticks([
            #     jointAngleBounds_inDegrees[0],
            #     0,
            #     jointAngleBounds_inDegrees[-1]
            # ])
            axs[i].set_xticks([-90,-45,0,45,90])
            # axs[i].set_yticks(ybin_edges,minor=True)
            # axs[i].set_yticks([jointStiffnessBounds[0],jointStiffnessBounds[-1]])
            axs[i].set_yticks(np.arange(10,ybin_edges[-1]+1e-3,10))
            axs[i].set_ylim([10,ybin_edges[-1]])
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            if i==2:
                axs[i].set_ylabel("Joint Stiffness (Nm/rad.)")
                axs[i].set_xlabel("Joint Angle (deg. from vertical)")
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
            cbar_ax.set_ylabel("Average " + metric+ " (deg.)")
            # fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[i])
    if returnFigs==True:
        if returnAverageError==True:
            return(figs,averageErrorTotal)
        else:
            return(figs)
    elif returnAverageError==True:
        return(averageErrorTotal)
        plt.show()
    else:
        plt.show()

# figs,AE = plot_2D_heatmap_of_error_wrt_desired_trajectory(experimentalData,"MAE",plant, xbins=20,ybins=20,jointStiffnessBounds=[20,100],returnAverageError=True)
#
# save_figures(
#     "visualizations/FBL_trajectories/angleStep_stiffStep/",
#     "heatmap_MAE",
#     {"metric":"MAE"},
#     subFolderName="Q1/",
#     figs = [figs[-1]],
#     saveAsPDF=True,
#     saveAsMD=True,
#     addNotes="Plotting the error as a function of joint angle and stiffness for the STEP/STEP trajectory (with the given params above)."
# )

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
        "Number of Epochs" : 10000,
        "Number of Trials" : 50,
    }

    ### Additional Arguments?
    parser = argparse.ArgumentParser(
        prog = "<filename>",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        build_NN_1DOF2DOA.py

        -----------------------------------------------------------------------------

        Build ANN for 1 DOF, 2 DOA tendon-driven system with nonlinear tendon
        elasticity in order to predict joint angle from different "sensory"
        states (like tendon tension or motor angle).

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
        default='MAE',
        help="Metrics to be compared. Should be either MAE, RMSE, or STD. Default is MAE."
    )
    parser.add_argument(
        '--consol',
        action="store_true",
        help='Consolidate all trials and generate comparison plots. Default is false.'
    )
    parser.add_argument(
        '--consolALL',
        action="store_true",
        help='Consolidate all trials and generate comparison plots. Default is false.'
    )
    args = parser.parse_args()
    if type(args.metrics)==str:
        metrics = [args.metrics]
    else:
        metrics = args.metrics
    for metric in metrics:
        assert metric in ["RMSE","MAE","STD"], "Invalid metric! Must be either 'RMSE', 'MAE', or 'STD'"

    if args.consol==True:
        plot_consolidated_data_babbling_duration_experiment(args.dur,metrics=metrics,includePSD=False)
    else:
        if args.consolALL==True:
            pathName = (
                'experimental_trials/'
            )
            folderName = (
                'All_Consolidated_Trials_'
                + '{:03d}'.format(int(args.trials))
                + '/'
            )
            plot_babbling_duration_vs_average_performance('RMSE')
            save_figures(
                pathName,
                babblingParams["Babbling Type"],
                {"Number of Trials":args.trials},
                subFolderName=folderName,
                saveAsPDF=True,
                saveAsMD=True,
                addNotes="Consolidated plot of babbling duration versus average performance ('RMSE' as performance measure)."
            )
            plt.close('all')
        else:
            angleQuadrants = [1,2,3,4]
            angleRanges = [
                [np.pi/2,3*np.pi/2], # Quadrant 1
                [np.pi/2,np.pi], # Quadrant 2
                [np.pi,3*np.pi/2], # Quadrant 3
                [3*np.pi/4,5*np.pi/4] # Quadrant 4
            ]
            frequencies = [1,2,3,4] # TEMP: Removed 5 Hz (need to add 0.5 Hz in the future)...
            stiffnessRange=[20,50]
            guesses = [
                [0,0],
                [6,6],
                [0,0],
                [0,0]
            ] # for 20 as the LB of stiffness
            delay = 0.3

            # allFileNames_SIN_SIN = list(map(
            #     lambda el: 'angleSin_stiffSin_Q%d_%02dHz_outputData.mat' % (el),
            #     itertools.chain(
            #         itertools.product(
            #             angleQuadrants,
            #             frequencies
            #         )
            #     )
            # ))
            #
            # allFileNames_SIN_STEP = list(map(
            #     lambda el: 'angleSin_stiffStep_Q%d_%02dHz_outputData.mat' % (el),
            #     itertools.chain(
            #         itertools.product(
            #             angleQuadrants,
            #             frequencies
            #         )
            #     )
            # ))
            #
            # allFileNames_STEP_SIN = [
            #     f'angleStep_stiffSin_{el:02d}Hz_outputData.mat' for el in frequencies
            # ]
            #
            # allFileNames_STEP_STEP = [
            #     'angleStep_stiffStep_outputData.mat'
            # ]
            # allFileNames = (
            #     allFileNames_SIN_SIN
            #     + allFileNames_SIN_STEP
            #     + allFileNames_STEP_SIN
            #     + allFileNames_STEP_STEP
            # )
            allFileNames = [
                "angleSin_stiffSin_outputData.mat",
                "angleSin_stiffStep_outputData.mat",
                "angleStep_stiffSin_outputData.mat",
                "angleStep_stiffStep_outputData.mat"
            ]
            allTrajectoriesGenerated = False

            numberOfTrajectories = len(allFileNames)

            startTime = time.time()
            trialStartTime = startTime
            for i in range(args.trials):
                plantParams["dt"] = args.dt
                ANNParams["Number of Epochs"] = args.epochs
                ANNParams["Number of Nodes"] = args.nodes

                ### Generate plant
                plant = plant_pendulum_1DOF2DOF(plantParams)

                ### Generate Sample Trajectories
                count = 0
                trajectoryCount = 0
                while allTrajectoriesGenerated==False:
                    basePath = "experimental_trials/"

                    ### Angle Step, Stiffness Step
                    print("Angle Step / Stiffness Step\n")
                    filePath = f"{basePath}angleStep_stiffStep_outputData.mat"
                    if path.exists(filePath):
                        print("ALREADY COMPLETED!!! (DELETE TO RUN AGAIN...)")
                    else:
                        stepDuration=1.0
                        numberOfSteps=200
                        trajectory = plant.generate_desired_trajectory_STEP_STEP(
                            stepDuration=stepDuration,
                            numberOfSteps=numberOfSteps, # NOTE: increased from 100 to 300
                            delay=delay,
                            angleRange=None,
                            stiffnessRange=stiffnessRange
                        )
                        x1d = trajectory[0,:]
                        sd = trajectory[1,:]
                        X_o = plant.return_X_o_given_s_o(x1d[0],sd[0],[0,0])
                        try:
                            X,U = generate_and_save_sensory_data(
                                plant,x1d,sd,X_o,
                                savePath=filePath,
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
                            save_figures(
                                "visualizations/",
                                "C4_diff_reference_trajectories",
                                {"Extra Steps" : 5*stepDuration,"stepDuration" : stepDuration, "numberOfSteps" : numberOfSteps, "delay" : delay, "angleRange" : None, "stiffnessRange" : stiffnessRange},
                                subFolderName="Generalization_Trajectories/angleStep_stiffStep/",
                                saveAsMD=True,
                                addNotes="Changed the reference trajectory to be C4 differentiable to see if it removes the transients in the control by removing the discontinuous higher derivatives in the reference trajectory (specifically the joint angle) as this will affect the feedback linearization inputs.\nAdditionally, the filter lengths were set to simulate a 3 Hz cutoff (some of these were previously calculated with a 5 Hz cutoff."
                            )
                            plt.close('all')
                            trajectoryCount+=1
                            try:
                                HAL.slack_post_message_update(
                                    trajectoryCount/numberOfTrajectories,
                                    time.time()-startTime
                                )
                            except:
                                pass
                        except:
                            pass

                    ### Angle Step, Stiffness Sinusoid
                    print("Angle Step / Stiffness Sinusoid\n")
                    for frequency in [1]: #frequencies:
                        print("All Angles, %d Hz Sinusoids\n" % frequency)
                        # filePath = (
                        #     "%sangleStep_stiffSin_%02dHz_outputData.mat"
                        #     % (basePath,frequency)
                        # )
                        filePath = (
                            "%sangleStep_stiffSin_outputData.mat"
                            % (basePath)
                        )
                        if path.exists(filePath):
                            print("ALREADY COMPLETED!!! (DELETE TO RUN AGAIN...)")
                        else:
                            trajectory = plant.generate_desired_trajectory_STEP_SIN(
                                frequency=frequency,
                                numberOfSteps=100,
                                stiffnessRange=stiffnessRange,
                                angleRange=None,
                                delay=delay
                            )
                            x1d = trajectory[0,:]
                            sd = trajectory[1,:]
                            X_o = plant.return_X_o_given_s_o(x1d[0],sd[0],[0,0])
                            try:
                                generate_and_save_sensory_data(
                                    plant,x1d,sd,X_o,
                                    savePath=filePath,
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
                                save_figures(
                                    "visualizations/",
                                    "C4_diff_reference_trajectories",
                                    {"Extra Steps":5,"Step Duration" : 2/frequency, "frequency" : frequency,"numberOfSteps" : 100,"stiffnessRange" : stiffnessRange,"angleRange" : None,"delay" : delay},
                                    subFolderName="Generalization_Trajectories/angleStep_stiffSin/",
                                    saveAsMD=True,
                                    addNotes="Changed the reference trajectory to be C4 differentiable to see if it removes the transients in the control by removing the discontinuous higher derivatives in the reference trajectory (specifically the joint angle) as this will affect the feedback linearization inputs.\nAdditionally, the filter lengths were set to simulate a 3 Hz cutoff (some of these were previously calculated with a 5 Hz cutoff."
                                )
                                plt.close('all')
                                trajectoryCount+=1
                                try:
                                    HAL.slack_post_message_update(
                                        trajectoryCount/numberOfTrajectories,
                                        time.time()-startTime
                                    )
                                except:
                                    pass
                            except:
                                pass

                    ### Angle Sinusoid, Stiffness Step
                    print("Angle Sinusoid / Stiffness Step\n")
                    for combo in list(
                            itertools.chain(itertools.product(
                                [angleQuadrants[3]],
                                [frequencies[0]]
                            ))):
                        # filePath = (
                        #     basePath
                        #     + "angleSin_stiffStep_Q%d_%02dHz_outputData.mat" %
                        #     combo
                        # )
                        filePath = (
                            basePath
                            + "angleSin_stiffStep_outputData.mat"
                        )
                        print("Angle Quadrant %d, %d Hz Sinusoid\n" % combo)
                        if path.exists(filePath):
                            print("ALREADY COMPLETED!!! (DELETE TO RUN AGAIN...)")
                        else:
                            trajectory = plant.generate_desired_trajectory_SIN_STEP(
                                angleRange=angleRanges[combo[0]-1],
                                frequency=combo[1],
                                stiffnessRange=stiffnessRange,
                                numberOfSteps=100,
                                delay=delay
                            )
                            x1d = trajectory[0,:]
                            sd = trajectory[1,:]
                            X_o = plant.return_X_o_given_s_o(
                                x1d[0],sd[0],guesses[combo[0]-1]
                            )
                            try:
                                X,U = generate_and_save_sensory_data(
                                    plant,x1d,sd,X_o,
                                    savePath=filePath,
                                    returnOutput=True,
                                    trim=delay+3*(2/combo[1])
                                )
                                actualTrajectory = np.array([
                                    X[0,:],
                                    np.array(list(map(plant.hs,X.T)))
                                ])
                                plant.plot_desired_trajectory_distribution_and_power_spectrum(
                                    actualTrajectory,
                                    cutoff=delay+3*(2/combo[1])
                                )
                                TEMPfig1 = plant.plot_states_and_inputs(
                                    X,U,
                                    inputString="Angle Sinusoidal / Stiffness Step",
                                    returnFig=True
                                )
                                TEMPfig2 = plant.plot_states_and_inputs(
                                    X[:,:int(3*(2/combo[1])/plant.dt)],
                                    U[:,:int(3*(2/combo[1])/plant.dt)-1],
                                    inputString="Angle Sinusoidal / Stiffness Step",
                                    returnFig=True
                                )
                                save_figures(
                                    "visualizations/",
                                    "C4_diff_reference_trajectories",
                                    {"Extra Steps" : 3, "Step Duration" : 2/combo[1], "angleRange" : angleRanges[combo[0]-1],"frequency" : combo[1],"stiffnessRange" : stiffnessRange,"numberOfSteps" : 100,"delay" : delay},
                                    subFolderName="Generalization_Trajectories/angleSin_stiffStep/",
                                    saveAsMD=True,
                                    addNotes="Changed the reference trajectory to be C4 differentiable to see if it removes the transients in the control by removing the discontinuous higher derivatives in the reference trajectory (specifically the joint angle) as this will affect the feedback linearization inputs.\nAdditionally, the filter lengths were set to simulate a 3 Hz cutoff (some of these were previously calculated with a 5 Hz cutoff."
                                )
                                plt.close('all')
                                trajectoryCount+=1
                                try:
                                    HAL.slack_post_message_update(
                                        trajectoryCount/numberOfTrajectories,
                                        time.time()-startTime
                                    )
                                except:
                                    pass
                            except:
                                pass

                    ### Angle Sinusoid, Stiffness Sinusoid
                    print("Angle Sinusoid / Stiffness Sinusoid\n")
                    for combo in list(
                            itertools.chain(itertools.product(
                                [angleQuadrants[3]],
                                [frequencies[0]]
                            ))):
                        filePath = (
                            basePath
                            + "angleSin_stiffSin_outputData.mat"
                        )
                        print("Angle Quadrant %d, %d Hz Sinusoid\n" % combo)
                        if path.exists(filePath):
                            print("ALREADY COMPLETED!!! (DELETE TO RUN AGAIN...)")
                        else:
                            trajectory = plant.generate_desired_trajectory_SIN_SIN(
                                stiffnessRange=stiffnessRange,
                                frequency=combo[1],
                                angleRange=angleRanges[combo[0]-1],
                                delay=delay
                            )
                            x1d = trajectory[0,:]
                            sd = trajectory[1,:]
                            X_o = plant.return_X_o_given_s_o(
                                x1d[0],sd[0],guesses[combo[0]-1]
                            )
                            try:
                                X,U = generate_and_save_sensory_data(
                                    plant,x1d,sd,X_o,
                                    savePath=filePath,
                                    returnOutput=True,
                                    trim=delay+(10-3)/combo[1] # only want the last three periods
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
                                save_figures(
                                    "visualizations/",
                                    "lower_stiffness",
                                    {"stiffnessRange" : stiffnessRange, "frequency" : combo[1], "angleRange" : angleRanges[combo[0]-1], "delay" : delay},
                                    subFolderName="Generalization_Trajectories/angleSin_stiffSin/",
                                    saveAsMD=True,
                                    addNotes="Rerunning with smaller stiffness range."
                                )
                                plt.close('all')
                                trajectoryCount+=1
                                try:
                                    HAL.slack_post_message_update(
                                        trajectoryCount/numberOfTrajectories,
                                        time.time()-startTime
                                    )
                                except:
                                    pass
                            except:
                                pass

                    if np.all([
                        path.exists(basePath+fileName)
                        for fileName in allFileNames
                    ]):
                        allTrajectoriesGenerated = True
                    else:
                        count+=1
                        assert count<10000, "Too many unsuccessful trials, please check code and run again."

                ### Generate babbling data and SAVE ALL FIGURES AND DATA IN SPECIFIC FOLDER
                print("Running Trial " + str(i+1) + "/" + str(args.trials))
                plantParams["Simulation Duration"] = args.dur # returned to original value.

                ANN = neural_network(ANNParams,babblingParams,plantParams)
                experimentalData = ANN.run_experimental_trial()

                # SAVE EXPERIMENTAL DATA TO TRIAL FOLDER
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

                ### Plot experimental data
                figs = plot_experimental_data(experimentalData,plantParams["dt"],returnFigs=True)

                for metric in metrics:
                    ### bar plots
                    fig = plot_bar_plots(experimentalData,metric=metric,yscale='log')

                    ### radial average error versus positions
                    figs = [fig]
                    newFigs=plot_all_polar_bar_plots(experimentalData,metric)
                    [figs.append(fig) for fig in newFigs]

                    ### save figs
                    save_figures(
                        ANN.trialPath,
                        babblingParams["Babbling Type"],
                        {"metric":metric},
                        figs=figs,
                        subFolderName=metric+"/",
                        saveAsMD=True
                    )
                    plt.close('all')
                print('\a')
                runTimeRaw = time.time()-startTime
                seconds = runTimeRaw % (24 * 3600)
                hour = seconds // 3600
                seconds %= 3600
                minutes = seconds // 60
                seconds %= 60
                runTime = "%d:%02d:%02d" % (hour, minutes, seconds)
                trialRunTime = time.time()-trialStartTime
                seconds = trialRunTime % (24 * 3600)
                hour = seconds // 3600
                seconds %= 3600
                minutes = seconds // 60
                seconds %= 60
                trialRunTime = "(+%d:%02d:%02d)" % (hour, minutes, seconds)
                trialStartTime = time.time()
                print('Run Time: ' + runTime + " " + trialRunTime + "\n")

                # if path.exists("slack_functions.py") and (i!=args.trials-1):
                #     HAL.slack_post_message_update((i+1)/args.trials,runTimeRaw)

            if path.exists("slack_functions.py"):
                message = (
                    '\n'
                    + '_Test Trial Finished!!!_ \n\n'
                    + 'Total Run Time: ' + runTime + '\n\n'
                    + '```params = {\n'
                    + '\t"Number of Trials" : ' + str(args.trials) + ',\n'
                    + '\t"Babbling Duration" : ' + str(args.dur) + ', # in seconds\n'
                    + '\t"Babbling Type" : "continuous"\n'
                    + '}```'
                )
                HAL.slack_post_message_code_completed(message)
