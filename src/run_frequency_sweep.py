from test_NN_1DOF2DOA import *
from matplotlib.patches import Patch
import matplotlib as mpl
from danpy.useful_functions import timer

movementTypes = ["angleSin_stiffSin","angleSin_stiffStep"]

def plot_frequency_sweep_bar_plots(
        experimentalData,
        metric="MAE",
        returnFig=True,
        includeYErrs=False
    ):
    prettyGroupNames = [
        "All\n Available\n States",
        "The\n Bio-Inspired\n Set",
        "Motor Position\n and\n Velocity Only",
        "All\n Motor\n States"
    ]

    frequencies = [0.5,1,2,4]
    freqStrings = [(f'f{el:0.1f}Hz').replace('.','_') for el in frequencies]

    labels = [
        "Sinusoidal Angle\nSinusoidal Stiffness",
        "Sinusoidal Angle\nStep Stiffness",
    ]
    movementTypes = ["angleSin_stiffSin","angleSin_stiffStep"]

    assert type(metric)==str and metric in ["MAE","STD","RMSE"], "metric must be either 'MAE' (default), 'RMSE' or 'STD'."
    if metric == "MAE":
        baseTitle = "Bar Plots of MAE by Movement Type at Different Frequencies"
        ylabel = "Mean Absolute Error (deg.)"
        valueKey = "experimentMAE"
    elif metric == 'STD':
        baseTitle = "Bar Plots of Error Std Dev by Movement Type at Different Frequencies"
        ylabel = "Error Standard Deviation (deg.)"
        valueKey = "experimentSTD"
    elif metric == 'RMSE':
        baseTitle = "Bar Plots of RMSE by Movement Type at Different Frequencies"
        ylabel = "Root Mean Squared Error (deg.)"
        valueKey = "experimentRMSE"

    allValue_SIN_SIN = [
        (180/np.pi)*experimentalData["angleSin_stiffSin"]["all"][frequency][valueKey]
        for frequency in freqStrings
    ] # in deg.

    allValue_SIN_STEP = [
        (180/np.pi)*experimentalData["angleSin_stiffStep"]["all"][frequency][valueKey]
        for frequency in freqStrings
    ] # in deg.

    bioValue_SIN_SIN = [
        (180/np.pi)*experimentalData["angleSin_stiffSin"]["bio"][frequency][valueKey]
        for frequency in freqStrings
    ] # in deg.

    bioValue_SIN_STEP = [
        (180/np.pi)*experimentalData["angleSin_stiffStep"]["bio"][frequency][valueKey]
        for frequency in freqStrings
    ] # in deg.

    kinapproxValue_SIN_SIN = [
        (180/np.pi)*experimentalData["angleSin_stiffSin"]["kinapprox"][frequency][valueKey]
        for frequency in freqStrings
    ] # in deg.

    kinapproxValue_SIN_STEP = [
        (180/np.pi)*experimentalData["angleSin_stiffStep"]["kinapprox"][frequency][valueKey]
        for frequency in freqStrings
    ] # in deg.

    allmotorValue_SIN_SIN = [
        (180/np.pi)*experimentalData["angleSin_stiffSin"]["allmotor"][frequency][valueKey]
        for frequency in freqStrings
    ] # in deg.

    allmotorValue_SIN_STEP = [
        (180/np.pi)*experimentalData["angleSin_stiffStep"]["allmotor"][frequency][valueKey]
        for frequency in freqStrings
    ] # in deg.

    xticks = np.arange(4)  # the label locations
    width = 0.45  # the width of the bars

    fig, ax = plt.subplots(figsize=(14,8))
    fig.subplots_adjust(bottom=0.2,top=0.9)
    rects1a = ax.bar(
        xticks - width/2, allValue_SIN_SIN, width,
        label="all", color=colors[0]
    )
    rects1b = ax.bar(
        xticks + width/2, allValue_SIN_STEP, width,
        label="all", hatch='/',edgecolor=colors[0],facecolor='w',linewidth=2
    )

    xticks += 5
    rects2a = ax.bar(
        xticks - width/2, bioValue_SIN_SIN, width,
        label="bio", color=colors[1]
    )
    rects2b = ax.bar(
        xticks + width/2, bioValue_SIN_STEP, width,
        label="bio", hatch='/',edgecolor=colors[1],facecolor='w',linewidth=2
    )

    xticks += 5
    rects3a = ax.bar(
        xticks - width/2, kinapproxValue_SIN_SIN, width,
        label="kinapprox", color=colors[2]
    )
    rects3b = ax.bar(
        xticks + width/2, kinapproxValue_SIN_STEP, width,
        label="kinapprox", hatch='/',edgecolor=colors[2],facecolor='w',linewidth=2
    )

    xticks += 5
    rects4a = ax.bar(
        xticks - width/2, allmotorValue_SIN_SIN, width,
        label="allmotor", color=colors[3]
    )
    rects4b = ax.bar(
        xticks + width/2, allmotorValue_SIN_STEP, width,
        label="allmotor", hatch='/',edgecolor=colors[3],facecolor='w',linewidth=2
    )


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel,fontsize=14)
    ax.set_yscale('log')
    ax.set_title(baseTitle,fontsize=24,y=1.05)
    xticks = list(range(xticks[-1]+1))
    xticks.remove(4)
    xticks.remove(9)
    xticks.remove(14)
    ax.set_xticks(xticks)
    # ax.set_xticklabels(["0.5 Hz","1 Hz", "2 Hz", "4 Hz"] + [""]*12)
    ax.set_xticklabels(["0.5 Hz","1 Hz", "2 Hz", "4 Hz"]*4)
    ax.text(
        (1.5-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0]),
        -0.15,
        prettyGroupNames[0],
        transform=ax.transAxes,
        color=colors[0],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16,
        bbox=dict(
            boxstyle='round',
            facecolor='w',
            edgecolor=colors[0]
        )
    )
    ax.text(
        (6.5-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0]),
        -0.15,
        prettyGroupNames[1],
        transform=ax.transAxes,
        color=colors[1],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16,
        bbox=dict(
            boxstyle='round',
            facecolor='w',
            edgecolor=colors[1]
        )
    )
    ax.text(
        (11.5-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0]),
        -0.15,
        prettyGroupNames[2],
        transform=ax.transAxes,
        color=colors[2],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16,
        bbox=dict(
            boxstyle='round',
            facecolor='w',
            edgecolor=colors[2]
        )
    )
    ax.text(
        (16.5-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0]),
        -0.15,
        prettyGroupNames[3],
        transform=ax.transAxes,
        color=colors[3],
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16,
        bbox=dict(
            boxstyle='round',
            facecolor='w',
            edgecolor=colors[3]
        )
    )
    labels = [
        Patch(facecolor='k',edgecolor='k',label="Angle Sinusoid\nStiffness Sinusoid"),
        Patch(facecolor='w',edgecolor='k',hatch="/",label="Angle Sinusoid\nStiffness Step"),
    ]
    leg = ax.legend(handles=labels,bbox_to_anchor=(0.05, 0.75, 0.4, .3),loc=3,fontsize=14,handlelength=3)
    for patch in leg.get_patches():
        patch.set_height(20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if returnFig==True:
        return(fig)
    else:
        plt.show()

def plot_polar_bar_plots_frequency_sweep(
        radial_bins,
        metric="MAE",
        addTitle=None,
        returnFig=False
    ):

    frequencies = [0.5,1,2,4]
    freqStrings = [(f'f{el:0.1f}Hz').replace('.','_') for el in frequencies]

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
    plt.suptitle(title,fontsize=24)
    ax1=plt.subplot(221)
    ax2=plt.subplot(222)
    ax3=plt.subplot(223)
    ax3.set_xlabel(xLabel,ha="center")
    ax3.xaxis.set_label_coords(0.8, -0.1)
    ax4=plt.subplot(224)
    axs=[ax1,ax2,ax3,ax4]

    slices = radial_bins['bins']
    thetaRays = (180/np.pi)*np.arange(0,np.pi+1e-3,np.pi/slices) # in degrees
    sectorWidth = (180/np.pi)*(np.pi/slices)/5 # in degrees
    thetaRays_SplitInFourths = [] # in degrees
    for j in range(len(thetaRays)-1):
        midAngle = (thetaRays[j+1]+thetaRays[j])/2 # in degrees
        thetaRays_SplitInFourths.append(
            [(midAngle + i*sectorWidth) for i in [1,0,-1,-2]]
        ) # in degrees
    thetaRays_SplitInFourths = np.concatenate(thetaRays_SplitInFourths)

    for i in range(4):
        for j in range(len(thetaRays)-1):
            count=0
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
            for k in range(len(frequencies)):
                if bin_name in radial_bins[groupNames[i]][freqStrings[k]]:
                    axs[i].add_patch(
                        Wedge(
                            (0,0),
                            np.log10(radial_bins[groupNames[i]][freqStrings[k]][bin_name][metric])+offset,
                            thetaRays_SplitInFourths[4*j+k],
                            thetaRays_SplitInFourths[4*j+k]+sectorWidth,
                            facecolor=mpl.colors.to_rgba(
                                colors[i],
                                [1,0.8,0.6,0.4][k]
                            ),
                            edgecolor=(0,0,0,0)
                        )
                    )
                else:
                    count+=1
                    if count==len(groupNames):
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

def return_radial_bins_frequency_sweep(experimentalData,movement,bins=12,metrics=["MAE"]):
    assert type(metrics)==list, "metrics must be a list."
    assert movement in ["angleSin_stiffSin","angleSin_stiffStep"], 'movement must be either "angleSin_stiffSin" or "angleSin_stiffStep".'

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
    """
    radial_bins
        ..bins
        ..max<Metric>
        ..min<Metric>
        ..<Group Name>
            ..<Frequency>
                ..<Bin Name>
                    ..errors
                    ..abs errors
                    ..<Metric>
    """

    for metric in metrics:
        radial_bins["max"+metric]=0
        radial_bins["min"+metric]=100

    for group in groupNames:
        for frequency in [0.5,1,2,4]:
            freqString = (f'f{frequency:0.1f}Hz').replace(".","_")
            radial_bins[group][freqString]={}
            expectedJointAngle = (180/np.pi)*(
                experimentalData[movement][group][freqString]['expectedJointAngle'] - np.pi/2
            ) # in degrees
            rawError = (180/np.pi)*experimentalData[movement][group][freqString]['rawError'] # in degrees
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
                    radial_bins[group][freqString][bin_name] = {}
                    radial_bins[group][freqString][bin_name]["errors"] = rawError[indices] # in degrees
                    radial_bins[group][freqString][bin_name]["abs errors"] = abs(
                        radial_bins[group][freqString][bin_name]["errors"]
                    ) # in degrees

                    ### Mean absolute error
                    if "MAE" in metrics:
                        radial_bins[group][freqString][bin_name]["MAE"] = \
                            radial_bins[group][freqString][bin_name]["abs errors"].mean() # in degrees
                        radial_bins["minMAE"] = min([
                            radial_bins["minMAE"],
                            radial_bins[group][freqString][bin_name]["MAE"]
                        ]) # in degrees
                        radial_bins["maxMAE"] = max([
                            radial_bins["maxMAE"],
                            radial_bins[group][freqString][bin_name]["MAE"]
                        ]) # in degrees


                    ### Root mean squared error
                    if "RMSE" in metrics:
                        radial_bins[group][freqString][bin_name]["RMSE"] = np.sqrt(
                            (radial_bins[group][freqString][bin_name]["errors"]**2).mean()
                        ) # in degrees
                        radial_bins["minRMSE"] = min([
                            radial_bins["minRMSE"],
                            radial_bins[group][freqString][bin_name]["RMSE"]
                        ]) # in degrees
                        radial_bins["maxRMSE"] = max([
                            radial_bins["maxRMSE"],
                            radial_bins[group][freqString][bin_name]["RMSE"]
                        ]) # in degrees
                    # radial_bins[group][freqString][bin_name]["abs error std"] = \
                    #     radial_bins[group][freqString][bin_name]["abs errors"].std() # in degrees
                    # radial_bins[group][freqString][bin_name]["min abs error"] = \
                    #     radial_bins[group][freqString][bin_name]["errors"].min() # in degrees
                    # radial_bins[group][freqString][bin_name]["max abs error"] = \
                    #     radial_bins[group][freqString][bin_name]["errors"].max() # in degrees
                    #
                    # radial_bins[group][freqString][bin_name]["avg error"] = \
                    #     radial_bins[group][freqString][bin_name]["errors"].mean() # in degrees
                    if "STD" in metrics:
                        radial_bins[group][freqString][bin_name]["STD"] = \
                            radial_bins[group][freqString][bin_name]["errors"].std() # in degrees
                        radial_bins["minSTD"] = min([
                            radial_bins["minSTD"],
                            radial_bins[group][freqString][bin_name]["STD"]
                        ]) # in degrees
                        radial_bins["maxSTD"] = max([
                            radial_bins["maxSTD"],
                            radial_bins[group][freqString][bin_name]["STD"]
                        ]) # in degrees
                # else:
                #     radial_bins[group][freqString][bin_name]["errors"] = None
                #     radial_bins[group][freqString][bin_name]["abs errors"] = None
                #     radial_bins[group][freqString][bin_name]["MAE"] = None
                #     radial_bins[group][freqString][bin_name]["RMSE"] = None
                #     radial_bins[group][freqString][bin_name]["abs error std"] = None
                #     radial_bins[group][freqString][bin_name]["min abs error"] = None
                #     radial_bins[group][freqString][bin_name]["max abs error"] = None
                #     radial_bins[group][freqString][bin_name]["avg error"] = None
                #     radial_bins[group][freqString][bin_name]["STD"] = None
    return(radial_bins)

def plot_all_polar_bar_plots_frequency_sweep(
        experimentalData,
        metric,
        totalRadialBins=None,
        returnFigs=True
        ):
    movementTypes = ["angleSin_stiffSin","angleSin_stiffStep"]
    labels = [
        "(Sinusoidal Angle / Sinusoidal Stiffness)",
        "(Sinusoidal Angle / Step Stiffness)",
    ]
    assert type(metric)==str and metric in ["MAE","STD","RMSE"], "metric must be either 'MAE' (default), 'RMSE' or 'STD'."

    ### radial average error versus positions

    if totalRadialBins is None:
        figs = []
        for i in range(len(movementTypes)):
            movement = movementTypes[i]
            radial_bins = return_radial_bins_frequency_sweep(experimentalData,movement,bins=12,metrics=[metric])
            tempFig = plot_polar_bar_plots_frequency_sweep(
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
            tempFig = plot_polar_bar_plots_frequency_sweep(
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

def plot_metric_distributions_frequency_sweep(outputData,metric,returnFigs=True):
    assert metric in ["MAE","RMSE","STD"], "metric must be either 'MAE', 'RMSE', or 'STD'."
    prettyGroupNames = [
        "All\n Available\n States",
        "The\n Bio-Inspired\n Set",
        "Motor Position\n and\n Velocity Only",
        "All\n Motor\n States"
    ]
    movementTypes=["angleSin_stiffSin","angleSin_stiffStep"]

    """
        You should create a 3,2 plot with the first row being just the overlapping KSE plots (hist=False), then you should do a normed hist for each of the following subplots for each group.
    """
    figs = []
    numberOfTrials = len(
        outputData[movementTypes[0]][groupNames[0]][freqStrings[0]]['experiment'+metric+"_list"]
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
            for k in range(len(freqStrings)):
                data = np.array(
                    outputData[movementTypes[i]][groupNames[j]][freqStrings[k]]['experiment'+metric+"_list"]
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
                        'weights' : np.ones(len(data))/len(data),
                        'alpha' : [1,0.8,0.6,0.4][k]
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

def plot_consolidated_data_frequency_sweep(metrics=None,returnPath=False):
    assert type(returnPath)==bool,"returnPath must be either True or False (default)."

    frequencies = [0.5,1,2,4]
    freqStrings = [(f'f{el:0.1f}Hz').replace('.','_') for el in frequencies]

    if metrics is None:
        metrics = ["MAE"]
        metricKeys = ["experimentMAE"]
    else:
        assert type(metrics)==list, "metrics must be a list of strings."
        for metric in metrics:
            assert metric in ["RMSE","STD","MAE"], 'Invalid metric! metrics must include "RMSE","STD", or "MAE".'
        metricKeys = ["experiment"+metric for metric in metrics]

    # angle in radians
    # if jointAngleBounds is not None:
    #     assert type(jointAngleBounds)==list and len(jointAngleBounds)==2, "jointAngleBounds must be a list."
    #     assert jointAngleBounds[1]>jointAngleBounds[0], "jointAngleBounds must be in ascending order."
    # else:
    #     jointAngleBounds = [
    #         plantParams["Joint Angle Bounds"]["LB"],
    #         plantParams["Joint Angle Bounds"]["UB"]
    #     ]
    #
    # if jointStiffnessBounds is not None:
    #     assert type(jointStiffnessBounds)==list and len(jointStiffnessBounds)==2, "jointStiffnessBounds must be a list."
    #     assert jointStiffnessBounds[1]>jointStiffnessBounds[0], "jointStiffnessBounds must be in ascending order."
    # else:
    #     jointStiffnessBounds = [10,plantParams["Maximum Joint Stiffness"]]

    ### get the testing trial directories
    directory = Path("experimental_trials/Sweep_Frequency_More_Damped")
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

    plot_training_performance(
        totalTrainingData,
        10000,
        numberOfTrials
    )

    fig = plot_training_epoch_bar_plots(
        totalTrainingData,
        addTitle="(Frequency Sweep Experiment)",
        returnFig=True
    )

    saveParams = {"Number of Trials" : numberOfTrials, "Motor Damping":plantParams["Motor Damping"]}
    save_figures(
        str(directory)+"/",
        "perf_v_epoch",
        saveParams,
        subFolderName=folderName,
        saveAsPDF=True,
        saveAsMD=True,
        addNotes="### Generated from `run_frequency_sweep()`"
    )
    plt.close('all')

    # Experimental Data

    totalOutputData = {}
    totalRadialBins = {}
    # totalAverageErrorBins = {}

    # xbins = 20
    # ybins = 20
    # xbin_width = (jointAngleBounds[1]-jointAngleBounds[0])/xbins # in radians
    # xbin_edges = np.arange(
    #     jointAngleBounds[0]-2*xbin_width,
    #     jointAngleBounds[1]+2*xbin_width+1e-3,
    #     xbin_width
    # )# in radians
    # ybin_width = (jointStiffnessBounds[1]-jointStiffnessBounds[0])/ybins
    # ybin_edges = np.arange(
    #     jointStiffnessBounds[0]-2*ybin_width,
    #     jointStiffnessBounds[1]+2*ybin_width+1e-3,
    #     ybin_width
    # )
    # x_indices,y_indices = return_average_error_bin_indices(
    #     xbin_edges,
    #     ybin_edges,
    #     plant
    # )

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
                            ..<Frequency>
                                expectedJointAngle (in rad.)
                                predictedJointAngle (in rad.)
                                rawError (in rad.)
                                experimentRMSE (in rad.)
                                experimentMAE (in rad.)
                                experimentSTD (in rad.)
            """
            for movement in ["angleSin_stiffSin","angleSin_stiffStep"]:
                totalRadialBins[movement] = {
                    "bins" : radialBins,
                    "all" : {},
                    "bio" : {},
                    "kinapprox" : {},
                    "allmotor" : {}
                }
                tempRadialBins = return_radial_bins_frequency_sweep(
                    tempOutputData,
                    movement,
                    bins=radialBins,
                    metrics=metrics
                )
                """
                tempRadialBins
                    ..bins
                    ..max<Metric>
                    ..min<Metric>
                    ..<Group Name>
                        ..<Frequency>
                            ..<Bin Name>
                                ..errors
                                ..abs errors
                                ..<Metric>
                """
                for group in groupNames:
                    for frequency in freqStrings:
                        totalRadialBins[movement][group][frequency]={}
                        for key in metricKeys:
                            totalOutputData[movement][group][frequency][key+"_list"] = [
                                totalOutputData[movement][group][frequency][key]
                            ]
                            totalOutputData[movement][group][frequency][key] = (
                                totalOutputData[movement][group][frequency][key]
                                / numberOfTrials
                            )
                            if key!="experimentSTD":
                                totalRadialBins[movement]["max"+key[10:]] = 0
                                totalRadialBins[movement]["min"+key[10:]] = 100
                                for i in range(len(thetaRays)-1):
                                    bin_name = f"{thetaRays[i]:0.1f} to {thetaRays[i+1]:0.1f}"
                                    if bin_name in tempRadialBins[group][frequency]:
                                        if bin_name not in totalRadialBins[movement][group][frequency]:
                                            totalRadialBins[movement][group][frequency][bin_name] = {}
                                        totalRadialBins[movement][group][frequency][bin_name][key[10:]+"_list"] = [
                                            tempRadialBins[group][frequency][bin_name][key[10:]]
                                        ]

                        # if includePSD==True:
                        #     freq, PSD = signal.welch(
                        #         totalOutputData[movement][group][frequency]["rawError"],
                        #         1/plantParams['dt']
                        #     )
                        #     totalOutputData[movement][group][frequency]["frequencies"] = freq
                        #     totalOutputData[movement][group][frequency]["avg_PSD"] = PSD/numberOfTrials
                        for key in [
                                "rawError",
                                "expectedJointAngle",
                                "predictedJointAngle",
                                "experimentMAE",
                                "experimentRMSE",
                                "experimentSTD"
                            ]:
                            if key not in metricKeys:
                                del(totalOutputData[movement][group][frequency][key])
        else:
            for movement in movementTypes:
                tempRadialBins = return_radial_bins_frequency_sweep(
                    tempOutputData,
                    movement,
                    bins=radialBins,
                    metrics=metrics
                )
                for group in groupNames:
                    for frequency in freqStrings:
                        for key in metricKeys:
                            totalOutputData[movement][group][frequency][key+"_list"].append(
                                tempOutputData[movement][group][frequency][key]
                            )
                            totalOutputData[movement][group][frequency][key] += (
                                tempOutputData[movement][group][frequency][key]
                                / numberOfTrials
                            )
                            if key!="experimentSTD":
                                for i in range(len(thetaRays)-1):
                                    bin_name = f"{thetaRays[i]:0.1f} to {thetaRays[i+1]:0.1f}"
                                    if bin_name in tempRadialBins[group][frequency]:
                                        totalRadialBins[movement][group][frequency][bin_name][key[10:]+"_list"].append(
                                            tempRadialBins[group][frequency][bin_name][key[10:]]
                                        )
                        # if includePSD==True:
                        #     _, PSD = signal.welch(
                        #         tempOutputData[movement][group][frequency]["rawError"],
                        #         1/plantParams['dt']
                        #     )
                        #     totalOutputData[movement][group][frequency]["avg_PSD"] += PSD/numberOfTrials

        # delete trial directory
        shutil.rmtree(trialDirectories[n])
        """
            totalOutputData
                ..<Movement Type>
                    ..<Group Name>
                        ..<Frequency>
                            # [frequencies] (in Hz.)
                            # [avg_PSD] (in rad.^2/Hz.)
                            experiment<Metric> (in rad.)
                            experiment<Metric>_list (in rad.)
        """
    for movement in movementTypes:
        for group in groupNames:
            for frequency in freqStrings:
                for i in range(len(thetaRays)-1):
                    bin_name = f"{thetaRays[i]:0.1f} to {thetaRays[i+1]:0.1f}"
                    if bin_name in totalRadialBins[movement][group][frequency]:
                        if "MAE" in metrics:
                            totalRadialBins[movement][group][frequency][bin_name]["MAE"] = np.mean(totalRadialBins[movement][group][frequency][bin_name]["MAE_list"])
                            totalRadialBins[movement]["minMAE"] = min([
                                totalRadialBins[movement]["minMAE"],
                                totalRadialBins[movement][group][frequency][bin_name]["MAE"]
                            ])
                            totalRadialBins[movement]["maxMAE"] = max([
                                totalRadialBins[movement]["maxMAE"],
                                totalRadialBins[movement][group][frequency][bin_name]["MAE"]
                            ])
                        if "RMSE" in metrics:
                            totalRadialBins[movement][group][frequency][bin_name]["RMSE"] = np.sqrt(np.mean([
                                el**2
                                for el in totalRadialBins[movement][group][frequency][bin_name]["RMSE_list"]
                            ]))
                            totalRadialBins[movement]["minRMSE"] = min([
                                totalRadialBins[movement]["minRMSE"],
                                totalRadialBins[movement][group][frequency][bin_name]["RMSE"]
                            ])
                            totalRadialBins[movement]["maxRMSE"] = max([
                                totalRadialBins[movement]["maxRMSE"],
                                totalRadialBins[movement][group][frequency][bin_name]["RMSE"]
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

    # if includePSD==True:
    #     figs = plot_average_error_signal_power_spectrums(totalOutputData,returnFigs=True)
    #
    #     save_figures(
    #         str(directory)+"/",
    #         "err_PSD",
    #         {},
    #         figs=figs,
    #         subFolderName=folderName+"/",
    #         saveAsMD=True,
    #         addNotes="Average error signal with power spectrum analysis."
    #     )
    #     plt.close('all')

    for metric in metrics:
        fig = plot_frequency_sweep_bar_plots(
            totalOutputData,
            metric=metric,
            returnFig=True
        )
        figs = [fig]
        tempFigs1 = plot_metric_distributions_frequency_sweep(
            totalOutputData,
            metric,
            returnFigs=True
        )
        [figs.append(fig) for fig in tempFigs1]
        if metric!="STD":
            tempFigs2 = plot_all_polar_bar_plots_frequency_sweep(
                None,
                metric,
                totalRadialBins,
                returnFigs=True
            )
            [figs.append(fig) for fig in tempFigs2]

            # tempFigs3 = plot_2D_heatmap_of_error_wrt_desired_trajectory(
            #     None,
            #     metric,
            #     plant,
            #     totalAverageErrorBins,
            #     returnFigs=True,
            #     xbins=xbins+4,
            #     ybins=ybins+4,
            #     jointAngleBounds=[
            #         xbin_edges[0],
            #         xbin_edges[-1]
            #     ],
            #     jointStiffnessBounds=[
            #         ybin_edges[0],
            #         ybin_edges[-1]
            #     ]
            # )
            # [figs.append(fig) for fig in tempFigs3]

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
            for frequency in freqStrings:
                consolidatedOutputData[movement][group][frequency]={}
                for metric in metricKeys:
                    consolidatedOutputData[movement][group][frequency][metric] = \
                        totalOutputData[movement][group][frequency][metric]
                    consolidatedOutputData[movement][group][frequency][metric+"_list"] = \
                        totalOutputData[movement][group][frequency][metric+"_list"]
    fileName = (
        "consolidatedOutputData.pkl"
    )
    with open(directory/folderName/fileName, 'wb') as handle:
        pickle.dump(
            consolidatedOutputData,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    if returnPath==True:
        return(str(directory/folderName))

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

    # NOTE: These values were chosen because they showed consistently good performance. Additionally, we effectively removed the upper limit for epochs to allow for convergence. We know that these parameters are a good choice because we see small standard deviations in the performance across trials and we noticed that, while fewer nodes may produce similar results, the average number of epochs was consistent. Similarly, we could argue that 10 seconds of babbling would be sufficient to get the desired performance for 15 internal nodes (instead of the default 15 seconds). But we again see that the number of epochs is consistent (meaning we would not be adding much more computational time). Therefore, the choice of 15 seconds was made in order to (1) minimize the standard deviation of the performance and (2) to avoid "poor babbling" trials that happen with some regularity for trials under 10 seconds.

    ### ANN parameters
    ANNParams = {
        "Number of Nodes" : 15,
        "Number of Epochs" : 10000,
        "Number of Trials" : 50,
    }

    ### plant parameters
    plantParams["Simulation Duration"] = 15
    plantParams["Motor Damping"] = 0.00462

    # ### babbling parameters
    # babblingParams["Cocontraction Standard Deviation"] = 2

    ### Additional Arguments?
    parser = argparse.ArgumentParser(
        prog = "<filename>",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        run_frequency_sweep.py

        -----------------------------------------------------------------------------

        FREQUENCY SWEEEEEEEEEEEEEEEEEEEEPPPPPPPP!!!!!!!

        -----------------------------------------------------------------------------'''),
        epilog=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        Written by Daniel A. Hagen (2020/05/13)

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
    args = parser.parse_args()
    if type(args.metrics)==str:
        metrics = [args.metrics]
    else:
        metrics = args.metrics
    for metric in metrics:
        assert metric in ["RMSE","MAE","STD"], "Invalid metric! Must be either 'RMSE', 'MAE', or 'STD'"
    plantParams["Simulation Duration"] = int(args.dur)
    ANNParams["Number of Nodes"] = int(args.nodes)
    numberOfTrials = args.trials

    frequencies = [0.5,1,2,4]
    freqStrings = [(f'f{el:0.1f}Hz').replace('.','_') for el in frequencies]
    angleRange = [3*np.pi/4,5*np.pi/4] # Quadrant 4

    # Define new plant
    plant = plant_pendulum_1DOF2DOF(plantParams)

    trialTimer = timer()
    totalTimer = timer()
    for i in range(numberOfTrials):
        trialTimer.reset()
        print(f"Running Trial {str(i+1)}/{str(numberOfTrials)}")

         # returned to original value.

        ANN = neural_network(ANNParams,babblingParams,plantParams)
        experimentalData = ANN.run_frequency_sweep_trial(
            basePath="experimental_trials/Sweep_Frequency_More_Damped/",
            upsample=False
        ) # TEMP: basePath chosen for More Damped Experiment.
        """
            experimentalData
                ..<Group Name>
                    ..<Movement Type>
                        ..<Frequency>
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
                for frequency in freqStrings:
                    formattedData[movement][group][frequency] = {}
                    for key in experimentalData[group][movement][frequency]:
                        if type(experimentalData[group][movement][frequency][key])==float:
                            formattedData[movement][group][frequency][key] = \
                                experimentalData[group][movement][frequency][key]
                        else:
                            formattedData[movement][group][frequency][key] = np.array(
                                experimentalData[group][movement][frequency][key]._data
                            )
        experimentalData = formattedData
        """
            experimentalData
                ..<Movement Type>
                    ..<Group Name>
                        ..<Frequency>
                            ..expectedJointAngle (in rad.)
                            ..predictedJointAngle (in rad.)
                            ..rawError (in rad.)
                            ..experimentRMSE (in rad.)
                            ..experimentMAE (in rad.)
                            ..experimentSTD (in rad.)
        """
        with open(path.join(ANN.trialPath,'experimentalData.pkl'), 'wb') as handle:
            pickle.dump(experimentalData, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for metric in metrics:
            ### bar plots
            fig = plot_frequency_sweep_bar_plots(experimentalData,metric=metric)

            ### radial average error versus positions
            figs = [fig]
            # newFigs=plot_all_polar_bar_plots_frequency_sweep(experimentalData,metric)
            # [figs.append(fig) for fig in newFigs]

            ### save figs
            save_figures(
                ANN.trialPath,
                "perf_vs_frequency",
                {"metric":metric},
                figs=figs,
                subFolderName=metric+"/",
                saveAsMD=True,
                addNotes="### For a single trial.\n\n"
            )
            plt.close('all')
        print('\a')
        trialTimer.end_trial()

    print("Consolidating Data from " + str(numberOfTrials) + " Trials (Frequency Sweep)...")
    pathName = plot_consolidated_data_frequency_sweep(
        metrics=metrics,
        returnPath=True
    )

    totalTimer.end_trial(0)
    print(f'Run Time for {numberOfTrials} trials (Frequency Sweep): {totalTimer.trialRunTimeStr}')

    if path.exists("slack_functions.py"):
        message = (
            '\n'
            + '_Frequency Sweep Finished!!!_ \n\n'
            + 'Total Run Time: ' + totalTimer.totalRunTimeStr + '\n\n'
            + '```params = {\n'
            + '\t"Number of Trials" : ' + str(args.trials) + ',\n'
            + '\t"Number of Nodes" : ' + str(args.nodes) + ',\n'
            + '\t"Babbling Duration" : ' + str(args.dur) + ',\n'
            + '\t"Babbling Type" : "continuous"\n'
            + '}```'
        )
        HAL.add_report_to_github_io(Path(pathName+"/README.md"))
        HAL.slack_post_message_code_completed(message)
