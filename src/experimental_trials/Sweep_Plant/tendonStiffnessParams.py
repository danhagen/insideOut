'''
NOTE: That joint stiffness is given by the square of the joint's moment arm times the two spring coefficients. Therefore, if we wish to keep the lower bounds of the stiffness the same, we need to respect the relationship:

plant.rj**2*plant.b_spr*plant.k_spr = CONSTANT = 5

'''

tendonStiffnessParams = {
    "1" : {
        "Spring Stiffness Coefficient" : 2000/20, # N
        "Spring Shape Coefficient" : 20, # unit-less
    },
    "2" : {
        "Spring Stiffness Coefficient" : 2000/60, # N
        "Spring Shape Coefficient" : 60, # unit-less
    },
    "3" : {
        "Spring Stiffness Coefficient" : 2000/125, # N
        "Spring Shape Coefficient" : 125, # unit-less
    }
}

if __name__=="__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from plant import *

    tendonDeformation = np.linspace(-0.02,0.10,1001)
    fig = plt.figure(figsize=(10,8))
    fig.subplots_adjust(bottom=0.2,top=0.9)
    ax = plt.gca()
    labels=[]
    lines=[]
    for i in range(3):
        plantParams["Spring Stiffness Coefficient"] = \
            tendonStiffnessParams[str(i+1)]["Spring Stiffness Coefficient"]
        plantParams["Spring Shape Coefficient"] = \
            tendonStiffnessParams[str(i+1)]["Spring Shape Coefficient"]

        plant = plant_pendulum_1DOF2DOF(plantParams)
        scaledTendonDeformation = tendonDeformation/plant.rm
        tendonForce = np.array(list(map(
            lambda x: plant.tendon_1_FL_func([0,0,x,0,0,0]),
            scaledTendonDeformation
        ))) # NOTE: by setting x1 = 0 and x3 as the scaled tendon deformation, we create a new function that is a function of tendon deformation only.
        label = (
            r"$k_{sp}$ = "
            + '{:0.2f}'.format(tendonStiffnessParams[str(i+1)]["Spring Stiffness Coefficient"])
            + "\n"
            + r"$b_{sp}$ = "
            + '{:0.2f}'.format(tendonStiffnessParams[str(i+1)]["Spring Shape Coefficient"])
        )
        labels.append(
            Line2D([0],[0],color="C"+str(i),label=label,lw=0,marker='|',linestyle=None,markersize=40, markeredgewidth=15)
        )
        line,=ax.plot(
            tendonDeformation*100,
            tendonForce,
            c="C"+str(i),
            label=label
        )
        lines.append(line)
    ax.legend(handles=labels,bbox_to_anchor=(0, -0.35, 1, .162), ncol=3, mode="expand", loc=3, borderaxespad=3,fontsize=14)
    # ax.legend(["this is a \n test","test","test 2"],loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Changes in Tendon Force Length Properties", fontsize=24, y=1.05)
    ax.set_xlabel("Tendon Deformation (cm)", fontsize=14)
    ax.set_ylabel("Tendon Force (N)", fontsize=14)
    ax.set_ylim([0,400])
    ax.text(
        0.3,0.8,
        "Note that the value of " + r"$r_j^2 k_{sp} b_{sp}$" + "\n was conserved to ensure consistent\njoint stiffness lower bounds.",
        fontsize=14,
        linespacing=2,
        transform=ax.transAxes,
        wrap=True,
        horizontalalignment='center',
        verticalalignment='center',
        color = "k",
        bbox=dict(
            boxstyle='round',
            edgecolor='k',
            facecolor='w'
        )
    )
    save_figures(
        "experimental_trials/",
        "tendon_stiffness_sweep",
        tendonStiffnessParams,
        subFolderName="Sweep_Plant/",
        saveAsMD=True,
        saveAsPDF=True,
        addNotes="Different values for the tendon force length curve. It is important to note that the lower bounds of the joint stiffness have been kept constant across these variations in plants to avoid problems with comparing similar reference trajectories. The default setting corresponds to the least step curve."
    )
    plt.close('all')
