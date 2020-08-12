'''
NOTE: That joint stiffness is given by the square of the joint's moment arm times the two spring coefficients. Therefore, if we wish to keep the lower bounds of the stiffness the same, we need to respect the relationship:

plant.rj**2*plant.b_spr*plant.k_spr = CONSTANT = 5

'''

tendonStiffnessParams = {"Spring Shape Coefficient" : 1000}
tendonStiffnessParams["Spring Stiffness Coefficient"] = (
    2000
    / tendonStiffnessParams["Spring Shape Coefficient"]
)

if __name__=="__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from plant import *

    tendonDeformation = np.linspace(-0.002,0.02,1001)
    fig = plt.figure(figsize=(10,8))
    fig.subplots_adjust(bottom=0.2,top=0.9)
    ax = plt.gca()
    labels=[]
    lines=[]

    plantParams["Spring Stiffness Coefficient"] = \
        tendonStiffnessParams["Spring Stiffness Coefficient"]
    plantParams["Spring Shape Coefficient"] = \
        tendonStiffnessParams["Spring Shape Coefficient"]

    plant = plant_pendulum_1DOF2DOF(plantParams)
    scaledTendonDeformation = tendonDeformation/plant.rm
    tendonForce = np.array(list(map(
        lambda x: plant.tendon_1_FL_func([0,0,x,0,0,0]),
        scaledTendonDeformation
    ))) # NOTE: by setting x1 = 0 and x3 as the scaled tendon deformation, we create a new function that is a function of tendon deformation only.
    UBidx = int(sum(tendonForce<=400))
    label = (
        r"$k_{sp}$ = "
        + '{:0.2f}'.format(tendonStiffnessParams["Spring Stiffness Coefficient"])
        + "\n"
        + r"$b_{sp}$ = "
        + '{:0.2f}'.format(tendonStiffnessParams["Spring Shape Coefficient"])
    )
    labels.append(
        Line2D([0],[0],color="C0",label=label,lw=0,marker='|',linestyle=None,markersize=40, markeredgewidth=15)
    )
    line,=ax.plot(
        tendonDeformation*100,
        tendonForce,
        c="C0",
        label=label
    )
    lines.append(line)
    ax.legend(handles=labels,bbox_to_anchor=(0, -0.35, 1, .162), ncol=3, mode="expand", loc=3, borderaxespad=3,fontsize=14)
    # ax.legend(["this is a \n test","test","test 2"],loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("High Stiffness Tendon Force Length Curve", fontsize=24, y=1.05)
    ax.set_xlabel("Tendon Deformation (cm)", fontsize=14)
    ax.set_ylabel("Tendon Force (N)", fontsize=14)
    ax.set_ylim([0,400])
    ax.set_xlim([-0.002*100,tendonDeformation[UBidx]*100])
    save_figures(
        "experimental_trials/",
        "tendon_stiffness_sweep",
        tendonStiffnessParams,
        subFolderName="High_Stiffness_Experiment/",
        saveAsMD=True,
        saveAsPDF=True,
        addNotes="Extremely high value of tendon stiffness."
    )
    plt.close('all')
