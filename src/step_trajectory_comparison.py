from plant import *

functions = {
    "x3" : lambda X: X[2,:],
    "dx3" : lambda X: X[3,:],
    "d2x3" : lambda X: np.gradient(X[3,:],plant.dt),
    "x5" : lambda X: X[4,:],
    "dx5" : lambda X: X[5,:],
    "d2x5" : lambda X: np.gradient(X[5,:],plant.dt),
    "fT1" : lambda X: np.array(list(map(plant.tendon_1_FL_func,X.T))),
    "dfT1" : lambda X: np.gradient(np.array(list(map(plant.tendon_1_FL_func,X.T))),plant.dt),
    "d2fT1" : lambda X: np.gradient(np.gradient(np.array(list(map(plant.tendon_1_FL_func,X.T))),plant.dt),plant.dt),
    "fT2" : lambda X: np.array(list(map(plant.tendon_2_FL_func,X.T))),
    "dfT2" : lambda X: np.gradient(np.array(list(map(plant.tendon_2_FL_func,X.T))),plant.dt),
    "d2fT2" : lambda X: np.gradient(np.gradient(np.array(list(map(plant.tendon_2_FL_func,X.T))),plant.dt),plant.dt)
}

keys = [
    "x3","dx3","d2x3",
    "x5","dx5","d2x5",
    "fT1","dfT1","d2fT1",
    "fT2","dfT2","d2fT2"
]

def plot_comparison(tempTime,X_new,X_old,title):
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9),(ax10,ax11,ax12)) = \
        plt.subplots(4,3,figsize=(24,20),sharex=True)
    plt.suptitle(title,fontsize=20)
    axs = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]
    axs[0].text(
        -0.2,0.5,
        "Motor 1",
        fontsize=14,
        horizontalalignment='center',
        verticalalignment='center',
        color = "C0",
        transform=axs[0].transAxes,
        bbox=dict(
            boxstyle='round',
            edgecolor='C0',
            facecolor='w',
            lw=0,
            alpha=0.6
        ),
        rotation=90
    )
    axs[3].text(
        -0.2,0.5,
        "Motor 2",
        fontsize=14,
        horizontalalignment='center',
        verticalalignment='center',
        color = "C1",
        transform=axs[3].transAxes,
        bbox=dict(
            boxstyle='round',
            edgecolor='C1',
            facecolor='w',
            lw=0,
            alpha=0.6
        ),
        rotation=90
    )
    axs[6].text(
        -0.2,0.5,
        "Tendon\nTension 1",
        fontsize=14,
        wrap=True,
        horizontalalignment='center',
        verticalalignment='center',
        color = "C2",
        transform=axs[6].transAxes,
        bbox=dict(
            boxstyle='round',
            edgecolor='C2',
            facecolor='w',
            lw=0,
            alpha=0.6
        ),
        rotation=90
    )

    axs[9].text(
        -0.2,0.5,
        "Tendon\nTension 2",
        fontsize=14,
        wrap=True,
        horizontalalignment='center',
        verticalalignment='center',
        color = "C3",
        transform=axs[9].transAxes,
        bbox=dict(
            boxstyle='round',
            edgecolor='C3',
            facecolor='w',
            lw=0,
            alpha=0.6
        ),
        rotation=90
    )

    axs[0].text(
        0.5,1.1,
        "Position",
        fontsize=14,
        horizontalalignment='center',
        verticalalignment='center',
        color = "k",
        transform=axs[0].transAxes,
        bbox=dict(
            boxstyle='round',
            edgecolor='k',
            facecolor='w',
            lw=0,
            alpha=0.6
        )
    )
    axs[1].text(
        0.5,1.1,
        r"1$^{st}$ Derivative",
        fontsize=14,
        horizontalalignment='center',
        verticalalignment='center',
        color = "k",
        transform=axs[1].transAxes,
        bbox=dict(
            boxstyle='round',
            edgecolor='k',
            facecolor='w',
            lw=0,
            alpha=0.6
        )
    )
    axs[2].text(
        0.5,1.1,
        r"2$^{nd}$ Derivative",
        fontsize=14,
        horizontalalignment='center',
        verticalalignment='center',
        color = "k",
        transform=axs[2].transAxes,
        bbox=dict(
            boxstyle='round',
            edgecolor='k',
            facecolor='w',
            lw=0,
            alpha=0.6
        )
    )
    units = ["(deg)","(deg/s)",r"(deg/s$^2$)","(deg)","(deg/s)",r"(deg/s$^2$)","(N)","(N/s)",r"(N/s$^2$)","(N)","(N/s)",r"(N/s$^2$)"]

    for i in range(12):
        if i < 6:
            axs[i].plot(
                tempTime,
                (180/np.pi)*functions[keys[i]](X_old),
                c="0.70",
                lw=3
            )
            axs[i].plot(
                tempTime,
                (180/np.pi)*functions[keys[i]](X_new),
                c="C"+str(int(i/3))
            )
        else:
            axs[i].plot(
                tempTime,
                functions[keys[i]](X_old),
                c="0.70",
                lw=3
            )
            axs[i].plot(
                tempTime,
                functions[keys[i]](X_new),
                c="C"+str(int(i/3))
            )
        axs[i].text(
            0.02,0.95,
            units[i],
            fontsize=12,
            horizontalalignment='left',
            verticalalignment='center',
            color = "k",
            transform=axs[i].transAxes
        )
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        if (i+1)%3==0:
            axs[i].legend(["Old","New"],loc="upper right")

plant = plant_pendulum_1DOF2DOF(plantParams)

##############################################
################# STEP STEP ##################
##############################################

print("ANGLE STEP / STIFF STEP")

delay = 0.3
numberOfSteps = 100
stepDuration = 1

angleRange = [
    plant.jointAngleBounds["LB"],
    plant.jointAngleBounds["UB"]
]
stiffnessRange = [
    plant.jointStiffnessBounds["LB"],
    plant.jointStiffnessBounds["UB"]
]

stepLength = int(stepDuration/plant.dt)
additionalSteps = 5
tempTime = np.arange(
    0,
    delay+(numberOfSteps+additionalSteps)*stepDuration+1e-4,
    plant.dt
)

minValue = [
    angleRange[0],
    stiffnessRange[0]
]
maxValue = [
    angleRange[1],
    stiffnessRange[1]
]
startValue = np.array([[
    np.mean(angleRange),
    stiffnessRange[0]
]])

## Point-to-Point for both joint angle and stiffness
trajectory = startValue.T*np.ones((2,len(tempTime)))
OLDtrajectory = startValue.T*np.ones((2,len(tempTime)))
transitionLength = int((1/2)/plant.dt)
start=int(delay/plant.dt)-int(transitionLength/2)
start_OLD=int(delay/plant.dt)
Xi = startValue.T
coeffs = [126,-420,540,-315,70]
# NOTE: by making the entire step equal to 1 second, you can create piecewise functions that do not have to deal with inbetween steps (i.e., the filterDuration is not a multiple of step.dt). Therefore "start" will always be a multiple of plant.dt
for i in range(numberOfSteps+additionalSteps+1):
    Xf = np.array([[
        np.random.uniform(
            minValue[0],
            maxValue[0]
        ),
        np.random.uniform(
            minValue[1],
            maxValue[1]
        )
    ]]).T
    end = start+stepLength
    end_OLD=start_OLD+stepLength
    transition_func = lambda t: (
        Xi + (Xf-Xi)*(
            coeffs[0]*((t-plant.dt*start)/(plant.dt*transitionLength))**5
            + coeffs[1]*((t-plant.dt*start)/(plant.dt*transitionLength))**6
            + coeffs[2]*((t-plant.dt*start)/(plant.dt*transitionLength))**7
            + coeffs[3]*((t-plant.dt*start)/(plant.dt*transitionLength))**8
            + coeffs[4]*((t-plant.dt*start)/(plant.dt*transitionLength))**9
        )
    )
    trajectory[:,start:start+transitionLength] = np.array(list(map(
        transition_func,
        tempTime[start:start+transitionLength]
    )))[:,:,0].T
    try:
        trajectory[:,start+transitionLength:start+stepLength] = (
            Xf*np.ones((2,stepLength-transitionLength))
        )
    except:
        # trajectory[:,-1] = transition_func(tempTime[-1]).flatten()
        pass

    try:
        OLDtrajectory[0,start_OLD:end_OLD] = [
            Xf[0][0]
        ]*stepLength
        OLDtrajectory[1,start_OLD:end_OLD] = [
            Xf[1][0]
        ]*stepLength
    except:
        OLDtrajectory[0,start_OLD:] = [
            Xf[0][0]
        ]*len(OLDtrajectory[0,start_OLD:])
        OLDtrajectory[1,start_OLD:] = [
            Xf[1][0]
        ]*len(OLDtrajectory[1,start_OLD:])
    start=end
    start_OLD=end_OLD
    Xi=Xf
# trajectory[:,-1]=trajectory[:,-2]

filterLength = int((1/3)/plant.dt/2)
b=np.ones(filterLength,)/(filterLength) #Finite Impulse Response (FIR) Moving Average (MA) filter with filter length (3 Hz^{-1} / dt) / 2
a=1
OLDtrajectory = signal.filtfilt(b, a, OLDtrajectory)
# FILTtrajectory = signal.filtfilt(b, a, trajectory)

X1d_new = np.zeros((5,len(trajectory[0,:])))
X1d_new[0,:] = trajectory[0,:]*180/np.pi
X1d_new[1,:] = np.gradient(X1d_new[0,:],plant.dt)
X1d_new[2,:] = np.gradient(X1d_new[1,:],plant.dt)
X1d_new[3,:] = np.gradient(X1d_new[2,:],plant.dt)
X1d_new[4,:] = np.gradient(X1d_new[3,:],plant.dt)

Sd_new = np.zeros((3,len(trajectory[1,:])))
Sd_new[0,:] = trajectory[1,:]
Sd_new[1,:] = np.gradient(Sd_new[0,:],plant.dt)
Sd_new[2,:] = np.gradient(Sd_new[1,:],plant.dt)

X1d_old = np.zeros((5,len(trajectory[0,:])))
X1d_old[0,:] = OLDtrajectory[0,:]*180/np.pi
X1d_old[1,:] = np.gradient(X1d_old[0,:],plant.dt)
X1d_old[2,:] = np.gradient(X1d_old[1,:],plant.dt)
X1d_old[3,:] = np.gradient(X1d_old[2,:],plant.dt)
X1d_old[4,:] = np.gradient(X1d_old[3,:],plant.dt)

Sd_old = np.zeros((3,len(trajectory[1,:])))
Sd_old[0,:] = OLDtrajectory[1,:]
Sd_old[1,:] = np.gradient(Sd_old[0,:],plant.dt)
Sd_old[2,:] = np.gradient(Sd_old[1,:],plant.dt)

# X1d_new_filt = np.zeros((5,len(trajectory[0,:])))
# X1d_new_filt[0,:] = FILTtrajectory[0,:]*180/np.pi
# X1d_new_filt[1,:] = np.gradient(X1d_new_filt[0,:],plant.dt)
# X1d_new_filt[2,:] = np.gradient(X1d_new_filt[1,:],plant.dt)
# X1d_new_filt[3,:] = np.gradient(X1d_new_filt[2,:],plant.dt)
# X1d_new_filt[4,:] = np.gradient(X1d_new_filt[3,:],plant.dt)

fig, axs = plt.subplots(5,1,figsize=(12,20),sharex=True)
axs[0].plot(tempTime,X1d_old[0,:],'0.70',lw=3)
axs[0].plot(tempTime,X1d_new[0,:],'C0')
# axs[0].plot(tempTime,X1d_new_filt[0,:],'C1')
axs[0].set_xlim([0,tempTime[-1]])
axs[0].set_title("Joint Angle (deg)")
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
# axs[0].legend(["Old","New","New (Filtered)"],loc='upper right')
axs[0].legend(["Old","New"],loc='upper right')

axs[1].plot(tempTime,X1d_old[1,:],'0.70',lw=3)
axs[1].plot(tempTime,X1d_new[1,:],'C0')
# axs[1].plot(tempTime,X1d_new_filt[1,:],'C1')
axs[1].set_title("Joint Angular Velocity (deg/s)")
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)

axs[2].plot(tempTime,X1d_old[2,:],'0.70',lw=3)
axs[2].plot(tempTime,X1d_new[2,:],'C0')
# axs[2].plot(tempTime,X1d_new_filt[2,:],'C1')
axs[2].set_title(r"Joint Angular Acceleration (deg/s$^2$)")
axs[2].spines["top"].set_visible(False)
axs[2].spines["right"].set_visible(False)

axs[3].plot(tempTime,X1d_old[3,:],'0.70',lw=3)
axs[3].plot(tempTime,X1d_new[3,:],'C0')
# axs[3].plot(tempTime,X1d_new_filt[3,:],'C1')
axs[3].set_title(r"Joint Angular Jerk (deg/s$^3$)")
axs[3].spines["top"].set_visible(False)
axs[3].spines["right"].set_visible(False)

axs[4].plot(tempTime,X1d_old[4,:],'0.70',lw=3)
axs[4].plot(tempTime,X1d_new[4,:],'C0')
# axs[4].plot(tempTime,X1d_new_filt[4,:],'C1')
axs[4].set_title(r"Joint Angular Snap (deg/s$^4$)")
axs[4].set_xlabel("Time (s)")
axs[4].spines["top"].set_visible(False)
axs[4].spines["right"].set_visible(False)

fig.tight_layout(pad=5.0)

X_o_old = plant.return_X_o_given_s_o(
    OLDtrajectory[0,0],OLDtrajectory[1,0],[0,0]
)
X_old,U_old,_,_ = plant.forward_simulation_FL(X_o_old,X1d_old*np.pi/180,Sd_old)

X_o_new = plant.return_X_o_given_s_o(trajectory[0,0],trajectory[1,0],[0,0])
X_new,U_new,_,_ = plant.forward_simulation_FL(X_o_new,X1d_new*np.pi/180,Sd_new)

plot_comparison(tempTime,X_new,X_old,"Angle Step / Stiffness Step")

##############################################
################## STEP SIN ##################
##############################################

print("ANGLE STEP / STIFF SIN")

frequency = 1

stepDuration = 3/frequency # three periods # TEMP : changed to see if this make the controller a little less crazy...
stepLength = int(stepDuration/plant.dt)
additionalSteps = 5
tempTime = np.arange(
    0,
    delay+(numberOfSteps+additionalSteps)*stepDuration+1e-4,
    plant.dt
)
trajectory = startValue.T*np.ones((2,len(tempTime)))
OLDtrajectory = startValue.T*np.ones((2,len(tempTime)))

## Point-to-Point Joint Angle
# Step duration is equal to the period of the sinusoid for fixed joint angles while modulating joint stiffess

transitionLength = int((1/2)/plant.dt)
start=int(delay/plant.dt)-int(transitionLength/2)
start_OLD=int(delay/plant.dt)
x1i = np.mean(angleRange)
coeffs = [126,-420,540,-315,70]
# NOTE: by making the entire step equal to 1 second, you can create piecewise functions that do not have to deal with inbetween steps (i.e., the filterDuration is not a multiple of step.dt). Therefore "start" will always be a multiple of plant.dt
for i in range(numberOfSteps+additionalSteps+1):
    x1f = np.random.uniform(
        angleRange[0],
        angleRange[1]
    )
    end = start+stepLength
    transition_func = lambda t: (
        x1i + (x1f-x1i)*(
            coeffs[0]*((t-plant.dt*start)/(plant.dt*transitionLength))**5
            + coeffs[1]*((t-plant.dt*start)/(plant.dt*transitionLength))**6
            + coeffs[2]*((t-plant.dt*start)/(plant.dt*transitionLength))**7
            + coeffs[3]*((t-plant.dt*start)/(plant.dt*transitionLength))**8
            + coeffs[4]*((t-plant.dt*start)/(plant.dt*transitionLength))**9
        )
    )
    trajectory[0,start:start+transitionLength] = np.array(list(map(
        transition_func,
        tempTime[start:start+transitionLength]
    )))
    try:
        trajectory[0,start+transitionLength:start+stepLength] = (
            x1f*np.ones((1,stepLength-transitionLength))
        )
    except:
        pass

    if i!=numberOfSteps+additionalSteps:
        end_OLD = start_OLD+stepLength
        OLDtrajectory[0,start_OLD:end_OLD] = [x1f]*stepLength
        start_OLD=end_OLD
    start=end
    x1i=x1f

filterLength = int((1/3)/plant.dt/2)
b=np.ones(filterLength,)/(filterLength) #Finite Impulse Response (FIR) Moving Average (MA) filter with filter length (5 Hz^{-1} / dt) / 2
a=1
OLDtrajectory[0,:] = signal.filtfilt(b, a, OLDtrajectory[0,:])

## Sinusoidal Joint Stiffness
stiffnessAmplitude = (stiffnessRange[1] - stiffnessRange[0])/2
stiffnessOffset = (stiffnessRange[1] + stiffnessRange[0])/2
sinusoidal_stiffness_func = lambda t:(
    stiffnessOffset
    - stiffnessAmplitude*np.cos(
        2*np.pi*frequency
        * (t-delay)
    )
)
trajectory[1,int(delay/plant.dt):] = np.array(list(map(
    sinusoidal_stiffness_func,
    tempTime[int(delay/plant.dt):]
)))
OLDtrajectory[1,int(delay/plant.dt):] = np.array(list(map(
    sinusoidal_stiffness_func,
    tempTime[int(delay/plant.dt):]
)))

X1d_new = np.zeros((5,len(trajectory[0,:])))
X1d_new[0,:] = trajectory[0,:]*180/np.pi
X1d_new[1,:] = np.gradient(X1d_new[0,:],plant.dt)
X1d_new[2,:] = np.gradient(X1d_new[1,:],plant.dt)
X1d_new[3,:] = np.gradient(X1d_new[2,:],plant.dt)
X1d_new[4,:] = np.gradient(X1d_new[3,:],plant.dt)

Sd_new = np.zeros((3,len(trajectory[1,:])))
Sd_new[0,:] = trajectory[1,:]
Sd_new[1,:] = np.gradient(Sd_new[0,:],plant.dt)
Sd_new[2,:] = np.gradient(Sd_new[1,:],plant.dt)

X1d_old = np.zeros((5,len(trajectory[0,:])))
X1d_old[0,:] = OLDtrajectory[0,:]*180/np.pi
X1d_old[1,:] = np.gradient(X1d_old[0,:],plant.dt)
X1d_old[2,:] = np.gradient(X1d_old[1,:],plant.dt)
X1d_old[3,:] = np.gradient(X1d_old[2,:],plant.dt)
X1d_old[4,:] = np.gradient(X1d_old[3,:],plant.dt)

Sd_old = np.zeros((3,len(trajectory[1,:])))
Sd_old[0,:] = OLDtrajectory[1,:]
Sd_old[1,:] = np.gradient(Sd_old[0,:],plant.dt)
Sd_old[2,:] = np.gradient(Sd_old[1,:],plant.dt)

fig, axs = plt.subplots(5,1,figsize=(12,20),sharex=True)
axs[0].plot(tempTime,X1d_old[0,:],'0.70',lw=3)
axs[0].plot(tempTime,X1d_new[0,:],'C0')
# axs[0].plot(tempTime,X1d_new_filt[0,:],'C1')
axs[0].set_xlim([0,tempTime[-1]])
axs[0].set_title("Joint Angle (deg)")
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
# axs[0].legend(["Old","New","New (Filtered)"],loc='upper right')
axs[0].legend(["Old","New"],loc='upper right')

axs[1].plot(tempTime,X1d_old[1,:],'0.70',lw=3)
axs[1].plot(tempTime,X1d_new[1,:],'C0')
# axs[1].plot(tempTime,X1d_new_filt[1,:],'C1')
axs[1].set_title("Joint Angular Velocity (deg/s)")
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)

axs[2].plot(tempTime,X1d_old[2,:],'0.70',lw=3)
axs[2].plot(tempTime,X1d_new[2,:],'C0')
# axs[2].plot(tempTime,X1d_new_filt[2,:],'C1')
axs[2].set_title(r"Joint Angular Acceleration (deg/s$^2$)")
axs[2].spines["top"].set_visible(False)
axs[2].spines["right"].set_visible(False)

axs[3].plot(tempTime,X1d_old[3,:],'0.70',lw=3)
axs[3].plot(tempTime,X1d_new[3,:],'C0')
# axs[3].plot(tempTime,X1d_new_filt[3,:],'C1')
axs[3].set_title(r"Joint Angular Jerk (deg/s$^3$)")
axs[3].spines["top"].set_visible(False)
axs[3].spines["right"].set_visible(False)

axs[4].plot(tempTime,X1d_old[4,:],'0.70',lw=3)
axs[4].plot(tempTime,X1d_new[4,:],'C0')
# axs[4].plot(tempTime,X1d_new_filt[4,:],'C1')
axs[4].set_title(r"Joint Angular Snap (deg/s$^4$)")
axs[4].set_xlabel("Time (s)")
axs[4].spines["top"].set_visible(False)
axs[4].spines["right"].set_visible(False)

fig.tight_layout(pad=5.0)

X_o_old = plant.return_X_o_given_s_o(
    OLDtrajectory[0,0],OLDtrajectory[1,0],[0,0]
)
X_old,U_old,_,_ = plant.forward_simulation_FL(X_o_old,X1d_old*np.pi/180,Sd_old)

X_o_new = plant.return_X_o_given_s_o(trajectory[0,0],trajectory[1,0],[0,0])
X_new,U_new,_,_ = plant.forward_simulation_FL(X_o_new,X1d_new*np.pi/180,Sd_new)

plot_comparison(tempTime,X_new,X_old,"Angle Step / Stiffness Sinusoid")


##############################################
################## SIN STEP ##################
##############################################

print("ANGLE SIN / STIFF STEP")
frequency = 1
angleRange = [3*np.pi/4,5*np.pi/4]

startValue = np.array([[
    np.mean(angleRange),
    stiffnessRange[0]
]])

filterDuration = 1/3/2 # (1/5 Hz) / 2
filterLength = int(filterDuration/plant.dt)

##  Sinusoidal Joint Angle
angleAmplitude = (angleRange[1] - angleRange[0])/2
angleOffset = (angleRange[1] + angleRange[0])/2
amplitudeCorrection = (
    np.pi*frequency
    / np.sin(np.pi*frequency*filterDuration)
)
smooth_transition_angle_func = lambda t:(
    (angleAmplitude*amplitudeCorrection/(2*np.pi*frequency))*(
        1 - np.cos(2*np.pi*frequency*(t + filterDuration/2 - delay))
    )
    + angleOffset
) # for delay-filterDuration/2 < t < delay + filterDuration/2
sinusoidal_angle_func = lambda t:(
    angleAmplitude*np.sin(
        2*np.pi*frequency
        * (t-delay)
    )
    + angleOffset
) # for t >= delay + T_filter

stepDuration = 3/frequency
stepLength = int(stepDuration/plant.dt)
additionalSteps = 3
tempTime = np.arange(
    0,
    delay+(numberOfSteps+additionalSteps)*stepDuration+1e-4,
    plant.dt
)
trajectory = startValue.T*np.ones((2,len(tempTime)))
OLDtrajectory = startValue.T*np.ones((2,len(tempTime)))

##  Sinusoidal Joint Angle
trajectory[0,int((delay-filterDuration/2)/plant.dt):int((delay+filterDuration/2)/plant.dt)] =\
    np.array(list(map(
        smooth_transition_angle_func,
        tempTime[int((delay-filterDuration/2)/plant.dt):int((delay+filterDuration/2)/plant.dt)]
    )))
trajectory[0,int((delay + filterDuration/2)/plant.dt):] = np.array(list(map(
    sinusoidal_angle_func,
    tempTime[int((delay + filterDuration/2)/plant.dt):]
)))
OLDtrajectory[0,:] = trajectory[0,:]

##  Point-to-Point Joint Stiffness
transitionLength = int((1/2)/plant.dt)
start=int(delay/plant.dt)-int(transitionLength/2)
start_OLD = int(delay/plant.dt)
si = stiffnessRange[0]
coeffs = [126,-420,540,-315,70]
# NOTE: by making the entire step equal to 1 second, you can create piecewise functions that do not have to deal with inbetween steps (i.e., the filterDuration is not a multiple of step.dt). Therefore "start" will always be a multiple of plant.dt
for i in range(numberOfSteps+additionalSteps+1):
    sf = np.random.uniform(
        stiffnessRange[0],
        stiffnessRange[1]
    )
    end = start+stepLength
    transition_func = lambda t: (
        si + (sf-si)*(
            coeffs[0]*((t-plant.dt*start)/(plant.dt*transitionLength))**5
            + coeffs[1]*((t-plant.dt*start)/(plant.dt*transitionLength))**6
            + coeffs[2]*((t-plant.dt*start)/(plant.dt*transitionLength))**7
            + coeffs[3]*((t-plant.dt*start)/(plant.dt*transitionLength))**8
            + coeffs[4]*((t-plant.dt*start)/(plant.dt*transitionLength))**9
        )
    )
    trajectory[1,start:start+transitionLength] = np.array(list(map(
        transition_func,
        tempTime[start:start+transitionLength]
    )))
    try:
        trajectory[1,start+transitionLength:start+stepLength] = (
            sf*np.ones((1,stepLength-transitionLength))
        )
    except:
        pass

    if i!=numberOfSteps+additionalSteps:
        end_OLD = start_OLD+stepLength
        OLDtrajectory[1,start_OLD:end_OLD] = [sf]*stepLength
        start_OLD=end_OLD
    start=end
    si=sf

filterLength = int((1/3)/plant.dt/2)
b=np.ones(filterLength,)/(filterLength) #Finite Impulse Response (FIR) Moving Average (MA) filter with filter length (5 Hz^{-1} / dt) / 2
a=1
OLDtrajectory[1,:] = signal.filtfilt(b, a, OLDtrajectory[1,:])

X1d_new = np.zeros((5,len(trajectory[0,:])))
X1d_new[0,:] = trajectory[0,:]*180/np.pi
X1d_new[1,:] = np.gradient(X1d_new[0,:],plant.dt)
X1d_new[2,:] = np.gradient(X1d_new[1,:],plant.dt)
X1d_new[3,:] = np.gradient(X1d_new[2,:],plant.dt)
X1d_new[4,:] = np.gradient(X1d_new[3,:],plant.dt)

Sd_new = np.zeros((3,len(trajectory[1,:])))
Sd_new[0,:] = trajectory[1,:]
Sd_new[1,:] = np.gradient(Sd_new[0,:],plant.dt)
Sd_new[2,:] = np.gradient(Sd_new[1,:],plant.dt)

X1d_old = np.zeros((5,len(trajectory[0,:])))
X1d_old[0,:] = OLDtrajectory[0,:]*180/np.pi
X1d_old[1,:] = np.gradient(X1d_old[0,:],plant.dt)
X1d_old[2,:] = np.gradient(X1d_old[1,:],plant.dt)
X1d_old[3,:] = np.gradient(X1d_old[2,:],plant.dt)
X1d_old[4,:] = np.gradient(X1d_old[3,:],plant.dt)

Sd_old = np.zeros((3,len(trajectory[1,:])))
Sd_old[0,:] = OLDtrajectory[1,:]
Sd_old[1,:] = np.gradient(Sd_old[0,:],plant.dt)
Sd_old[2,:] = np.gradient(Sd_old[1,:],plant.dt)

fig, axs = plt.subplots(5,1,figsize=(12,20),sharex=True)
axs[0].plot(tempTime[int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],X1d_old[0,int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],'0.70',lw=3)
axs[0].plot(tempTime[int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],X1d_new[0,int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],'C0')
# axs[0].plot(tempTime[int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],X1d_new_filt[0,:],'C1')
axs[0].set_xlim([0,tempTime[int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength][-1]])
axs[0].set_title("Joint Angle (deg)")
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
# axs[0].legend(["Old","New","New (Filtered)"],loc='upper right')
axs[0].legend(["Old","New"],loc='upper right')

axs[1].plot(tempTime[int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],X1d_old[1,int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],'0.70',lw=3)
axs[1].plot(tempTime[int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],X1d_new[1,int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],'C0')
# axs[1].plot(tempTime[int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],X1d_new_filt[1,:],'C1')
axs[1].set_title("Joint Angular Velocity (deg/s)")
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)

axs[2].plot(tempTime[int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],X1d_old[2,int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],'0.70',lw=3)
axs[2].plot(tempTime[int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],X1d_new[2,int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],'C0')
# axs[2].plot(tempTime[int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],X1d_new_filt[2,:],'C1')
axs[2].set_title(r"Joint Angular Acceleration (deg/s$^2$)")
axs[2].spines["top"].set_visible(False)
axs[2].spines["right"].set_visible(False)

axs[3].plot(tempTime[int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],X1d_old[3,int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],'0.70',lw=3)
axs[3].plot(tempTime[int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],X1d_new[3,int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],'C0')
# axs[3].plot(tempTime[int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],X1d_new_filt[3,:],'C1')
axs[3].set_title(r"Joint Angular Jerk (deg/s$^3$)")
axs[3].spines["top"].set_visible(False)
axs[3].spines["right"].set_visible(False)

axs[4].plot(tempTime[int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],X1d_old[4,int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],'0.70',lw=3)
axs[4].plot(tempTime[int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],X1d_new[4,int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],'C0')
# axs[4].plot(tempTime[int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],X1d_new_filt[4,:],'C1')
axs[4].set_title(r"Joint Angular Snap (deg/s$^4$)")
axs[4].set_xlabel("Time (s)")
axs[4].spines["top"].set_visible(False)
axs[4].spines["right"].set_visible(False)

fig.tight_layout(pad=5.0)

X_o_old = plant.return_X_o_given_s_o(
    OLDtrajectory[0,0],OLDtrajectory[1,0],[0,0]
)
X_old,U_old,_,_ = plant.forward_simulation_FL(X_o_old,X1d_old*np.pi/180,Sd_old)

X_o_new = plant.return_X_o_given_s_o(trajectory[0,0],trajectory[1,0],[0,0])
X_new,U_new,_,_ = plant.forward_simulation_FL(X_o_new,X1d_new*np.pi/180,Sd_new)

plot_comparison(
    tempTime[int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],
    X_new[:,int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],
    X_old[:,int(delay/plant.dt)+(additionalSteps-1)*stepLength:-stepLength],
    "Angle Sinusoid / Stiffness Step"
)

plt.show()
