import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from danpy.sb import dsb
from danpy.useful_functions import save_figures,is_number
from scipy import signal
import numdifftools as nd
import scipy as sp
from plantParams import *
import argparse
import textwrap
from animate import *
import scipy.io as sio
import pandas as pd
from scipy.signal import savgol_filter

def LP_filt(filterLength, x):
    """
    Finite Impulse Response (FIR) Moving Average (MA) Low-Pass Filter
    """
    b=np.ones(filterLength,)/(filterLength) #Finite Impulse Response (FIR) Moving Average (MA) filter with one second filter length
    a=1
    y = signal.filtfilt(b, a, x)
    return y

class plant_pendulum_1DOF2DOF:
    def __init__(self,plantParams):
        self.params = plantParams

        self.Ij = plantParams.get("Joint Inertia", 1.15e-2) # kg⋅m²
        is_number(self.Ij,"Joint Inertia",default=1.15e-2)

        self.bj = plantParams.get("Joint Damping", 0.001) # N⋅s⋅m⁻¹
        is_number(self.bj,"Joint Damping",default=0.001)

        self.mj = plantParams.get("Joint Mass", 0.541) # kg
        is_number(self.mj,"Joint Mass",default=0.541)

        self.rj = plantParams.get("Joint Moment Arm", 0.05) # m
        is_number(self.rj,"Joint Moment Arm",default=0.05)

        self.Lcm = plantParams.get("Link Center of Mass", 0.085) # m
        is_number(self.Lcm,"Link Center of Mass",default=0.085)

        self.L = plantParams.get("Link Length", 0.3) # m
        is_number(self.L,"Link Length",default=0.3)

        self.Jm = plantParams.get("Motor Inertia", 6.6e-5) # kg⋅m²
        is_number(self.Jm,"Motor Inertia",default=6.6e-5)

        self.bm = plantParams.get("Motor Damping", 0.00462) # N⋅s⋅m⁻¹
        is_number(self.bm,"Motor Damping",default=0.00462)

        self.rm = plantParams.get("Motor Moment Arm", 0.01) # m
        is_number(self.rm,"Motor Moment Arm",default=0.01)

        self.k_spr = plantParams.get("Spring Stiffness Coefficient",1) # N
        is_number(self.k_spr,"",default=1)

        self.b_spr = plantParams.get("Spring Shape Coefficient",100) # unit-less
        is_number(self.b_spr,"",default=1)

        self.simulationDuration = plantParams.get("Simulation Duration", 1000)
        is_number(self.simulationDuration,"Simulation Duration")

        self.dt = plantParams.get("dt", 0.01)
        is_number(self.dt,"dt")

        self.k0 = plantParams.get(
            "Position Gains",
            {
                0 : 3162.3,
                1 : 1101.9,
                2 : 192.0,
                3 : 19.6
            }
        )
        self.ks = plantParams.get(
            "Stiffness Gains",
            {
                0 : 316.2,
                1 : 25.1
            }
        )

        self.jointAngleBounds = plantParams.get(
            "Joint Angle Bounds",
            {
                "LB" : np.pi/2,
                "UB" : 3*np.pi/2
            }
        )
        self.jointAngleRange = (
            self.jointAngleBounds["UB"]
            - self.jointAngleBounds["LB"]
        )
        self.jointAngleMidPoint = (
            self.jointAngleBounds["UB"]
            + self.jointAngleBounds["LB"]
        )/2

        self.jointStiffnessBounds = {
            "LB" : 2*(self.rj**2)*self.k_spr*self.b_spr
            }
        self.jointStiffnessBounds["UB"] = plantParams.get("Maximum Joint Stiffness",100)
        is_number(self.jointStiffnessBounds["UB"],"Maximum Joint Stiffness",default=100)
        self.jointStiffnessBounds["LB"] = 15
        self.jointStiffnessRange = (
            self.jointStiffnessBounds["UB"]
            - self.jointStiffnessBounds["LB"]
        )

        self.jointStiffnessMidPoint = (
            self.jointStiffnessBounds["UB"]
            + self.jointStiffnessBounds["LB"]
        )/2

        self.boundaryFrictionWeight = plantParams.get("Boundary Friction Weight",0.1)
        is_number(self.boundaryFrictionWeight,"Boundary Friction Weight",
            default=0.1,
            notes="It appears that values less than 1 will produce steeper boundaries for the periodic solution, while values greater than 1 will produce sharper bounderies for the quadratic penalty function."
        )
        self.boundaryFrictionGain = plantParams.get("Boundary Friction Gain",1)
        is_number(self.boundaryFrictionGain,"Boundary Friction Gain",default=1)

        self.time = np.arange(
            0,
            self.simulationDuration+self.dt,
            self.dt
        )

    def return_X_o(self,x1o,U_o):
        """
        Returns the initial state X_o where the positions of the motors depends on the initial position of the pendulum and the amount of torque on the motors. This assumes we are at (or near) equilibrium at the start of a simulation to get us close to the actual initial state (and therefore minimize the large transitions at the start).

        U_o should be an array with 2 elements and x1o should be a number given in radians between 0 and 2 pi.
        """
        lTo1 = np.log((U_o[0]/(self.k_spr*self.rm))+1)/self.b_spr
        lTo2 = np.log((U_o[1]/(self.k_spr*self.rm))+1)/self.b_spr
        x35o = (
            (1/(2*self.rm))
            * np.matrix([[1,1],[1,-1]])
            * np.matrix([
                [lTo1+lTo2],
                [2*self.rj*x1o + lTo1 - lTo2]
            ])
        )

        return([x1o,0,x35o[0,0],0,x35o[1,0],0])

    def return_X_o_given_s_o(self,x1o,s_o,guess):
        """
        Returns the initial state X_o where the positions of the motors depends on the initial position of the pendulum and the amount of torque on the motors. This assumes we are at (or near) equilibrium at the start of a simulation to get us close to the actual initial state (and therefore minimize the large transitions at the start).

        U_o should be an array with 2 elements and x1o should be a number given in radians between 0 and 2 pi.
        """
        if abs(x1o-np.pi)>1e-3:
            def equations(p):
                x3, x5 = p
                eqn1 = (self.rj**2)*self.k_spr*self.b_spr*(
                        np.exp(self.b_spr*(self.rm*x3 - self.rj*x1o))
                            * ((self.rm*x3 - self.rj*x1o)>=0)
                        + np.exp(self.b_spr*(self.rm*x5 + self.rj*x1o))
                            * ((self.rm*x5 + self.rj*x1o)>=0)
                    ) - s_o
                eqn2 = (
                    -self.Lcm*self.mj*gr*np.sin(x1o) # gravitational torque
                    + self.rj*self.k_spr * (
                        (np.exp(self.b_spr*(self.rm*x3 - self.rj*x1o))-1)
                            * ((self.rm*x3 - self.rj*x1o)>=0)
                        - (np.exp(self.b_spr*(self.rm*x5 + self.rj*x1o))-1)
                            * ((self.rm*x5 + self.rj*x1o)>=0)
                    )
                )
                return (eqn1,eqn2)
            x3o,x5o = sp.optimize.fsolve(equations,guess)
        else:
            x3o = (
                self.rj*x1o/self.rm
                + (1/(self.rm*self.b_spr))
                * np.log(
                    s_o / (2*self.rj**2*self.k_spr*self.b_spr)
                )
            )
            x5o = x3o - 2*self.rj*x1o/self.rm
        # print("f2 output: " + str(self.f2_func([x1o,0,x3o,0,x5o,0])))
        # print("hs output: " + str(self.hs([x1o,0,x3o,0,x5o,0])))
        return([x1o,0,x3o,0,x5o,0])

    def C(self,X):
        """
        Returns zero until the effects are quantified
        """
        return(
            0
        )
    def dCdx1(self,X):
        return(0)
    def d2Cdx12(self,X):
        return(0)
    def d2Cdx1x2(self,X):
        return(0)
    def dCdx2(self,X):
        return(0)
    def d2Cdx22(self,X):
        return(0)

    def update_state_variables(self,X):

        #>>>> State functions

        self.f1 = self.f1_func(X)
        self.f2 = self.f2_func(X)
        self.f3 = self.f3_func(X)
        self.f4 = self.f4_func(X)
        self.f5 = self.f5_func(X)
        self.f6 = self.f6_func(X)

        #>>>> State functions first gradient

        # self.df1dx1 = 0
        self.df1dx2 = 1
        # self.df1dx3 = 0
        # self.df1dx4 = 0
        # self.df1dx5 = 0
        # self.df1dx6 = 0

        self.df2dx1 = self.df2dx1_func(X)
        self.df2dx2 = self.df2dx2_func(X)
        self.df2dx3 = self.df2dx3_func(X)
        # self.df2dx4 = 0
        self.df2dx5 = self.df2dx5_func(X)
        # self.df2dx6 = 0

        # self.df3dx1 = 0
        # self.df3dx2 = 0
        # self.df3dx3 = 0
        self.df3dx4 = 1
        # self.df3dx5 = 0
        # self.df3dx6 = 0

        # self.df4dx1 = N/A
        # self.df4dx2 = N/A
        # self.df4dx3 = N/A
        # self.df4dx4 = N/A
        # self.df4dx5 = N/A
        # self.df4dx6 = N/A

        # self.df5dx1 = 0
        # self.df5dx2 = 0
        # self.df5dx3 = 0
        # self.df5dx4 = 0
        # self.df5dx5 = 0
        self.df5dx6 = 1

        # self.df6dx1 = N/A
        # self.df6dx2 = N/A
        # self.df6dx3 = N/A
        # self.df6dx4 = N/A
        # self.df6dx5 = N/A
        # self.df6dx6 = N/A

        #>>>> State functions second gradient

        self.d2f2dx12 = self.d2f2dx12_func(X)
        self.d2f2dx1x2 = self.d2f2dx1x2_func(X)
        self.d2f2dx1x3 = self.d2f2dx1x3_func(X)
        self.d2f2dx1x5 = self.d2f2dx1x5_func(X)

        self.d2f2dx22 = self.d2f2dx22_func(X)

        self.d2f2dx32 = self.d2f2dx32_func(X)

        self.d2f2dx52 = self.d2f2dx52_func(X)

    # def motor_coupling_function(self,X,motorNumber):
    #     return(
    #         self.rm*self.k_spr*(
    #             np.exp(
    #                 self.b_spr*(
    #                     self.rm*X[2+2*(motorNumber-1)]
    #                     + ((1.5-motorNumber)/0.5)*self.rj*X[0]
    #                 )
    #             )
    #             -1
    #         )
    #     )
    def tendon_1_FL_func(self,X):
        return(
            self.k_spr*(
                np.exp(self.b_spr*(self.rm*X[2]-self.rj*X[0]))
                - 1
            ) * ((self.rm*X[2]-self.rj*X[0])>=0)
        )
    def tendon_2_FL_func(self,X):
        return(
            self.k_spr*(
                np.exp(self.b_spr*(self.rm*X[4]+self.rj*X[0]))
                - 1
            ) * ((self.rm*X[4]+self.rj*X[0])>=0)
        )

    def f1_func(self,X):
        return(X[1])

    def f2_func(self,X):
        return(
            (
                -self.C(X) # Coriolis and centrifugal torques (zero)
                - self.bj*X[1] # damping torque
                - self.Lcm*self.mj*gr*np.sin(X[0]) # gravitational torque
                + self.rj*self.k_spr * (
                    (np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))-1)
                        * ((self.rm*X[2] - self.rj*X[0])>=0)
                    - (np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))-1)
                        * ((self.rm*X[4] + self.rj*X[0])>=0)
                ) # total coupling torque between motors and joint
            )/self.Ij
        )
    def df2dx1_func(self,X):
        result = (
            (
                -self.dCdx1(X) # Coriolis and centrifugal torques (zero)
                - self.Lcm*self.mj*gr*np.cos(X[0]) # gravitational torque
                - (self.rj**2)*self.k_spr*self.b_spr * (
                    np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
                        * ((self.rm*X[2] - self.rj*X[0])>=0)
                    + np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
                        * ((self.rm*X[4] + self.rj*X[0])>=0)
                ) # total coupling torque between motors and joint
            )/self.Ij
        )
        return(result)
    def d2f2dx12_func(self,X):
        return(
            (
                -self.d2Cdx12(X) # Coriolis and centrifugal torques (zero)
                + self.Lcm*self.mj*gr*np.sin(X[0]) # gravitational torque
                + (self.rj**3)*self.k_spr*(self.b_spr**2) * (
                    np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
                        * ((self.rm*X[2] - self.rj*X[0])>=0)
                    - np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
                        * ((self.rm*X[4] + self.rj*X[0])>=0)
                ) # total coupling torque between motors and joint
            )/self.Ij
        )
    def d2f2dx1x2_func(self,X):
        return(
            (
                -self.d2Cdx1x2(X) # Coriolis and centrifugal torques (zero)
            )/self.Ij
        )
    def d2f2dx1x3_func(self,X):
        """
        This is equivalently -dSda/Ij
        """
        return(
            -(self.rj**2)*self.rm*self.k_spr*(self.b_spr**2) * (
                np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
                    * ((self.rm*X[2] - self.rj*X[0])>=0)
            ) / self.Ij
        )
    def d2f2dx1x5_func(self,X):
        """
        This is equivalently dSdb/Ij
        """
        return(
            -(self.rj**2)*self.rm*self.k_spr*(self.b_spr**2) * (
                np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
                    * ((self.rm*X[4] + self.rj*X[0])>=0)
            ) / self.Ij
        )
    def df2dx2_func(self,X):
        result = (
            (
                -self.dCdx2(X) # Coriolis and centrifugal torques (zero)
                - self.bj # damping torque
            )/self.Ij
        )
        return(result)
    def d2f2dx22_func(self,X):
        return(
            (
                -self.d2Cdx22(X) # Coriolis and centrifugal torques (zero)
            )/self.Ij
        )
    def df2dx3_func(self,X):
        """
        Equivalently, this is the negative value of -Q_{11}/Ij
        """
        result = (
            self.rj*self.rm*self.k_spr*self.b_spr * (
                np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
                    * ((self.rm*X[2] - self.rj*X[0])>=0)
            ) / self.Ij
        )
        return(result)
    def d2f2dx32_func(self,X):
        return(
            self.rj*(self.rm**2)*self.k_spr*(self.b_spr**2) * (
                np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
                    * ((self.rm*X[2] - self.rj*X[0])>=0)
            ) / self.Ij
        )
    def df2dx5_func(self,X):
        """
        Equivalently, this is the negative value of -Q_{12}/Ij
        """
        result = (
            -self.rj*self.rm*self.k_spr*self.b_spr * (
                np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
                    * ((self.rm*X[4] + self.rj*X[0])>=0)
            ) / self.Ij
        )
        return(result)
    def d2f2dx52_func(self,X):
        return(
            -self.rj*(self.rm**2)*self.k_spr*(self.b_spr**2) * (
                np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
                    * ((self.rm*X[4] + self.rj*X[0])>=0)
            ) / self.Ij
        )

    def f3_func(self,X):
        return(X[3])

    def f4_func(self,X):
        return(
            (
                -self.bm*X[3]
                - self.rm*self.k_spr*(
                    np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
                    -1
                ) * ((self.rm*X[2] - self.rj*X[0])>=0)
            )/self.Jm
        )

    def f5_func(self,X):
        return(X[5])

    def f6_func(self,X):
        return(
            (
                -self.bm*X[5]
                - self.rm*self.k_spr*(
                    np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
                    -1
                ) * ((self.rm*X[4] + self.rj*X[0])>=0)
            )/self.Jm
        )

    def f(self,X):
        result = np.zeros((6,1))
        result[0,0] = self.f1
        result[1,0] = self.f2
        result[2,0] = self.f3
        result[3,0] = self.f4
        result[4,0] = self.f5
        result[5,0] = self.f6
        return(result)
    def g(self,X):
        result = np.matrix(np.zeros((6,2)))
        result[3,0] = 1/self.Jm
        result[5,1] = 1/self.Jm
        return(result)
    def h(self,X):
        result = np.zeros((2,))
        result[0] = X[0]
        result[1] = (self.rj**2)*self.k_spr*self.b_spr*(
            np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
            + np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
        )
        return(result)

    def forward_simulation(self,X_o,U=None,addTitle=None):
        """
        Building our own f_array to reduce the number of calls for f_funcs by making it a static call for each iteration in the FBL instance.
        """
        assert len(X_o)==6, "X_o must have 6 elements, not " + str(len(X_o)) + "."
        if addTitle is None:
            addTitle = "Custom"
        else:
            assert type(addTitle)==str, "addTitle must be a str."

        dt = self.time[1]-self.time[0]
        if U is None:
            U = np.zeros((2,len(self.time)-1))
        else:
            assert np.shape(U)==(2,len(self.time)-1), "U must be either None (default) of have shape (2,len(self.time)-1), not " + str(np.shape(U)) + "."
        X = np.zeros((6,len(self.time)))
        Y = np.zeros((2,len(self.time)))
        X[:,0] = X_o
        Y[:,0] = self.h(X[:,0])
        statusbar=dsb(0,len(self.time)-1,title="Forward Simulation (" + addTitle + ")")
        for i in range(len(self.time)-1):
            X[0,i+1] = X[0,i] + self.dt*self.f1_func(X[:,i])
            X[1,i+1] = X[1,i] + self.dt*self.f2_func(X[:,i])
            if X[0,i+1]<=self.jointAngleBounds["LB"]:
                X[0,i+1] = self.jointAngleBounds["LB"]
                # if X[1,i+1]<0: X[1,i+1] = 0
                if X[1,i+1]<0: X[1,i+1] = (X[0,i+1]-X[0,i])/self.dt
                # X[1,i+1] = X[1,i]
            elif X[0,i+1]>=self.jointAngleBounds["UB"]:
                X[0,i+1] = self.jointAngleBounds["UB"]
                # if X[1,i+1]>0: X[1,i+1] = 0
                if X[1,i+1]>0: X[1,i+1] = (X[0,i+1]-X[0,i])/self.dt
                # X[1,i+1] = X[1,i]
            # X[1,i+1] = X[1,i] + self.dt*self.f2_func(X[:,i])
            X[2,i+1] = X[2,i] + self.dt*self.f3_func(X[:,i])
            X[3,i+1] = X[3,i] + self.dt*(self.f4_func(X[:,i]) + U[0,i]/self.Jm)
            X[4,i+1] = X[4,i] + self.dt*self.f5_func(X[:,i])
            X[5,i+1] = X[5,i] + self.dt*(self.f6_func(X[:,i]) + U[1,i]/self.Jm)

            Y[:,i+1] = self.h(X[:,i+1])
            statusbar.update(i)
        return(X,U,Y)

    def h0(self,X):
        return(X[0])
    def Lfh0(self,X):
        return(X[1])
    def Lf2h0(self,X):
        return(self.f2)
    def Lf3h0(self,X):
        result = (
            self.df2dx1*self.f1
            + self.df2dx2*self.f2
            + self.df2dx3*self.f3
            + self.df2dx5*self.f5
        )
        return(result)
    def Lf4h0(self,X):
        return(
            (
                self.d2f2dx12*self.f1
                + self.d2f2dx1x2*self.f2
                + self.df2dx2*self.df2dx1
                + self.d2f2dx1x3*self.f3
                + self.d2f2dx1x5*self.f5
            ) * self.f1
            + (
                self.d2f2dx1x2*self.f1
                + self.df2dx1
                + self.d2f2dx22*self.f2
                + (self.df2dx2**2)
            ) * self.f2
            + (
                self.d2f2dx1x3*self.f1
                + self.df2dx2*self.df2dx3
                + self.d2f2dx32*self.f3
            ) * self.f3
            + (
                self.df2dx3
            ) * self.f4
            + (
                self.d2f2dx1x5*self.f1
                + self.df2dx2*self.df2dx5
                + self.d2f2dx52*self.f5
            ) * self.f5
            + (
                self.df2dx5
            ) * self.f6
        )

    def hs(self,X):
        return(
            (self.rj**2)*self.k_spr*self.b_spr*(
                np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
                    * ((self.rm*X[2] - self.rj*X[0])>=0)
                + np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
                    * ((self.rm*X[4] + self.rj*X[0])>=0)
            )
        )
    def Lfhs(self,X):
        return(
            (self.rj**2)*self.k_spr*(self.b_spr**2)*(
                -(self.rj*self.f1 - self.rm*self.f3)*(
                    np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
                        * ((self.rm*X[2] - self.rj*X[0])>=0)
                )
                + (self.rj*self.f1 + self.rm*self.f5)*(
                    np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
                        * ((self.rm*X[4] + self.rj*X[0])>=0)
                )
            )
        )
    def Lf2hs(self,X):
        return(
            (self.rj**2)*self.k_spr*(self.b_spr**2)*(
                (
                    self.b_spr*(self.rj*self.f1 - self.rm*self.f3)**2
                    - self.rj*self.f2
                    + self.rm*self.f4
                ) * np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
                    * ((self.rm*X[2] - self.rj*X[0])>=0)
                + (
                    self.b_spr*(self.rj*self.f1 + self.rm*self.f5)**2
                    + self.rj*self.f2
                    + self.rm*self.f6
                ) * np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
                    * ((self.rm*X[4] + self.rj*X[0])>=0)
            )
        )

    # def Phi(self,X):
    #     return(
    #         np.matrix([[
    #             self.h0(X),
    #             self.Lfh0(X),
    #             self.Lf2h0(X),
    #             self.Lf3h0(X),
    #             self.hs(X),
    #             self.Lfhs(X)
    #         ]]).T
    #     )
    def v0(self,X,x1d):
        result = (
            x1d[4]
            + self.k0[3]*(x1d[3]-self.Lf3h0(X))
            + self.k0[2]*(x1d[2]-self.Lf2h0(X))
            + self.k0[1]*(x1d[1]-self.Lfh0(X))
            + self.k0[0]*(x1d[0]-self.h0(X))
        )
        return(result)
    def vs(self,X,Sd):
        result =(
            Sd[2]
            + self.ks[1]*(Sd[1]-self.Lfhs(X))
            + self.ks[0]*(Sd[0]-self.hs(X))
        )
        return(result)

    def Q(self,X):
        B = np.matrix([
            [1/(self.Jm*self.Ij),0],
            [0,1/self.Jm]
        ])
        W = self.rj*self.rm*self.k_spr*self.b_spr*np.matrix([
            [
                np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))* ((self.rm*X[2] - self.rj*X[0])>=0),
                -np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))* ((self.rm*X[4] + self.rj*X[0])>=0)
            ],
            [
                self.rj*self.b_spr*(
                    np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))* ((self.rm*X[2] - self.rj*X[0])>=0)
                ),
                self.rj*self.b_spr*(
                    np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))* ((self.rm*X[4] + self.rj*X[0])>=0)
                )
            ]
        ])
        return(B*W)
    def return_input(self,X,x1d,Sd):
        Q_inv = self.Q(X)**(-1)
        return(
            Q_inv
            * (
                np.matrix([[-self.Lf4h0(X),-self.Lf2hs(X)]]).T
                + np.matrix([[self.v0(X,x1d),self.vs(X,Sd)]]).T
            )
        )

    def plot_desired_trajectory_distribution_and_power_spectrum(
            self,
            trajectory,
            cutoff=3
        ):

        fig1, (ax1a,ax1b) = plt.subplots(2,1,sharex=True,figsize=(5,7))
        time = np.array(list(range(len(trajectory[0,:int(cutoff/self.dt)]))))*self.dt
        ax1a.plot(
            [time[0],time[-1]],
            [(180/np.pi)*self.jointAngleBounds["LB"]]*2,
            'k--',
            label='_nolegend_'
        )
        ax1a.plot(
            [time[0],time[-1]],
            [(180/np.pi)*self.jointAngleBounds["UB"]]*2,
            'k--',
            label='_nolegend_'
        )
        ax1a.set_ylabel('Joint Angle (deg.)')
        ax1a.set_ylim([
            (180/np.pi)*(self.jointAngleBounds["LB"]-0.1*self.jointAngleRange),
            (180/np.pi)*(self.jointAngleBounds["UB"]+0.1*self.jointAngleRange)
        ])
        ax1a.set_xlim([time[0],time[-1]])
        plt.setp(ax1a.get_xticklabels(), visible=False)
        ax1a.spines["right"].set_visible(False)
        ax1a.spines["top"].set_visible(False)

        ax1b.plot(
            [time[0],time[-1]],
            [self.jointStiffnessBounds["LB"]]*2,
            'k--',
            label='_nolegend_'
        )
        ax1b.plot(
            [time[0],time[-1]],
            [self.jointStiffnessBounds["UB"]]*2,
            'k--',
            label='_nolegend_'
        )
        ax1b.set_ylabel('Joint Stiffness (Nm/rad)')
        ax1b.set_ylim([
            self.jointStiffnessBounds["LB"] - 0.1*self.jointStiffnessRange,
            self.jointStiffnessBounds["UB"] + 0.1*self.jointStiffnessRange
        ])
        ax1b.set_xlim([time[0],time[-1]])
        ax1b.set_xlabel('Time (s)')
        ax1b.spines["right"].set_visible(False)
        ax1b.spines["top"].set_visible(False)

        ax1a.plot(
            time,
            (180/np.pi)*trajectory[0,:int(cutoff/self.dt)],
            "C0"
        )
        ax1b.plot(
            time,
            trajectory[1,:int(cutoff/self.dt)],
            "C1"
        )

        # PSD
        fig2, (ax2a,ax2b) = plt.subplots(1,2,figsize=(15,5))
        plt.suptitle('PSD: Power Spectral Densities',fontsize=18)
        ax2a.set_xlabel('Frequency',fontsize=14)
        ax2b.set_xlabel('Frequency',fontsize=14)
        ax2a.set_ylabel('Power',fontsize=14)

        freqs_x1,psd_x1 = signal.welch(
            (180/np.pi)*trajectory[0,:]-180,
            1/self.dt
        )
        ax2a.semilogx(freqs_x1,psd_x1,c='C0')
        ax2a.text(
            np.sqrt(ax2a.get_xlim()[1]*ax2a.get_xlim()[0]),
            0.9*np.diff(ax2a.get_ylim())[0] + ax2a.get_ylim()[0],
            "Joint Angle",
            fontsize=18,
            wrap=True,
            horizontalalignment='center',
            verticalalignment='center',
            color = "C0",
            bbox=dict(
                boxstyle='round',
                facecolor='C0',
                lw=0,
                alpha=0.2
            )
        )
        ax2a.spines["top"].set_visible(False)
        ax2a.spines["right"].set_visible(False)

        freqs_S,psd_S = signal.welch(
            trajectory[1,:],
            1/self.dt
        )
        ax2b.semilogx(freqs_S,psd_S,c='C1')
        ax2b.text(
            np.sqrt(ax2b.get_xlim()[1]*ax2b.get_xlim()[0]),
            0.9*np.diff(ax2b.get_ylim())[0] + ax2b.get_ylim()[0],
            "Joint Stiffness",
            fontsize=18,
            wrap=True,
            horizontalalignment='center',
            verticalalignment='center',
            color = "C1",
            bbox=dict(
                boxstyle='round',
                facecolor='C1',
                lw=0,
                alpha=0.2
            )
        )
        ax2b.spines["top"].set_visible(False)
        ax2b.spines["right"].set_visible(False)
        # fig2 = plt.figure(figsize=(5, 4))
        # ax2=plt.gca()
        # plt.title('PSD: power spectral density')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel(r'Power')
        # plt.tight_layout()
        #
        # angle_freqs,angle_psd = signal.welch(
        #     (180/np.pi)*trajectory[0,:],
        #     1/self.dt
        # )
        # ax2.semilogx(angle_freqs,angle_psd,c='C0')
        #
        # stiffness_freqs,stiffness_psd = signal.welch(
        #     trajectory[1,:],
        #     1/self.dt
        # )
        # ax2.semilogx(stiffness_freqs,stiffness_psd,c='C1')
        #
        # ax2.legend(
        #     ["Joint Angle","Joint Stiffness"],
        #     loc="upper right"
        # )

        fig3, (ax3a,ax3b) = plt.subplots(1,2,figsize=(15,7))
        ax3a.set_ylabel('Percentage')
        ax3a.set_xlabel('Joint Angle (deg. w.r.t. vertical)',fontsize=14,color="C0")
        x1_min = np.floor((180/np.pi)*(trajectory[0,:]-np.pi).min()/45)*45
        x1_max = np.ceil((180/np.pi)*(trajectory[0,:]-np.pi).max()/45)*45
        ax3a.set_xticks(list(np.arange(x1_min,x1_max+1,45)))
        ax3a.spines["right"].set_visible(False)
        ax3a.spines["top"].set_visible(False)

        ax3b.set_xlabel('Joint Stiffness (Nm/rad)',fontsize=14,color="C1")
        ax3b.spines["right"].set_visible(False)
        ax3b.spines["top"].set_visible(False)
        sns.distplot(
            (180/np.pi)*trajectory[0,:]-180,
            hist=False,
            color="C0",
            ax=ax3a
        )
        x1bins_min = np.floor((180/np.pi)*(trajectory[0,:]-np.pi).min()/15)*15
        x1bins_max = np.ceil((180/np.pi)*(trajectory[0,:]-np.pi).max()/15)*15
        x1_bins = list(np.arange(x1bins_min,x1bins_max+1,15))

        sns.distplot(
            (180/np.pi)*trajectory[0,:]-180,
            bins=x1_bins,
            hist=True,
            kde=False,
            color="C0",
            hist_kws={
                'weights': np.ones(len(trajectory[0,:]))/len(trajectory[0,:])
            },
            ax=ax3a
        )
        sns.distplot(
            trajectory[1,:],
            hist=False,
            color="C1",
            ax=ax3b
        )

        # sbins_min = np.floor(trajectory[1,:].min()/15)*15
        # sbins_max = np.ceil(trajectory[1,:].max()/15)*15
        # s_bins = list(np.arange(sbins_min,sbins_max+1,15))
        sns.distplot(
            trajectory[1,:],
            bins=len(x1_bins),
            hist=True,
            kde=False,
            color="C1",
            hist_kws={
                'weights': np.ones(len(trajectory[1,:]))/len(trajectory[1,:])
            },
            ax=ax3b
        )
        ax3a.set_yticklabels(["{:.1f}%".format(100*el) for el in ax3a.get_yticks()])
        ax3b.set_yticklabels(["{:.1f}%".format(100*el) for el in ax3b.get_yticks()])

        data1 = np.array([
            (180/np.pi)*trajectory[0,:]-180,
            trajectory[1,:]
        ]).T
        df1 = pd.DataFrame(
            data1,
            columns=[
                "Joint Angle (deg w.r.t. vertical)",
                "Joint Stiffness (Nm/rad)"
            ]
        )
        plot1 = sns.jointplot(
            x="Joint Angle (deg w.r.t. vertical)",
            y="Joint Stiffness (Nm/rad)",
            data=df1,
            kind="kde",
            color="C0"
        )

    def generate_desired_trajectory_SIN_SIN(
            self,
            delay=0.3,
            frequency=1,
            angleRange=None,
            stiffnessRange=None
        ):
        # NOTE: We will generate 20 different training sets, (4 angle ranges over 5 frequencies. To remove the effect of FBL controller on the generalizability, we will choose the last 3 periods as the error of the ANN will be period for a purely period trajectory (10 periods are generated).
        is_number(frequency,"frequency",default=1,notes="Should be positive.")
        if angleRange is not None:
            assert (type(angleRange)==list
                    and len(angleRange)==2
                    and angleRange[1]>angleRange[0]), \
                "angleRange must be a list of length 2 in ascending order."
        else:
            angleRange = [
                self.jointAngleBounds["LB"],
                self.jointAngleBounds["UB"]
            ]
        if stiffnessRange is not None:
            assert (type(stiffnessRange)==list
                    and len(stiffnessRange)==2
                    and stiffnessRange[1]>stiffnessRange[0]), \
                "stiffnessRange must be a list of length 2 in ascending order."
        else:
            stiffnessRange = [
                self.jointStiffnessBounds["LB"],
                self.jointStiffnessBounds["UB"]
            ]

        startValue = np.array([[
            np.mean(angleRange),
            stiffnessRange[0]
        ]])
        tempTime = self.dt*np.array(range(int((delay + 10/frequency)/self.dt)))
        filterDuration = 1/5/2 # (1/5 Hz) / 2

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

        ##  Sinusoidal Joint Stiffness
        stiffnessAmplitude = (stiffnessRange[1] - stiffnessRange[0])/2
        stiffnessOffset = (stiffnessRange[1] + stiffnessRange[0])/2
        sinusoidal_stiffness_func = lambda t:(
            stiffnessOffset
            - stiffnessAmplitude*np.cos(
                2*np.pi*(2*frequency)
                * (t-delay)
            )
        )

        trajectory = startValue.T*np.ones((2,len(tempTime)))
        trajectory[0,int((delay-filterDuration/2)/self.dt):int((delay+filterDuration/2)/self.dt)] =\
            np.array(list(map(
                smooth_transition_angle_func,
                tempTime[int((delay-filterDuration/2)/self.dt):int((delay+filterDuration/2)/self.dt)]
            )))
        trajectory[0,int((delay + filterDuration/2)/self.dt):] = np.array(list(map(
            sinusoidal_angle_func,
            tempTime[int((delay + filterDuration/2)/self.dt):]
        )))

        trajectory[1,int(delay/self.dt):] = np.array(list(map(
            sinusoidal_stiffness_func,
            tempTime[int(delay/self.dt):]
        )))
        return(trajectory)

    def generate_desired_trajectory_SIN_STEP(
            self,
            delay=0.3, # s
            # stepRange=[0.2,1],# 1-5 Hz
            # fixedStep=False,
            frequency=1, # Hz
            angleRange=None,
            stiffnessRange=None,
            numberOfSteps=100,
            filtered=True
        ):
        # NOTE: 3 additional steps have been added to account for any issues with the FBL algorithm. These will be trimmed when the data is saved.
        is_number(numberOfSteps,"numberOfSteps",default=100,notes="Should be an int")
        assert numberOfSteps%1<1e-5,"numberOfSteps should be an int."

        if angleRange is not None:
            assert (type(angleRange)==list
                    and len(angleRange)==2
                    and angleRange[1]>angleRange[0]), \
                "angleRange must be a list of length 2 in ascending order."
        else:
            angleRange = [
                self.jointAngleBounds["LB"],
                self.jointAngleBounds["UB"]
            ]

        # assert (type(stepRange)==list
        #         and len(stepRange)==2
        #         and stepRange[1]>stepRange[0]), \
        #     "stepRange must be a list of length 2 in ascending order."
        #
        # assert type(fixedStep)==bool, "fixedStep must be either True or False (default). If true, the point-to-point step size will be equal to the period of the sinusoid."

        is_number(frequency,"frequency",default=1,notes="Should be positive.")

        if stiffnessRange is not None:
            assert (type(stiffnessRange)==list
                    and len(stiffnessRange)==2
                    and stiffnessRange[1]>stiffnessRange[0]), \
                "stiffnessRange must be a list of length 2 in ascending order."
        else:
            stiffnessRange = [
                self.jointStiffnessBounds["LB"],
                self.jointStiffnessBounds["UB"]
            ]

        assert type(filtered)==bool, "filtered must either be true (default) or false."

        startValue = np.array([[
            np.mean(angleRange),
            stiffnessRange[0]
        ]])

        filterDuration = 1/3/2 # (1/5 Hz) / 2
        filterLength = int(filterDuration/self.dt)

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

        # minStep = stepRange[0]/2 # default: 200 ms / 2 = 100 ms
        # maxStep = stepRange[1]/2 # default: 1000 ms / 2 = 500 ms

        stepDuration = 3/frequency
        stepLength = int(stepDuration/self.dt)
        additionalSteps = 3
        tempTime = np.arange(
            0,
            delay+(numberOfSteps+additionalSteps)*stepDuration+1e-4,
            self.dt
        )
        trajectory = startValue.T*np.ones((2,len(tempTime)))

        ##  Sinusoidal Joint Angle
        trajectory[0,int((delay-filterDuration/2)/self.dt):int((delay+filterDuration/2)/self.dt)] =\
            np.array(list(map(
                smooth_transition_angle_func,
                tempTime[int((delay-filterDuration/2)/self.dt):int((delay+filterDuration/2)/self.dt)]
            )))
        trajectory[0,int((delay + filterDuration/2)/self.dt):] = np.array(list(map(
            sinusoidal_angle_func,
            tempTime[int((delay + filterDuration/2)/self.dt):]
        )))

        ##  Point-to-Point Joint Stiffness
        transitionLength = int((1/2)/self.dt)
        start=int(delay/self.dt)-int(transitionLength/2)
        si = stiffnessRange[0]
        coeffs = [126,-420,540,-315,70]
        # NOTE: by making the entire step equal to 1 second, you can create piecewise functions that do not have to deal with inbetween steps (i.e., the filterDuration is not a multiple of step.dt). Therefore "start" will always be a multiple of self.dt
        for i in range(numberOfSteps+additionalSteps+1):
            sf = np.random.uniform(
                stiffnessRange[0],
                stiffnessRange[1]
            )
            end = start+stepLength
            transition_func = lambda t: (
                si + (sf-si)*(
                    coeffs[0]*((t-self.dt*start)/(self.dt*transitionLength))**5
                    + coeffs[1]*((t-self.dt*start)/(self.dt*transitionLength))**6
                    + coeffs[2]*((t-self.dt*start)/(self.dt*transitionLength))**7
                    + coeffs[3]*((t-self.dt*start)/(self.dt*transitionLength))**8
                    + coeffs[4]*((t-self.dt*start)/(self.dt*transitionLength))**9
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
            start=end
            si=sf
        # start=int(delay/self.dt)
        # for i in range(numberOfSteps+additionalSteps):
        #     end = start+stepLength
        #     trajectory[1,start:end] = [
        #         np.random.uniform(
        #             stiffnessRange[0],
        #             stiffnessRange[1]
        #         )
        #     ]*stepLength
        #     start=end
        # # allDone=False
        # # i=int(delay/self.dt)
        # # while allDone == False:
        # #     if fixedStep==True:
        # #         nextStepLength = int(1/frequency/self.dt)
        # #     else: # nextStepLength is random within a range of frequencies
        # #         nextStepLength = int(
        # #             np.random.uniform(minStep,maxStep)/self.dt
        # #         )
        # #     if nextStepLength<len(self.time)-i-1:
        # #         trajectory[1,i:i+nextStepLength] = [
        # #             np.random.uniform(
        # #                 stiffnessRange[0],
        # #                 stiffnessRange[1]
        # #             )
        # #         ]*nextStepLength
        # #         i+=nextStepLength
        # #     else:
        # #         trajectory[1,i:] = [
        # #             np.random.uniform(
        # #                 stiffnessRange[0],
        # #                 stiffnessRange[1]
        # #             )
        # #         ]*(len(self.time)-i)
        # #         allDone=True
        # if filtered:
        #     filterLength = int((1/5)/self.dt/2)
        #     b=np.ones(filterLength,)/(filterLength) #Finite Impulse Response (FIR) Moving Average (MA) filter with filter length (5 Hz^{-1} / dt) / 2
        #     a=1
        #     trajectory[1,:] = signal.filtfilt(b, a, trajectory[1,:])

        return(trajectory)

    def generate_desired_trajectory_STEP_SIN(
            self,
            delay=0.3,
            frequency=1,
            stiffnessRange=None,
            angleRange=None,
            numberOfSteps=100
        ):
        # NOTE: An additional 5 steps have been added that will be removed afterwards to hopefully remove any artifacts from the FBL algorithm.
        is_number(numberOfSteps,"numberOfSteps",default=100,notes="Should be an int")
        assert numberOfSteps%1<1e-5,"numberOfSteps should be an int."

        if angleRange is not None:
            assert (type(angleRange)==list
                    and len(angleRange)==2
                    and angleRange[1]>angleRange[0]), \
                "angleRange must be a list of length 2 in ascending order."
        else:
            angleRange = [
                self.jointAngleBounds["LB"],
                self.jointAngleBounds["UB"]
            ]

        if stiffnessRange is not None:
            assert (type(stiffnessRange)==list
                    and len(stiffnessRange)==2
                    and stiffnessRange[1]>stiffnessRange[0]), \
                "stiffnessRange must be a list of length 2 in ascending order."
        else:
            stiffnessRange = [
                self.jointStiffnessBounds["LB"],
                self.jointStiffnessBounds["UB"]
            ]

        startValue = np.array([[
            np.mean(angleRange),
            stiffnessRange[0]
        ]])

        stepDuration = 3/frequency # three periods
        stepLength = int(stepDuration/self.dt)
        additionalSteps = 5
        tempTime = np.arange(
            0,
            delay+(numberOfSteps+additionalSteps)*stepDuration+1e-4,
            self.dt
        )
        trajectory = startValue.T*np.ones((2,len(tempTime)))

        ## Point-to-Point Joint Angle
        # NOTE: Step duration is equal to the period of the sinusoid for fixed joint angles while modulating joint stiffess

        transitionLength = int((1/2)/self.dt)
        start=int(delay/self.dt)-int(transitionLength/2)
        x1i = np.mean(angleRange)
        coeffs = [126,-420,540,-315,70]
        # NOTE: by making the entire step equal to 1 second, you can create piecewise functions that do not have to deal with inbetween steps (i.e., the filterDuration is not a multiple of step.dt). Therefore "start" will always be a multiple of self.dt
        for i in range(numberOfSteps+additionalSteps+1):
            x1f = np.random.uniform(
                angleRange[0],
                angleRange[1]
            )
            end = start+stepLength
            transition_func = lambda t: (
                x1i + (x1f-x1i)*(
                    coeffs[0]*((t-self.dt*start)/(self.dt*transitionLength))**5
                    + coeffs[1]*((t-self.dt*start)/(self.dt*transitionLength))**6
                    + coeffs[2]*((t-self.dt*start)/(self.dt*transitionLength))**7
                    + coeffs[3]*((t-self.dt*start)/(self.dt*transitionLength))**8
                    + coeffs[4]*((t-self.dt*start)/(self.dt*transitionLength))**9
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
            start=end
            x1i=x1f

        # start=int(delay/self.dt)
        # for i in range(numberOfSteps+additionalSteps):
        #     end = start+stepLength
        #     trajectory[0,start:end] = [
        #         np.random.uniform(
        #             angleRange[0],
        #             angleRange[1]
        #         )
        #     ]*stepLength
        #     start=end
        # filterLength = int((1/5)/self.dt/2)
        # b=np.ones(filterLength,)/(filterLength) #Finite Impulse Response (FIR) Moving Average (MA) filter with filter length (5 Hz^{-1} / dt) / 2
        # a=1
        # trajectory[0,:] = signal.filtfilt(b, a, trajectory[0,:])

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
        trajectory[1,int(delay/self.dt):] = np.array(list(map(
            sinusoidal_stiffness_func,
            tempTime[int(delay/self.dt):]
        )))

        return(trajectory)

    def generate_desired_trajectory_STEP_STEP(
            self,
            delay=0.3,
            numberOfSteps=100,
            stepDuration=1.0,
            angleRange=None,
            stiffnessRange=None
        ):
        # NOTE: 5 additional steps have been added. These will be removed when the data is saved for testing.
        is_number(numberOfSteps,"numberOfSteps",default=100,notes="Should be an int")
        assert numberOfSteps%1<1e-5,"numberOfSteps should be an int."

        is_number(stepDuration,"stepDuration",default=0.3,notes="(in seconds).")
        assert abs((stepDuration*10000)%(self.dt*10000))<1e-5, "stepDuration should be a multiple of the timestep (self.dt = " + str(self.dt) + ")."

        if angleRange is not None:
            assert (type(angleRange)==list
                    and len(angleRange)==2
                    and angleRange[1]>angleRange[0]), \
                "angleRange must be a list of length 2 in ascending order."
        else:
            angleRange = [
                self.jointAngleBounds["LB"],
                self.jointAngleBounds["UB"]
            ]

        if stiffnessRange is not None:
            assert (type(stiffnessRange)==list
                    and len(stiffnessRange)==2
                    and stiffnessRange[1]>stiffnessRange[0]), \
                "stiffnessRange must be a list of length 2 in ascending order."
        else:
            stiffnessRange = [
                self.jointStiffnessBounds["LB"],
                self.jointStiffnessBounds["UB"]
            ]

        stepLength = int(stepDuration/self.dt)
        additionalSteps = 5
        tempTime = np.arange(
            0,
            delay+(numberOfSteps+additionalSteps)*stepDuration+1e-4,
            self.dt
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
        transitionLength = int((1/2)/self.dt)
        start=int(delay/self.dt)-int(transitionLength/2)
        Xi = startValue.T
        coeffs = [126,-420,540,-315,70]
        # NOTE: by making the entire step equal to 1 second, you can create piecewise functions that do not have to deal with inbetween steps (i.e., the filterDuration is not a multiple of step.dt). Therefore "start" will always be a multiple of self.dt
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
            transition_func = lambda t: (
                Xi + (Xf-Xi)*(
                    coeffs[0]*((t-self.dt*start)/(self.dt*transitionLength))**5
                    + coeffs[1]*((t-self.dt*start)/(self.dt*transitionLength))**6
                    + coeffs[2]*((t-self.dt*start)/(self.dt*transitionLength))**7
                    + coeffs[3]*((t-self.dt*start)/(self.dt*transitionLength))**8
                    + coeffs[4]*((t-self.dt*start)/(self.dt*transitionLength))**9
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
                pass
            start=end
            Xi=Xf

        # filterLength = int((1/3)/self.dt/2)
        # b=np.ones(filterLength,)/(filterLength) #Finite Impulse Response (FIR) Moving Average (MA) filter with filter length (3 Hz^{-1} / dt) / 2
        # a=1
        # # OLDtrajectory = signal.filtfilt(b, a, OLDtrajectory)
        # trajectory = signal.filtfilt(b,a,trajectory)
        # start=int(delay/self.dt)
        # for i in range(numberOfSteps+additionalSteps):
        #     end = start+stepLength
        #     trajectory[0,start:end] = [
        #         np.random.uniform(
        #             minValue[0],
        #             maxValue[0]
        #         )
        #     ]*stepLength
        #     trajectory[1,start:end] = [
        #         np.random.uniform(
        #             minValue[1],
        #             maxValue[1]
        #         )
        #     ]*stepLength
        #     start=end
        # filterLength = int((1/3)/self.dt/2)
        # b=np.ones(filterLength,)/(filterLength) #Finite Impulse Response (FIR) Moving Average (MA) filter with filter length (3 Hz^{-1} / dt) / 2
        # a=1
        # trajectory = signal.filtfilt(b, a, trajectory)
        return(trajectory)

    def forward_simulation_FL(self,X_o,X1d,Sd):
        assert len(X_o)==6, "X_o must have 6 elements, not " + str(len(X_o)) + "."
        dt = self.time[1]-self.time[0]
        U = np.zeros((2,np.shape(X1d)[1]-1),dtype=np.float64)
        X = np.zeros((6,np.shape(X1d)[1]),dtype=np.float64)
        X_measured = np.zeros((6,np.shape(X1d)[1]),dtype=np.float64)
        Y = np.zeros((2,np.shape(X1d)[1]),dtype=np.float64)
        X[:,0] = X_o
        Y[:,0] = self.h(X[:,0])
        self.update_state_variables(X_o)
        statusbar=dsb(0,np.shape(X1d)[1]-1,title="Forward Simulation (FBL)")
        self.desiredOutput = np.array([X1d[0,:],Sd[0,:]])
        for i in range(np.shape(X1d)[1]-1):
            if i>0:
                X_measured[0,i] = X[0,i]
                X_measured[1,i] = (X[0,i]-X[0,i-1])/self.dt
                X_measured[2,i] = X[2,i]
                X_measured[3,i] = (X[2,i]-X[2,i-1])/self.dt
                X_measured[4,i] = X[4,i]
                X_measured[5,i] = (X[4,i]-X[4,i-1])/self.dt
            else:
                X_measured[:,i] = X[:,i]
            U[:,i] = (self.return_input(X[:,i],X1d[:,i],Sd[:,i])).flatten()

            # X[0,i+1] = X[0,i] + self.dt*self.f1_func(X[:,i])
            # X[1,i+1] = X[1,i] + self.dt*self.f2_func(X[:,i])
            # if X[0,i+1]<=self.jointAngleBounds["LB"]:
            #     X[0,i+1] = self.jointAngleBounds["LB"]
            #     if X[1,i+1]<0: X[1,i+1] = 0
            #     # X[1,i+1] = X[1,i]
            # elif X[0,i+1]>=self.jointAngleBounds["UB"]:
            #     X[0,i+1] = self.jointAngleBounds["UB"]
            #     if X[1,i+1]>0: X[1,i+1] = 0
            #     # X[1,i+1] = X[1,i]
            # # X[1,i+1] = X[1,i] + self.dt*self.f2_func(X[:,i])
            # X[2,i+1] = X[2,i] + self.dt*self.f3_func(X[:,i])
            # X[3,i+1] = X[3,i] + self.dt*(self.f4_func(X[:,i]) + U[0,i]/self.Jm)
            # X[4,i+1] = X[4,i] + self.dt*self.f5_func(X[:,i])
            # X[5,i+1] = X[5,i] + self.dt*(self.f6_func(X[:,i]) + U[1,i]/self.Jm)

            X[:,i+1] = (
                X[:,i]
                + self.dt*(
                    self.f(X[:,i])
                    + self.g(X[:,i])@U[:,np.newaxis,i]
                ).T
            )
            Y[:,i+1] = self.h(X[:,i+1])
            self.update_state_variables(X[:,i+1])
            statusbar.update(i)
        return(X,U,Y,X_measured)

    def plot_tendon_tension_deformation_curves(self,X,
            returnValues=False,
            addTitle=None):

        assert type(returnValues)==bool, "returnValues must be either True or False (default)."

        if addTitle is None:
            addTitle = ""
        else:
            assert type(addTitle)==str, "addTitle can either be None (default) or a string."
            addTitle = "\n " + addTitle

        tendonTension1 = np.array(list(map(self.tendon_1_FL_func,X.T)))
        tendonDeformation1 = np.array([-self.rj,0,self.rm,0,0,0])@X
        tendonTension2= np.array(list(map(self.tendon_2_FL_func,X.T)))
        tendonDeformation2 = np.array([self.rj,0,0,0,self.rm,0])@X

        tempTime = self.dt*np.array(list(range(len(tendonTension1))))

        minimumDeformation = min([
            tendonDeformation1.min(),
            tendonDeformation2.min()
        ])
        maximumDeformation = max([
            tendonDeformation1.max(),
            tendonDeformation2.max()
        ])
        deformationRange = maximumDeformation - minimumDeformation
        deformationArray = np.linspace(
            0,
            maximumDeformation + 0.1*deformationRange,
            1001
        )
        actualForceLengthCurve = np.array(list(map(
            lambda x3: self.tendon_1_FL_func([0,0,x3/self.rm,0,0,0]),
            deformationArray
        )))

        fig, ((ax1,ax3),(ax2,ax4)) = plt.subplots(
            2,2,figsize=(10,8),sharex=True)
        # ax1 = fig.add_subplot(221) # FL 1
        # ax2 = fig.add_subplot(223) # self.time v Deformation 1
        # ax3 = fig.add_subplot(222) # FL 2
        # ax4 = fig.add_subplot(224) # self.time v Deformation 2
        plt.suptitle("Tendon Deformation vs. Tension" + addTitle)
        tAxes = [[ax1,ax2],[ax3,ax4]]
        tendonDeformation = [tendonDeformation1,tendonDeformation2]
        tendonTension = [tendonTension1,tendonTension2]
        colors = ["C0","C0"]
        for i in range(2):
            tAxes[i][0].plot(np.linspace(-1,0,1001),np.zeros((1001,)),'0.70')
            tAxes[i][0].plot(deformationArray,actualForceLengthCurve,'0.70')
            tAxes[i][0].plot(tendonDeformation[i],tendonTension[i],c=colors[i])
            tAxes[i][0].set_xlim([
                minimumDeformation - 0.1*deformationRange,
                maximumDeformation + 0.1*deformationRange
            ])
            tAxes[i][0].set_xlabel("Tendon Deformation (m)")
            tAxes[i][0].set_ylabel("Tendon " + str(i+1) + " Tension (N)")
            tAxes[i][0].spines['right'].set_visible(False)
            tAxes[i][0].spines['top'].set_visible(False)

            tAxes[i][1].plot(tendonDeformation[i],-tempTime,c=colors[i])
            tAxes[i][1].set_ylabel("Time (s)")
            tAxes[i][1].set_xlim([
                minimumDeformation - 0.1*deformationRange,
                maximumDeformation + 0.1*deformationRange
            ])
            plt.setp(tAxes[i][0].get_xticklabels(), visible=False)
            # tAxes[i][0].set_xticklabels([
            #     "" for tick in tAxes[i][0].get_xticks()
            # ])
            tAxes[i][1].set_yticks([-tempTime[0],-tempTime[-1]])
            tAxes[i][1].set_yticklabels([tempTime[0],tempTime[-1]])
            tAxes[i][1].xaxis.tick_top()
            tAxes[i][1].spines['right'].set_visible(False)
            tAxes[i][1].spines['bottom'].set_visible(False)

        if returnValues==True:
            return(tendonDeformation,tendonTension)

    def plot_motor_angles(self,X):
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(self.time,X[2,:]*180/np.pi,'C0')
        ax.plot(self.time,X[4,:]*180/np.pi,'C0',ls='--')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Motor Angles (deg)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.legend(["Motor 1","Motor 2"],loc="upper right")

    def plot_states_and_inputs(self,X,U,returnFig=False,**kwargs):
        import numpy as np
        import matplotlib.pyplot as plt

        inputString = kwargs.get("inputString",None)
        assert inputString is None or type(inputString)==str, "InputString must either be None or a str."

        tempTime = np.array(range(np.shape(X)[1]))*self.dt

        fig, axes = plt.subplots(5,1,figsize=(12,20),sharex=True)
        if inputString is not None:
            fig.suptitle(inputString,fontsize=18)
        for i in range(len(axes)):
            axes[i].spines["top"].set_visible(False)
            if i<len(axes)-1:
                plt.setp(axes[i].get_xticklabels(), visible=False)

        ax1b = axes[0].twinx()
        ax2b = axes[1].twinx()
        ax3b = axes[2].twinx()
        axes_rt = [ax1b,ax2b,ax3b]
        for ax in axes_rt:
            ax.spines['top'].set_visible(False)

        axes[0].plot(tempTime,180*X[0,:]/np.pi - 180,"k")
        ax1b.plot(tempTime,180*X[1,:]/np.pi,"k",linestyle=":")
        # axes[0].set_title("Pendulum")
        axes[0].text(
            -0.16667*tempTime[-1],np.mean(axes[0].get_ylim()),
            "Pendulum",
            fontsize=14,
            wrap=True,
            horizontalalignment='center',
            verticalalignment='center',
            color = "k",
            bbox=dict(
                boxstyle='round',
                facecolor='k',
                lw=0,
                alpha=0.2
            ),
            rotation=90
        )
        axes[0].set_ylabel("Angle (deg.)")
        ax1b.set_ylabel("Angular Velocity\n(deg./s)")

        axes[1].plot(tempTime,180*X[2,:]/np.pi - 180,"r")
        ax2b.plot(tempTime,180*X[3,:]/np.pi,"r",linestyle=":")
        axes[1].text(
            -0.16667*tempTime[-1],np.mean(axes[1].get_ylim()),
            "Motor 1",
            fontsize=14,
            wrap=True,
            horizontalalignment='center',
            verticalalignment='center',
            color = "r",
            bbox=dict(
                boxstyle='round',
                facecolor='r',
                lw=0,
                alpha=0.2
            ),
            rotation=90
        )
        axes[1].set_ylabel("Angle (deg.)")
        ax2b.set_ylabel("Angular Velocity\n(deg./s)")

        axes[2].plot(tempTime,180*X[4,:]/np.pi - 180,"b")
        ax3b.plot(tempTime,180*X[5,:]/np.pi,"b",linestyle=":")
        axes[2].text(
            -0.16667*tempTime[-1],np.mean(axes[2].get_ylim()),
            "Motor 1",
            fontsize=14,
            wrap=True,
            horizontalalignment='center',
            verticalalignment='center',
            color = "b",
            bbox=dict(
                boxstyle='round',
                facecolor='b',
                lw=0,
                alpha=0.2
            ),
            rotation=90
        )
        axes[2].set_ylabel("Angle (deg.)")
        ax3b.set_ylabel("Angular Velocity\n(deg./s)")

        fT1 = np.array(list(map(self.tendon_1_FL_func,X.T)))
        fT2 = np.array(list(map(self.tendon_2_FL_func,X.T)))
        axes[3].plot(tempTime,fT1,'r',tempTime,fT2,'b')
        axes[3].spines["right"].set_visible(False)
        axes[3].text(
            -0.17*tempTime[-1],np.mean(axes[3].get_ylim()),
            "Tendon\nTensions",
            fontsize=14,
            wrap=True,
            horizontalalignment='center',
            verticalalignment='center',
            color = "k",
            bbox=dict(
                boxstyle='round',
                facecolor='k',
                lw=0,
                alpha=0.2
            ),
            rotation=90
        )
        axes[3].set_ylabel("Tension (N)")

        axes[4].plot(tempTime[:-1],U[0,:],'r',tempTime[:-1],U[1,:],'b')
        axes[4].spines["right"].set_visible(False)
        # axes[4].set_title("Inputs")
        axes[4].text(
            -0.1667*tempTime[-1],np.mean(axes[4].get_ylim()),
            "Inputs",
            fontsize=14,
            wrap=True,
            horizontalalignment='center',
            verticalalignment='center',
            color = "k",
            bbox=dict(
                boxstyle='round',
                facecolor='k',
                lw=0,
                alpha=0.2
            ),
            rotation=90
        )
        axes[4].set_ylabel("Motor Torques (Nm)")
        axes[4].set_xlabel("Time (s)")

        if returnFig==False:
            plt.show()
        else:
            return(fig)

    def plot_states(self,X,**kwargs):
        """
        Take the numpy.ndarray for the state space (X) of shape (M,N), where M is the number of states and N is the same length as time t. Returns a plot of the states.

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        **kwargs
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        1) Return - must be a bool. Determines if the function returns a function handle. Default is False.

        2) InputString - must be a string. Input to the DescriptiveTitle that can be used to personalize the title. Default is None.

        """
        import numpy as np
        import matplotlib.pyplot as plt
        tempTime = np.array(range(np.shape(X)[1]))*self.dt
        # assert (np.shape(X)[0] in [6,8]) \
        #             and (np.shape(X)[1] == len(self.time)) \
        #                 and (str(type(X)) == "<class 'numpy.ndarray'>"), \
        #         "X must be a (6,N) or (8,N) numpy.ndarray, where N is the length of t."


        Return = kwargs.get("Return",False)
        assert type(Return)==bool, "Return must be either True or False."

        InputString = kwargs.get("InputString",None)
        assert InputString is None or type(InputString)==str, "InputString must either be None or a str."

        NumStates = np.shape(X)[0]
        X[:6,:] = 180*X[:6,:]/np.pi # converting to deg and deg/s
        X[0,:] -= 180 # centering joint angle at 0 deg.
        if NumStates == 6:
            NumColumns = 2
            NumRows = 3
        else:
            NumColumns = 4
            NumRows = 2

        ColumnNumber = [el%2 for el in np.arange(0,NumStates,1)]
        RowNumber = [int(el/2) for el in np.arange(0,NumStates,1)]
        Units = [
            "(Deg)","(Deg/s)",
            "(Deg)","(Deg/s)",
            "(Deg)","(Deg/s)",
            "(N)","(N)"]
        if InputString is None:
            DescriptiveTitle = "Plotting States vs. Time"
        else:
            assert type(InputString)==str, "InputString must be a string"
            DescriptiveTitle = InputString + " Driven"
        if NumRows == 1:
            FigShape = (NumColumns,)
        else:
            FigShape = (NumRows,NumColumns)
        Figure = kwargs.get("Figure",None)
        assert (Figure is None) or \
                    (    (type(Figure)==tuple) and \
                        (str(type(Figure[0]))=="<class 'matplotlib.figure.Figure'>") and\
                        (np.array([str(type(ax))=="<class 'matplotlib.axes._subplots.AxesSubplot'>" \
                            for ax in Figure[1].flatten()]).all()) and \
                        (Figure[1].shape == FigShape)\
                    ),\
                         ("Figure can either be left blank (None) or it must be constructed from data that has the same shape as X.\ntype(Figure) = " + str(type(Figure)) + "\ntype(Figure[0]) = " + str(type(Figure[0])) + "\nFigure[1].shape = " + str(Figure[1].shape) + " instead of (" + str(NumRows) + "," + str(NumColumns) + ")" + "\ntype(Figure[1].flatten()[0]) = " + str(type(Figure[1].flatten()[0])))
        if Figure is None:
            fig, axes = plt.subplots(NumRows,NumColumns,figsize=(3.5*NumColumns,2*NumRows + 2),sharex=True)
            plt.subplots_adjust(top=0.85,bottom=0.15,left=0.075,right=0.975)
            plt.suptitle(DescriptiveTitle,Fontsize=20,y=0.975)
            for j in range(NumStates):
                axes[RowNumber[j],ColumnNumber[j]].spines['right'].set_visible(False)
                axes[RowNumber[j],ColumnNumber[j]].spines['top'].set_visible(False)
                axes[RowNumber[j],ColumnNumber[j]].plot(tempTime,X[j,:])
                if not(RowNumber[j] == RowNumber[-1] and ColumnNumber[j]==0):
                    plt.setp(axes[RowNumber[j],ColumnNumber[j]].get_xticklabels(), visible=False)
                    # axes[RowNumber[j],ColumnNumber[j]].set_xticklabels(\
                    #                     [""]*len(axes[RowNumber[j],ColumnNumber[j]].get_xticks()))
                else:
                    axes[RowNumber[j],ColumnNumber[j]].set_xlabel("Time (s)")
                axes[RowNumber[j],ColumnNumber[j]].set_title(r"$x_{" + str(j+1) + "}$ "+ Units[j])
                # if NumStates%5!=0:
                #     [fig.delaxes(axes[RowNumber[-1],el]) for el in range(ColumnNumber[-1]+1,5)]
        else:
            fig = Figure[0]
            axes = Figure[1]
            for i in range(NumStates):
                if NumRows != 1:
                    axes[RowNumber[i],ColumnNumber[i]].plot(tempTime,X[i,:])
                else:
                    axes[ColumnNumber[i]].plot(tempTime,X[i,:])
        X[0,:] += 180 # returning to original frame
        X[:6,:] = np.pi*X[:6,:]/180 # returning to radians
        if Return == True:
            return((fig,axes))
        else:
            plt.show()

    def plot_output_power_spectrum_and_distribution(self,X,**kwargs):
        returnFigs = kwargs.get("returnFigs",False)
        assert type(returnFigs)==bool, "returnFigs must be either true or false (default)."

        S = np.array(list(map(self.hs,X.T)))

        # PSD
        fig1, (ax1a,ax1b) = plt.subplots(1,2,figsize=(15,5))
        plt.suptitle('PSD: Power Spectral Densities',fontsize=18)
        ax1a.set_xlabel('Frequency',fontsize=14)
        ax1b.set_xlabel('Frequency',fontsize=14)
        ax1a.set_ylabel('Power',fontsize=14)

        freqs_x1,psd_x1 = signal.welch(
            X[0,:],
            1/self.dt
        )
        ax1a.semilogx(freqs_x1,psd_x1,c='C0')
        ax1a.text(
            np.sqrt(ax1a.get_xlim()[1]*ax1a.get_xlim()[0]),
            0.9*np.diff(ax1a.get_ylim())[0] + ax1a.get_ylim()[0],
            "Joint Angle",
            fontsize=18,
            wrap=True,
            horizontalalignment='center',
            verticalalignment='center',
            color = "C0",
            bbox=dict(
                boxstyle='round',
                facecolor='C0',
                lw=0,
                alpha=0.2
            )
        )
        ax1a.spines["top"].set_visible(False)
        ax1a.spines["right"].set_visible(False)

        freqs_S,psd_S = signal.welch(
            S,
            1/self.dt
        )
        ax1b.semilogx(freqs_S,psd_S,c='C1')
        ax1b.text(
            np.sqrt(ax1b.get_xlim()[1]*ax1b.get_xlim()[0]),
            0.9*np.diff(ax1b.get_ylim())[0] + ax1b.get_ylim()[0],
            "Joint Stiffness",
            fontsize=18,
            wrap=True,
            horizontalalignment='center',
            verticalalignment='center',
            color = "C1",
            bbox=dict(
                boxstyle='round',
                facecolor='C1',
                lw=0,
                alpha=0.2
            )
        )
        ax1b.spines["top"].set_visible(False)
        ax1b.spines["right"].set_visible(False)

        fig2, (ax2a,ax2b) = plt.subplots(1,2,figsize=(15,5))
        ax2a.set_ylabel("Percentage",fontsize=14)
        ax2a.set_xlabel('Joint Angle (deg.)',fontsize=14,color="C0")
        ax2a.spines["right"].set_visible(False)
        ax2a.spines["top"].set_visible(False)

        X[0,:] -= np.pi # shifting vertical position to 0 rad.
        hist,bin_edges=np.histogram(X[0,:],bins=48)
        percentNearBoundaries = (hist[0]+hist[-1])/len(X[0,:])*100 # percent within 180/48 = 3.75 deg of the boundaries.
        _,_,_ = ax2a.hist(
            x=X[0,:]*(180/np.pi),
            bins=12,
            color='C0',
            alpha=0.7,
            weights=np.ones(len(X[0,:]*(180/np.pi))) / len(X[0,:]*(180/np.pi))
        )
        _,yMax = ax2a.get_ylim()
        ax2a.text(
            0,0.9*yMax,
            "{:.2f}".format(percentNearBoundaries) + "%" + " of Data\n" + r"$<3.75^\circ$ from Boundaries",
            fontsize=14,
            wrap=True,
            horizontalalignment='center',
            verticalalignment='center',
            color = "C0",
            bbox=dict(
                boxstyle='round',
                facecolor='C0',
                lw=0,
                alpha=0.2
            )
        )
        ax2a.set_xticks([-90,-45,0,45,90])
        ax2a.set_xticklabels([str(int(tick))+r"$^\circ$" for tick in ax2a.get_xticks()])
        ax2a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))

        ax2b.set_xlabel('Joint Stiffness (Nm/rad)',fontsize=14,color="C1")
        ax2b.spines["right"].set_visible(False)
        ax2b.spines["top"].set_visible(False)

        hist,bin_edges=np.histogram(S,bins=48)
        _,_,_ = ax2b.hist(
            x=S,
            bins=12,
            color='C1',
            alpha=0.7,
            weights=np.ones(len(S))/len(S)
        )
        # ax2b.set_xticks(np.arange(10,51,10))
        ax2b.set_xticklabels([int(tick) for tick in ax2b.get_xticks()])
        ax2b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))

        data1 = np.array([
            (180/np.pi)*X[0,:],
            (180/np.pi)*X[1,:]
        ]).T
        df1 = pd.DataFrame(
            data1,
            columns=[
                "Joint Angle (deg)",
                "Joint Angular Velocity (deg/s)"
            ]
        )
        plot1 =sns.jointplot(
            x="Joint Angle (deg)",
            y="Joint Angular Velocity (deg/s)",
            data=df1,
            kind="kde",
            color="C0"
        )

        data2 = np.array([
            (180/np.pi)*X[0,:],
            S
        ]).T
        df2 = pd.DataFrame(
            data2,
            columns=[
                "Joint Angle (deg w.r.t. vertical)",
                "Joint Stiffness (Nm/rad)"
            ]
        )
        plot2 = sns.jointplot(
            x="Joint Angle (deg w.r.t. vertical)",
            y="Joint Stiffness (Nm/rad)",
            data=df2,
            kind="kde",
            color="C0"
        )
        plot2.ax_marg_x.set_xlim(-100, 100)
        plot2.ax_marg_y.set_ylim(self.jointStiffnessBounds['LB'], self.jointStiffnessBounds['UB'])
        X[0,:] += np.pi # shifting back

        if returnFigs==False:
            plt.show()
        else:
            return([fig1,fig2,plot1,plot2])

    def save_data(self,X,U,additionalDict=None,filePath=None):
        fT1 = np.array(list(map(self.tendon_1_FL_func,X.T)))
        fT2 = np.array(list(map(self.tendon_2_FL_func,X.T)))
        outputData = {
            "Time" : self.dt*np.array(list(range(len(U[0,:])))),
            "u1" : U[0,:],
            "du1" : np.gradient(U[0,:],self.dt),
            "u2" : U[1,:],
            "du2" : np.gradient(U[1,:],self.dt),
            "x1" : X[0,:],
            "dx1" : X[1,:],
            "d2x1" : np.gradient(X[1,:],self.dt),
            "x3" : X[2,:],
            "dx3" : X[3,:],
            "d2x3" : np.gradient(X[3,:],self.dt),
            "x5" : X[4,:],
            "dx5" : X[5,:],
            "d2x5" : np.gradient(X[5,:],self.dt),
            "fT1" : fT1,
            "dfT1" : np.gradient(fT1,self.dt),
            "d2fT1" : np.gradient(np.gradient(fT1,self.dt),self.dt),
            "fT2" : fT2,
            "dfT2" : np.gradient(fT2,self.dt),
            "d2fT2" : np.gradient(np.gradient(fT2,self.dt),self.dt)
        }
        # outputData = {
        #     "Time" : self.dt*np.array(list(range(len(U[0,:])))),
        #     "u1" : U[0,:],
        #     "du1" : np.gradient(U[0,:],self.dt),
        #     "u2" : U[1,:],
        #     "du2" : np.gradient(U[1,:],self.dt),
        #     "x1" : X[0,:],
        #     "dx1" : X[1,:],
        #     "d2x1" : np.gradient(X[1,:],self.dt),
        #     "x3" : X[2,:],
        #     "dx3" : X[3,:],
        #     "d2x3" : savgol_filter(np.gradient(X[3,:],self.dt),11,3),
        #     "x5" : X[4,:],
        #     "dx5" : X[5,:],
        #     "d2x5" : savgol_filter(np.gradient(X[5,:],self.dt),11,3),
        #     "fT1" : fT1,
        #     "dfT1" : np.gradient(fT1,self.dt),
        #     "d2fT1" : savgol_filter(np.gradient(np.gradient(fT1,self.dt),self.dt),11,3),
        #     "fT2" : fT2,
        #     "dfT2" : np.gradient(fT2,self.dt),
        #     "d2fT2" : savgol_filter(np.gradient(np.gradient(fT2,self.dt),self.dt),11,3)
        # }

        if additionalDict is not None:
            outputData.update(additionalDict)

        if filePath is not None:
            assert type(filePath)==str, "filePath must be a str."
            assert filePath[-4:]==".mat", "filePath must end in .mat"
            sio.savemat(filePath,outputData)
        else:
            sio.savemat("outputData.mat",outputData)

def sweep_plant():
    plantParams["dt"]=0.01
    plantParams["Stage Duration"] = 10
    plantParams["Number of Stiffness Stages"] = 100
    plantParams["Number of Angle Stages"] = 100
    plantParams["Simulation Duration"] = plantParams["Stage Duration"]*(
        plantParams["Number of Stiffness Stages"]
        + plantParams["Number of Angle Stages"]
    )
    plant = plant_pendulum_1DOF2DOF(plantParams)

    stiffnessMinimum = 50
    stiffnessMaximum = 150
    angleMinimum = np.pi/2 + np.pi/9
    angleMaximum = 3*np.pi/2 - np.pi/9
    x1o = np.pi
    X_o = [x1o,0,plant.rj*x1o/plant.rm,0,-plant.rj*x1o/plant.rm,0]
    plantParams["X_o"] = X_o

    # X,U,Y = plant.forward_simulation(X_o)

    desiredAngle = np.zeros((5,len(plant.time)))
    desiredStiffness = np.zeros((3,len(plant.time)))
    timeBreaks = np.array(list(range(
        0,len(plant.time),int(plantParams["Stage Duration"]/plantParams["dt"]
    ))))
    breakDuration = int(plantParams["Stage Duration"]/plantParams["dt"])

    angleSweep = np.concatenate([
        (angleMaximum-np.pi)*np.linspace(0,1,int(breakDuration/2)) + np.pi,
        (angleMinimum-np.pi)*np.linspace(0,1,breakDuration -int(breakDuration/2)) + np.pi
    ])
    constantAngleValues = np.linspace(angleMinimum,angleMaximum,plantParams["Number of Angle Stages"])
    stiffnessSweep = (stiffnessMaximum-stiffnessMinimum)*np.linspace(0,1,breakDuration) + stiffnessMinimum
    constantStiffnessValues = np.linspace(stiffnessMinimum,stiffnessMaximum,plantParams["Number of Stiffness Stages"])
    # Sweep Angles at different stiffness values
    for i in range(plantParams["Number of Stiffness Stages"]):
        desiredAngle[0,timeBreaks[i]:timeBreaks[i+1]] = angleSweep
        desiredStiffness[0,timeBreaks[i]:timeBreaks[i+1]] = constantStiffnessValues[i]*np.ones(breakDuration)
    for i in range(plantParams["Number of Angle Stages"]):
        j = plantParams["Number of Stiffness Stages"] + i
        desiredAngle[0,timeBreaks[j]:timeBreaks[j+1]] = constantAngleValues[i]*np.ones(breakDuration)
        desiredStiffness[0,timeBreaks[j]:timeBreaks[j+1]] = stiffnessSweep
    desiredAngle[0,-1]=desiredAngle[0,-2]
    desiredStiffness[0,-1]=desiredStiffness[0,-2]

    desiredAngle[0,:] = LP_filt(100, desiredAngle[0,:])
    desiredAngle[1,:] = np.gradient(desiredAngle[0,:],plantParams["dt"])
    desiredAngle[2,:] = np.gradient(desiredAngle[1,:],plantParams["dt"])
    desiredAngle[3,:] = np.gradient(desiredAngle[2,:],plantParams["dt"])
    desiredAngle[4,:] = np.gradient(desiredAngle[3,:],plantParams["dt"])

    desiredStiffness[0,:] = LP_filt(100, desiredStiffness[0,:])
    desiredStiffness[1,:] = np.gradient(desiredStiffness[0,:],plantParams["dt"])
    desiredStiffness[2,:] = np.gradient(desiredStiffness[1,:],plantParams["dt"])

    X_FBL,U_FBL,Y_FBL,X_measured = plant.forward_simulation_FL(X_o,desiredAngle,desiredStiffness)
    fig1 = plt.figure(figsize=(10,8))
    ax1=plt.gca()

    ax1.plot(plant.time,(180/np.pi)*Y_FBL[0,:].T,c="C0")
    ax1.plot(plant.time,(180/np.pi)*desiredAngle[0,:],c="C0",linestyle="--")
    ax1.set_title(r"$-$ Actual; --- Desired", fontsize=16)
    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="C0")
    ax1.set_ylabel('Position (deg.)', color="C0")
    # y1_min = np.floor((Y_FBL[0,:].min()*180/np.pi)/22.5)*22.5
    # y1_min = min([y1_min,np.floor((X1d[0,:].min()*180/np.pi)/22.5)*22.5])
    # y1_max = np.ceil((Y_FBL[0,:].max()*180/np.pi)/22.5)*22.5
    # y1_max = max([y1_max,np.ceil((X1d[0,:].max()*180/np.pi)/22.5)*22.5])
    y1_min = 0
    y1_max = 360
    yticks = np.arange(y1_min,y1_max+22.5,22.5)
    yticklabels = []
    for el in yticks:
        if el%45==0:
            yticklabels.append(str(int(el)) + r"$^\circ$")
        else:
            yticklabels.append("")
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)
    ax2 = ax1.twinx()
    ax2.plot(plant.time,Y_FBL[1,:].T,c="C1")
    ax2.plot(plant.time,desiredStiffness[0,:],c="C1",linestyle="--")
    ax2.tick_params(axis='y', labelcolor="C1")
    ax2.set_ylabel('Stiffness (Nm/rad.)', color="C1")

    fig2 = plt.figure(figsize=(10,8))
    ax3=plt.gca()
    ax3.plot(plant.time,(180/np.pi)*(Y_FBL[0,:]-desiredAngle[0,:]).T,c="C0")
    ax3.set_title("Error", fontsize=16)
    ax3.set_xlabel("Time (s)")
    ax3.tick_params(axis='y', labelcolor="C0")
    ax3.set_ylabel('Positional Error (deg.)', color="C0")
    yticklabels = [str(el)+r"$^\circ$" for el in ax3.get_yticks()]
    ax3.set_yticklabels(yticklabels)
    ax4 = ax3.twinx()
    ax4.plot(plant.time,Y_FBL[1,:] - desiredStiffness[0,:],c="C1")
    ax4.tick_params(axis='y', labelcolor="C1")
    ax4.set_ylabel('Stiffness Error (Nm/rad.)', color="C1")
    ax4.set_ylim([-0.1,0.1])
    ax4.set_yticks([-0.1,-0.05,0,0.05,0.1])

    # plant1.plot_tendon_tension_deformation_curves(X,addTitle="(Unforced)")
    out = plant.plot_tendon_tension_deformation_curves(
        X_FBL,
        returnValues=True,
        addTitle="(Feedback Linearization)"
    )
    tendonDeformation1_FBL,tendonDeformation2_FBL = out[0]
    tendonTension1_FBL,tendonTension2_FBL = out[1]

    plant.plot_boundary_friction_func(X_FBL)

    plant.plot_motor_angles(X_FBL)

    plt.show()
    return(plant.time,X_FBL,U_FBL,Y_FBL,out[1],plant)

def test_plant(plantParams):
    plant1 = plant_pendulum_1DOF2DOF(plantParams)
    plant2 = plant_pendulum_1DOF2DOF(plantParams)

    x1o = np.pi
    X_o = [x1o,0,plant2.rj*x1o/plant2.rm,0,-plant2.rj*x1o/plant2.rm,0]
    plantParams["X_o"] = X_o

    X,U,Y = plant1.forward_simulation(X_o)

    delay = 3
    X1d = np.zeros((5,len(X[0,:])))
    X1d[0,:] = np.pi*np.ones((1,len(X[0,:])))
    ### first transition after delay
    X1d[0,int(delay/plant1.dt)] = (
        plant1.jointAngleRange*np.random.uniform(0,1) # random number inside
        + plant1.jointAngleBounds["LB"]
    )
    Sd = np.zeros((3,len(X[0,:])))
    Sd[0,:] = plant1.jointStiffnessBounds["LB"]*np.ones((1,len(X[0,:])))
    Sd[0,int(delay/plant1.dt)] = (
        plant1.jointStiffnessRange*np.random.uniform(0,1) # random number inside
        + plant1.jointStiffnessBounds["LB"]
    )
    for i in range(int(delay/plant1.dt)+1, len(X[0,:])):
        if np.random.uniform() < 0.00025: # change offset
            X1d[0,i] = (
                plant1.jointAngleRange*np.random.uniform(0,1) # random number inside
                + plant1.jointAngleBounds["LB"]
            )
            Sd[0,i] = (
                plant1.jointStiffnessRange*np.random.uniform(0,1) # random number inside
                + plant1.jointStiffnessBounds["LB"]
            )
        else: # stay at previous input
            X1d[0,i] = X1d[0,i-1]
            Sd[0,i] = Sd[0,i-1]

    X1d[0,:] = LP_filt(100,X1d[0,:])
    X1d[1,:] = np.gradient(X1d[0,:],plant1.dt)
    X1d[2,:] = np.gradient(X1d[1,:],plant1.dt)
    X1d[3,:] = np.gradient(X1d[2,:],plant1.dt)
    X1d[4,:] = np.gradient(X1d[3,:],plant1.dt)

    Sd[0,:] = LP_filt(100,Sd[0,:])
    Sd[1,:] = np.gradient(Sd[0,:],plant1.dt)
    Sd[2,:] = np.gradient(Sd[1,:],plant1.dt)

    # timeBreaks = [
    #     int(el*plantParams["Simulation Duration"]/plantParams["dt"])
    #     for el in [0, 0.13333, 0.21667, 0.41667, .57, .785, 1]
    # ]
    # breakDurations = np.diff(timeBreaks)
    #
    # X1d[0,timeBreaks[0]:timeBreaks[1]] = np.pi*np.ones(breakDurations[0])
    # X1d[0,timeBreaks[1]:timeBreaks[2]] = np.pi*np.ones(breakDurations[1]) - 1
    # X1d[0,timeBreaks[2]:timeBreaks[3]] = (
    #     np.pi
    #     + 0.5*np.sin(
    #         3*np.pi*np.arange(
    #             0,plant1.time[breakDurations[2]],plantParams["dt"]
    #         ) / 5
    #     )
    # )
    # X1d[0,timeBreaks[3]:timeBreaks[4]] = np.pi*np.ones(breakDurations[3]) + 1
    # X1d[0,timeBreaks[4]:timeBreaks[5]] = np.pi*np.ones(breakDurations[4]) + 0.5
    # X1d[0,timeBreaks[5]:timeBreaks[6]] = np.pi*np.ones(breakDurations[5])
    #
    # X1d[0,:] = LP_filt(100, X1d[0,:])
    # X1d[1,:] = np.gradient(X1d[0,:],plant2.dt)
    # X1d[2,:] = np.gradient(X1d[1,:],plant2.dt)
    # X1d[3,:] = np.gradient(X1d[2,:],plant2.dt)
    # X1d[4,:] = np.gradient(X1d[3,:],plant2.dt)
    #
    # Sd = np.zeros((3,len(X[0,:])))
    # Sd[0,:] = 32 - 20*np.cos(16*np.pi*plant1.time/25)
    # Sd[1,:] = np.gradient(Sd[0,:],plant1.dt)
    # Sd[2,:] = np.gradient(Sd[1,:],plant1.dt)


    # Sd[0,:] = 80 - 20*np.cos(16*np.pi*plant1.time/25)
    # Sd[1,:] = 64*np.pi*np.sin(16*np.pi*plant1.time/25)/5
    # Sd[2,:] = (4**5)*(np.pi**2)*np.cos(16*np.pi*plant1.time/25)/(5**3)

    X_FBL,U_FBL,Y_FBL,X_measured = plant2.forward_simulation_FL(X_o,X1d,Sd)
    fig1 = plt.figure(figsize=(10,8))
    ax1=plt.gca()

    ax1.plot(plant1.time,(180/np.pi)*Y_FBL[0,:].T,c="C0")
    ax1.plot(plant1.time,(180/np.pi)*X1d[0,:],c="C0",linestyle="--")
    ax1.set_title(r"$-$ Actual; --- Desired", fontsize=16)
    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="C0")
    ax1.set_ylabel('Position (deg.)', color="C0")
    # y1_min = np.floor((Y_FBL[0,:].min()*180/np.pi)/22.5)*22.5
    # y1_min = min([y1_min,np.floor((X1d[0,:].min()*180/np.pi)/22.5)*22.5])
    # y1_max = np.ceil((Y_FBL[0,:].max()*180/np.pi)/22.5)*22.5
    # y1_max = max([y1_max,np.ceil((X1d[0,:].max()*180/np.pi)/22.5)*22.5])
    y1_min = 0
    y1_max = 360
    yticks = np.arange(y1_min,y1_max+22.5,22.5)
    yticklabels = []
    for el in yticks:
        if el%45==0:
            yticklabels.append(str(int(el)) + r"$^\circ$")
        else:
            yticklabels.append("")
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)
    ax2 = ax1.twinx()
    ax2.plot(plant1.time,Y_FBL[1,:].T,c="C1")
    ax2.plot(plant1.time,Sd[0,:],c="C1",linestyle="--")
    ax2.tick_params(axis='y', labelcolor="C1")
    ax2.set_ylabel('Stiffness (Nm/rad.)', color="C1")

    fig2 = plt.figure(figsize=(10,8))
    ax3=plt.gca()
    ax3.plot(plant1.time,(180/np.pi)*(Y_FBL[0,:]-X1d[0,:]).T,c="C0")
    ax3.set_title("Error", fontsize=16)
    ax3.set_xlabel("Time (s)")
    ax3.tick_params(axis='y', labelcolor="C0")
    ax3.set_ylabel('Positional Error (deg.)', color="C0")
    yticklabels = [str(el)+r"$^\circ$" for el in ax3.get_yticks()]
    ax3.set_yticklabels(yticklabels)
    ax4 = ax3.twinx()
    ax4.plot(plant1.time,Y_FBL[1,:] - Sd[0,:],c="C1")
    ax4.tick_params(axis='y', labelcolor="C1")
    ax4.set_ylabel('Stiffness Error (Nm/rad.)', color="C1")
    ax4.set_ylim([-0.1,0.1])
    ax4.set_yticks([-0.1,-0.05,0,0.05,0.1])

    # plant1.plot_tendon_tension_deformation_curves(X,addTitle="(Unforced)")
    out = plant2.plot_tendon_tension_deformation_curves(
        X_FBL,
        returnValues=True,
        addTitle="(Feedback Linearization)"
    )
    tendonDeformation1_FBL,tendonDeformation2_FBL = out[0]
    tendonTension1_FBL,tendonTension2_FBL = out[1]

    plant2.plot_motor_angles(X_FBL)

    plt.show()
    return(plant1.time,X1d,Sd,X_FBL,U_FBL,Y_FBL,plant1,plant2)

if __name__ == '__main__':
    ### Additional Arguments?
    parser = argparse.ArgumentParser(
        prog = "<filename>",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        plant.py

        -----------------------------------------------------------------------------

        A 1 DOF, 2 DOA tendon-driven system with nonlinear tendon
        elasticity in order to predict joint angle from different "sensory"
        states (like tendon tension or motor angle). This system can be controlled via Feedback linearization or by forward integration. The system will also stop when its

        -----------------------------------------------------------------------------'''),
        epilog=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        Written by Daniel A. Hagen (2020/01/29)

        -----------------------------------------------------------------------------'''
        )
    )
    parser.add_argument(
        '-dt',
        metavar='timestep',
        type=float,
        nargs="?",
        help='Time step for the simulation (float). Default is given by plantParams.py',
        default=plantParams["dt"]
    )
    parser.add_argument(
        '-dur',
        metavar='duration',
        type=float,
        nargs="?",
        help='Duration of the simulation (float). Default is given by plantParams.py',
        default=plantParams["Simulation Duration"]
    )
    parser.add_argument(
        '--savefigs',
        action="store_true",
        help='Option to save figures for babbling trial. Default is false.'
    )
    parser.add_argument(
        '--animate',
        action="store_true",
        help='Option to animate trial. Default is false.'
    )

    args = parser.parse_args()
    plantParams["dt"] = args.dt
    plantParams["Simulation Duration"] = args.dur
    saveFigures = args.savefigs
    animate = args.animate

    time,X1d,Sd,X,U,Y,plant1,plant2 = test_plant(plantParams)
    if saveFigures==True:
        save_figures(
            "visualizations/",
            "v1",
            plantParams,
            returnPath=False,
            saveAsPDF=True,
            saveAsMD=True
        )
    if animate==True:
        downsamplingFactor = int(0.3/plant1.dt)
        Yd = np.concatenate([X1d[0,:,np.newaxis],Sd[0,:,np.newaxis]],axis=1).T
        ani = animate_pendulum(
            time[::downsamplingFactor],
            X[:,::downsamplingFactor],
            U[:,::downsamplingFactor],
            Y[:,::downsamplingFactor],
            Yd[:,::downsamplingFactor],
            **plantParams
        )
        ani.start(downsamplingFactor)
