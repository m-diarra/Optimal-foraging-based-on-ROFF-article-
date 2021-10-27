# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:41:14 2021

@author: Hélène
"""

import numpy as np
import matplotlib.pyplot as plt
import random 
from scipy.stats import binom 

"""Scenario 1 :  A different terminal fitness"""

def Fitness_1 (X,Xcritical,Xmax,patch_p,Fvectors):
    """Fitness return the fitness of one patch
    Attributes: 
        scalar : actuel stat ,critical state, maximum state
        list : patch parameters
        matrix : Fvectors 
    Return: 
        scalar egal to the fitness of the patch"""
        
    XFood = X - patch_p[3] + patch_p[0] # value of X when food is found : X - cost + benefit
    XFood = min(XFood,Xmax) # If Xfood is superior than Xmax, it becomes Xmax 
    XFood = max(XFood,Xcritical) # In the same way, Xfood must not be inferior to Xcritical 
    XNoFood = X - patch_p[3] # value of X when food is not found : X - cost 
    XNoFood = max(XNoFood,Xcritical)
    Term1 = patch_p[1] * Fvectors[XFood - 1,1] #Pbenefice * probability to obtain food 
    Term2 = (1-patch_p[1]) * Fvectors[XNoFood - 1,1] #(1-Pbenefit) * probability to not obtain food
    W = (1-patch_p[2]) * (Term1 + Term2) #(1-Pmortality) * (Term1 + Term2)
    return W 


def OVER_PATCHES_1 (X, Fvectors, Xcritical, Xmax, Npatch, dictpatch):
    """OVER_Patches return the fitness of all patchs and select the optimal patch
    Attributes: 
        scalar : actual state
        matrix : Fvectors
        sacalr : critical state, maximum state, number of patchs
        dictionnary : patchs parameters
    Return: 
        matrix of 2 columns 'temp' """
    RHS = []
    for i in range(1,Npatch+1): # Cycle over patches
        # Call Fitness function
        RHS.append(Fitness_1(X, Xcritical, Xmax, dictpatch[i], Fvectors))
    # Now find optimal patch 
    value = max(RHS) # Fitness value of the optimal patch 
    Fvectors[X-1,0] = value  
    BestPatch = RHS.index(value) + 1 # return the number of the optimal patch 
    # Concatenate F(x,t) and the optimal patch number
    Temp = [Fvectors[X-1,0], BestPatch]
    Temp = np.append(Fvectors,[Temp],axis= 0) # Add the optimal batch number at the end of F(x,t)
    return (Temp)


def OVER_STATES_1(Fvectors, Xcritical, Xmax, Npatch, dictpatch):
    """OVER_STATES return the optimal decision for all states and the associated probability
    Attributes: 
        matrix : Fvectors
        saclar : critical state, maximum state, number of patchs
        dictionnary : patchs parameters
    Return: 
        matrix of 4 colums : 'temp' """
    Store = np.zeros((Xmax,2))
    for X in range(Xcritical + 1, Xmax + 1):
        # For given X call Over.Patches to determine F(x,t) and best patch
        Temp = OVER_PATCHES_1(X, Fvectors, Xcritical, Xmax, Npatch, dictpatch)
        n = Temp.shape[0] - 1  # Number total of lines - 1 
        Fvectors = Temp[0:n,:]
        Store[X-1,:] = Temp[n,:]
        Temp = np.c_[Fvectors, Store]
     # Add Store values to end of F.vectors for pass back to main program
     # Combined by columns
    return(Temp) # Return actual state, probability of each optimal patch, futur state and its associated probability 

print('SCENARIO 1')

# MAIN PROGRAM
random.seed(2) # Set random number seed

# Initialize parameters
Xmax = 10 # Maximum value of X
Xcritical = 3 # Value of X at which death occurs
Xmin = Xcritical + 1 # Smallest value of X allowed


#For each patch we have : 
## type: [benefit, Pbenefit, Pmortality, Cost]
# benefit if food is discovered
# Pbenefit: probability of finding food
# Pmortality: probability of mortality 
# Cost : cost of period 
dictpatch = {1:[0,1,0,1],2:[3,0.4,0.004,1],3:[5,0.6,0.02,1]}

Npatch = 3 # Number of patches

Horizon = 20 # Number of time step

# Set up matrix for fitnesses
# Column 1 is F(x, t+1). Column 2 is F(x,t)
Fvectors = np.zeros((Xmax,2)) # Set all values to zero
#initialize the values of the first column between Xmin and Xmax 
Fvectors[Xmin-1:Xmax,1] = [i for i in range(Xmin,Xmax +1)]

"""------------------------Decision matrix---------------------------------"""

# Create matrices for output
FxtT = np.zeros((Horizon,Xmax)) # F(x,t)
Best_Patch = np.zeros((Horizon,Xmax)) # Best patch number


Time = Horizon #Initialize Time

while Time>0:
    Time = Time - 1 # Decrement Time by 1 unit
    # Call OVER.STATES to get best values for this time step
    Temp = OVER_STATES_1(Fvectors, Xcritical, Xmax, Npatch, dictpatch)
    # Extract F.vectors
    TempF = Temp[:,0:1]
    # Update F1
    for i in range(Xmin, Xmax+1):
        Fvectors[i-1,1] = TempF[i-1,0]
        
    # Store results
    Best_Patch[Time-1,:] = Temp[:,3]
    FxtT[Time-1,:] = Temp[:,2]
    


# Output information. For display add states to last row of matrices
X = np.arange(1,Xmax+1)
Best_Patch = np.append(Best_Patch,[X], axis=0)
FxtT[Horizon - 1, : ] = X

print('Decision matrix :')
print(Best_Patch[: ,Xmin-1:Xmax])
print("Associated fitness matrix :")
print(FxtT[: ,Xmin-1:Xmax])



"""------------------Decisions simulation of one individual----------------"""

Output = np.zeros((Horizon,10)) # Matrix to hold output
Time = np.arange(1,Horizon + 1) # Values for x axis in plot
evolutionX = np.zeros((Horizon,10))
for Replicate in range(1,11) :
    X = 4  # Animal starts in state 4
    evolutionX[0,Replicate -1] =4
    for i in range(1,Horizon + 1) :

        if X > Xcritical :  # Iterate over time
            evolutionX[i-1,Replicate-1] = X
            Patch = Best_Patch[i-1,X-1] # Select patch
            # Check if animal survives predation
            # Generate random number
            if random.random() < dictpatch[Patch][2] :
                X = Xcritical # animal was eaten by a predator 
            else :
                # Now find new weight
                # Set multiplier to zero, which corresponds to no food found
                Index = 0
            
                if random.random() < dictpatch[Patch][1]:
                    Index = 1 # food is discovered
                
                X = X - dictpatch[Patch][3] + dictpatch[Patch][0] * Index
                # If X greater than Xmax then X must be set to Xmax
                X = min(X, Xmax)
                # If X less than X then animal dies
                    
                Output[i-1,Replicate-1] = Patch # Store data
                
                        
#Graphics representation 
Axe_combinaison=[[0,0],[1,0],[2,0],[3,0],[4,0],[0,1],[1,1],[2,1],[3,1],[4,1]]
i=0 # counter
fig, axs = plt.subplots(5, 2, figsize=(10, 10))
fig.suptitle('Patch value during the simulations for scenario 1')
for elt in Axe_combinaison: 
    
    axs[elt[0],elt[1]].plot(Time,Output[:,i])
    axs[elt[0],elt[1]].set_ylabel("Selected Patch")
    axs[elt[0],elt[1]].set_xlabel("Time")
    
    i=i+1 # The second column of OUTPUT will be selectioned
    
Axe_combinaison=[[0,0],[1,0],[2,0],[3,0],[4,0],[0,1],[1,1],[2,1],[3,1],[4,1]]
i=0 # counter
fig, axs = plt.subplots(5, 2, figsize=(10, 10))
fig.suptitle('State value during the simulations for scenario 1')
    
for elt in Axe_combinaison: 
    
    axs[elt[0],elt[1]].plot(Time,evolutionX[:,i],color ='green')
    axs[elt[0],elt[1]].set_ylabel("State")
    axs[elt[0],elt[1]].set_xlabel("Time")
    
    i=i+1 # The second column of OUTPUT will be selectioned

plt.show()
"""---------------------Transition density matrix---------------------------"""

T = 2
Trans_density = np.zeros((Xmax, Xmax))
for z in range (Xmin,Xmax + 1):
    K = Best_Patch[T-1, z-1] #Best patch for a given time and a given state 
    #Food is found 
    x = min(z - dictpatch[K][3] + dictpatch[K][0], Xmax)
    #Assign probability
    Trans_density[z-1, x-1] = (1-dictpatch[K][2]) * (dictpatch[K][1])
    # Food not found
    x = z - dictpatch[K][3]
    if x > Xcritical:
        # Animal survives
        Trans_density[z-1,x-1] = (1- dictpatch[K][2]) * (1-dictpatch[K][1])
       
        # Animal does not survive
        Trans_density [z-1, Xcritical-1] = dictpatch[K][2]
    else:
        Trans_density[z-1, Xcritical-1] = (dictpatch[K][2]) + \
            ((1-dictpatch[K][2]) * (1 - dictpatch[K][1]))
            
print("Transition density matrix :")
print(Trans_density)


print('\n' +'Please print enter in the console to run the scenario 2')
Next = input()

"""Scenario 2: To forage or not to forage when patches become options """

print('SCENARIO 2')

def Fitness_2 (X,Xcritical,Xmax,patch_p,Pmor,Fvectors):
    """Fitness return the fitness of the patch
    Attributes: 
        scalar : actuel state, critical state, maximum state, mortality probability
        probability matrix 
    Return: 
        scalar equals to the fitness of the patch"""
        
    XFood = X + patch_p[0] #Cost is not count 
    XFood = min(XFood,Xmax)# If XFood is superior than Xmax, it becomes Xmax 
    XFood = max(XFood,Xcritical) # In the same way, XFood must not be inferior to Xcritical
    XNoFood = X #Cost is not count 
    XNoFood = max(XNoFood,Xcritical)
    Term1 = patch_p[1] * Fvectors[XFood - 1,1] #Pbenefit * probability to obtain Food 
    Term2 = (1-patch_p[1]) * Fvectors[XNoFood - 1,1] #(1-Pbenefit) * to not obtain Food
    W = (1-Pmor) * (Term1 + Term2) #(1-Pmortality) * (Term1 + Term2)
    return W 


def OVER_PATCHES_2 (X, Fvectors, Xcritical, Xmax, Npatch, dictpatch):
    """OVER_Patches return the fitness of all patchs and select the optimal patch
    Attributes: 
        scalar : actual state
        matrix : Fvectors
        saclar : critical state, maximum state, number of patches
        dictionnary : patchs parameters
    Return: 
        matrix of 2 columns 'Temp' """
    RHS = []
    for i in range(1,Npatch+1): # Cycle over patches
        Pmor = Pmortality[i-1,X-1]
        # Call Fitness function
        RHS.append(Fitness_2(X, Xcritical, Xmax, dictpatch[i],Pmor, Fvectors))
    # Now find optimal patch 
    value=max(RHS) # Fitness value of the optimal patch 
    Fvectors[X-1,0] = value  
    BestPatch=RHS.index(value) + 1 # return the number of the optimal patch 
    # Concatenate F(x,t) and the optimal patch number
    Temp = [Fvectors[X-1,0], BestPatch]
    Temp = np.append(Fvectors,[Temp],axis= 0)# Add the optimal batch number at the end of F(x,t)
    return (Temp)


def OVER_STATES_2(Fvectors, Xcritical, Xmax, Npatch, dictpatch):
    """OVER_STATES return the optimal decision for all states and the associated probability
    Attributes: 
        matrix : Fvectors
        saclar : critical state, maximum state, number of patches
        dictionnary : patches parameters
    Return: 
        matrix of 4 columns : 'Temp' """
    Store = np.zeros((Xmax,2))
    for X in range(Xcritical + 1, Xmax + 1):
        # For given X call Over.Patches to determine F(x,t) and best patch
        Temp = OVER_PATCHES_2(X, Fvectors, Xcritical, Xmax, Npatch, dictpatch)
        n = Temp.shape[0] - 1  # Number total of lignes - 1 
        Fvectors =Temp[0:n,:]
        Store[X-1,:] =Temp[n,:]
        Temp = np.c_[Fvectors, Store]
     # Add Store values to end of F.vectors to pass back to the main program
     # Combined by columns
    return(Temp) # Return actual state, probability of each optimal patch, futur state and its associated probability 



# MAIN PROGRAM
random.seed(2) # Set random number seed

# Initialize parameters
Xmax = 7 # Maximum value of X
Xcritical = 1 # Value of X at which death occurs
Xmin = Xcritical + 1 # Smallest value of X allowed
random.seed(10) # Set random number seed


"""Particularity of scenario 2"""
#For the two patches (Not Forage, Forage) we have : 
## type: [benefit, Pbenefit, Pmortality]
# benefit if food is discovered
# Pbenefit: probability of finding food
# Pmortality: probability of mortality which depend of the state
# Cost : not applicable in this scenario

Pmin = 0
Pmax = 0.1

Pnoforage = np.zeros(Xmax)
Pforage =np.arange(Pmin,Pmax + 0.00001,round((Pmax-Pmin)/(Xmax-2),3))
Pforage = np.concatenate(([0],Pforage),axis = 0)
Pmortality = np.array([Pnoforage,Pforage])

dictpatch = {1:[-1,0.4,0,1],2:[1,0.8,0.02,1]}

Npatch = 2 # Number of patches : Not Forage, Forage 

Horizon = 6 # Number of time steps

# Set up matrix for fitnesses
# Column 1 is F(x, t). Column 2 is F(x,t+1)
Fvectors = np.zeros((Xmax,2)) # Set all values to zero
##Initialize the values of the first column between Xmin and Xmax 
Fvectors[Xmin-1:Xmax,1] = [i for i in range(Xmin,Xmax +1)] 

"""------------------------Decision matrix---------------------------------"""

# Create matrices for output
FxtT = np.zeros((Horizon,Xmax)) # F(x,t)
Best_Patch = np.zeros((Horizon,Xmax)) # Best patch number


Time = Horizon #Initialize Time

while Time>0:
    Time = Time - 1 # Decrement Time by 1 unit
    # Call OVER.STATES to get best values for this time step
    Temp = OVER_STATES_2(Fvectors, Xcritical, Xmax, Npatch, dictpatch)
    # Extract F.vectors
    TempF = Temp[:,0:1]
    # Update F1
    for i in range(Xmin, Xmax+1):
        Fvectors[i-1,1] = TempF[i-1,0]
        
    # Store results
    Best_Patch[Time-1,:] = Temp[:,3]
    FxtT[Time-1,:] = Temp[:,2]
    


# Output information. For display add states to last row of matrices
X = np.arange(1,Xmax+1)
Best_Patch = np.append(Best_Patch,[X], axis=0)
FxtT[Horizon - 1, : ] = X

print("Decision matrix :")
print(Best_Patch[: ,Xmin-1:Xmax]) 

"""------------------Decisions simulation of one individual----------------"""

Output = np.zeros((Horizon,10)) # Matrix to hold output
Time = np.arange(1,Horizon + 1) # Values for x axis in plot
evolutionX = np.zeros((Horizon,10))
for Replicate in range(1,11) :
    X = 4 # Animal starts in state 4
    for i in range(1,Horizon + 1) :

        if X > Xcritical :  # Iterate over time
            evolutionX[i-1,Replicate-1] = X
            Patch = Best_Patch[i-1,X-1] # Select patch

            # Now find new weight
            # Set multiplier to zero, which corresponds to no food found
            Index = 0
            
            if random.random() < dictpatch[Patch][1]:
                Index = 1 # food is discovered
            
            X = X - dictpatch[Patch][3] + dictpatch[Patch][0] * Index
            # If X greater than Xmax then X must be set to Xmax
            X = min(X, Xmax)
            # If X less than X then animal dies
            #if (X < Xmin):
                
            Output[i-1,Replicate-1] = Patch # Store data
                        
#Graphics representation 
Axe_combinaison=[[0,0],[1,0],[2,0],[3,0],[4,0],[0,1],[1,1],[2,1],[3,1],[4,1]]
i=0 # counter
fig, axs = plt.subplots(5, 2, figsize=(10, 10))
fig.suptitle('Simulations using the decision matrix for scenario 2')
for elt in Axe_combinaison: 
    
    axs[elt[0],elt[1]].plot(Time,Output[:,i])
    axs[elt[0],elt[1]].set_ylabel("Selected Patch")
    axs[elt[0],elt[1]].set_xlabel("Time")
    
    i=i+1 # The second column of OUTPUT will be selected
    
Axe_combinaison=[[0,0],[1,0],[2,0],[3,0],[4,0],[0,1],[1,1],[2,1],[3,1],[4,1]]
i=0 # counter
fig, axs = plt.subplots(5, 2, figsize=(10, 10))
fig.suptitle('State value during the simulations for scenario 2')
    
for elt in Axe_combinaison: 
    
    axs[elt[0],elt[1]].plot(Time,evolutionX[:,i],color ='green')
    axs[elt[0],elt[1]].set_ylabel("State")
    axs[elt[0],elt[1]].set_xlabel("Time")
    
    i=i+1 # The second column of OUTPUT will be selected

plt.show()

"""---------------------Transition density matrix---------------------------"""

T = 2
Trans_density = np.zeros((Xmax, Xmax))
for z in range (Xmin,Xmax + 1):
    K = Best_Patch[T-1, z-1] #Best patch for a given time and a given state
    #Food is found
    x = min(z + dictpatch[K][0], Xmax)
    #Assign probability
    Pmor = Pmortality[int(K-1),x-1]
    Trans_density[z-1, x-1] = (1-Pmor) * (dictpatch[K][1])
    # Food not found
    x = z 
    if x > Xcritical:
        # Animal survives
        Trans_density[z-1,x-1] = (1-Pmor) * (1-dictpatch[K][1])
       
        # Animal does not survive
        Trans_density [z-1, Xcritical-1] = Pmor
    else:
        Trans_density[z-1, Xcritical-1] = Pmor + ((1-Pmor) * (1 - dictpatch[K][1]))

print('Density matrix :')
print(Trans_density)


print('\n' +'Please print enter in the console to run the scenario 3')
Next = input()

"""Scenario 3: Testing for equivalent choices, indexing, and interpolation """

print('SCENARIO 3')

def Fitness_3 (X,Xcritical,Xmax,Xinc,Cost,Benefit,Pbenefit,Fvectors):
    """Fitness return the fitness of the patch
    Attributes: 
        scalar : actuel state, critical state, increment, maximum state
        list : patch parameters
        probability matrix 
    Return: 
        scalar equals to the fitness of the patch"""
    
    Max_Index = int(1 + (Xmax-Xcritical)/Xinc)
    W = 0
    Xstore = X 
    for kill in range(0,kmax+1): #Cycle over the possible number of kill
        X = Xstore - Cost + Benefit[kill]
        X = min(X,Xmax) # If Xfood is superior than Xmax, it becomes Xmax  
        X = max(X,Xcritical) # In the same way, Xfood must not be inferior to Xcritical 
        Index = 1 +(X-Xcritical)/Xinc
        Index_lower = int(Index//1) #Index must be an integer
        Index_upper = int(Index//1 + 1)
        Index_upper = min(Index_upper, Max_Index) # If index_upper is superior than Max_Index, it becomes Max_Index  
        Qx = X - X//1
        W = W + Pbenefit[kill]*(Qx*Fvectors[Index_upper-1,1] \
                                + (1-Qx)*Fvectors[Index_lower-1,1])
    return W 


def OVER_PATCHES_3 (X, Fvectors, Xcritical, Xmax, Xinc, Npatch, Cost, Benefit, Pbenefit):
    """OVER_Patches return the fitness of all patchs and select the optimal patch
    Attributes: 
        scalar : actual state
        matrix : Fvectors
        saclar : critical state, maximum state, increment, number of patches,
                 Cost, Benefit, Pbenefit 
    Return: 
        matrix of 2 columns 'Temp' """
    RHS = []
    
    for i in range(1,Npatch+1): # Cycle over patches
        # Call Fitness function
        RHS.append(Fitness_3(X,Xcritical,Xmax,Xinc,Cost,Benefit[i-1],Pbenefit[i-1],Fvectors))
        # Now find optimal patch Best row is in Best[1]
    value=max(RHS) # Fitness value of the optimal patch 
    Fvectors[X-1,0] = value  
    BestPatch=RHS.index(value)  # return the number of the optimal kill (could be 0)
    # Concatenate F(x,t) and the optimal patch number
    Temp = [Fvectors[X-1,0], BestPatch]
    Temp = np.append(Fvectors,[Temp],axis= 0)
    Choice = np.array([0,0])
    if RHS.count(value) >= 2:
        Choice = np.array([1,1]) # Equal fitnesses
    Temp = np.append(Temp,[Choice], axis = 0)
    return (Temp)


def OVER_STATES_3(Fvectors, Xcritical, Xmax, Xinc, Npatch, Cost, Benefit, Pbenefit, Max_Index):
    """OVER_STATES return the optimal decision for all states and the associated probability
    Attributes: 
        matrix : Fvectors
        saclar : critical state, maximum state, incrementation, number of patches,
                 Cost, Benefit, Pbenefit, Max_Index 
    Return: 
        matrix of 4 columns : 'Temp' """
    Store = np.zeros((Xmax,2))
        
    Store = np.zeros((Max_Index,3))
    for Index in range(1, Max_Index+1):
        # For given X call Over.Patches to determine F(x,t) and best patch
        X =  (Index-1)*Xinc + Xcritical
        Temp = OVER_PATCHES_3(X, Fvectors, Xcritical, Xmax, Xinc, Npatch, Cost, Benefit, Pbenefit)
    
        n = Temp.shape[0] - 2  # Number total of lignes - 1 
        Fvectors =Temp[0:n,:]
        Store[Index-1,0:2] = Temp[n ,:] # Save F(x,t,T) and best patch
        Store[Index-1,2] = Temp[n + 1,0] # Save Flag for several choices
       
    Temp = np.c_[Fvectors, Store]
     # Add Store values to end of F.vectors for pass back to main program
     # Combined by columns
    return(Temp) # Return actual state, probability of each optimal patch, futur state and its associated probability 



# MAIN PROGRAM
random.seed(2) # Set random number seed

# Initialize parameters
Xmax = 30 # Maximum value of X
Xcritical = 0 # Value of X at which death occurs
Xmin = Xcritical + 1 # Smallest value of X allowed
Xinc = 1 #increment in state variable
Max_Index = int(1 + (Xmax-Xcritical)/Xinc)
Cost = 6 
Npatch = 4
Horizon = 31
kmax = 3 #maximum number of kills in a day

"""Calcul des paramètres"""
Benefit = np.zeros((Npatch,kmax+1),dtype = float) # Rows = pack size, Columns = number of kills 
Pbenefit = np.zeros((Npatch,kmax+1))
Pi = np.array([0.15, 0.31, 0.33, 0.33]) # Probability of single kill for pack size
Y = 11.25 # Size of single prey
k = np.array([0,1,2,3]) # Number of kills
Size_max = 4

for PackSize in range(1,Size_max + 1): # Iterate over pack sizes
    # Calculate binomial probabilities using binomal function
    Pbenefit[PackSize -1,] = [binom.pmf(kill,kmax,Pi[PackSize - 1]) for kill in k ]
    # Calculate benefits
    Benefit[PackSize - 1, 1:Npatch] = k[1:Npatch]*Y/PackSize


# Set up matrix for fitnesses
# Column 1 is F(x, t). Column 2 is F(x,t+1)
Fvectors = np.zeros((Max_Index,2)) # Set all values to zero
Fvectors[1:Max_Index-1,1] = 1


# Create matrices for output
FxtT = np.zeros((Horizon,Max_Index)) # F(x,t)
Best_Patch = np.zeros((Horizon,Max_Index)) # Best patch number
# Matrix of choices: 0  for only one choice, 1  for more than one choice
Choices = np.zeros((Horizon,Max_Index))

Time = Horizon #Initialize Time

while Time>0:
    Time = Time - 1 # Decrement Time by 1 unit
    # Call OVER.STATES to get best values for this time step
    Temp = OVER_STATES_3(Fvectors, Xcritical, Xmax, Xinc, Npatch, Cost, Benefit, Pbenefit, Max_Index)
    # Extract F.vectors
    TempF = Temp[:,0:2]
    # Update F1
    for i in range(1, Max_Index):
        Fvectors[i,1] = TempF[i,0]
        
    # Store results
    Best_Patch[Time-1,:] = Temp[:,3]
    FxtT[Time-1,:] = Temp[:,2]
    Choices[Time-1,:] = Temp[:,4]
    


# Output information. 
Index = np.arange(1,Max_Index+1)
Best_Patch[Horizon -1 ,:] = (Index-1)*Xinc + Xcritical
FxtT[Horizon -1 ,:] = (Index-1)*Xinc + Xcritical
#print(Best_Patch[:,1:Max_Index]) # Print Decision matrix
#print(np.round(FxtT[:,1:Max_Index],3)) # Print Fxt of Decision matrix: 3 sig places
#print(Choices[:,1:Max_Index] )# Print matrix indicating choice flag

# Plot data
y = Best_Patch[Horizon-1,1:Max_Index]
x = np.outer(np.linspace(1, 30, 30), np.ones(30))

fig = plt.figure()
 
# syntax for 3-D projection
ax = plt.axes(projection ='3d')

# plotting
ax.plot_surface(x,y,Best_Patch[0:30,0:30],cmap ='viridis')
ax.set_xlabel("Time")
ax.set_ylabel("State")
ax.set_zlabel("Optimal pack size")
ax.view_init(40,-100)

plt.show()
