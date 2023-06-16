

import numpy as np
import time
from numba import njit, prange
import numba
import math
from astropy.table import Table




@njit(cache=True)
def get_cell_pairs(n_cells):

    #--building every possible cells:
    L_cells=np.empty((n_cells[0]*n_cells[1]*n_cells[2],3))
    i=0
    for a in prange(n_cells[0]): 
        for b in range(n_cells[1]): 
            for c in range(n_cells[2]):
                L_cells[i] = [a,b,c]
                i+=1

    L_cell_pair=[]
    for i in prange(L_cells.shape[0]):
        for j in range(i+1,L_cells.shape[0]):
            dist = L_cells[i]-L_cells[j]
            if -1<=dist[0]<=1 and -1<=dist[1]<=1 and -1<=dist[2]<=1:
                L_cell_pair.append(list(L_cells[i])+list(L_cells[j]))
    return np.array(L_cell_pair),L_cells




######################## 1D BINNING ############################

@njit(parallel=False,cache=True)
def auto_count_1d(X,weights,bin_min,bin_max,n_bin):
    count = np.zeros(n_bin)
    for i in prange(X.shape[0]):
        for j in prange(i+1,X.shape[0]):
            dist = math.sqrt((X[i][0]-X[j][0])**2 + (X[i][1]-X[j][1])**2 + (X[i][2]-X[j][2])**2)
            bin_index = (dist-bin_min)/(bin_max-bin_min)*n_bin
            #--lower and upper exclusion    
            if n_bin>bin_index>=0 :
                count[int(bin_index)] += weights[i]*weights[j]
    return count

@njit(parallel=False,cache=True)
def cross_count_1d(X1,X2,weights1,weights2,bin_min,bin_max,n_bin):
    count = np.zeros(n_bin)
    for i in prange(len(X1)):#.shape[0]):
        for j in prange(len(X2)):#.shape[0]):
            dist = math.sqrt((X1[i][0]-X2[j][0])**2 + (X1[i][1]-X2[j][1])**2 + (X1[i][2]-X2[j][2])**2)
            bin_index = (dist-bin_min)/(bin_max-bin_min)*n_bin
            if n_bin>bin_index>=0 : #--lower and upper exclusion
                count[int(bin_index)] += weights1[i]*weights2[j]
    return count


@njit(parallel=True,cache=True)
def auto_counter_1d(X,weights,bins_s,n_cells,indices_cells,L_cell_pair,L_cells):
    """
    Make sure numba has been initialized with the correct number 
    of threads : numba.set_num_threads(Nthread) 
    Note : dim of bins depends on the mode
    """

    N = len(L_cell_pair) #--number of pairs
    bin_min = bins_s[0]
    bin_max = bins_s[-1]
    n_bin = len(bins_s)-1
    count = np.zeros(n_bin)
    #--Count pairs within the same cell:
    indices_cells_flat = indices_cells[:,0]*(n_cells[1]*n_cells[2]) + indices_cells[:,1]*(n_cells[2]) +indices_cells[:,2]
    
    for i_cell in prange(L_cells.shape[0]):
        w = (indices_cells_flat==i_cell)
        count += auto_count_1d(X[w],weights[w],bin_min,bin_max,n_bin)

    #--Count pairs between pair of cells       
    L_cell_pair_flat_1 = L_cell_pair[:,0]*(n_cells[1]*n_cells[2]) + L_cell_pair[:,1]*(n_cells[2]) +L_cell_pair[:,2]      
    L_cell_pair_flat_2 = L_cell_pair[:,3]*(n_cells[1]*n_cells[2]) + L_cell_pair[:,4]*(n_cells[2]) +L_cell_pair[:,5]      
                             
    for i_pair in prange(L_cell_pair.shape[0]):
        #-- acces the positions in one cell-pair :
        w1 = (indices_cells_flat==L_cell_pair_flat_1[i_pair])
        w2 = (indices_cells_flat==L_cell_pair_flat_2[i_pair])
                 
        count += cross_count_1d(X[w1],X[w2],weights[w1],weights[w2],bin_min,bin_max,n_bin)
        
    return count#*2



@njit(parallel=True,cache=True)
def cross_counter_1d(X1,X2,weights1,weights2,bins_s,n_cells,indices_cells1,indices_cells2,L_cell_pair,L_cells):
    """
    Make sure numba has been initialized with the correct number 
    of threads : numba.set_num_threads(Nthread) 
    Note : dim of bins depends on the mode
    """

    N = len(L_cell_pair) #--number of pairs
    bin_min = bins_s[0]
    bin_max = bins_s[-1]
    n_bin = len(bins_s)-1
    count = np.zeros(n_bin)
    
    #--Count pairs within the same cell:
    L_cells_flat =  L_cells[:,0]*(n_cells[1]*n_cells[2]) + L_cells[:,1]*(n_cells[2]) +L_cells[:,2]#
    indices_cells_flat_1 = indices_cells1[:,0]*(n_cells[1]*n_cells[2]) + indices_cells1[:,1]*(n_cells[2]) +indices_cells1[:,2]
    indices_cells_flat_2 = indices_cells2[:,0]*(n_cells[1]*n_cells[2]) + indices_cells2[:,1]*(n_cells[2]) +indices_cells2[:,2]

    for i_cell in prange(L_cells.shape[0]):
        w1 = (indices_cells_flat_1==i_cell)
        w2 = (indices_cells_flat_2==i_cell)
        count += cross_count_1d(X1[w1],X2[w2],weights1[w1],weights2[w2],bin_min,bin_max,n_bin)

    #--Count pairs between pair of cells       
    L_cell_pair_flat_1 = L_cell_pair[:,0]*(n_cells[1]*n_cells[2]) + L_cell_pair[:,1]*(n_cells[2]) +L_cell_pair[:,2]      
    L_cell_pair_flat_2 = L_cell_pair[:,3]*(n_cells[1]*n_cells[2]) + L_cell_pair[:,4]*(n_cells[2]) +L_cell_pair[:,5]      
                             
    for i_pair in prange(L_cell_pair.shape[0]):
        #-- acces the positions in one cell-pair :
        w11 = (indices_cells_flat_1==L_cell_pair_flat_1[i_pair])
        w22 = (indices_cells_flat_2==L_cell_pair_flat_2[i_pair])
        count += cross_count_1d(X1[w11],X2[w22],weights1[w11],weights2[w22],bin_min,bin_max,n_bin)
        
    for i_pair in prange(L_cell_pair.shape[0]):
        #-- acces the positions in one cell-pair :
        w12 = (indices_cells_flat_1==L_cell_pair_flat_2[i_pair])
        w21 = (indices_cells_flat_2==L_cell_pair_flat_1[i_pair])
        count += cross_count_1d(X1[w12],X2[w21],weights1[w12],weights2[w21],bin_min,bin_max,n_bin)

    return count

######################## 2D BINNING ############################
@njit(parallel=False,cache=True)
def auto_count_2d(X,weights,bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin):
    count = np.zeros((n_s_bin,n_mu_bin))
    for i in prange(X.shape[0]):
        for j in prange(i+1,X.shape[0]):
            dist = math.sqrt((X[i][0]-X[j][0])**2 + (X[i][1]-X[j][1])**2 + (X[i][2]-X[j][2])**2)
            bin_s_index = (dist-bin_s_min)/(bin_s_max-bin_s_min)*n_s_bin
            #--lower and upper exclusion    
            if n_s_bin>bin_s_index>=0 :
                ##-- Inline developpement of dot(los,sep) with x1=X1[i], x2=X1[j]
                dist_mu = 1/math.sqrt(2) *( (math.sqrt(X[i][0]**2+X[i][1]**2+X[i][2]**2) - math.sqrt(X[j][0]**2+X[j][1]**2+X[j][2]**2) ) * 
                                           ( math.sqrt((X[i][0]**2+X[i][1]**2+X[i][2]**2)*(X[j][0]**2+X[j][1]**2+X[j][2]**2)) + X[i][0]*X[j][0]+X[i][1]*X[j][1]+X[i][2]*X[j][2] ) ) / math.sqrt(
                    ( (X[i][0]**2+X[i][1]**2+X[i][2]**2)*(X[j][0]**2+X[j][1]**2+X[j][2]**2) + (X[i][0]*X[j][0]+X[i][1]*X[j][1]+X[i][2]*X[j][2])*math.sqrt((X[i][0]**2+X[i][1]**2+X[i][2]**2)*(X[j][0]**2+X[j][1]**2+X[j][2]**2)) ) *
                    ( (X[i][0]**2+X[i][1]**2+X[i][2]**2)+(X[j][0]**2+X[j][1]**2+X[j][2]**2) -2*(X[i][0]*X[j][0]+X[i][1]*X[j][1]+X[i][2]*X[j][2]) ) )

                bin_mu_index = (dist_mu-bin_mu_min)/(bin_mu_max-bin_mu_min)*n_mu_bin
                count[int(bin_s_index),int(bin_mu_index)] += weights[i]*weights[j]               
                
    return count

@njit(parallel=False,cache=True)
def cross_count_2d(X1,X2,weights1,weights2,bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin):
    count = np.zeros((n_s_bin,n_mu_bin))
    for i in prange(len(X1)):#.shape[0]):
        for j in prange(len(X2)):#.shape[0]):
            dist = math.sqrt((X1[i][0]-X2[j][0])**2 + (X1[i][1]-X2[j][1])**2 + (X1[i][2]-X2[j][2])**2)
            bin_s_index = (dist-bin_s_min)/(bin_s_max-bin_s_min)*n_s_bin
            if n_s_bin>bin_s_index>=0 : #--lower and upper exclusion
                ##-- Inline developpement of dot(los,sep) with x1=X1[i], x2=X1[j]
                dist_mu = 1/math.sqrt(2) *( (math.sqrt(X1[i][0]**2+X1[i][1]**2+X1[i][2]**2) - math.sqrt(X2[j][0]**2+X2[j][1]**2+X2[j][2]**2) ) * 
                                           ( math.sqrt((X1[i][0]**2+X1[i][1]**2+X1[i][2]**2)*(X2[j][0]**2+X2[j][1]**2+X2[j][2]**2)) + X1[i][0]*X2[j][0]+X1[i][1]*X2[j][1]+X1[i][2]*X2[j][2] ) ) / math.sqrt(
                    ( (X1[i][0]**2+X1[i][1]**2+X1[i][2]**2)*(X2[j][0]**2+X2[j][1]**2+X2[j][2]**2) + (X1[i][0]*X2[j][0]+X1[i][1]*X2[j][1]+X1[i][2]*X2[j][2])*math.sqrt((X1[i][0]**2+X1[i][1]**2+X1[i][2]**2)*(X2[j][0]**2+X2[j][1]**2+X2[j][2]**2)) ) *
                    ( (X1[i][0]**2+X1[i][1]**2+X1[i][2]**2)+(X2[j][0]**2+X2[j][1]**2+X2[j][2]**2) -2*(X1[i][0]*X2[j][0]+X1[i][1]*X2[j][1]+X1[i][2]*X2[j][2]) ) )

                bin_mu_index = (dist_mu-bin_mu_min)/(bin_mu_max-bin_mu_min)*n_mu_bin
                count[int(bin_s_index),int(bin_mu_index)] += weights1[i]*weights2[j]
                
    return count

@njit(parallel=True,cache=True)
def auto_counter_2d(X,weights,bins_s,bins_mu,n_cells,indices_cells,L_cell_pair,L_cells):
    """
    Make sure numba has been initialized with the correct number 
    of threads : numba.set_num_threads(Nthread) 
    Note : dim of bins depends on the mode
    """

    bin_s_min = bins_s[0]
    bin_s_max = bins_s[-1]
    n_s_bin = len(bins_s)-1

    bin_mu_min = bins_mu[0]
    bin_mu_max = bins_mu[-1]
    n_mu_bin = len(bins_mu)-1

    count = np.zeros((n_s_bin,n_mu_bin))
    #--Count pairs within the same cell:
    indices_cells_flat = indices_cells[:,0]*(n_cells[1]*n_cells[2]) + indices_cells[:,1]*(n_cells[2]) +indices_cells[:,2]
    
    for i_cell in prange(L_cells.shape[0]):
        w = (indices_cells_flat==i_cell)
        count += auto_count_2d(X[w],weights[w],bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin)

    #--Count pairs between pair of cells       
    L_cell_pair_flat_1 = L_cell_pair[:,0]*(n_cells[1]*n_cells[2]) + L_cell_pair[:,1]*(n_cells[2]) +L_cell_pair[:,2]      
    L_cell_pair_flat_2 = L_cell_pair[:,3]*(n_cells[1]*n_cells[2]) + L_cell_pair[:,4]*(n_cells[2]) +L_cell_pair[:,5]      
                             
    for i_pair in prange(L_cell_pair.shape[0]):
        #-- acces the positions in one cell-pair :
        w1 = (indices_cells_flat==L_cell_pair_flat_1[i_pair])
        w2 = (indices_cells_flat==L_cell_pair_flat_2[i_pair])
                 
        count += cross_count_2d(X[w1],X[w2],weights[w1],weights[w2],bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin)
        
    return count #+ count[:,::-1]

@njit(parallel=True,cache=True)
def cross_counter_2d(X1,X2,weights1,weights2,bins_s,bins_mu,n_cells,indices_cells1,indices_cells2,L_cell_pair,L_cells):
    """
    Make sure numba has been initialized with the correct number 
    of threads : numba.set_num_threads(Nthread) 
    Note : dim of bins depends on the mode
    """
    bin_s_min = bins_s[0]
    bin_s_max = bins_s[-1]
    n_s_bin = len(bins_s)-1

    bin_mu_min = bins_mu[0]
    bin_mu_max = bins_mu[-1]
    n_mu_bin = len(bins_mu)-1

    count = np.zeros((n_s_bin,n_mu_bin))
    
    #--Count pairs within the same cell:
    L_cells_flat =  L_cells[:,0]*(n_cells[1]*n_cells[2]) + L_cells[:,1]*(n_cells[2]) +L_cells[:,2]#
    indices_cells_flat_1 = indices_cells1[:,0]*(n_cells[1]*n_cells[2]) + indices_cells1[:,1]*(n_cells[2]) +indices_cells1[:,2]
    indices_cells_flat_2 = indices_cells2[:,0]*(n_cells[1]*n_cells[2]) + indices_cells2[:,1]*(n_cells[2]) +indices_cells2[:,2]

    for i_cell in prange(L_cells.shape[0]):
        w1 = (indices_cells_flat_1==i_cell)
        w2 = (indices_cells_flat_2==i_cell)
        count += cross_count_2d(X1[w1],X2[w2],weights1[w1],weights2[w2],bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin)

    #--Count pairs between pair of cells       
    L_cell_pair_flat_1 = L_cell_pair[:,0]*(n_cells[1]*n_cells[2]) + L_cell_pair[:,1]*(n_cells[2]) +L_cell_pair[:,2]      
    L_cell_pair_flat_2 = L_cell_pair[:,3]*(n_cells[1]*n_cells[2]) + L_cell_pair[:,4]*(n_cells[2]) +L_cell_pair[:,5]      
                             
    for i_pair in prange(L_cell_pair.shape[0]):
        #-- acces the positions in one cell-pair :
        w11 = (indices_cells_flat_1==L_cell_pair_flat_1[i_pair])
        w22 = (indices_cells_flat_2==L_cell_pair_flat_2[i_pair])
        count += cross_count_2d(X1[w11],X2[w22],weights1[w11],weights2[w22],bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin)
        
    for i_pair in prange(L_cell_pair.shape[0]):
        #-- acces the positions in one cell-pair :
        w12 = (indices_cells_flat_1==L_cell_pair_flat_2[i_pair])
        w21 = (indices_cells_flat_2==L_cell_pair_flat_1[i_pair])
        count += cross_count_2d(X1[w12],X2[w21],weights1[w12],weights2[w21],bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin)

    return count



######################## 3D BINNING ############################

@njit(parallel=False,cache=True)
def auto_count_3d(X,weights,bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin,bin_theta_min,bin_theta_max,n_theta_bin):
    count = np.zeros((n_s_bin,n_mu_bin,n_theta_bin))
    for i in prange(X.shape[0]):
        for j in prange(i+1,X.shape[0]):
            dist = math.sqrt((X[i][0]-X[j][0])**2 + (X[i][1]-X[j][1])**2 + (X[i][2]-X[j][2])**2)
            bin_s_index = (dist-bin_s_min)/(bin_s_max-bin_s_min)*n_s_bin
            #--lower and upper exclusion    
            if n_s_bin>bin_s_index>=0 :
                ##-- Inline developpement of dot(los,sep) with x1=X1[i], x2=X1[j]
                dist_mu = 1/math.sqrt(2) *( (math.sqrt(X[i][0]**2+X[i][1]**2+X[i][2]**2) - math.sqrt(X[j][0]**2+X[j][1]**2+X[j][2]**2) ) * 
                                           ( math.sqrt((X[i][0]**2+X[i][1]**2+X[i][2]**2)*(X[j][0]**2+X[j][1]**2+X[j][2]**2)) + X[i][0]*X[j][0]+X[i][1]*X[j][1]+X[i][2]*X[j][2] ) ) / math.sqrt(
                    ( (X[i][0]**2+X[i][1]**2+X[i][2]**2)*(X[j][0]**2+X[j][1]**2+X[j][2]**2) + (X[i][0]*X[j][0]+X[i][1]*X[j][1]+X[i][2]*X[j][2])*math.sqrt((X[i][0]**2+X[i][1]**2+X[i][2]**2)*(X[j][0]**2+X[j][1]**2+X[j][2]**2)) ) *
                    ( (X[i][0]**2+X[i][1]**2+X[i][2]**2)+(X[j][0]**2+X[j][1]**2+X[j][2]**2) -2*(X[i][0]*X[j][0]+X[i][1]*X[j][1]+X[i][2]*X[j][2]) ) )
                bin_mu_index = (dist_mu-bin_mu_min)/(bin_mu_max-bin_mu_min)*n_mu_bin         
            
                #-- Inline developpement of dot(x1,x2)/norm
                dist_theta = (X[i][0]*X[j][0]+X[i][1]*X[j][1]+X[i][2]*X[j][2]) / math.sqrt( (X[i][0]**2+X[i][1]**2+X[i][2]**2)*(X[j][0]**2+X[j][1]**2+X[j][2]**2) )
                #dist_theta = math.sqrt((1+dist_theta)/2)
                bin_theta_index = (dist_theta-bin_theta_min)/(bin_theta_max-bin_theta_min)*n_theta_bin
                
                #dist_mu = ((X[i][0]**2+X[i][1]**2+X[i][2]**2) - (X[i][0]*X[j][0]+X[i][1]*X[j][1]+X[i][2]*X[j][2])) / math.sqrt( ((X[i][0]**2+X[i][1]**2+X[i][2]**2))*((X[i][0]-X[j][0])**2+(X[i][1]-X[j][1])**2+(X[i][2]-X[j][2])**2) )
                #bin_mu_index = (dist_mu-bin_mu_min)/(bin_mu_max-bin_mu_min)*n_mu_bin         
            
                #dist_theta = ((X[j][0]**2+X[j][1]**2+X[j][2]**2) - (X[i][0]*X[j][0]+X[i][1]*X[j][1]+X[i][2]*X[j][2])) / math.sqrt( ((X[j][0]**2+X[j][1]**2+X[j][2]**2))*((X[i][0]-X[j][0])**2+(X[i][1]-X[j][1])**2+(X[i][2]-X[j][2])**2) )
                #bin_theta_index = (dist_theta-bin_theta_min)/(bin_theta_max-bin_theta_min)*n_theta_bin
                count[int(bin_s_index),int(bin_mu_index),int(bin_theta_index)] += weights[i]*weights[j]

    return count

@njit(parallel=False,cache=True)
def cross_count_3d(X1,X2,weights1,weights2,bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin,bin_theta_min,bin_theta_max,n_theta_bin):
    count = np.zeros((n_s_bin,n_mu_bin,n_theta_bin))
    for i in prange(len(X1)):#.shape[0]):
        for j in prange(len(X2)):#.shape[0]):
            dist = math.sqrt((X1[i][0]-X2[j][0])**2 + (X1[i][1]-X2[j][1])**2 + (X1[i][2]-X2[j][2])**2)
            bin_s_index = (dist-bin_s_min)/(bin_s_max-bin_s_min)*n_s_bin
            if n_s_bin>bin_s_index>=0 : #--lower and upper exclusion
                ##-- Inline developpement of dot(los,sep) with x1=X1[i], x2=X1[j]
                dist_mu = 1/math.sqrt(2) *( (math.sqrt(X1[i][0]**2+X1[i][1]**2+X1[i][2]**2) - math.sqrt(X2[j][0]**2+X2[j][1]**2+X2[j][2]**2) ) * 
                                           ( math.sqrt((X1[i][0]**2+X1[i][1]**2+X1[i][2]**2)*(X2[j][0]**2+X2[j][1]**2+X2[j][2]**2)) + X1[i][0]*X2[j][0]+X1[i][1]*X2[j][1]+X1[i][2]*X2[j][2] ) ) / math.sqrt(
                    ( (X1[i][0]**2+X1[i][1]**2+X1[i][2]**2)*(X2[j][0]**2+X2[j][1]**2+X2[j][2]**2) + (X1[i][0]*X2[j][0]+X1[i][1]*X2[j][1]+X1[i][2]*X2[j][2])*math.sqrt((X1[i][0]**2+X1[i][1]**2+X1[i][2]**2)*(X2[j][0]**2+X2[j][1]**2+X2[j][2]**2)) ) *
                    ( (X1[i][0]**2+X1[i][1]**2+X1[i][2]**2)+(X2[j][0]**2+X2[j][1]**2+X2[j][2]**2) -2*(X1[i][0]*X2[j][0]+X1[i][1]*X2[j][1]+X1[i][2]*X2[j][2]) ) )
                
                bin_mu_index = (dist_mu-bin_mu_min)/(bin_mu_max-bin_mu_min)*n_mu_bin
            
                ##-- Inline developpement of dot(x1,x2)/norm
                dist_theta = (X1[i][0]*X2[j][0]+X1[i][1]*X2[j][1]+X1[i][2]*X2[j][2]) / math.sqrt( (X1[i][0]**2+X1[i][1]**2+X1[i][2]**2)*(X2[j][0]**2+X2[j][1]**2+X2[j][2]**2) )
                #dist_theta = math.sqrt((1+dist_theta)/2)
                bin_theta_index = (dist_theta-bin_theta_min)/(bin_theta_max-bin_theta_min)*n_theta_bin
                   
                #dist_mu = ((X1[i][0]**2+X1[i][1]**2+X1[i][2]**2) - (X1[i][0]*X2[j][0]+X1[i][1]*X2[j][1]+X1[i][2]*X2[j][2])) / math.sqrt( (X1[i][0]**2+X1[i][1]**2+X1[i][2]**2)*((X1[i][0]-X2[j][0])**2+(X1[i][1]-X2[j][1])**2+(X1[i][2]-X2[j][2])**2) )
                #bin_mu_index = (dist_mu-bin_mu_min)/(bin_mu_max-bin_mu_min)*n_mu_bin         

                #dist_theta = ((X2[j][0]**2+X2[j][1]**2+X2[j][2]**2) - (X1[i][0]*X2[j][0]+X1[i][1]*X2[j][1]+X1[i][2]*X2[j][2])) / math.sqrt( ((X2[j][0]**2+X2[j][1]**2+X2[j][2]**2))*((X1[i][0]-X2[j][0])**2+(X1[i][1]-X2[j][1])**2+(X1[i][2]-X2[j][2])**2) )
                #bin_theta_index = (dist_theta-bin_theta_min)/(bin_theta_max-bin_theta_min)*n_theta_bin
                count[int(bin_s_index),int(bin_mu_index),int(bin_theta_index)] += weights1[i]*weights2[j]
                      
    return count

@njit(parallel=True,cache=True)
def auto_counter_3d(X,weights,bins_s,bins_mu,bins_theta,n_cells,indices_cells,L_cell_pair,L_cells):
    """
    Make sure numba has been initialized with the correct number 
    of threads : numba.set_num_threads(Nthread) 
    Note : dim of bins depends on the mode
    """
   
    bin_s_min = bins_s[0]
    bin_s_max = bins_s[-1]
    n_s_bin = len(bins_s)-1

    bin_mu_min = bins_mu[0]
    bin_mu_max = bins_mu[-1]
    n_mu_bin = len(bins_mu)-1
    
    bin_theta_min = bins_theta[0]
    bin_theta_max = bins_theta[-1]
    n_theta_bin = len(bins_theta)-1

    count = np.zeros((n_s_bin,n_mu_bin,n_theta_bin))
    #--Count pairs within the same cell:
    indices_cells_flat = indices_cells[:,0]*(n_cells[1]*n_cells[2]) + indices_cells[:,1]*(n_cells[2]) +indices_cells[:,2]
    
    for i_cell in prange(L_cells.shape[0]):
        w = (indices_cells_flat==i_cell)
        count += auto_count_3d(X[w],weights[w],bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin,bin_theta_min,bin_theta_max,n_theta_bin)

    #--Count pairs between pair of cells       
    L_cell_pair_flat_1 = L_cell_pair[:,0]*(n_cells[1]*n_cells[2]) + L_cell_pair[:,1]*(n_cells[2]) +L_cell_pair[:,2]      
    L_cell_pair_flat_2 = L_cell_pair[:,3]*(n_cells[1]*n_cells[2]) + L_cell_pair[:,4]*(n_cells[2]) +L_cell_pair[:,5]      
                             
    for i_pair in prange(L_cell_pair.shape[0]):
        #-- acces the positions in one cell-pair :
        w1 = (indices_cells_flat==L_cell_pair_flat_1[i_pair])
        w2 = (indices_cells_flat==L_cell_pair_flat_2[i_pair])
                 
        count += cross_count_3d(X[w1],X[w2],weights[w1],weights[w2],bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin,bin_theta_min,bin_theta_max,n_theta_bin)
        
    return count #+ count[:,::-1,::-1]

@njit(parallel=True,cache=True)
def cross_counter_3d(X1,X2,weights1,weights2,bins_s,bins_mu,bins_theta,n_cells,indices_cells1,indices_cells2,L_cell_pair,L_cells):
    """
    Make sure numba has been initialized with the correct number 
    of threads : numba.set_num_threads(Nthread) 
    Note : dim of bins depends on the mode
    """
   
    bin_s_min = bins_s[0]
    bin_s_max = bins_s[-1]
    n_s_bin = len(bins_s)-1

    bin_mu_min = bins_mu[0]
    bin_mu_max = bins_mu[-1]
    n_mu_bin = len(bins_mu)-1
    
    bin_theta_min = bins_theta[0]
    bin_theta_max = bins_theta[-1]
    n_theta_bin = len(bins_theta)-1

    count = np.zeros((n_s_bin,n_mu_bin,n_theta_bin))
    
    #--Count pairs within the same cell:
    L_cells_flat =  L_cells[:,0]*(n_cells[1]*n_cells[2]) + L_cells[:,1]*(n_cells[2]) +L_cells[:,2]#
    indices_cells_flat_1 = indices_cells1[:,0]*(n_cells[1]*n_cells[2]) + indices_cells1[:,1]*(n_cells[2]) +indices_cells1[:,2]
    indices_cells_flat_2 = indices_cells2[:,0]*(n_cells[1]*n_cells[2]) + indices_cells2[:,1]*(n_cells[2]) +indices_cells2[:,2]

    for i_cell in prange(L_cells.shape[0]):
        w1 = (indices_cells_flat_1==i_cell)
        w2 = (indices_cells_flat_2==i_cell)
        count += cross_count_3d(X1[w1],X2[w2],weights1[w1],weights2[w2],bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin,bin_theta_min,bin_theta_max,n_theta_bin)

    #--Count pairs between pair of cells       
    L_cell_pair_flat_1 = L_cell_pair[:,0]*(n_cells[1]*n_cells[2]) + L_cell_pair[:,1]*(n_cells[2]) +L_cell_pair[:,2]      
    L_cell_pair_flat_2 = L_cell_pair[:,3]*(n_cells[1]*n_cells[2]) + L_cell_pair[:,4]*(n_cells[2]) +L_cell_pair[:,5]      
                             
    for i_pair in prange(L_cell_pair.shape[0]):
        #-- acces the positions in one cell-pair :
        w11 = (indices_cells_flat_1==L_cell_pair_flat_1[i_pair])
        w22 = (indices_cells_flat_2==L_cell_pair_flat_2[i_pair])
        count += cross_count_3d(X1[w11],X2[w22],weights1[w11],weights2[w22],bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin,bin_theta_min,bin_theta_max,n_theta_bin)
        
    for i_pair in prange(L_cell_pair.shape[0]):
        #-- acces the positions in one cell-pair :
        w12 = (indices_cells_flat_1==L_cell_pair_flat_2[i_pair])
        w21 = (indices_cells_flat_2==L_cell_pair_flat_1[i_pair])
        count += cross_count_3d(X1[w12],X2[w21],weights1[w12],weights2[w21],bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin,bin_theta_min,bin_theta_max,n_theta_bin)

    return count


######################## Turner Code ############################


@njit(parallel=False,cache=True)
def psi1_auto(X,nbar,v,v_sig,alpha,v_amp,bin_s_min,bin_s_max,n_s_bin):

    num = np.zeros((n_s_bin))
    den = np.zeros((n_s_bin))
    eta = v/alpha
    sum_w = np.zeros(1)
    for i in prange(X.shape[0]):
        sum_w[0] +=  1 / (alpha[i]*v_amp*nbar[i] + v_sig[i]**2/alpha[i])
        for j in prange(i+1,X.shape[0]):
            
            deltax = X[i][0] - X[j][0]
            deltay = X[i][1] - X[j][1]
            deltaz = X[i][2] - X[j][2]
            #Norms of r, ra, rb vectors
            norm_r = math.sqrt((deltax)**2 + (deltay)**2 + (deltaz)**2)
            norm_ri = math.sqrt((X[i][0])**2 + (X[i][1])**2 + (X[i][2])**2)
            norm_rj = math.sqrt((X[j][0])**2 + (X[j][1])**2 + (X[j][2])**2) 
            #Geometry calculations
            cosAB = (X[i][0]/norm_ri * X[j][0]/norm_rj) + (X[i][1]/norm_ri * X[j][1]/norm_rj) + (X[i][2]/norm_ri * X[j][2]/norm_rj)
            #weight
            w_i = 1 / (alpha[i]*v_amp*nbar[i] + v_sig[i]**2/alpha[i])
            w_j = 1 / (alpha[j]*v_amp*nbar[j] + v_sig[j]**2/alpha[j])

            bin_s_index = (norm_r-bin_s_min)/(bin_s_max-bin_s_min)*n_s_bin
            if n_s_bin>bin_s_index>=0 :
                num[int(bin_s_index)] += w_i*w_j * cosAB * eta[i]*eta[j]
                den[int(bin_s_index)] += w_i*w_j * cosAB** 2 

    return num,den,sum_w

@njit(parallel=False,cache=True)
def psi1_cross(X1,X2,nbar1,nbar2,v1,v2,v1_sig,v2_sig,alpha1,alpha2,v1_amp,v2_amp,bin_s_min,bin_s_max,n_s_bin):

    num = np.zeros((n_s_bin))
    den = np.zeros((n_s_bin))
    eta1 = v1/alpha1
    eta2 = v2/alpha2
    sum_w = np.zeros(1)
    for i in prange(len(X1)):
        sum_w[0] +=  1 / (alpha1[i]*v1_amp*nbar1[i] + v1_sig[i]**2/alpha1[i])
        for j in prange(len(X2)):
            
            deltax = X1[i][0] - X2[j][0]
            deltay = X1[i][1] - X2[j][1]
            deltaz = X1[i][2] - X2[j][2]
            #Norms of r, ra, rb vectors
            norm_r = math.sqrt((deltax)**2 + (deltay)**2 + (deltaz)**2)
            norm_ri = math.sqrt((X1[i][0])**2 + (X1[i][1])**2 + (X1[i][2])**2)
            norm_rj = math.sqrt((X2[j][0])**2 + (X2[j][1])**2 + (X2[j][2])**2) 
            #Geometry calculations
            cosAB = (X1[i][0]/norm_ri * X2[j][0]/norm_rj) + (X1[i][1]/norm_ri * X2[j][1]/norm_rj) + (X1[i][2]/norm_ri * X2[j][2]/norm_rj)
            #weight
            w_i = 1 / (alpha1[i]*v1_amp*nbar1[i] + v1_sig[i]**2/alpha1[i])
            w_j = 1 / (alpha2[j]*v2_amp*nbar2[j] + v2_sig[j]**2/alpha2[j])

            bin_s_index = (norm_r-bin_s_min)/(bin_s_max-bin_s_min)*n_s_bin
            if n_s_bin>bin_s_index>=0 :
                num[int(bin_s_index)] += w_i*w_j * cosAB * eta1[i]*eta2[j]
                den[int(bin_s_index)] += w_i*w_j * cosAB** 2 

    return num,den,sum_w

@njit(parallel=False,cache=True)
def psi2_auto(X,nbar,v,v_sig,alpha,v_amp,bin_s_min,bin_s_max,n_s_bin):

    num = np.zeros((n_s_bin))
    den = np.zeros((n_s_bin))
    eta = v/alpha
    sum_w = np.zeros(1)
    for i in prange(X.shape[0]):
        sum_w[0] +=  1 / (alpha[i]*v_amp*nbar[i] + v_sig[i]**2/alpha[i])
        for j in prange(i+1,X.shape[0]):
            
            deltax = X[i][0] - X[j][0]
            deltay = X[i][1] - X[j][1]
            deltaz = X[i][2] - X[j][2]
            #Norms of r, ra, rb vectors
            norm_r = math.sqrt((deltax)**2 + (deltay)**2 + (deltaz)**2)
            norm_ri = math.sqrt((X[i][0])**2 + (X[i][1])**2 + (X[i][2])**2)
            norm_rj = math.sqrt((X[j][0])**2 + (X[j][1])**2 + (X[j][2])**2) 
            #Geometry calculations
            cosA  = (X[i][0]/norm_ri * deltax/norm_r)   + (X[i][1]/norm_ri * deltay/norm_r)   + (X[i][2]/norm_ri * deltaz/norm_r)
            cosB  = (X[j][0]/norm_rj * deltax/norm_r)   + (X[j][1]/norm_rj * deltay/norm_r)   + (X[j][2]/norm_rj * deltaz/norm_r)
            cosAB = (X[i][0]/norm_ri * X[j][0]/norm_rj) + (X[i][1]/norm_ri * X[j][1]/norm_rj) + (X[i][2]/norm_ri * X[j][2]/norm_rj)
            #weight
            w_i = 1 / (alpha[i]*v_amp*nbar[i] + v_sig[i]**2/alpha[i])
            w_j = 1 / (alpha[j]*v_amp*nbar[j] + v_sig[j]**2/alpha[j])

            bin_s_index = (norm_r-bin_s_min)/(bin_s_max-bin_s_min)*n_s_bin
            if n_s_bin>bin_s_index>=0 :
                num[int(bin_s_index)] += w_i*w_j * cosA * cosB * eta[i]*eta[j]
                den[int(bin_s_index)] += w_i*w_j * cosA * cosB *cosAB

    return num,den,sum_w

@njit(parallel=False,cache=True)
def psi2_cross(X1,X2,nbar1,nbar2,v1,v2,v1_sig,v2_sig,alpha1,alpha2,v1_amp,v2_amp,bin_s_min,bin_s_max,n_s_bin):

    num = np.zeros((n_s_bin))
    den = np.zeros((n_s_bin))
    eta1 = v1/alpha1
    eta2 = v2/alpha2
    sum_w = np.zeros(1)
    for i in prange(len(X1)):
        sum_w[0] +=  1 / (alpha1[i]*v1_amp*nbar1[i] + v1_sig[i]**2/alpha1[i])
        for j in prange(len(X2)):
            
            deltax = X1[i][0] - X2[j][0]
            deltay = X1[i][1] - X2[j][1]
            deltaz = X1[i][2] - X2[j][2]
            #Norms of r, ra, rb vectors
            norm_r = math.sqrt((deltax)**2 + (deltay)**2 + (deltaz)**2)
            norm_ri = math.sqrt((X1[i][0])**2 + (X1[i][1])**2 + (X1[i][2])**2)
            norm_rj = math.sqrt((X2[j][0])**2 + (X2[j][1])**2 + (X2[j][2])**2) 
            #Geometry calculations
            cosA  = (X1[i][0]/norm_ri * deltax/norm_r)   + (X1[i][1]/norm_ri * deltay/norm_r)   + (X1[i][2]/norm_ri * deltaz/norm_r)
            cosB  = (X2[j][0]/norm_rj * deltax/norm_r)   + (X2[j][1]/norm_rj * deltay/norm_r)   + (X2[j][2]/norm_rj * deltaz/norm_r)
            cosAB = (X1[i][0]/norm_ri * X2[j][0]/norm_rj) + (X1[i][1]/norm_ri * X2[j][1]/norm_rj) + (X1[i][2]/norm_ri * X2[j][2]/norm_rj)
            #weight
            w_i = 1 / (alpha1[i]*v1_amp*nbar1[i] + v1_sig[i]**2/alpha1[i])
            w_j = 1 / (alpha2[j]*v2_amp*nbar2[j] + v2_sig[j]**2/alpha2[j])

            bin_s_index = (norm_r-bin_s_min)/(bin_s_max-bin_s_min)*n_s_bin
            if n_s_bin>bin_s_index>=0 :
                num[int(bin_s_index)] += w_i*w_j * cosA * cosB * eta1[i]*eta2[j]
                den[int(bin_s_index)] += w_i*w_j * cosA * cosB *cosAB

    return num,den,sum_w

@njit(parallel=False,cache=True)
def psi3_auto(X_v,X_d,n_v,n_d,v,v_sig,alpha,v_amp,d_amp,bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin):

    num = np.zeros((n_s_bin,n_mu_bin))
    den = np.zeros((n_s_bin,n_mu_bin))
    eta = v/alpha
    sum_w_v = np.zeros(1)
    sum_w_d = np.zeros(1)

    for i in prange(len(X_v)):
        sum_w_v[0] +=   1 / (alpha[i]*v_amp*n_v[i] + v_sig[i]**2/alpha[i])
    for i in prange(len(X_d)):
        sum_w_d[0] += 1 / (1 + n_d[i] *d_amp)
        for j in prange(len(X_v)):

            deltax = X_d[i][0] - X_v[j][0]
            deltay = X_d[i][1] - X_v[j][1]
            deltaz = X_d[i][2] - X_v[j][2]
            #Norms of r, ra, rb vectors
            norm_r = math.sqrt((deltax)**2 + (deltay)**2 + (deltaz)**2)
            norm_ri = math.sqrt((X_d[i][0])**2 + (X_d[i][1])**2 + (X_d[i][2])**2)
            norm_rj = math.sqrt((X_v[j][0])**2 + (X_v[j][1])**2 + (X_v[j][2])**2) 

            bin_s_index = (norm_r-bin_s_min)/(bin_s_max-bin_s_min)*n_s_bin
            
            if n_s_bin>bin_s_index>=0 :
                #angle bisector
                rmu_x = (norm_ri*norm_rj)/(norm_ri+norm_rj) * (X_d[i][0]/norm_ri + X_v[j][0]/norm_rj)
                rmu_y = (norm_ri*norm_rj)/(norm_ri+norm_rj) * (X_d[i][1]/norm_ri + X_v[j][1]/norm_rj)
                rmu_z = (norm_ri*norm_rj)/(norm_ri+norm_rj) * (X_d[i][2]/norm_ri + X_v[j][2]/norm_rj)
                norm_mu =  math.sqrt((rmu_x)**2 + (rmu_y)**2 + (rmu_z)**2)

                cosmu =  (rmu_x/norm_mu)*(deltax/norm_r) + (rmu_y/norm_mu)*(deltay/norm_r) + (rmu_z/norm_mu)*(deltaz/norm_r)
                #weight
                w_i = 1 / (1 + n_d[i] *d_amp)
                w_j = 1 / (alpha[j]*v_amp*n_v[j] + v_sig[j]**2/alpha[j])

                bin_mu_index = (cosmu-bin_mu_min)/(bin_mu_max-bin_mu_min)*n_mu_bin

                num[int(bin_s_index),int(bin_mu_index)] += w_i * w_j * v[j]
                den[int(bin_s_index),int(bin_mu_index)] += w_i * w_j

    return num,den,sum_w_d,sum_w_v

@njit(parallel=False,cache=True)
def psi3_cross(X_v,X_d,n_v,n_d,v,v_sig,alpha,v_amp,d_amp,bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin):

    num = np.zeros((n_s_bin,n_mu_bin))
    den = np.zeros((n_s_bin,n_mu_bin))
    eta = v/alpha
    sum_w_v = np.zeros(1)
    sum_w_d = np.zeros(1)

    for i in prange(len(X_v)):
        sum_w_v[0] +=   1 / (alpha[i]*v_amp*n_v[i] + v_sig[i]**2/alpha[i])
    for i in prange(len(X_d)):
        sum_w_d[0] += 1 / (1 + n_d[i] *d_amp)
        for j in prange(len(X_v)):

            deltax = X_d[i][0] - X_v[j][0]
            deltay = X_d[i][1] - X_v[j][1]
            deltaz = X_d[i][2] - X_v[j][2]
            #Norms of r, ra, rb vectors
            norm_r = math.sqrt((deltax)**2 + (deltay)**2 + (deltaz)**2)
            norm_ri = math.sqrt((X_d[i][0])**2 + (X_d[i][1])**2 + (X_d[i][2])**2)
            norm_rj = math.sqrt((X_v[j][0])**2 + (X_v[j][1])**2 + (X_v[j][2])**2) 

            bin_s_index = (norm_r-bin_s_min)/(bin_s_max-bin_s_min)*n_s_bin
            
            if n_s_bin>bin_s_index>=0 :
                #angle bisector
                rmu_x = (norm_ri*norm_rj)/(norm_ri+norm_rj) * (X_d[i][0]/norm_ri + X_v[j][0]/norm_rj)
                rmu_y = (norm_ri*norm_rj)/(norm_ri+norm_rj) * (X_d[i][1]/norm_ri + X_v[j][1]/norm_rj)
                rmu_z = (norm_ri*norm_rj)/(norm_ri+norm_rj) * (X_d[i][2]/norm_ri + X_v[j][2]/norm_rj)
                norm_mu =  math.sqrt((rmu_x)**2 + (rmu_y)**2 + (rmu_z)**2)

                cosmu =  (rmu_x/norm_mu)*(deltax/norm_r) + (rmu_y/norm_mu)*(deltay/norm_r) + (rmu_z/norm_mu)*(deltaz/norm_r)
                #weight
                w_i = 1 / (1 + n_d[i] *d_amp)
                w_j = 1 / (alpha[j]*v_amp*n_v[j] + v_sig[j]**2/alpha[j])

                bin_mu_index = (cosmu-bin_mu_min)/(bin_mu_max-bin_mu_min)*n_mu_bin

                num[int(bin_s_index),int(bin_mu_index)] += w_i * w_j * v[j]
                den[int(bin_s_index),int(bin_mu_index)] += w_i * w_j

    return num,den,sum_w_d,sum_w_v


@njit(parallel=True,cache=True)
def psi1_counter(X,nbar,v,v_sig,alpha,v_amp,bins_s,n_cells,indices_cells,L_cell_pair,L_cells):
    """
    Make sure numba has been initialized with the correct number 
    of threads : numba.set_num_threads(Nthread) 
    Note : dim of bins depends on the mode
    """

    bin_s_min = bins_s[0]
    bin_s_max = bins_s[-1]
    n_s_bin = len(bins_s)-1

    num = np.zeros((n_s_bin))
    den = np.zeros((n_s_bin))
    sum_w =  np.zeros((1))
    #--Count pairs within the same cell:
    indices_cells_flat = indices_cells[:,0]*(n_cells[1]*n_cells[2]) + indices_cells[:,1]*(n_cells[2]) +indices_cells[:,2]
    
    for i_cell in prange(L_cells.shape[0]):
        w = (indices_cells_flat==i_cell)
        count = psi1_auto(X[w],nbar[w],v[w],v_sig[w],alpha[w],v_amp,bin_s_min,bin_s_max,n_s_bin)
        num += count[0]
        den += count[1]
        sum_w += count[2]
    #--Count pairs between pair of cells       
    L_cell_pair_flat_1 = L_cell_pair[:,0]*(n_cells[1]*n_cells[2]) + L_cell_pair[:,1]*(n_cells[2]) +L_cell_pair[:,2]      
    L_cell_pair_flat_2 = L_cell_pair[:,3]*(n_cells[1]*n_cells[2]) + L_cell_pair[:,4]*(n_cells[2]) +L_cell_pair[:,5]      
                             
    for i_pair in prange(L_cell_pair.shape[0]):
        #-- acces the positions in one cell-pair :
        w1 = (indices_cells_flat==L_cell_pair_flat_1[i_pair])
        w2 = (indices_cells_flat==L_cell_pair_flat_2[i_pair])

        count = psi1_cross(X[w1],X[w2],nbar[w1],nbar[w2],v[w1],v[w2],v_sig[w1],v_sig[w2],alpha[w1],alpha[w2],v_amp,v_amp,bin_s_min,bin_s_max,n_s_bin)        
        num += count[0]
        den += count[1]
        sum_w += count[2]
        
    return num,den,sum_w

@njit(parallel=True,cache=True)
def psi2_counter(X,nbar,v,v_sig,alpha,v_amp,bins_s,n_cells,indices_cells,L_cell_pair,L_cells):
    """
    Make sure numba has been initialized with the correct number 
    of threads : numba.set_num_threads(Nthread) 
    Note : dim of bins depends on the mode
    """

    bin_s_min = bins_s[0]
    bin_s_max = bins_s[-1]
    n_s_bin = len(bins_s)-1

    num = np.zeros((n_s_bin))
    den = np.zeros((n_s_bin))
    sum_w =  np.zeros((1))
    #--Count pairs within the same cell:
    indices_cells_flat = indices_cells[:,0]*(n_cells[1]*n_cells[2]) + indices_cells[:,1]*(n_cells[2]) +indices_cells[:,2]
    
    for i_cell in prange(L_cells.shape[0]):
        w = (indices_cells_flat==i_cell)
        count = psi2_auto(X[w],nbar[w],v[w],v_sig[w],alpha[w],v_amp,bin_s_min,bin_s_max,n_s_bin)
        num += count[0]
        den += count[1]
        sum_w += count[2]
    #--Count pairs between pair of cells       
    L_cell_pair_flat_1 = L_cell_pair[:,0]*(n_cells[1]*n_cells[2]) + L_cell_pair[:,1]*(n_cells[2]) +L_cell_pair[:,2]      
    L_cell_pair_flat_2 = L_cell_pair[:,3]*(n_cells[1]*n_cells[2]) + L_cell_pair[:,4]*(n_cells[2]) +L_cell_pair[:,5]      
                             
    for i_pair in prange(L_cell_pair.shape[0]):
        #-- acces the positions in one cell-pair :
        w1 = (indices_cells_flat==L_cell_pair_flat_1[i_pair])
        w2 = (indices_cells_flat==L_cell_pair_flat_2[i_pair])

        count = psi2_cross(X[w1],X[w2],nbar[w1],nbar[w2],v[w1],v[w2],v_sig[w1],v_sig[w2],alpha[w1],alpha[w2],v_amp,v_amp,bin_s_min,bin_s_max,n_s_bin)         
        num += count[0]
        den += count[1]
        sum_w += count[2]
        
    return num,den,sum_w

@njit(parallel=True,cache=True)
def psi3_counter(X_v,X_d,n_v,n_d,v,v_sig,alpha,v_amp,d_amp,bins_s,bins_mu,n_cells,indices_cells1,indices_cells2,L_cell_pair,L_cells):
    """
    Make sure numba has been initialized with the correct number 
    of threads : numba.set_num_threads(Nthread) 
    Note : dim of bins depends on the mode
    """
    bin_s_min = bins_s[0]
    bin_s_max = bins_s[-1]
    n_s_bin = len(bins_s)-1

    bin_mu_min = bins_mu[0]
    bin_mu_max = bins_mu[-1]
    n_mu_bin = len(bins_mu)-1

    num = np.zeros((n_s_bin,n_mu_bin))
    den = np.zeros((n_s_bin,n_mu_bin))
    sum_w_v = np.zeros((1))
    sum_w_d = np.zeros((1))
    
    #--Count pairs within the same cell:
    L_cells_flat =  L_cells[:,0]*(n_cells[1]*n_cells[2]) + L_cells[:,1]*(n_cells[2]) +L_cells[:,2]#
    indices_cells_flat_1 = indices_cells1[:,0]*(n_cells[1]*n_cells[2]) + indices_cells1[:,1]*(n_cells[2]) +indices_cells1[:,2]
    indices_cells_flat_2 = indices_cells2[:,0]*(n_cells[1]*n_cells[2]) + indices_cells2[:,1]*(n_cells[2]) +indices_cells2[:,2]
    for i_cell in prange(L_cells.shape[0]):
        w1 = (indices_cells_flat_1==i_cell)
        w2 = (indices_cells_flat_2==i_cell)
        count = psi3_cross(X_v[w1],X_d[w2],n_v[w1],n_d[w2],v[w1],v_sig[w1],alpha[w1],v_amp,d_amp,bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin)
        num += count[0]
        den += count[1]
        sum_w_d += count[2]
        sum_w_v += count[3]

    #--Count pairs between pair of cells       
    L_cell_pair_flat_1 = L_cell_pair[:,0]*(n_cells[1]*n_cells[2]) + L_cell_pair[:,1]*(n_cells[2]) +L_cell_pair[:,2]      
    L_cell_pair_flat_2 = L_cell_pair[:,3]*(n_cells[1]*n_cells[2]) + L_cell_pair[:,4]*(n_cells[2]) +L_cell_pair[:,5]      
                             
    for i_pair in prange(L_cell_pair.shape[0]):
        #-- acces the positions in one cell-pair :
        w11 = (indices_cells_flat_1==L_cell_pair_flat_1[i_pair])
        w22 = (indices_cells_flat_2==L_cell_pair_flat_2[i_pair])
        count = psi3_cross(X_v[w11],X_d[w22],n_v[w11],n_d[w22],v[w11],v_sig[w11],alpha[w11],v_amp,d_amp,bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin)
        num += count[0]
        den += count[1]
        sum_w_d += count[2]
        sum_w_v += count[3]

    for i_pair in prange(L_cell_pair.shape[0]):
        #-- acces the positions in one cell-pair :
        w12 = (indices_cells_flat_1==L_cell_pair_flat_2[i_pair])
        w21 = (indices_cells_flat_2==L_cell_pair_flat_1[i_pair])
        count = psi3_cross(X_v[w12],X_d[w21],n_v[w12],n_d[w21],v[w12],v_sig[w12],alpha[w12],v_amp,d_amp,bin_s_min,bin_s_max,n_s_bin,bin_mu_min,bin_mu_max,n_mu_bin)
        num += count[0]
        den += count[1]
        sum_w_d += count[2]
        sum_w_v += count[3]
    return num,den,sum_w_d,sum_w_v


######################## Correlation Function ############################

class CorrelationFunctionPV:
    def __init__(self,edges,
                 dat_pos,dat_n,dat_v,dat_vsig,dat_alpha,dat_vamp,
                 ran_pos,ran_n,ran_v,ran_vsig,ran_alpha,ran_vamp,
                 RR=None,Nthread=32):
            
            
        self.edges = edges
        self.Nthread = Nthread

        numba.set_num_threads(Nthread)
        self.s_max = edges[-1]

        self.dat_pos,self.ran_pos = dat_pos,ran_pos
        self.dat_n,self.ran_n = dat_n,ran_n
        self.dat_v,self.ran_v = dat_v,ran_v
        self.dat_vsig,self.ran_vsig = dat_vsig,ran_vsig
        self.dat_alpha,self.ran_alpha = dat_alpha,ran_alpha
        self.dat_vamp,self.ran_vamp = dat_vamp,ran_vamp

        self.n_cells = np.int64(np.floor((np.max(ran_pos,axis=0)-np.min(ran_pos,axis=0))/self.s_max))
        self.indices_cells_dat = np.int64(np.floor(self.n_cells * (dat_pos-np.min(ran_pos,axis=0))/(np.max(ran_pos,axis=0)-np.min(ran_pos,axis=0))))
        self.indices_cells_ran = np.int64(np.floor(self.n_cells * (ran_pos-np.min(ran_pos,axis=0))/(np.max(ran_pos,axis=0)-np.min(ran_pos,axis=0))))
         
        L_cell_pair,L_cells = get_cell_pairs(self.n_cells)
        self.L_cells = np.int64(L_cells)
        self.L_cell_pair = np.int64(L_cell_pair)

    def DD(self,mode):
        self.mode = mode

        if mode=='psi1':
             num,den,sum_w = psi1_counter(self.dat_pos,self.dat_n,self.dat_v,self.dat_vsig,self.dat_alpha,self.dat_vamp,
                          self.edges,self.n_cells,self.indices_cells_dat,self.L_cell_pair,self.L_cells)
        if mode =='psi2':
             num,den,sum_w = psi2_counter(self.dat_pos,self.dat_n,self.dat_v,self.dat_vsig,self.dat_alpha,self.dat_vamp,
                          self.edges,self.n_cells,self.indices_cells_dat,self.L_cell_pair,self.L_cells)
        return num,den,sum_w
        
    def RR(self,mode):
        self.mode = mode

        if mode=='psi1':
             num,den,sum_w = psi1_counter(self.ran_pos,self.ran_n,self.ran_v,self.ran_vsig,self.ran_alpha,self.ran_vamp,
                          self.edges,self.n_cells,self.indices_cells_ran,self.L_cell_pair,self.L_cells)
        if mode =='psi2':
             num,den,sum_w = psi2_counter(self.ran_pos,self.ran_n,self.ran_v,self.ran_vsig,self.ran_alpha,self.ran_vamp,
                          self.edges,self.n_cells,self.indices_cells_ran,self.L_cell_pair,self.L_cells)
        return num,den,sum_w
        
    def psi(self,mode):

        self.mode = mode
        
        DD = self.DD(mode)
        RR = self.RR(mode)

        psi = (RR[2]/DD[2])**2 * DD[0]/RR[1]

        return psi

class CorrelationFunctionPVD:
    def __init__(self,edges,
                 dat_vpos,dat_vn,dat_v,dat_vsig,dat_valpha,dat_vamp,
                 dat_dpos,dat_dn,dat_damp,
                 ran_vpos,ran_vn,ran_v,ran_vsig,ran_valpha,ran_vamp,
                 ran_dpos,ran_dn,ran_damp,
                 Nthread=32):
            
            
        self.edges = edges
        self.Nthread = Nthread

        numba.set_num_threads(Nthread)
        self.s_max = edges[0][-1]

        self.dat_vpos,self.ran_vpos = dat_vpos,ran_vpos
        self.dat_dpos,self.ran_dpos = dat_dpos,ran_dpos

        self.dat_vn,self.ran_vn = dat_vn,ran_vn
        self.dat_dn,self.ran_dn = dat_dn,ran_dn

        self.dat_vamp,self.ran_vamp = dat_vamp,ran_vamp
        self.dat_damp,self.ran_damp = dat_damp,ran_damp

        self.dat_v,self.ran_v = dat_v,ran_v
        self.dat_vsig,self.ran_vsig = dat_vsig,ran_vsig
        self.dat_valpha,self.ran_valpha = dat_valpha,ran_valpha

        self.n_cells = np.int64(np.floor((np.max(self.ran_dpos,axis=0)-np.min(self.ran_dpos,axis=0))/self.s_max))
        self.indices_cells_dat_v = np.int64(np.floor(self.n_cells * (dat_vpos-np.min(ran_vpos,axis=0))/(np.max(ran_vpos,axis=0)-np.min(ran_vpos,axis=0))))
        self.indices_cells_dat_d = np.int64(np.floor(self.n_cells * (dat_dpos-np.min(ran_dpos,axis=0))/(np.max(ran_dpos,axis=0)-np.min(ran_dpos,axis=0))))
        self.indices_cells_ran_v = np.int64(np.floor(self.n_cells * (ran_vpos-np.min(ran_vpos,axis=0))/(np.max(ran_vpos,axis=0)-np.min(ran_vpos,axis=0))))
        self.indices_cells_ran_d = np.int64(np.floor(self.n_cells * (ran_dpos-np.min(ran_dpos,axis=0))/(np.max(ran_dpos,axis=0)-np.min(ran_dpos,axis=0))))

        L_cell_pair,L_cells = get_cell_pairs(self.n_cells)
        self.L_cells = np.int64(L_cells)
        self.L_cell_pair = np.int64(L_cell_pair)

    def DD(self):        
        num,den,sum_w_d,sum_w_v = psi3_counter(self.dat_vpos,self.dat_dpos,
                                     self.dat_vn,self.dat_dn,
                                     self.dat_v,self.dat_vsig,
                                     self.dat_valpha,
                                     self.dat_vamp,self.dat_damp,
                                     self.edges[0],self.edges[1],
                                     self.n_cells,self.indices_cells_dat_v,self.indices_cells_dat_d,self.L_cell_pair,self.L_cells)
       
        return num,den,sum_w_d,sum_w_v
        
    def RR(self):        
        num,den,sum_w_d,sum_w_v = psi3_counter(self.ran_vpos,self.ran_dpos,
                                     self.ran_vn,self.ran_dn,
                                     self.ran_v,self.ran_vsig,
                                     self.ran_valpha,
                                     self.ran_vamp,self.ran_damp,
                                     self.edges[0],self.edges[1],
                                     self.n_cells,self.indices_cells_ran_v,self.indices_cells_ran_d,self.L_cell_pair,self.L_cells)
       
        return num,den,sum_w_d,sum_w_v

    def DR(self):        
        num,den,sum_w_d,sum_w_v = psi3_counter(self.dat_vpos,self.ran_dpos,
                                     self.dat_vn,self.ran_dn,
                                     self.dat_v,self.dat_vsig,
                                     self.dat_valpha,
                                     self.dat_vamp,self.ran_damp,
                                     self.edges[0],self.edges[1],
                                     self.n_cells,self.indices_cells_dat_v,self.indices_cells_ran_d,self.L_cell_pair,self.L_cells)
       
        return num,den,sum_w_d,sum_w_v

    def RD(self):        
        num,den,sum_w_d,sum_w_v = psi3_counter(self.ran_vpos,self.dat_dpos,
                                     self.ran_vn,self.dat_dn,
                                     self.ran_v,self.ran_vsig,
                                     self.ran_valpha,
                                     self.ran_vamp,self.dat_damp,
                                     self.edges[0],self.edges[1],
                                     self.n_cells,self.indices_cells_ran_v,self.indices_cells_dat_d,self.L_cell_pair,self.L_cells)
       
        return num,den,sum_w_d,sum_w_v
    
    def psi3(self):

        DD = self.DD()
        DR = self.RD()
        RD = self.RD()
        RR = self.RR()

        psi3 = ((    RR[2]*RR[3]/DD[2]*DD[3])* DD[0]/RR[1]
                -2 * RR[2]/DD[2] * RD[0]/RR[1]
                -2 * RR[3]/DD[3] * DR[0]/RR[1]
                   + RR[0]/ RR[1])


        return psi3
class CorrelationFunction3D:
    """
    Note : bisectrice los
    """
    def __init__(self,mode,edges,data_positions,random_positions,data_weights=None,random_weights=None,RR=None,Nthread=32):
        
        
        if mode=='s':
            print('1d : binning along s')
            self.s_max = edges[-1]
        elif mode=='smu':
            print('2d : binning along (s,mu)')
            self.s_max = edges[0][-1]
        elif mode=='smutheta':
            print('3d : binning along (s,mu,theta)')
            self.s_max = edges[0][-1]
        else:
            print('Error : mode argument must be s smu or smutheta')
            
        self.edges = edges
        self.mode = mode
                
        self.data_positions = data_positions
        self.random_positions = random_positions
        self.Nthread = Nthread
        numba.set_num_threads(Nthread)
        
        if data_weights is None:
            data_weights = np.ones(data_positions.shape[0])
        if random_weights is None:
            random_weights = np.ones(random_positions.shape[0])
        
        self.N_g = np.sum(data_weights)
        self.N_r = np.sum(random_weights)   
        
        self.data_weights = data_weights
        self.random_weights = random_weights
        self.RR = RR
        
        ##-- Grid partionning
        self.n_cells = np.int64(np.floor((np.max(random_positions,axis=0)-np.min(random_positions,axis=0))/self.s_max))
        self.indices_cells_data = np.int64(np.floor(self.n_cells * (data_positions-np.min(random_positions,axis=0))/(np.max(random_positions,axis=0)-np.min(random_positions,axis=0))))
        self.indices_cells_random = np.int64(np.floor(self.n_cells * (random_positions-np.min(random_positions,axis=0))/(np.max(random_positions,axis=0)-np.min(random_positions,axis=0))))
         
        L_cell_pair,L_cells = get_cell_pairs(self.n_cells)
        self.L_cells = np.int64(L_cells)
        self.L_cell_pair = np.int64(L_cell_pair)
        
    def auto_pair_count(self,positions,weights,edges,indices_cells):
        
        start = time.time()
             
        if self.mode=='s':
            count = auto_counter_1d(positions,weights,edges,self.n_cells,indices_cells,self.L_cell_pair,self.L_cells)
     
        elif self.mode=='smu':
            count = auto_counter_2d(positions,weights,edges[0],edges[1],self.n_cells,indices_cells,self.L_cell_pair,self.L_cells)
         
        elif self.mode=='smutheta':
            count = auto_counter_3d(positions,weights,edges[0],edges[1],edges[2],self.n_cells,indices_cells,self.L_cell_pair,self.L_cells)

        print('autocount',time.time()-start)

        return count
            
    def cross_pair_count(self,positions1,positions2,weights1,weights2,edges,indices_cells1,indices_cells2):
        start = time.time()
        
        if self.mode=='s':
            count=cross_counter_1d(positions1,positions2,weights1,weights2,edges,
                 self.n_cells,indices_cells1,indices_cells2,self.L_cell_pair,self.L_cells)     
        elif self.mode=='smu':
            count=cross_counter_2d(positions1,positions2,weights1,weights2,edges[0],edges[1],
                 self.n_cells,indices_cells1,indices_cells2,self.L_cell_pair,self.L_cells)         
        elif self.mode=='smutheta':
            count=cross_counter_3d(positions1,positions2,weights1,weights2,edges[0],edges[1],edges[2],
                 self.n_cells,indices_cells1,indices_cells2,self.L_cell_pair,self.L_cells)
            
        print('crosscount',time.time()-start)

        return count
        
        
    def run(self):
        
        self.DD = self.auto_pair_count(self.data_positions,self.data_weights,self.edges,self.indices_cells_data)
        self.DR = self.cross_pair_count(self.data_positions,self.random_positions,self.data_weights,self.random_weights,self.edges,self.indices_cells_data,self.indices_cells_random)

        if self.mode=='s':
            self.RD = self.DR
            self.DD = 2*self.DD
        elif self.mode=='smu':
            self.RD = self.DR[:,::-1]
            self.DD = self.DD+self.DD[:,::-1]
        elif self.mode=='smutheta':
            #self.RD = np.swapaxes(self.DR,1,2)[:,::-1,::-1]
            #self.DD = self.DD+np.swapaxes(self.DD,1,2)[:,::-1,::-1]
            self.RD = self.DR[:,::-1,:]
            self.DD = self.DD+self.DD[:,::-1,:]

        if self.RR is None:
            self.RR = self.auto_pair_count(self.random_positions,self.random_weights,self.edges,self.indices_cells_random)
            if self.mode=='s':
                self.RR = 2*self.RR
            elif self.mode=='smu':
                self.RR = self.RR+self.RR[:,::-1]
            elif self.mode=='smutheta':
                #self.RR = self.RR+np.swapaxes(self.RR,1,2)[:,::-1,::-1]
                self.RR = self.RR+self.RR[:,::-1,:]

            self.RR[np.where(self.RR==0)]=1
       

        return ( self.DD * (self.N_r/self.N_g)**2 - (self.DR+self.RD) * (self.N_r/self.N_g) ) / self.RR + 1    
    
    
