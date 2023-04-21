import numpy as np
import matplotlib.pyplot as plt
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import abacusnbody.metadata
import fitsio
import time
from scipy.spatial import cKDTree
import scipy as sc
import torch
from sklearn import metrics
from numba import njit, prange
import numba
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import legendre




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
    
    


