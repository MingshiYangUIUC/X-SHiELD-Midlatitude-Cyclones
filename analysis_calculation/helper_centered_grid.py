"""
Author:
Mingshi Yang (mingshi3@illinois.edu)

Date:
2026-01-16

Project:
Characteristics of Midlatitude Cyclones under Climate Change in a Global Storm-Resolving Model

Description:
This script defines helper functions for constructing a storm-centered grid and
performing horizontal bilinear interpolation using weighted averaging.
"""

import numpy as np
from scipy.interpolate import interp2d
import great_circle_calculator.great_circle_calculator as gcc # pip install great-circle-calculator
import time

# return destination lat lon given initial point, distance traveled, and bearing angle
def p1top2(lon1,lat1,d,brng):
    brng = np.deg2rad(brng)
    R = 6371 #Radius of the Earth

    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)

    lat2 = np.arcsin(np.sin(lat1)*np.cos(d/R) +np.cos(lat1)*np.sin(d/R)*np.cos(brng))

    lon2 = lon1 + np.arctan2(np.sin(brng)*np.sin(d/R)*np.cos(lat1),np.cos(d/R)-np.sin(lat1)*np.sin(lat2))

    lat2 = np.rad2deg(lat2)
    lon2 = np.rad2deg(lon2)

    return lon2,lat2

# calculate bearing angle given initial and destination points
def get_bearing(lon1,lat1,lon2,lat2):
    dLon = (lon2 - lon1)
    dLon = np.deg2rad(dLon)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    x = np.cos(lat2) * np.sin(dLon)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    brng = np.arctan2(x,y)
    brng = np.rad2deg(brng)

    return brng

# vectorized function to construct a storm centered grid
def get_grid_new(center_lon,center_lat,radius,grid_dist,beering=0): # unit in km
    ncells= radius//grid_dist*2+1
    dd = np.arange(-radius,radius+1,grid_dist)
    d0,b0 = np.meshgrid(dd,(dd*0+beering)%360)
    d1,b1 = np.meshgrid(dd,(dd*0-90+beering)%360)
    d1 = d1.T[::-1]
    lon0,lat0 = np.zeros_like(d0)+center_lon,np.zeros_like(d0)+center_lat
    
    lon1,lat1 = p1top2(lon0,lat0,d0,b0)
    #print(lon1,lat1)
    lon1 = lon1 % 360

    b1 = (get_bearing(lon1,lat1,lon0,lat0)+90) % 360
    #print(b1)
    b1[:,int((ncells-1)/2)] = beering+270
    #b1[np.where(np.abs(lon1-center_lon%360)<90)]*= -1
    #print(b1[0])
    b1[:,int((ncells-1)/2):] += 180

    lon2,lat2 = p1top2(lon1,lat1,d1,b1)
    
    #lon2[:,:int((ncells-1)/2)] = np.nan

    gnew = np.zeros((ncells,ncells,2))
    lon2[lon2>180] -= 360
    gnew[:,:,0] = np.round(lon2,10)
    gnew[:,:,1] = np.round(lat2,10)
    #gnew[:,:,0] = np.round(lon1,10)
    #gnew[:,:,1] = np.round(lat1,10)
    #print(t01-t00,t02-t01,t03-t02)
    return gnew

# get storm centered grid given storm center location and grid specs
def get_centered_lonlat(center_lon,center_lat,radius,grid_dist,grid=False,mode='new',beering=0):

    if mode == 'old': # same as method in Stoll et al. (2021) https://doi.org/10.5194/wcd-2-19-2021
        ncells= radius//grid_dist
        if center_lat ==90:
            center_lat -= 0.00001
        if center_lon > 180:
            center_lon -= 360
        
        #beering =0
        tanax= [list(gcc.point_given_start_and_bearing((center_lon, center_lat), beering, n*grid_dist*1E3)) for n in np.arange(-ncells, ncells+1)]
        tanax= np.array(tanax)
        point0= gcc.point_given_start_and_bearing((center_lon, center_lat), beering, -(radius+grid_dist)*1E3) #point one before the start of the tangential axis, used for calculation of beering_axis
        beering_axis= [gcc.bearing_at_p2((point0), (tanax[m,0], tanax[m,1])) for m in range(len(tanax))] #the wind direction along the tangential axis
        center_grid= [[list(gcc.point_given_start_and_bearing((tuple(tanax[m])), (beering_axis[m]-90)%360, n*grid_dist*1E3)) for m in range(len(tanax))] for n in np.arange(-ncells, ncells+1)]
        center_grid= np.array(center_grid)
    else: # vectorized and faster method
        center_grid = get_grid_new(center_lon,center_lat,radius,grid_dist,beering)

    lons = (center_grid[:,:,0]%360).flatten()

    lats = center_grid[:,:,1].flatten()
    if grid == True:
        return lons,lats,center_grid
    else:
        return lons,lats

# bilinear interpolation core that calculate weighted average given data and weights
def get_centered_data_with_weight(Data,w11,w12,w21,w22,ix1,ix2,iy1,iy2,iymax=720):
    #fill with nan if out of range
    iyM = max(np.max(iy1),np.max(iy2))
    if iyM > iymax:
        Data_na = np.zeros((iyM+1,len(Data[0])))
        Data_na[:len(Data)] += Data
        Data_na[len(Data):] += np.nan
        Data = Data_na
    # changed!
    D_mean = w11*Data[iy1,ix1]+w21*Data[iy1,ix2]+w12*Data[iy2,ix1]+w22*Data[iy2,ix2]
    
    return D_mean.flatten()

