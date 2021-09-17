# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 04:38:30 2021

@author: BigIs
"""

import numpy as np
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord 
from astropy import units as u
#import matplotlib.pyplot as pl

#import the table
#path = 'C:\\Users\\BigIs\\Documents\\School Stuff\\Coding/'
#data = Table.read(path + 'act_clusters.fits')
#data2 = Table.read(path + 'Clash_Clusters.csv')
#data3 = Table.read(path + 'Herschel_Clusters.csv')

# path = 'C:/Users/BigIs/Documents/Summer Research/Cluster Table Code/Catalogs/fix'
# data4 = Table.read(path + 'NIKAClustersTest.csv')


def check_z(clusters,zmin,zmax): #FUNCTION FOR CHECKING IF WITHIN Z RANGE
    col = np.ones(len(clusters['z']))
    for i in range(len(clusters['z'])):
        if clusters['z'][i] > zmin and clusters['z'][i] < zmax:
            col[i] = 1
        else:
            col[i] = 0
            
    clusters.add_column(col, name= 'InZRange',)
    
    return clusters


def compare_clusters(clusters1, clusters2, instrument): #FUNCTION FOR CHECKING FOR DATA FROM OTHER INSTRUMENTS
    # Clusters1 is the data set that you wish to add columns to
    # Clusters2 is the data set you are comparing with 
    # RA column for data sets should be 'RA' and Declination column should be 'Dec.'
    # Values for RA and Dec should be in degrees
    col1 = np.zeros(len(clusters1['RADeg']))        

    #Set allowance for match
    upboundRA = clusters1['RADeg'] + 0.09
    lowboundRA = clusters1['RADeg'] - 0.09
        
    upboundDEC = clusters1['decDeg'] + 0.04
    lowboundDEC = clusters1['decDeg'] - 0.04
    
    #compare every cluster in the primary data to every cluster in the anciliary data set to find matches
    
    for i in range(len(col1)):
        for m in range(len(clusters2['RA'])):
            if clusters2['RA'][m] <= upboundRA[i] and clusters2['RA'][m] >= lowboundRA[i] and clusters2['DEC'][m] <= upboundDEC[i] and clusters2['DEC'][m] >= lowboundDEC[i]:
                col1[i] = 1
                break
            else:
                continue        
    
    clusters1.add_column(col1, name=instrument,)
    
    #If the instrument has data for the cluster, it will be assigned a 1 in the list
    #After all clusters are checked, the list will automatically be made into a column
    #then the function returns the array with the new column added
            
    return clusters1


#function to convert units from HHMMSS to degrees for a catalog

def fixUnits(clusters):
    ra = clusters['RA']
    dec = clusters['DEC']
    for i in range(len(clusters)):
        c = SkyCoord(ra[i],dec[i], unit=(u.hourangle, u.deg))
        clusters['RA'][i] = c.ra.deg
        clusters['DEC'][i] = c.dec.deg
        
    return clusters
    

"""TESTING STUFF - WILL BE REMOVED AT A LATER DATE"""

#print(redshift_range(data,0.3,0.8))
#print(data['z'])

#print(compare_clusters(data,data2,'Clash'))
#print(compare_clusters(data,data3,'Herschel'))


#compare_clusters(data,data2,'Clash')
#compare_clusters(data,data3,'Herschel')

#data.write('act_clusters_test5.csv', format = 'csv')



