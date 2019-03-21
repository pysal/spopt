import BaseClass as bc
from libpysal.weights.util import get_points_array_from_shapefile
from libpysal.io.fileio import FileIO as psopen
import numpy

import geopandas as gp
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist

class BCMaxPExact(bc.BaseSpOptExactSolver):    
       
    def constructModel(self, stdAttrDistMatrix, stdSpatialDistMatrix, extensiveAttr, extensiveThre, attrWeight):
        """
        structure the optimization model using or-tools
    
        """        
        self.num_units = extensiveAttr.size
        self.y = {}
        self.x = {}
    
        for i in range(self.num_units):
            for j in range(self.num_units):
                self.x[i, j] = self.spOptSolver.BoolVar('x[%i,%i]' % (i, j))  
                self.y[i, j] = self.spOptSolver.BoolVar('y[%i,%i]' % (i, j))  

        self.spOptSolver.Minimize(self.spOptSolver.Sum([(attrWeight * stdAttrDistMatrix[i][j] + (1-attrWeight) * stdSpatialDistMatrix[i][j]) * self.y[i,j] 
                                    for i in range(self.num_units)
                                  for j in range(self.num_units)]))    

        # Each unit is assigned to exactly one region.
    
        for i in range(self.num_units):
            self.spOptSolver.Add(self.spOptSolver.Sum([self.x[i, k] for k in range(self.num_units)]) == 1)
    
        # Each region's total extensive attributes reach the threshold.
    
        for k in range(self.num_units):
            for i in range(self.num_units):
                #self.spOptSolver.Add(self.spOptSolver.Sum([extensiveAttr[i] * x[i, k] for i in range(self.num_units)]) - extensiveThre * self.spOptSolver.Sum([x[i, k] for i in range(self.num_units)]) >= 0)
                self.spOptSolver.Add(self.spOptSolver.Sum([extensiveAttr[i] * self.x[i, k] for i in range(self.num_units)]) - extensiveThre * self.x[i, k] >= 0)
    
        # link yij with xik.
        for k in range(self.num_units):
            for i in range(self.num_units):
                for j in range(self.num_units):     
                    self.spOptSolver.Add(self.x[i, k] + self.x[j, k] - self.y[i, j] <= 1)
                    
    def solve(self):
        """
        override the solve function
    
        """        
        super().solve()
        #lpfile = open('result\\lpfile.txt', 'w')
        #lpfile.write(self.spOptSolver.ExportModelAsLpFormat(True))
        print('Total cost = ', self.spOptSolver.Objective().Value())
        self.labels = numpy.zeros(numObs, dtype=int)
        for i in range(self.num_units):
            for k in range(self.num_units):
                if self.x[i, k].solution_value() > 0:
                    self.labels[i] = k + 1
                    print('area %d assigned to region %d.' % (i, k+1))   
                    
class BCMaxPHeuristic(bc.BaseSpOptHeuristicSolver): 
    
    def __init__(self, stdAttrDistMatrix, stdSpatialDistMatrix, extensiveAttr, extensiveThre, attrWeight):
        self.stdAttrDistMatrix = stdAttrDistMatrix
        self.stdSpatialDistMatrix = stdSpatialDistMatrix
        self.extensiveAttr = extensiveAttr
        self.extensiveThre = extensiveThre
        self.attrWeight = attrWeight
        self.num_units = extensiveAttr.size
    
    def solve(self):
        ro = numpy.arange(self.num_units)
        numpy.random.shuffle(ro)   
        self.labels = numpy.zeros(self.num_units, dtype=int)
        regionList = []
        unAssigned = numpy.arange(self.num_units)
        regionExtensiveAttr = []
        attrDistWithinRegionList = []
        spatialDistWithinRegionList = []
        totalDistWithinRegionList = []
        C = 0

        for i in ro:
  
            if not (self.labels[i] == 0):
                continue
            
            if extensiveAttr[unAssigned].sum() > extensiveThre:           
                C += 1
                labeledID, unAssigned, extensiveAttrTotal, attrDistWithinRegion, spatialDistWithinRegion, totalDistWithinRegion = self.growCluster(i, C, unAssigned)
                regionList.append(labeledID)
                regionExtensiveAttr.append(extensiveAttrTotal)
                attrDistWithinRegionList.append(attrDistWithinRegion)
                spatialDistWithinRegionList.append(spatialDistWithinRegion)   
                totalDistWithinRegionList.append(totalDistWithinRegion)
            else:
                attrDistWithinRegionList, spatialDistWithinRegionList, totalDistWithinRegionList = self.assignEnclaves(unAssigned, regionList, attrDistWithinRegionList, spatialDistWithinRegionList, totalDistWithinRegionList, regionExtensiveAttr)
              
        print("The number of regions is %d" %(len(regionList)))
        #print(str(regionList))
        print("The region extensive attribute is %s" %(str(regionExtensiveAttr))) 
        #print(totalDistWithinRegionList.sum())
        print("Attribute distance is %f" %(sum(attrDistWithinRegionList)))
        print("Spatial distance is %f" %(sum(spatialDistWithinRegionList))) 
        print("Attribute distance is %s" %(str(attrDistWithinRegionList)))
        print("Spatial distance is %s" %(str(spatialDistWithinRegionList))) 
        
        for i, il in enumerate(self.labels):
            print('area %d assigned to region %d.' % (i, il))
        
    def growCluster(self, i, C, unAssigned):
        # Assign the cluster label to the seed area.
        self.labels[i] = C
        labeledID = numpy.array([i])
        extensiveAttrTotal = self.extensiveAttr[i]
         
        unAssigned = numpy.delete(unAssigned, numpy.where(unAssigned == i))   
        totalDistWithinRegion = 0
        
        while extensiveAttrTotal < self.extensiveThre and unAssigned.size > 0:  
            
            workingMatrix = self.stdAttrDistMatrix[labeledID[:, None], unAssigned] * self.attrWeight + self.stdSpatialDistMatrix[labeledID[:, None], unAssigned] * (1 - self.attrWeight)        
            columnSum = numpy.sum(workingMatrix, axis = 0)        
            curIndex = columnSum.argmin()
            totalDistWithinRegion += numpy.amin(columnSum)
            closestUnit = unAssigned[curIndex]   
            
            self.labels[closestUnit] = C
            labeledID = numpy.append(labeledID, closestUnit)
            extensiveAttrTotal += extensiveAttr[closestUnit]
            unAssigned = numpy.delete(unAssigned, numpy.where(unAssigned == closestUnit))   
            
        attrDistWithinRegion = numpy.sum(self.stdAttrDistMatrix[labeledID[:, None], labeledID])/2.0
            
        spatialDistWithinRegion = numpy.sum(self.stdSpatialDistMatrix[labeledID[:, None], labeledID])/2.0        
        #print(labeledID)
        #print(extensiveAttrTotal)
        #print(attrDistWithinRegion)
        #print(totalDistWithinRegion)
        
        #print('grow cluster time is %f' %(time.process_time()-t1))
        #print('unassigned is %d' %(unAssigned.size))
      
        return labeledID, unAssigned, extensiveAttrTotal, attrDistWithinRegion, spatialDistWithinRegion, totalDistWithinRegion
    
    def assignEnclaves(self, enclave, regionList, attrDistWithinRegionList, spatialDistWithinRegionList, totalDistWithinRegionList, regionExtensiveAttr):       
        
        for ec in enclave:
            minDistance = numpy.Inf
            assignedRegion = 0
            for iur, ur in enumerate(regionList):
                attrDistArray = self.stdAttrDistMatrix[ec, ur]
                spatialDistArray = self.stdSpatialDistMatrix[ec, ur]
                
                attrDistToRegion = attrDistArray.sum()
                spatialDistToRegion = spatialDistArray.sum()
                            
                distToRegion = attrDistToRegion * attrWeight + spatialDistToRegion * (1-attrWeight)
                
                if distToRegion < minDistance:
                    assignedRegion = (iur + 1)
                    minDistance = distToRegion
                    addedAttrDist = attrDistToRegion
                    addedSpatialDist = spatialDistToRegion
                    
            self.labels[ec] = assignedRegion
            regionList[assignedRegion-1] = numpy.append(regionList[assignedRegion-1], ec)
            #print('Add enclave %d to region %d' %(ec, assignedRegion))
            totalDistWithinRegionList[assignedRegion-1] += minDistance
            attrDistWithinRegionList[assignedRegion-1] += addedAttrDist
            spatialDistWithinRegionList[assignedRegion-1] += addedSpatialDist
            regionExtensiveAttr[assignedRegion-1] += extensiveAttr[ec]
        
        return attrDistWithinRegionList, spatialDistWithinRegionList, totalDistWithinRegionList
            
                        
                   

if __name__ != 'main':
    filePath = 'data\\soil_precip_temp_field_projected_sample_exact2.dbf'
    attrWeight = 0.5
    dbfReader = psopen(filePath)   
    attrsName = ['nccpi2cs']
    extensiveAttrName = 'field_ct'
    extensiveThre = 3
    
    attrs = numpy.array(dbfReader.by_col(attrsName[0]))
    attrs = numpy.reshape(attrs, (-1,1))
    numObs = attrs.size
    
    if len(attrsName) > 1:
        for an in attrsName[1:]:            
            attr = numpy.array(dbfReader.by_col(an))
            attr = numpy.reshape(attr, (-1,1))
            attrs = numpy.concatenate((attrs, attr), axis=1)

    extensiveAttr = numpy.array(dbfReader.by_col(extensiveAttrName))
    spatialAttr = get_points_array_from_shapefile(filePath.split('.')[0] +'.shp') 
    
    attrDistMatrix = pdist(attrs, metric='cityblock')
    attrDistMatrix = squareform(attrDistMatrix) 
    minAD = numpy.amin(attrDistMatrix)
    maxAD = numpy.amax(attrDistMatrix)    

    stdAttrDistMatrix = (attrDistMatrix - minAD)/(maxAD - minAD)
    
    spatialDistMatrix = pdist(spatialAttr, metric='euclidean')
    spatialDistMatrix = squareform(spatialDistMatrix)
    minSD = numpy.amin(spatialDistMatrix)
    maxSD = numpy.amax(spatialDistMatrix)    
    
    stdSpatialDistMatrix = (spatialDistMatrix - minSD)/(maxSD - minSD)    
    bcmaxp = BCMaxPExact('bcmaxpSolver')
    bcmaxp.constructModel(stdAttrDistMatrix, stdSpatialDistMatrix, 
                         extensiveAttr, 
                         extensiveThre, attrWeight)
    bcmaxp.solve()
    
    gp_shp = gp.read_file(filePath.split('.')[0] +'.shp')
    gp_shp['regions'] = bcmaxp.labels
    gp_shp.plot(column='regions', legend = True)
    plt.show()           
    #gp_shp.to_file('result\\AW%d_%d.shp' %(attrWeight*10, iteration), driver="ESRI Shapefile", schema=None)    
    #for variable in variable_list:
        #print('%s = %d' % (variable.name(), variable.solution_value()))
        
    bcmaxpH = BCMaxPHeuristic(stdAttrDistMatrix, stdSpatialDistMatrix, 
                             extensiveAttr, 
                             extensiveThre, 
                             attrWeight)
    bcmaxpH.solve()
    gp_shp = gp.read_file(filePath.split('.')[0] +'.shp')
    gp_shp['regions'] = bcmaxpH.labels
    gp_shp.plot(column='regions', legend = True)
    plt.show()       
