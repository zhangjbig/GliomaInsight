from functools import reduce

import numpy
import math
import operator
import collections

class GeometricalMeasures:
  
  def __init__(self, labelNode, parameterMatrix, parameterMatrixCoordinates, parameterValues, allKeys):
    # need non-linear scaling of surface heights for normalization (reduce computational time)
    self.GeometricalMeasures = collections.OrderedDict()
    self.GeometricalMeasures["Extruded Surface Area"] = "self.extrudedSurfaceArea(self.labelNode, self.extrudedMatrix, self.extrudedMatrixCoordinates, self.parameterValues)"
    self.GeometricalMeasures["Extruded Volume"] = "self.extrudedVolume(self.extrudedMatrix, self.extrudedMatrixCoordinates, self.cubicMMPerVoxel)"
    self.GeometricalMeasures["Extruded Surface:Volume Ratio"] = "self.extrudedSurfaceVolumeRatio(self.labelNode, self.extrudedMatrix, self.extrudedMatrixCoordinates, self.parameterValues, self.cubicMMPerVoxel)"
       
    self.labelNode = labelNode
    self.parameterMatrix = parameterMatrix
    self.parameterMatrixCoordinates = parameterMatrixCoordinates
    self.parameterValues = parameterValues
    self.keys = set(allKeys).intersection(self.GeometricalMeasures.keys())
    
    if self.keys:
      self.cubicMMPerVoxel = reduce(lambda x,y: x*y, self.labelNode.GetSpacing())
      self.extrudedMatrix, self.extrudedMatrixCoordinates = self.extrudeMatrix(self.parameterMatrix, self.parameterMatrixCoordinates, self.parameterValues)
    
  def extrudedSurfaceArea(self, labelNode, extrudedMatrix, extrudedMatrixCoordinates, parameterValues):
    x, y, z = labelNode.GetSpacing()
       
    # surface areas of directional connections
    xz = x*z
    yz = y*z
    xy = x*y
    fourD = (2*xy + 2*xz + 2*yz)
       
    totalVoxelSurfaceArea4D = (2*xy + 2*xz + 2*yz + 2*fourD)
    totalSA = parameterValues.size * totalVoxelSurfaceArea4D
    
    # in matrixSACoordinates
    # i: height (z), j: vertical (y), k: horizontal (x), l: 4th or extrusion dimension   
    i, j, k, l = 0, 0, 0, 0
    extrudedSurfaceArea = 0
    
    # vectorize
    for i,j,k,l_slice in zip(*extrudedMatrixCoordinates):
      for l in range(l_slice.start, l_slice.stop):
        fxy = numpy.array([ extrudedMatrix[i+1,j,k,l], extrudedMatrix[i-1,j,k,l] ]) == 0
        fyz = numpy.array([ extrudedMatrix[i,j+1,k,l], extrudedMatrix[i,j-1,k,l] ]) == 0
        fxz = numpy.array([ extrudedMatrix[i,j,k+1,l], extrudedMatrix[i,j,k-1,l] ]) == 0  
        f4d = numpy.array([ extrudedMatrix[i,j,k,l+1], extrudedMatrix[i,j,k,l-1] ]) == 0
               
        extrudedElementSurface = (numpy.sum(fxz) * xz) + (numpy.sum(fyz) * yz) + (numpy.sum(fxy) * xy) + (numpy.sum(f4d) * fourD)     
        extrudedSurfaceArea += extrudedElementSurface
    return (extrudedSurfaceArea)
  
  def extrudedVolume(self, extrudedMatrix, extrudedMatrixCoordinates, cubicMMPerVoxel):
    extrudedElementsSize = extrudedMatrix[numpy.where(extrudedMatrix == 1)].size
    return(extrudedElementsSize * cubicMMPerVoxel)
      
  def extrudedSurfaceVolumeRatio(self, labelNode, extrudedMatrix, extrudedMatrixCoordinates, parameterValues, cubicMMPerVoxel):
    extrudedSurfaceArea = self.extrudedSurfaceArea(labelNode, extrudedMatrix, extrudedMatrixCoordinates, parameterValues) 
    extrudedVolume = self.extrudedVolume(extrudedMatrix, extrudedMatrixCoordinates, cubicMMPerVoxel)    
    return(extrudedSurfaceArea/extrudedVolume)
    
  def extrudeMatrix(self, parameterMatrix, parameterMatrixCoordinates, parameterValues):
    # extrude 3D image into a binary 4D array with the intensity or parameter value as the 4th Dimension
    # need to normalize CT images with a shift of 120 Hounsfield units 
	
    parameterValues = numpy.abs(parameterValues)
	
    # maximum intensity/parameter value appended as shape of 4th dimension    
    extrudedShape = parameterMatrix.shape + (numpy.max(parameterValues),)
    
    # pad shape by 1 unit in all 8 directions
    extrudedShape = tuple(map(operator.add, extrudedShape, [2,2,2,2]))
    
    extrudedMatrix = numpy.zeros(extrudedShape)   
    extrudedMatrixCoordinates = tuple(map(operator.add, parameterMatrixCoordinates, ([1,1,1]))) + (numpy.array([slice(1,value+1) for value in parameterValues]),)   
    for slice4D in zip(*extrudedMatrixCoordinates):
      extrudedMatrix[slice4D] = 1      
    return (extrudedMatrix, extrudedMatrixCoordinates)
    
  def EvaluateFeatures(self):
    # Evaluate dictionary elements corresponding to user-selected keys 
    if not self.keys:
      return(self.GeometricalMeasures)
      
    for key in self.keys:
      self.GeometricalMeasures[key] = eval(self.GeometricalMeasures[key])
    return(self.GeometricalMeasures)

# Example usage
labelNode = './BraTS-GLI-0009_0000.nii.gz' # Your label node
parameterMatrix = None  # Your parameter matrix
parameterMatrixCoordinates = None  # Your parameter matrix coordinates
parameterValues = None  # Your parameter values
allKeys = ["Extruded Surface Area", "Extruded Volume", "Extruded Surface:Volume Ratio"]

geo_measurements = GeometricalMeasures(labelNode, parameterMatrix, parameterMatrixCoordinates, parameterValues, allKeys)
results = geo_measurements.EvaluateFeatures()
print(results)

