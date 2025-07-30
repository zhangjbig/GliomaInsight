import slicer
import collections

class NodeInformation:

  def __init__(self, dataNode, labelNode, allKeys):
    self.nodeInformation = collections.OrderedDict()
    self.nodeInformation["Node"] = "self.nodeName(self.dataNode)"
    
    self.dataNode = dataNode
    self.labelNode = labelNode
    self.keys = set(allKeys).intersection(self.nodeInformation.keys())
             
  def nodeName (self, dataNode):
    return (dataNode.GetName())
    
  def EvaluateFeatures(self):
    # Evaluate dictionary elements corresponding to user-selected keys
       
    if not self.keys:
      return(self.nodeInformation)
       
    for key in self.keys:
      self.nodeInformation[key] = eval(self.nodeInformation[key])
    return(self.nodeInformation)


class NodeInformation:

    def __init__(self, dataNode, labelNode, allKeys):
        self.nodeInformation = collections.OrderedDict()
        self.nodeInformation["Node"] = dataNode.GetName() if dataNode else ""
        self.dataNode = dataNode
        self.labelNode = labelNode
        self.keys = set(allKeys).intersection(self.nodeInformation.keys())

    def EvaluateFeatures(self):
        # Evaluate dictionary elements corresponding to user-selected keys
        if not self.keys:
            return self.nodeInformation
        for key in self.keys:
            self.nodeInformation[key] = eval(self.nodeInformation[key])
        return self.nodeInformation

# Example usage:
# Create some dummy data for demonstration
dataNode = slicer.vtkMRMLScalarVolumeNode()
dataNode.SetName("Data Node")

labelNode = slicer.vtkMRMLScalarVolumeNode()
labelNode.SetName("Label Node")

allKeys = ["Node"]

# Create an instance of NodeInformation
nodeInfo = NodeInformation(dataNode, labelNode, allKeys)

# Evaluate features
result = nodeInfo.EvaluateFeatures()

# Print the result
print(result)
