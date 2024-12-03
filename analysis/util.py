import os

def getDataPath(file):
    datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data-collection")
    return os.path.join(datapath, file)

def writeFilePath(file):
    datapath = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(datapath, file)