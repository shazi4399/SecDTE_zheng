import pickle
import random
# from common.helper import *
from benchmarkOption import *

# For dummy feature
FEATURE_USED = {
    "linnerud": [0, 1, 2],
    "cancer": [0, 1, 2],
    "wine": [0, 1, 2],
    "digits-10": [0, 1, 2],
    "digits-12": [0, 1, 2],
    "digits-15": [0, 1, 2],
    "diabets-18": [0, 1, 2],
    "iris":[0, 0, 0]
}


class Node:
    def __init__(self, pNode, threshold, leafValue, feature):
        self.left = None
        self.right = None
        self.isLeaf = False
        self.classId = None
        self.height = 0
        self.featureIdx = feature
        self.id = None
        self.parent = pNode
        if self.parent is not None:
            self.height = self.parent.getHeight() + 1

        self.threshold = threshold
        self.condition = 1  # Denotes lessThan operation or equal operation
        self.leafValue = leafValue

    def setAsLeaf(self):
        self.isLeaf = True
        value, index = max((value, index) for index,value in enumerate(self.leafValue))
        self.classId = index

    def getHeight(self):
        return self.height


# print(clf.tree_.threshold)
# print(clf.tree_.value)
# totalNum = 2**(clf.get_depth()+1) -1
class BinaryTree:
    def __init__(self, name):
        clf = None
        self.modelName = name
        with open("./model/" + name + ".model", 'rb') as fid:
            clf = pickle.load(fid)

        self.thresholds = clf.tree_.threshold
        self.values = clf.tree_.value
        self.features = clf.tree_.feature
        self.dummyLeafCnt = 0

        print("binary tree features are: ", self.features)
        # self.featureUsed = clf.n_features_in_int
        print("nodes num is: ",len(self.thresholds))
        print(self.values)

        # count used feature number here
        usedFeatures = {}
        cnt = 0
        for f in self.features:
            if f != -2:
                if str(f) not in usedFeatures.keys():
                    usedFeatures[str(f)] = cnt
                    cnt += 1
        self.usedFeatureLen = len(usedFeatures.keys())
        print("#Used_features is: ", self.usedFeatureLen)
        print(usedFeatures)

        self.usedFeaturesMapping = usedFeatures.copy()

        renamedFeatures = []
        # self.features.copy()
        print("bTree's features",self.features)
        print("usedFeatures is ", usedFeatures)
        for v in self.features:
            if v == -2:
                renamedFeatures.append(-2)
            else:
                renamedFeatures.append(usedFeatures[str(v)])
        self.features = renamedFeatures
        print(self.features)

        print(clf.tree_.feature)
        self.maxHeight = clf.get_depth()
        self.dummyNodesCnt = 2 ** (self.maxHeight + 1) - 1 - len(self.thresholds)
        # nodesNum = len(thresholds)
        # print(totalNum)
        idx, self.root = self.middleOrderCreate(None, 0)

    def getMaxHeight(self):
        return self.maxHeight

    def getUsedFeatures(self):
        return self.usedFeatureLen

    def getFeatureIdxMapping(self):
        return self.usedFeaturesMapping

    def middleOrderCreate(self,parent,idx):
        newIdx, middleNode = 0,None
        leafValue = self.values[idx][0]
        leafValue = [int(v) for v in leafValue]
        #print("leafValue is: ",leafValue)

        threshold = self.thresholds[idx]
        if threshold == -2:
            #rndIndex = random.choice(FEATURE_USED[self.modelName])
            newIdx, middleNode = idx + \
                1, Node(parent, parent.threshold, leafValue, parent.featureIdx)
            # print("leafValue: ",leafValue)
            # self.dummyNodesCnt -= 1
            self.addDummyNode(middleNode,leafValue)
        else:
            newIdx, middleNode = idx + \
                1, Node(parent, threshold, leafValue, self.features[idx])
            newIdx,leftNode = self.middleOrderCreate(middleNode,newIdx)
            newIdx,rightNode = self.middleOrderCreate(middleNode,newIdx)
            middleNode.left =leftNode
            middleNode.right =rightNode
        return newIdx,middleNode

    # def addDummyNode_rec(self,parent:Node, leafValue, id):
    #     if parent.getHeight() < self.getMaxHeight():
    #         newId, middleNodeDummy = id+1, Node(parent, -2, leafValue, id)
    #         newId, leftNodeDummy = self.addDummyNode(middleNodeDummy, leafValue, newId)
    #         newId, rightNodeDummy = self.addDummyNode(middleNodeDummy, leafValue, newId)
    #         middleNodeDummy.left = leftNodeDummy
    #         middleNodeDummy.right = rightNodeDummy
    #     else:
    #         parent.setAsLeaf()
    #     return newId, middleNodeDummy

    def addDummyNode(self, parent, leafValue):
        # self.dummyNodesCnt += 1
        if parent.getHeight() < self.getMaxHeight():  # Automatically insert dummy nodes with nonleaf node

            leftNode = Node(parent, parent.threshold, leafValue,
                            parent.featureIdx)
            rightNode = Node(parent, parent.threshold, leafValue,
                             parent.featureIdx)
            parent.left = leftNode
            parent.right = rightNode
            self.addDummyNode(leftNode, leafValue)
            self.addDummyNode(rightNode, leafValue)
        else:
            # self.dummyNodesCnt -= 1
            # self.dummyLeafCnt += 1
            parent.setAsLeaf()

    def getNodesInfo(self, CONVERT_FACTOR):
        print(self.modelName," is with ",self.dummyNodesCnt," dummy nodes.")
        print(self.modelName, " is with ", self.maxHeight, " depth.")
        id = 0
        queue = [self.root]
        nonLeafNodes = []
        leafNodes = []
        while len(queue) > 0:
            top = queue.pop(0)
            top.id = id
            id = id + 1
            if top.isLeaf:
                leafNodes.append(top.classId)
            else:
                queue.append(top.left)
                queue.append(top.right)
                t = int(top.threshold * CONVERT_FACTOR)
                # if t == -200:
                #     t=1
                tuple = (top.featureIdx, t)
                nonLeafNodes.append(tuple)

        print("nonLeafNodes len is: ", len(nonLeafNodes))
        print("leafNodes len is: ", len(leafNodes))
        print("nonLeafNodes is :", nonLeafNodes)
        print("leafNodes is: ", leafNodes)
        return leafNodes, nonLeafNodes

    def find_paths(self):
        def dfs(node:Node, path, paths):
            if node == None:
                return
            path.append(node.id)
            if not node.left and not node.right:
                paths.append(path.copy())
            else:
                dfs(node.left, path, paths)
                dfs(node.right, path, paths)
            path.pop()

        paths = []
        dfs(self.root, [], paths)
        return paths

