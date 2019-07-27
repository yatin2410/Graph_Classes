from collections import defaultdict
from itertools import combinations 
import math
import queue
import sys


#class Graph
class Graph:
    
    def __init__(self,lst=defaultdict(list)):
        self.graphList = lst
        self.directed = False
        self.nodes = 0
        self.edges = 0
        self.chromaticNumber = 0
        self.Color = []
        self.cliques = []

    def changeDirected(self):
        self.directed = True

    def setNodes(self,n):
        self.nodes = n
        self.Color = [0]*(self.nodes+1)
    
    def setEdges(self,n):
        self.edges = n
    
    def setChrometicNumber(self,m):
        self.chromaticNumber = m

    def setColors(self,FinalColor):
        self.Color = FinalColor.copy()

    def addEdge(self,u,v):
        if self.directed == False:
            self.graphList[u].append(v)
            self.graphList[v].append(u)
        else:
            self.graphList[u].append(v)

    def isEdgeLessGraph(self):
        if self.edges == 0:
            return True
        else:
            return False

    def isCycleUtil(self, node, visited, parent):
        visited[node] = True
        for adj in self.graphList[node]:
            if visited[adj] == False:
                if self.isCycleUtil(adj, visited, node) == True:
                    return True
            elif parent != adj:
                return True
        return False

    def isCycle(self):
        visited = [False]*(self.nodes+1)
        for node in range(self.nodes):
            if visited[node+1] == False:
                if self.isCycleUtil(node+1, visited, -1) == True:
                    return True
        return False

    def isSimpleGraph(self):
        edgeSet = set()
        for node in range(1,self.nodes+1):
            for adj in self.graphList[node]:
                if node == adj:
                    return False
                if (node,adj) in edgeSet:
                    return False
                edgeSet.add((node,adj))
        return True

    def isMultiGraph(self):
        return True
          
    def isRegularGraph(self,wantToPrint):
        isDegreeSame = True
        Degree = -1

        if self.directed == False:
            for i in range(2,self.nodes+1):
                if len(self.graphList[i-1]) != len(self.graphList[i]):
                    isDegreeSame = False
                    return Degree

            Degree = len(self.graphList[1])
            if isDegreeSame == True and wantToPrint:
                print(Degree , '- Regular Graph')
        else:
            for i in range(2,self.nodes+1):
                if len(self.graphList[i-1]) != len(self.graphList[i]):
                    isDegreeSame = False
                    return Degree

            inDegrees = [0]*(self.nodes+1)
            for i in range(1,self.nodes+1):
                for j in range(0,len(self.graphList[i])):
                    inDegrees[self.graphList[i][j]] += 1

            for i in range(2,self.nodes+1):
                if inDegrees[i-1] != inDegrees[i]:
                    isDegreeSame = False
                    return Degree 

            inDegree = inDegrees[1]
            outDegree = len(self.graphList[1])
            Degree = inDegree + outDegree      
                            
            if isDegreeSame == True  and wantToPrint:
                print(Degree, '- Regular Graph')
                print('InDegree : ', inDegree)
                print('OutDegree : ', outDegree)
        return Degree

    def isStronglyRegularGraph(self,wantToPrint):
        Degree = self.isRegularGraph(False)
        result = [-1]*3
        if Degree == -1:
            return result

        if self.isEdgeLessGraph()==True:
            if wantToPrint:
                print('srg(nodes = ',self.nodes,', degree = 0 , lemmda = 0 , mu = 0 )')
            return [0,0,0]

        edgeSet = set()

        for i in range(1,self.nodes+1):
            for j in range(i+1,self.nodes+1):
                edgeSet.add((i,j))

        lemmda = -1

        for i in range(1,self.nodes+1):
            for j in self.graphList[i]:
                if (i,j) in edgeSet:
                    edgeSet.remove((i,j))
                    Set1 = set()
                    Set2 = set()
                    Set1.clear()
                    Set2.clear()
                    for k in self.graphList[i]:
                        Set1.add(k)
                    for k in self.graphList[j]:
                        Set2.add(k)

                    Set1 = Set1.intersection(Set2)

                    if lemmda == -1:
                        lemmda = len(Set1)
                    elif lemmda != len(Set1):
                        return result
        
        MU = -1
        rightTerm = Degree * (Degree - lemmda - 1)
        leftTerm = (self.nodes - Degree - 1)

        if self.nodes == 1 or lemmda == -1 or (rightTerm !=0 and leftTerm == 0):
            return result

        if rightTerm == 0:
            MU = 0
            if wantToPrint:
                print('srg(Nodes = ',self.nodes,', Degree = ',Degree,', Lemmda = ',lemmda,', Mu = ',MU,')')
            result[0] = Degree
            result[1] = lemmda
            result[2] = MU
            return result
        elif rightTerm % leftTerm == 0:
            MU = rightTerm // leftTerm
            if wantToPrint:
                print('srg(Nodes = ',self.nodes,', Degree = ',Degree,', Lemmda = ',lemmda,', Mu = ',MU,')')
            result[0] = Degree
            result[1] = lemmda
            result[2] = MU
            return result
        else:
            return result

    def isPathPossible(self,start,end,visited):
        for u in self.graphList[start]:
            if u == end:
                return True
            if visited[u] == False:
                visited[u] = True
                if self.isPathPossible(u,end,visited) == True:
                    return True
                visited[u] = False
        return False

    def isPlanarGraph(self):
        node_array = []
        for i in range(1,self.nodes+1):
            node_array.append(i)

        comb = list(combinations(node_array,5))

        for i in list(comb):
            Pair = list(combinations(i,2))

            visited = [False]*(self.nodes+1)

            for j in list(Pair):
                visited[j[0]] = True
                visited[j[1]] = True

            isK5 = True
            for j in list(Pair):
                if self.isPathPossible(j[0],j[1],visited) == False:
                    isK5 = False
                    break
            if isK5 == True:
                return False

        comb = list(combinations(node_array,6))
        for i in list(comb):
            j = list(combinations(i,3))
            for Tuple in list(j):
                otherTuple = []
                for k in i:
                    if k not in Tuple:
                        otherTuple.append(k)
                visited = [False]*(self.nodes+1)

                for k1 in Tuple:
                    for k2 in otherTuple:
                        visited[k1] = True
                        visited[k2] = True 

                isK33 = True
                for k1 in Tuple:
                    for k2 in otherTuple:
                        if self.isPathPossible(k1,k2,visited) == False:
                            isK33 = False
                            break
                    if isK33 == False:
                        break

                if isK33 == True:
                    return False
        return True

    def bfs(self,root,visited,distance,parent):
        nodeQueue = queue.Queue()
        nodeQueue.put(root)

        while nodeQueue.empty()==False:
            current = nodeQueue.get()
            visited[current] = True

            for adj in self.graphList[current]:
                if visited[adj] == False:
                    nodeQueue.put(adj)
                    parent[adj] = current
                    distance[adj] = distance[current] + 1
                    visited[adj] = True

    def shortestcycle(self,root,visited,distance,parent):
        nodeQueue = queue.Queue()
        nodeQueue.put(root)

        while nodeQueue.empty()==False:
            current = nodeQueue.get()
            visited[current] = True

            for adj in self.graphList[current]:
                if visited[adj] == False:
                    nodeQueue.put(adj)
                    visited[adj] = True
                if parent[current]==adj or parent[adj]==current:
                    continue
                return distance[current]+distance[adj]+1

        return int(sys.maxsize)

    def eccentricity(self,root):
        visited = [False]*(self.nodes+1)
        distance = [0]*(self.nodes+1)
        parent = [-1]*(self.nodes+1)

        self.bfs(root,visited,distance,parent)

        maxDistance = 0
        for i in range(1,self.nodes+1):
            maxDistance = max(maxDistance,distance[i])

        visited = [False]*(self.nodes+1)
        girth = self.shortestcycle(root,visited,distance,parent)

        result = []
        result.append(maxDistance)
        result.append(girth)

        return result

    def DFS(self, marked, n, vert, start, count): 
        marked[vert] = True
        if n == 0:  
            marked[vert] = False
            if start in self.graphList[vert]: 
                count = count + 1
                return count 
            else: 
                return count 

        for i in self.graphList[vert]: 
            if marked[i] == False:  
                count = self.DFS(marked, n-1, i, start, count) 
        marked[vert] = False
        return count 

    def countCycles(self,n): 
        marked = [False]*(self.nodes+1)
        count = 0
        for i in range(1,self.nodes-(n-1)+1): 
            count = self.DFS(marked, n-1, i, i, count) 
            marked[i] = True
          
        return int(count/2) 

    def isMooreGraph(self):
        Degree = self.isRegularGraph(False)

        if Degree == -1:
            return False

        Diameter = -1
        girth = int(sys.maxsize)
        for i in range(1,self.nodes+1):
            result = self.eccentricity(i)
            Diameter = max(Diameter,result[0])
            girth = min(girth,result[1])

        if Diameter == -1 or girth == int(sys.maxsize):
            return False
 
        totalCycles =  self.countCycles(girth)

        numerator = self.nodes*(self.edges-self.nodes+1)
        if girth == 2*Diameter+1 and numerator%girth==0 and totalCycles == numerator/girth:
            count = 0

            for i in range(0,Diameter):
                count += (Degree-1)**i

            count *= Degree
            count += 1

            if math.ceil(count)==self.nodes:
                return True
            else:
                return False
        else:
            return False

    def isLineGraph(self):
        occurence = [0]*(self.nodes+1)

        edgeSet = set()

        for i in range(1,self.nodes+1):
            for j in self.graphList[i]:
                if i < j:
                    edgeSet.add((i,j))
                else:
                    edgeSet.add((j,i))

        while len(edgeSet) > 0:
            selectedEdge = edgeSet.pop()
            currentClique = set()
            currentClique.add(int(selectedEdge[0]))
            currentClique.add(int(selectedEdge[1]))

            removeEdgeSet = set()

            for currentEdge in edgeSet:
                x = currentEdge[0]
                y = currentEdge[1]

                if ((x in currentClique) or (y in currentClique))==False:
                    continue

                if (x in currentClique) and (y in currentClique):
                    removeEdgeSet.add((currentEdge))

                if y in currentClique:
                    temp = x
                    x = y
                    y = temp

                adjacentNode = set()

                for node in self.graphList[y]:
                    adjacentNode.add(node)

                flag = False
                for node in currentClique:
                    if node not in adjacentNode:
                        flag = True
                        break 

                if flag == True:
                    continue

                currentClique.add(y)
                removeEdgeSet.add(currentEdge)

            for edge in removeEdgeSet:
                edgeSet.remove(edge)

            for node in currentClique:
                occurence[node] += 1

            removeEdgeSet.clear()
            currentClique.clear()

        for i in range(1,self.nodes+1):
            if occurence[i] > 2:
                return False

        return True


    def isCubicGraph(self):
        Degree = self.isRegularGraph(False)

        if Degree == 3:
            return True
        else:
            return False

    def isPaleyGraph(self):
        result = self.isStronglyRegularGraph(False)

        Degree = result[0]
        lemmda = result[1]
        MU = result[2]

        if self.nodes % 4 == 1:
            if (self.nodes-1)//2 == Degree and (self.nodes-5)//4 == lemmda and (self.nodes-1)//4 == MU:
                return True
            else:
                return False
        else:
            return False

    def isCubeGraph(self):
        k = 0
        while 2**k < self.nodes:
            k += 1
        if 2**k == self.nodes:
            Degree = self.isRegularGraph(False)
            if Degree == k:
                return True
            else:
                return False
        else:
            return False

    def isHararyGraph(self):
        Degree = self.isRegularGraph(False)

        if Degree > 1:
            print("H(k,n) = ( k =",Degree,", n =",self.nodes,")")
            return True
        elif Degree == -1:
            DegreeSet = set()
            for i in range(1,self.nodes+1):
                DegreeSet.add(len(self.graphList[i]))

            if len(DegreeSet)==2:
                DegreeArray = []
                for i in DegreeSet:
                    DegreeArray.append(i)

                cnt = 0
                for i in range(1,self.nodes+1):
                    if len(self.graphList[i]) == DegreeArray[1]:
                        cnt +=1

                if cnt==1 and (DegreeArray[1] - DegreeArray[0] == 1) and DegreeArray[0] >=2 and math.ceil((DegreeArray[0]*self.nodes)/2) == self.edges:
                    print("H(k,n) = ( k =",DegreeArray[0],", n =",self.nodes,")")
                    return True
                else:
                    return False
            else:
                return False
                    
        else:
            return False


    def BipartedDFS(self,node,visited,color):
        for u in self.graphList[node]:
            if visited[u] == False:
                visited[u] = True
                color[u] = 1 - color[node]
                if self.BipartedDFS(u,visited,color)==False:
                    return False
            elif color[u]==color[node]:
                return False
        return True

    def isBipartedGraph(self):
        if self.isConnectedGraph() == False:
            return False
        visited = [False]*(self.nodes+1)
        color = [False]*(self.nodes+1)
        visited[1] = True
        color[1] = 0
        if self.BipartedDFS(1,visited,color):
            return True
        else:
            return False

    def isCompleteBipartedGraph(self):
        if self.isConnectedGraph() == False:
            return False
        visited = [False]*(self.nodes+1)
        color = [False]*(self.nodes+1)
        visited[1] = True
        color[1] = 0
        if self.BipartedDFS(1,visited,color) == False:
            return False
        s1 = set()
        s2 = set()
        for i in range(1,self.nodes+1):
            if color[i] == 0:
                s1.add(i)
            else:
                s2.add(i)
        len1 = len(s1)
        len2 = len(s2)
        for u in s1:
            if len(self.graphList[u]) != len2:
                return False
        for u in s2:
            if len(self.graphList[u]) != len1:
                return False
        return True
        

    def isCycleGraph(self):
        if self.isConnectedGraph() == False:
            return False
        if self.nodes != self.edges:
            return False
        for i in range(1, self.nodes+1):
            if len(self.graphList[i]) != 2:
                return False 
        return True

    def isWheelorStarGraph(self, e):
        if self.isConnectedGraph() == False:
            return False
        nodeCount = 0
        CenterNodeCount = 0
        for i in range(1,self.nodes+1):
            if len(self.graphList[i]) == e:
                nodeCount += 1
            elif len(self.graphList[i]) == self.nodes-1:
                CenterNodeCount += 1
        if nodeCount == self.nodes-1 and CenterNodeCount == 1:
            return True
        return False 

    def isWheelGraph(self):
        return self.isWheelorStarGraph(3) 

    def isStarGraph(self):
        return self.isWheelorStarGraph(1)

    def isCompleteGraph(self):
        for i in range(1,self.nodes+1):
            if len(self.graphList[i]) != self.nodes-1:
                return False
        return True

    def isCyclicGraph(self):
        return self.isCycle()
          
    def ConnectedDFS(self,visited,node):
        visited[node] = True
        for u in self.graphList[node]:
            if visited[u] == False:
                self.ConnectedDFS(visited,u)

    def isConnectedGraph(self):
        if self.directed == True:
            return False
        visited = [False]*(self.nodes+1)
        self.ConnectedDFS(visited,1)
        for i in range(1,self.nodes+1):
            if visited[i] == False:
                return False
        return True

    def DFSDirected(self,node,stack,visited):
        visited[node] = True
        for u in self.graphList[node]:
            if visited[u] == False:
                self.DFSDirected(u,stack,visited)
        stack.append(node)

    def DFSComponent(self,node,visited,component,tempList):
        visited[node] = True
        component.append(node)
        for u in tempList[node]:
            if visited[u] == False:
                self.DFSComponent(u,visited,component,tempList)


    def isStronglyConnectedGraph(self):
        if self.directed == False:
            return False
        stack = []
        visited = [False]*(self.nodes+1)
        for i in range(1,self.nodes+1):
            if visited[i] == False:
                self.DFSDirected(1,stack,visited)
        tempList = defaultdict(list)
        for i in range(1,self.nodes+1):
            for u in self.graphList[i]:
                tempList[u].append(i)
        visited = [False]*(self.nodes+1)
        while len(stack) > 0 :
            u = stack.pop()
            if visited[u] == True:
                continue
            component = []
            self.DFSComponent(u,visited,component,tempList)
            print("component is : ",end=' ')
            for i in component:
                print(i,end=' ')
            print()
        return True
        

    def isTreeGraph(self):
        if self.directed == False and self.isCycle() == False and self.isConnectedGraph() == True:
            return True
        return False

    def isForestGraph(self):
        if self.directed == False and self.isCycle() == False:
            return True
        return False

    def isRooksGraph(self):
        Degree = self.isRegularGraph(False)
        if Degree == -1:
            return False
        for i in range(1, self.nodes):
            if i*i > self.nodes:
                break
            else:
                if self.nodes%i == 0:
                    if Degree == (i-1) + (self.nodes/i - 1):
                        return True
        return False

    def isThresholdGraph(self):
        DegreeList = []
        for i in range(1, self.nodes+1):
            DegreeList.append(len(self.graphList[i]))
        DominatingNodeDegree = self.nodes-1
        IsolatedNodeDegree = 0
        while len(DegreeList) > 1:
            if DegreeList.count(DominatingNodeDegree) > 0 and DegreeList.count(IsolatedNodeDegree) == 0:
                DegreeList.remove(DominatingNodeDegree)
                IsolatedNodeDegree += 1
            elif DegreeList.count(DominatingNodeDegree) == 0 and DegreeList.count(IsolatedNodeDegree) > 0:
                DegreeList.remove(IsolatedNodeDegree)
                DominatingNodeDegree -= 1
            else:
                return False 
        if DegreeList.count(IsolatedNodeDegree) != DegreeList.count(DominatingNodeDegree):
            return False
        return True

    def isSafeColor(self,node,color,c):
        for u in self.graphList[node]:
            if c == color[u]:
                return False
        return True

    def graphColoringUtil(self,m,color,node):
        if node==self.nodes+1:
            return True
        for i in range(1,m+1):
            if(self.isSafeColor(node,color,i)):
                color[node] = i
                # print(node," ",i)
                if self.graphColoringUtil(m,color,node+1) == True:
                    return True
                color[node] = 0
        return False

    def isKPartiteGraph(self,m,color):
        return self.graphColoringUtil(m,color,1)

    def isMultiPartiteGraph(self):
        if self.chromaticNumber != 0:
            return True
        low = 2
        high = self.nodes
        m = high
        FinalColor = [0]*(self.nodes+1)
        while low<=high:
            mid = (low+high)//2
            color = [0]*(self.nodes+1)
            # print(mid)
            if self.isKPartiteGraph(mid,color):
                m = mid
                # print("in",m)
                FinalColor = color.copy()
                high = mid -1
            else:
                low = mid + 1
        self.setChrometicNumber(m)
        self.setColors(FinalColor)
        print("\nChromatic number is : ",self.chromaticNumber)
        for i in range(1,self.nodes+1):
            print("node : ",i,", color :  ",self.Color[i])
        print()
        return True

    def isCompleteMultiPartitieGraph(self):
        self.isMultiPartiteGraph()
        cntColor = [0]*(self.chromaticNumber+1)
        for i in range(1,self.nodes+1):
            cntColor[self.Color[i]]+=1
        sum = 0
        for i in range(1,self.chromaticNumber+1):
            sum += cntColor[i]
        for i in range(1,self.nodes+1):
            s = set()
            for u in self.graphList[i]:
                s.add(u)
            if (sum-cntColor[self.Color[i]]) != len(s):
                return False
        return True

    def isKneserGraph(self):
        self.isMultiPartiteGraph()
        chromaticNumber = self.chromaticNumber
        if chromaticNumber == 1:
            return True
        Degree = self.isRegularGraph(False)
        if Degree == -1 or Degree*self.nodes != 2*self.edges:
            return False
        for k in range(1,100):
            n = 2*k + chromaticNumber - 2
            if self.nCr(n, k) == self.nodes and self.nCr(n, k)*self.nCr(n-k, k) == 2*self.edges and self.nCr(n-k, k) == Degree:
                print("K(n,k) = ( n =",n,", k =",k,")")
                return True
        return False

    def isJohnsonGraph(self):
        Degree = self.isRegularGraph(False)
        if Degree == -1 or Degree*self.nodes != 2*self.edges:
            return False
        for k in range(1,100):
            if Degree%k != 0:
                continue
            n = Degree/k + k
            if self.nCr(n, k) == self.nodes and self.nCr(n, k)*k*(n-k) == 2*self.edges and k*(n-k) == Degree:
                print("J(n,k) = ( n =",n,", k =",k,")")
                return True
        return False

    def nCr(self, n, r):
        f = math.factorial
        return f(n)/f(r)/f(n-r)

    def isHammingGraph(self):
        Degree = self.isRegularGraph(False)
        if Degree == -1 or Degree*self.nodes != 2*self.edges:
            return False
        q = self.chromaticNumber
        for d in range(1,100):
            if Degree%d != 0:
                continue
            if q**d == self.nodes and d*(q-1)*(q**d) == 2*self.edges and d*(q-1) == Degree:
                print("H(d,q) = ( d =",d,", q =",q,")")
                return True
        return False

    def findSimplicialNode(self, graphList):
        if self.directed == True:
            return False
        for simplicialNode in graphList:
            isSimplicialNode = True
            list = []
            list.append(simplicialNode)
            for adj in graphList[simplicialNode]:
                list.append(adj)
            for i in list:
                for j in list:
                    if i == j:
                        continue
                    if j not in graphList[i]:
                        isSimplicialNode = False
                        break
                if isSimplicialNode == False:
                    break
            if isSimplicialNode == True:
                return simplicialNode
        return -1

    def isChordalGraph(self):
        graphList = defaultdict(list)
        for i in self.graphList:
            for j in self.graphList[i]:
                graphList[i].append(j)
        length = len(graphList)
        perfectEliminationOrder = []
        while len(perfectEliminationOrder) != length:
            simplicialNode = self.findSimplicialNode(graphList)
            if simplicialNode == -1:
                return False
            if simplicialNode in graphList:
                graphList.pop(simplicialNode, None)
            for i in graphList:
                while simplicialNode in graphList[i]:
                    graphList[i].remove(simplicialNode)
            perfectEliminationOrder.append(simplicialNode)
        for i in self.graphList:
            if len(self.graphList[i]) == 0:
                perfectEliminationOrder.append(i)
        print("Perfect Elimination Order: ", perfectEliminationOrder)
        return True

    def Find_All_Cliques(self):
        '''
        Implements Bron-Kerbosch algorithm, Version 2
        '''
        Cliques=[]
        Stack=[]
        nd=None
        Nodes = []
        for i in range(1,self.nodes+1):
            Nodes.append(i)
        disc_num=len(Nodes)
        search_node=(set(),set(Nodes),set(),nd,disc_num) 
        Stack.append(search_node)
        while len(Stack)!=0:
            (c_compsub,c_candidates,c_not,c_nd,c_disc_num)=Stack.pop()
            if len(c_candidates)==0 and len(c_not)==0:
                if len(c_compsub)>2:
                    Cliques.append(c_compsub)
                    continue
            for u in list(c_candidates):
                if (c_nd==None) or (not c_nd in self.graphList[u]): #self.are_adjacent(u, c_nd)):
                    c_candidates.remove(u)
                    Nu=self.graphList[u]                                
                    new_compsub=set(c_compsub)
                    new_compsub.add(u)
                    new_candidates=set(c_candidates.intersection(Nu))
                    new_not=set(c_not.intersection(Nu))                    
                    if c_nd!=None:
                        if c_nd in new_not:
                            new_disc_num=c_disc_num-1
                            if new_disc_num>0:
                                new_search_node=(new_compsub,new_candidates,new_not,c_nd,new_disc_num)                        
                                Stack.append(new_search_node)
                        else:
                            new_disc_num=len(Nodes)
                            new_nd=c_nd
                            for cand_nd in new_not:
                                cand_disc_num=len(new_candidates)-len(new_candidates.intersection(self.graphList[cand_nd])) 
                                if cand_disc_num<new_disc_num:
                                    new_disc_num=cand_disc_num
                                    new_nd=cand_nd
                            new_search_node=(new_compsub,new_candidates,new_not,new_nd,new_disc_num)                        
                            Stack.append(new_search_node)                
                    else:
                        new_search_node=(new_compsub,new_candidates,new_not,c_nd,c_disc_num)
                        Stack.append(new_search_node)
                    c_not.add(u) 
                    new_disc_num=0
                    for x in c_candidates:
                        if not u in self.graphList[x]: #self.are_adjacent(x, u):
                            new_disc_num+=1
                    if new_disc_num<c_disc_num and new_disc_num>0:
                        new1_search_node=(c_compsub,c_candidates,c_not,u,new_disc_num)
                        Stack.append(new1_search_node)
                    else:
                        new1_search_node=(c_compsub,c_candidates,c_not,c_nd,c_disc_num)
                        Stack.append(new1_search_node)     
        return Cliques

    def isSplitGraph(self):
        for clique in self.cliques:
            vertices = []
            for i in range(1,self.nodes+1):
                if i not in clique:
                    vertices.append(i)
            for u in vertices:
                for v in vertices:
                    if v in self.graphList[u]:
                        return False
        return True

    def isColorCriticalGraph(self):
        for i in range(1,self.nodes+1):
            tempGraph = defaultdict(list)
            for u in range(1,self.nodes+1):
                if u == i:
                    continue
                for v in self.graphList[u]:
                    if v == i:
                        continue
                    if v > i and u > i:
                        tempGraph[u-1].append(v-1)
                    elif v < i and u > i :
                        tempGraph[u-1].append(v)
                    elif v < i and u < i :
                        tempGraph[u].append(v)
                    elif v > i and u < i :
                        tempGraph[u].append(v-1)

            tGraph = Graph(tempGraph)
            tGraph.setNodes(self.nodes-1)
            # print(tGraph.graphList)
            # print(tGraph.isMultiPartiteGraph())
            self.isMultiPartiteGraph()
            if tGraph.chromaticNumber+1!=self.chromaticNumber:
                return False
        return True

def run():
    #Driver code
    graph = Graph()

    isUndirected = input('Is Graph Undirected (y/n)?: ')

    if isUndirected == 'n':
        graph.changeDirected()

    nodes = int(input('Enter total number of nodes: '))
    edges = int(input('Enter total number of edges: '))
    graph.setNodes(nodes)
    graph.setEdges(edges)

    for i in range(0,edges):
        u,v = input('Enter edge nodes u and v: ').split(" ")
        u = int(u)
        v = int(v)
        if u <= 0 or v <= 0 or u > nodes or v > nodes:
            print("Invalid")
            i = i-1
            continue     
        graph.addEdge(u,v)

    print('isSimpleGraph : ' , graph.isSimpleGraph(),'\n')
    print('isMultiGraph : ',graph.isMultiGraph(),'\n')
    print('isEdgeLessGraph : ' , graph.isEdgeLessGraph(),'\n')

    if graph.isRegularGraph(True) == -1:
        print('isRegularGraph : False\n')
    else:
        print('isRegularGraph : True\n') 

    if graph.isStronglyRegularGraph(True)[0] == -1:
        print('isStronglyRegularGraph: False\n')
    else:
        print('isStronglyRegularGraph : True\n')       

    print('isCubicGraph : ' , graph.isCubicGraph(),'\n')
    print('isBipartedGraph : ',graph.isBipartedGraph(),'\n')
    print('isCycleGraph : ' , graph.isCycleGraph(),'\n')
    print('isWheelGraph : ' , graph.isWheelGraph(),'\n')
    print('isStarGraph : ' , graph.isStarGraph(),'\n')
    print('isCompleteGraph : ' , graph.isCompleteGraph(),'\n')
    print('isCyclicGraph : ' , graph.isCyclicGraph(),'\n')
    print('isConnectedGraph : ',graph.isConnectedGraph(),'\n')
    print('isStronglyConnectedGraph : ',graph.isStronglyConnectedGraph(),'\n')
    print('isTreeGraph : ',graph.isTreeGraph(),'\n')
    print('isForestGraph : ',graph.isForestGraph(),'\n')
    print('isRooksGraph : ',graph.isRooksGraph(),'\n')
    print('isCompleteBipartedGraph : ',graph.isCompleteBipartedGraph(),'\n')
    print('isThresholdGraph : ',graph.isThresholdGraph(),'\n')
    print('isPlanarGraph : ',graph.isPlanarGraph(),'\n')
    print('isMultiPartiteGraph : ',graph.isMultiPartiteGraph(),'\n')
    print('isCompleteMultiPartitieGraph :',graph.isCompleteMultiPartitieGraph(),'\n')
    print('isPaleyGraph : ',graph.isPaleyGraph(),'\n')
    print('isCubeGraph : ',graph.isCubeGraph(),'\n')
    print('isHararyGraph : ',graph.isHararyGraph(),'\n')
    print('isKneserGraph : ',graph.isKneserGraph(),'\n')
    print('isJohnsonGraph : ',graph.isJohnsonGraph(),'\n')
    print('isHammingGraph : ',graph.isHammingGraph(),'\n')
    print('isChordalGraph : ',graph.isChordalGraph(),'\n')
    print('isMooreGraph : ',graph.isMooreGraph(),'\n')
    print('isLineGraph : ',graph.isLineGraph(),'\n')


    graph.cliques = graph.Find_All_Cliques()
    for clique in graph.cliques:
        print("Maximal clique is :",end=" ")
        print(clique)

    print('isSplitGraph : ',graph.isSplitGraph(),'\n')
    print('isColorCriticalGraph : ',graph.isColorCriticalGraph(),'\n')


    print("---------DONE------------")

# run()