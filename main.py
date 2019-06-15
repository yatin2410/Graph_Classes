from collections import defaultdict

#class Graph
class Graph:
    
    def __init__(self):
        self.graph = defaultdict(list)
        self.directed = False
        self.nodes = 0
        self.edges = 0

    def changeDirected(self):
        self.directed = True

    def setNodes(self,n):
        self.nodes = n
    
    def setEdges(self,n):
        self.edges = n

    def addEdge(self,u,v):
        if self.directed == False:
            self.graph[u].append(v)
            self.graph[v].append(u)
        else:
            self.graph[u].append(v)

    def isEdgeLessGraph(self):
        if edges == 0:
            return True
        else:
            return False

#Driver code
gr = Graph()
tmp = input('Is Graph Undirected (Y/n)?: ')
if tmp == 'n':
    gr.changeDirected()

nodes = int(input('Enter total number of nodes: '))
edges = int(input('Enter total number of edges: '))
gr.setNodes(nodes)
gr.setEdges(edges)

for i in range(0,edges):
    u,v = input('Enter edge nodes u and v: ').split(" ")
    u = int(u)
    v = int(v)
    gr.addEdge(u,v)

print(gr.isEdgeLessGraph())

print("---------DONE------------")