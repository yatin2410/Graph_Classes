# Check Graph

You can use this package for recognition of various graph classes.

We made python library for graph classes recognition algorithm which takes directed or undirected graph as input and provides recognition of particular graph which mentioned earlier in terms of true or false.

[Click here](https://pypi.org/project/check-graph) to view this project on pypi.org

[Click here](https://bit.ly/33JDG36) to view PPT

## Installation

```
pip install check-graph
```

## Usage

Use Run function for taking input and calling every methods:

```
import check_graph as cg
cg.run()
```

For custom run, you have to create a graph using check_graph library:

```
    import check_graph as cg
    graph = cg.Graph()
```

And then you can set numbers of nodes and edges as you want::

```
    graph.setNodes(5)
    graph.setEdges(4)
```
 
You can add Edge between two nodes using below method::

```
    graph.addEdge(1,2)
    graph.addEdge(2,3)
    graph.addEdge(3,4)
    graph.addEdge(4,5)
```

## Methods

You can use below methods for recognition of various Graph classes.


* Return types of following methods are True/False:


```
      graph.isSimpleGraph()
      graph.isMultiGraph()
      graph.isEdgeLessGraph()
      graph.isCubicGraph()
      graph.isBipartedGraph()
      graph.isCycleGraph()
      graph.isWheelGraph()
      graph.isStarGraph()
      graph.isCompleteGraph()
      graph.isCyclicGraph()
      graph.isConnectedGraph()
      graph.isStronglyConnectedGraph()
      graph.isTreeGraph()
      graph.isForestGraph()
      graph.isRooksGraph()
      graph.isCompleteBipartedGraph()
      graph.isThresholdGraph()
      graph.isPlanarGraph()
      graph.isMultiPartiteGraph()
      graph.isCompleteMultiPartitieGraph()
      graph.isPaleyGraph()
      graph.isCubeGraph()
      graph.isHararyGraph()
      graph.isKneserGraph()
      graph.isJohnsonGraph()
      graph.isHammingGraph()
      graph.isChordalGraph()
      graph.isMooreGraph()
      graph.isLineGraph()
      graph.isSplitGraph()
      graph.isColorCriticalGraph()
 ```
 
 
 * Return type of following method is Integer(If it is Regular Graph then it will return Degree,otherwise it will return -1):
 
 
 ```    
      graph.isRegularGraph(True)   # If argument is True then it will print details otherwise not.
  ```
 
 
 * Return type of following method is Triplet(If it is Strongly Regular Graph then it will return Degree,Lemmda and MU otherwise             it will return -1):
 
 
      ```
      graph.isStronglyRegularGraph(True)   # If argument is True then it will print details otherwise not.
      ```
      
 *  Return type of following method is list of Maximal Clique:
 
 
 ``` 
      cliques = graph.getCliques()
 ```
 
 
 * Return type of following method is integer which is chromatic number of graph:
 
 
 ```
    ch = graph.getChromaticNumber()
 ```
 
 
 * Return type of following method is list of color of every node:
 
 
 ```
    colors = graph.getColors()
 ```


## Authors 

* [Yatin Patel](https://github.com/yatin2410)
* [Milan Dungarani](https://github.com/milandungrani)
* [Pratik Rajani](https://github.com/PratikRajani)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Thanks ❤
