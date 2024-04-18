from matplotlib import pyplot as plt
from roads.QuadTree import QuadTree
from roads import Util
from roads import globals
import numpy as np
from random import randrange

class RoadGenerator(object):
    def __init__(self, size, max_nodes) -> None:
        self.size = size
        self.max_nodes = max_nodes
        self.QT = None
        
        globals.init()

    def randomize(self):
        globals.init()

        X=Util.getPlane(self.size)

        mins = (0.0, 0.0)
        maxs = (self.size-1.0, self.size-1.0)

        self.QT = QuadTree(X, mins, maxs, 0,0)

        self.QT.add_square()

        # for high density choose ones counter depth with highest number of squares randomly
        while(True):
            node=randrange(max(globals.node_index))
            if len(globals.node_index) > self.max_nodes: # limit network generation by number of nodes
                break

            Util.add_square_at(self.QT,node)

    def plot(self, output=False):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.set_xlim(0, self.size-1.0)
        ax.set_ylim(0, self.size-1.0)

        # for each depth generate squares
        for d in range(0,len(globals.node_index)):
            self.QT.draw_rectangle(ax, depth=d)

        if output:
            print("writing data to files...")
            fn = open('../node-list','w')
            fe = open('../edge-list','w')

            edgeCount=0 #directed edge count
            for point in globals.edges:
                if point in globals.coord_id:
                    fn.write(str(globals.coord_id[point])+","+str(point.x)+","+str(point.y)+"\n")

                for edge in globals.edges[point]:
                    fe.write(str(globals.coord_id[point])+","+str(globals.coord_id[edge])+"\n")
                    edgeCount=edgeCount+1

            fn.close()
            fe.close()

        plt.savefig('./images/rand-quad-road-network.png')
