import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

##  Class to represent a graph
class Graph:
    def __init__(self,vertices):
        self.graph = defaultdict(list) ##  Dictionary containing adjacency List
        self.V = vertices ##  No. of vertices
        self.vert_dict = {} ## Dictionary of dictionaries for nodes and weights
        self.top_order = [] ## Stores result of topologicalSort

##  Function to add an edge to graph
    def addEdge(self,u,v,cost):
        if cost > 0:
            self.graph[u].append(v)
        if u not in self.vert_dict:
            self.vert_dict[u] = {}
        self.vert_dict[u][v] = cost

##  The function to do Topological Sort.
    def topologicalSort(self):

##  Create a vector to store indegrees of all
##  vertices. Initialize all indegrees as 0.
        in_degree = [0]*(self.V)

##  Traverse adjacency lists to fill indegrees of
##  vertices.  This step takes O(V+E) time
        for i in self.graph:
            for j in self.graph[i]:
                in_degree[j] += 1

##  Create an queue and enqueue all vertices with
##  indegree 0
        queue = []
        for i in range(self.V):
            if in_degree[i] == 0:
                queue.append(i)

##  Initialize count of visited vertices
        cnt = 0

##  One by one dequeue vertices from queue and enqueue
##  adjacents if indegree of adjacent becomes 0
        while queue:

##  Extract front of queue (or perform dequeue)
##  and add it to topological order
            u = queue.pop(0)
            self.top_order.append(u)

##  Iterate through all neighbouring nodes
##  of dequeued node u and decrease their in-degree
##  by 1
            for i in self.graph[u]:
                in_degree[i] -= 1
##  If in-degree becomes zero, add it to queue
                if in_degree[i] == 0:
                    queue.append(i)

            cnt += 1

##  Check if there was a cycle
        if cnt != self.V:
            print("There exists a cycle in the graph")
        #else :
            #Print topological order
        #    print(self.top_order)

    def displayWeight(self, u):
        if u in self.vert_dict:
            for key in self.vert_dict[u]:
                print('Node', u, 'flows to node', key, 'and the volume is', self.vert_dict[u][key])
        else:
            print('Node', u, 'does not flow to any other node')

    def dfs(self, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        if start in self.vert_dict:
            for key in self.vert_dict[start]:
                #print(self.vert_dict[key])
                if self.vert_dict[start][key] < 0:
                    if key in visited:
                        continue
                    else:
                        #print(str(start) + ' is downstream from ' + str(key))
                        self.dfs(key, visited)
        return visited

    def fractionThrough(self, start):
##  Calls dfs and returns set of upstream nodes from start
        visited = self.dfs(start)
##  Filter topological order so that only upstream nodes are displayed
        filt_top_order = [x for x in self.top_order if x in visited]
        #print(filt_top_order)
##  Dictionary to hold nodes as keys and fractions as values
        percents = {}
##  As long as there are upstream nodes calculate their fraction through
        for i in reversed(filt_top_order):
            down1 = 0
            down2 = 0
            down3 = 0
            down4 = 0
            down1mod = 0
            down2mod = 0
            down3mod = 0
            down4mod = 0
            down_frac = 0
            if i == start:
                percents[filt_top_order.pop()] = 1.0
                continue
            if i in self.vert_dict:
                for key in self.vert_dict[i]:
                    node_flow = 0
                    if self.vert_dict[i][key] > 0:
## This (below) checks if the neighbor's fraction has already
## been calculated, this is why the top sort is useful
                        if key in percents:
                            node_flow = percents[key]
                        if down1 == 0:
                            down1 = self.vert_dict[i][key]
## The mod variables take into account whether or not the downstream
## neighbor leads to the target node, if it does then the % of
## that flow is applied, if it does not then it's multiplied by 0
                            down1mod = down1 * node_flow
                            #print(down1, ' ', down1mod)
                        elif down2 == 0:
                            down2 = self.vert_dict[i][key]
                            down2mod = down2 * node_flow
                            #print(down2, ' ', down2mod)
                        elif down3 == 0:
                            down3 = self.vert_dict[i][key]
                            down3mod = down3 * node_flow
                            #print(down3, ' ', down3mod)
                        else:
                            down4 = self.vert_dict[i][key]
                            down4mod = down4 * node_flow
                            #print(down4, ' ', down4mod)
                down_frac = (down1mod + down2mod + down3mod + down4mod)/(down1 + down2 + down3 + down4)
                #print(i, ' ', down_frac)

            percents[filt_top_order.pop()] = down_frac

##  Create list to graph from -- Initialize to 0
        nodes = [0]*int(num_nodes)
        print('Percentage of water that flows from each cell through cell ' + str(start))
##  Replace relevant values
        for key in percents:
            nodes[key] = percents[key]*100
##  Split into smaller lists
        chunks = [nodes[x:x+(int(num_columns))] for x in range(0, len(nodes), int(num_columns))]
        x = np.array(chunks)

        fig, ax = plt.subplots()

        #x = np.ma.masked_where(x < 0.05, x)
        cmap = plt.cm.ocean_r
        #cmap.set_bad(color='black')
        cax = ax.imshow(x, cmap=cmap, interpolation='none')
        ax.set_title('Fraction Through')

##  Start the ticks to start at -0.5 otherwise
##  the grid lines will cut the nodes in half
        ax.set_xticks(np.arange(-0.5, int(num_columns), 1))
        ax.set_yticks(np.arange(-0.5, int(num_nodes)/int(num_columns), 1))

##  This will hide the tick labels
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        cbar = fig.colorbar(cax, ticks=[0, int(max(nodes)/4), int(max(nodes)/2), int(max(nodes)*.75), int(max(nodes))])
        cbar.ax.set_yticklabels(['0', int(max(nodes)/4), int(max(nodes)/2), int(max(nodes)*.75), int(max(nodes))])

        plt.rc('grid', linestyle='solid', color='black')
        ax.grid(linewidth=2)
        plt.grid(True)
        plt.show()
        #for key in percents:
        #    print(str(key) + ': ' + str(percents[key]*100))

    def volumeThrough(self, start):
##  Calls dfs and returns set of upstream nodes from start
        visited = self.dfs(start)
##  Filter topological order so that only upstream nodes are displayed
        filt_top_order = [x for x in self.top_order if x in visited]
        #print(filt_top_order)
##  Dictionary to hold nodes as keys and fractions as values
        percents = {}
        vols = {}
##  Similar process to fractionThrough()
        for i in reversed(filt_top_order):
            if i not in self.vert_dict:
                continue
            down1 = 0
            down2 = 0
            down3 = 0
            down4 = 0
            down1mod = 0
            down2mod = 0
            down3mod = 0
            down4mod = 0
            down_frac = 0
            if i == start:
                popped = filt_top_order.pop()
                percents[popped] = 1.0
## Upstream flows are only used when i == start
                up1 = 0
                up2 = 0
                up3 = 0
                up4 = 0
                for key in self.vert_dict[i]:
                    if self.vert_dict[i][key] < 0:
                        if up1 == 0:
                            up1 = self.vert_dict[i][key]
                        elif up2 == 0:
                            up2 = self.vert_dict[i][key]
                        elif up3 == 0:
                            up3 = self.vert_dict[i][key]
                        else:
                            up4 = self.vert_dict[i][key]
                vols[popped] = abs(up1) + abs(up2)
                continue
            if i in self.vert_dict:
                for key in self.vert_dict[i]:
                    node_flow = 0
                    if self.vert_dict[i][key] > 0:
                        if key in percents:
                            node_flow = percents[key]
                        if down1 == 0:
                            down1 = self.vert_dict[i][key]
                            down1mod = down1 * node_flow
                            #print(down1, ' ', down1mod)
                        elif down2 == 0:
                            down2 = self.vert_dict[i][key]
                            down2mod = down2 * node_flow
                            #print(down2, ' ', down2mod)
                        elif down3 == 0:
                            down3 = self.vert_dict[i][key]
                            down3mod = down3 * node_flow
                            #print(down3, ' ', down3mod)
                        else:
                            down4 = self.vert_dict[i][key]
                            down4mod = down4 * node_flow
                            #print(down4, ' ', down4mod)
                down_frac = (down1mod + down2mod + down3mod + down4mod)/(down1 + down2 + down3 + down4)
                vol = (down1mod + down2mod + down3mod + down4mod)
                #print(i, ' ', down_frac, ' ', vol)
            popped = filt_top_order.pop()
            vols[popped] = vol
            percents[popped] = down_frac
##  Create list to graph from -- Initialize to 0
        nodes = [0]*int((num_nodes))
        print('Volume of water that flows from each cell through cell ' + str(start))
##  Replace relevant values
        for key in vols:
            nodes[key] = vols[key]
##  Split into smaller lists
        chunks = [nodes[x:x+(int(num_columns))] for x in range(0, len(nodes), int(num_columns))]
        X = np.array(chunks)
        fig, ax = plt.subplots()
        cax = ax.imshow(X, cmap='ocean_r', interpolation='none')
        ax.set_title('Volume Through')

##  Start the ticks to start at -0.5 otherwise
##  the grid lines will cut the nodes in half
        ax.set_xticks(np.arange(-0.5, int(num_columns), 1))
        ax.set_yticks(np.arange(-0.5, int(num_nodes)/int(num_columns), 1))

##  This will hide the tick labels
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        #print(max(nodes))
        if max(nodes) == 0:
            cbar = fig.colorbar(cax, ticks=[0])
            cbar.ax.set_yticklabels(['0'])
        else:
            cbar = fig.colorbar(cax, ticks=[0, int(max(nodes)/4), int(max(nodes)/2), int(max(nodes)*.75), int(max(nodes))])
            cbar.ax.set_yticklabels(['0', int(max(nodes)/4), int(max(nodes)/2), int(max(nodes)*.75), int(max(nodes))])

        plt.rc('grid', linestyle='solid', color='black')
        ax.grid(linewidth=2)
        plt.show()

        #for key in vols:
        #    print(str(key) + ': ' + str(vols[key]))

    def volumeFrom(self, start):
        visited = self.dfs(start)
        filt_top_order = [x for x in self.top_order if x in visited]
        #print(filt_top_order)
        percents = {}
        vols = {}
## Similar to two functions above, except the difference between
## upstream and downstream nodes must be calculated, hence the up1-4 variables
        for i in reversed(filt_top_order):
            down1 = 0
            down2 = 0
            down3 = 0
            down4 = 0
            down1mod = 0
            down2mod = 0
            down3mod = 0
            down4mod = 0
            up1 = 0
            up2 = 0
            up3 = 0
            up4 = 0
            if i == start:
                popped = filt_top_order.pop()
                percents[popped] = 1.0
                if i in self.vert_dict:
                    for key in self.vert_dict[i]:
                        if self.vert_dict[i][key] < 0:
                            if up1 == 0:
                                up1 = abs(self.vert_dict[i][key])
                            elif up2 == 0:
                                up2 = abs(self.vert_dict[i][key])
                            elif up3 == 0:
                                up3 = abs(self.vert_dict[i][key])
                            else:
                                up4 = abs(self.vert_dict[i][key])
                        else:
                            if down1 == 0:
                                down1 = self.vert_dict[i][key]
                            elif down2 == 0:
                                down2 = self.vert_dict[i][key]
                            elif down3 == 0:
                                down3 = self.vert_dict[i][key]
                            else:
                                down4 = self.vert_dict[i][key]
                val = ((down1 + down2 + down3 + down4) - (up1 + up2 + up3 + up4)) * percents[popped]
                if val > 0:
                    vols[popped] = val
                else:
                    vols[popped] = 0
                continue
            if i in self.vert_dict:
                for key in self.vert_dict[i]:
                    node_flow = 0
                    if self.vert_dict[i][key] > 0:
                        if key in percents:
                            node_flow = percents[key]
                        if down1 == 0:
                            down1 = self.vert_dict[i][key]
                            down1mod = down1 * node_flow
                            #print(down1, ' ', down1mod)
                        elif down2 == 0:
                            down2 = self.vert_dict[i][key]
                            down2mod = down2 * node_flow
                            #print(down2, ' ', down2mod)
                        elif down3 == 0:
                            down3 = self.vert_dict[i][key]
                            down3mod = down3 * node_flow
                            #print(down3, ' ', down3mod)
                        else:
                            down4 = self.vert_dict[i][key]
                            down4mod = down4 * node_flow
                            #print(down4, ' ', down4mod)
                    else:
                        if up1 == 0:
                            up1 = abs(self.vert_dict[i][key])
                        elif up2 == 0:
                            up2 = abs(self.vert_dict[i][key])
                        elif up3 == 0:
                            up3 = abs(self.vert_dict[i][key])
                        else:
                            up4 = abs(self.vert_dict[i][key])
                down_frac = (down1mod + down2mod + down3mod + down4mod)/(down1 + down2 + down3 + down4)
            popped = filt_top_order.pop()
            percents[popped] = down_frac
            val = ((down1 + down2 + down3 + down4) - (up1 + up2 + up3 + up4)) * percents[popped]
            if val > 0:
                vols[popped] = val
            else:
                vols[popped] = 0

##  Create list to graph from -- Initialize to 0
        nodes = [0]*(int(num_nodes))
        print('Volume of water that originates from each cell and flows to target cell ' + str(start))
##  Replace non-zero values
        for key in vols:
            nodes[key] = vols[key]
##  Split into smaller lists
        chunks = [nodes[x:x+(int(num_columns))] for x in range(0, len(nodes), int(num_columns))]
        X = np.array(chunks)
        fig, ax = plt.subplots()
        cax = ax.imshow(X, cmap='ocean_r', interpolation='none')
        ax.set_title('Volume From')

##  Start the ticks at -0.5, otherwise
##  the grid lines will cut the nodes in half
        ax.set_xticks(np.arange(-0.5, int(num_columns), 1))
        ax.set_yticks(np.arange(-0.5, int(num_nodes)/int(num_columns), 1))

##  This will hide the tick labels
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        if max(nodes) == 0:
            cbar = fig.colorbar(cax, ticks=[0])
            cbar.ax.set_yticklabels(['0'])
        else:
            cbar = fig.colorbar(cax, ticks=[0, int(max(nodes)/4), int(max(nodes)/2), int(max(nodes)*.75), int(max(nodes))])
            cbar.ax.set_yticklabels(['0', int(max(nodes)/4), int(max(nodes)/2), int(max(nodes)*.75), int(max(nodes))])

        plt.rc('grid', linestyle='solid', color='black')
        ax.grid(linewidth=2)
        plt.show()

        #for key in vols:
        #    print(str(key) + ': ' + str(vols[key]))


periods = input('Number of time periods:\n')
num_nodes = input('Number of nodes:\n')
num_columns = input('What is the width of the graph:\n')

g = []

for i in range(int(periods)):
    g.append(Graph(int(num_nodes)))

##  The empty list that will hold the flow values
    r_flows = []
    f_flows = []

##  Parse the right facing flow values from file.
##  The expected naming convention for the files
##  is r_flows[n].txt where n is the time period
    for line in open('r_flows' + str(i) + '.txt', 'r'):
        for char in '[]':
            line = line.replace(char, '')
            #print(line)
        words = line.split(' ')
        words = [x for x in words if x != '']
        words = [x for x in words if x != '\n']
        words = [x for x in words if x != ' ']
        #print(words)
        for string in words:
            r_flows.append(float(string))
        #print(r_flows)

    if int(num_nodes) != len(r_flows):
        print('Expected ', num_nodes, ' nodes, but ', len(r_flows), ' were given in r_flows.')
        quit()

##  Front Face
    for line in open('f_flows' + str(i) + '.txt', 'r'):
        for char in '[]':
            line = line.replace(char, '')
        #print(line)
        words = line.split(' ')
        words = [x for x in words if x != '']
        words = [x for x in words if x != '\n']
        words = [x for x in words if x != ' ']
        #print(words)
        for string in words:
            f_flows.append(float(string))
        #print(f_flows)

    if int(num_nodes) != len(f_flows):
        print('Expected ', num_nodes, ' nodes, but ', len(f_flows), ' were given in f_flows.')
        quit()

    for j in range (0, (int(num_nodes)-1)):
        if r_flows[j] != 0:
            g[i].addEdge(j, j+1, r_flows[j])
            g[i].addEdge(j+1, j, -1*(r_flows[j]))
        if f_flows[j] != 0:
            g[i].addEdge(j, j+int(num_columns), f_flows[j])
            g[i].addEdge(j+int(num_columns), j, -1*(f_flows[j]))

    g[i].topologicalSort()

loop = True
if len(g) == 1:
    period = 1
else:
    period = -1

while(loop):
    if period == -1:
        period = input('Which time period do you want to look at?\n')
    case = input('\nWhich command would you like to run?\n'
    'Enter the number of your desired command:\n'
    '1 - Display Neighbors\n'
    '2 - Fraction Through\n'
    '3 - Volume Through\n'
    '4 - Volume From\n'
    '5 - Change time period\n'
    'Type \'quit\' to quit\n')
    if case == '1':
        x = input('Which node? \n')
        g[int(period)-1].displayWeight(int(x))
    elif case == '2':
        x = input('Which node? \n')
        g[int(period)-1].fractionThrough(int(x))
    elif case == '3':
        x = input('Which node? \n')
        g[int(period)-1].volumeThrough(int(x))
    elif case == '4':
        x = input('Which node? \n')
        g[int(period)-1].volumeFrom(int(x))
    elif case == '5':
        if len(g) > 1:
            print('You are currently on period', period)
            period = input('Which time period do you want to look at?\n')
        else:
            period = 1
            print('Only 1 time period')
    elif case == 'quit':
        loop = False
    else:
        print('\n\nUNKNOWN COMMAND\n\n')
