import random, numpy as np

def create_world(n=4):
w=np.full((n,n)," ")
w[random.randrange(n),random.randrange(n)]="W"
for _ in range(random.randint(1,n)):
x,y=random.randrange(n),random.randrange(n)
if w[x,y]==" ":w[x,y]="P"
x,y=random.randrange(n),random.randrange(n)
if w[x,y]==" ":w[x,y]="G"
return w,(0,0)

def percept(w,pos):
x,y=pos;n=len(w)
adj=[(i,j) for i,j in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)] if 0<=i smell=breeze=glitter=False
for i,j in adj:
if w[i,j]=="W":smell=True
if w[i,j]=="P":breeze=True
if w[x,y]=="G":glitter=True
return {"Smell":smell,"Breeze":breeze,"Glitter":glitter}

def play():
w,a=create_world()
print("World (hidden):\n",w)
print("Agent starts at",a)
for s in range(10):
print("Step",s,"Percepts:",percept(w,a))
if w[a]=="W":print("Eaten by Wumpus!");break
if w[a]=="P":print("Fell into a pit!");break
if w[a]=="G":print("Found the gold!");break
a=(random.randrange(len(w)),random.randrange(len(w)))
else:print("Agent wandered aimlessly.")

if __name__=="__main__":play()

import networkx as nx

def neighbors(a, b, capA, capB):
yield (capA, b), "Fill A"
yield (a, capB), "Fill B"
yield (0, b), "Empty A"
yield (a, 0), "Empty B"
pour = min(a, capB-b); yield (a-pour, b+pour), "A→B"
pour = min(b, capA-a); yield (a+pour, b-pour), "B→A"

def water_jug(capA, capB, t):
G = nx.DiGraph()
start = (0,0)
stack = [start]
while stack:
a,b = stack.pop()
for (na,nb), action in neighbors(a,b,capA,capB):
if (na,nb) not in G:
G.add_edge((a,b),(na,nb),action=action)
stack.append((na,nb))
for node in G:
if t in node or sum(node)==t:
path = nx.shortest_path(G, start, node)
actions = [G[u][v]['action'] for u,v in zip(path,path[1:])]
return list(zip(path, ["Start"]+actions))
return None

print(water_jug(4,3,2))

import networkx as nx
from itertools import product

def valid(m,c):return 0<=m<=3 and 0<=c<=3 and (m in(0,3)or m>=c)and((3-m)in(0,3)or(3-m)>=(3-c))
def next_states(s):
m,c,b=s;d=-1 if b else 1
for dm,dc in[(1,0),(2,0),(0,1),(0,2),(1,1)]:
nm,nc,b2=m+dm*d,c+dc*d,1-b
if valid(nm,nc):yield(nm,nc,b2)
def solve(start=(3,3,1),goal=(0,0,0)):
G=nx.DiGraph();S=[start];seen={start}
while S:
s=S.pop()
for t in next_states(s):
G.add_edge(s,t)
if t not in seen:seen|={t};S.append(t)
return nx.shortest_path(G,start,goal)

for s in solve():print(s)

import networkx as nx

# Example graph
G = nx.Graph()
edges = [
('A','B',1), ('A','C',3),
('B','D',3), ('B','E',1),
('C','F',5), ('D','G',2),
('E','G',1), ('F','G',2)
]
G.add_weighted_edges_from(edges)

# Heuristic
h = {'A':6,'B':4,'C':5,'D':2,'E':1,'F':2,'G':0}

path = nx.astar_path(G, 'A', 'G', heuristic=lambda u,v: h[u], weight='weight')
cost = nx.astar_path_length(G, 'A', 'G', heuristic=lambda u,v: h[u], weight='weight')
print("Path:", path, "Cost:", cost)

class Graph:
def __init__(self):self.g={};self.h={};self.best={}
def add(self,n,c):self.g[n]=c
def set_h(self,h):self.h=h
def ao_star(self,n):
if not self.g.get(n):return self.h[n], [n]
mc,bp=float('inf'),[]
for cs in self.g[n]:
cost,paths=0,[n]
for ch in cs:
c,p=self.ao_star(ch)
cost+=c;paths+=p
if cost self.h[n]=mc;self.best[n]=bp;return mc,bp

g=Graph()
g.add('A',[('B','C'),('D',)])
g.add('B',[('E',),('F',)])
for n in 'CDEF':g.add(n,[])
g.set_h({'A':0,'B':2,'C':3,'D':4,'E':2,'F':3})

cost,path=g.ao_star('A')
print("Cost by AO* algorithm for root node A is:",cost)
print("Optimal path:", " → ".join(path))

import itertools

def solve_nqueens(n):
for perm in itertools.permutations(range(n)):
if len({i+perm[i] for i in range(n)}) == n and len({i-perm[i] for i in range(n)}) == n:
yield perm

n = 5
solutions = list(solve_nqueens(n))
print("Number of solutions:", len(solutions))
for sol in solutions:
for i in sol:
print(" ".join("Q" if j==i else "." for j in range(n)))
print()

import networkx as nx

g = [
[0,10,15,20],
[10,0,35,25],
[15,35,0,30],
[20,25,30,0]
]

G = nx.complete_graph(len(g))
for i in G.nodes:
for j in G.nodes:
if i != j: G[i][j]["weight"] = g[i][j]

path = nx.approximation.traveling_salesman_problem(G, cycle=True)
cost = sum(G[u][v]["weight"] for u,v in zip(path, path[1:]))
print("Path:", path)
print("Cost:", cost)
