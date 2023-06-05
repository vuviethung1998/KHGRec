## Data Preparation Step For GraphSAGE:
```
 1. Read graph data from files and Generate edgelist 
    python utils/graph/get_graph.py --mode=kg
    python utils/graph/get_graph.py --mode=interaction
 2. Merge graphs into one unified graph 
    python utils/graph/merge_graphs.py
 3. Create corrupted dataset and convert to .mat type
 4. Create corrupted edge list 
 5. Split dataset 
```