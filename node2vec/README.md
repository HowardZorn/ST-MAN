# Node2Vec
## Adjacency Matrix
- PeMS04: Adj(PeMS04).txt
- PeMS08: Adj(PeMS08).txt
- Loop: Adj(Loop).txt
## Requirements
- numpy
- networkx
- gensim
## How to run
> Notice: We have included the output node embeddings of node2vec for the PeMS04, PeMS08 and Loop datasets at path '../data', respectively. You can choose to use the given output directly or run the following command to get the new embeddings.
- PeMS04:
    ```
    python3 generateSE.py -i '../data/Adj(PeMS04).txt' -o '../data/SE(PeMS04).txt' 
    ```
- PeMS08:
    ```
    python3 generateSE.py -i '../data/Adj(PeMS08).txt' -o '../data/SE(PeMS08).txt' 
    ```
- Loop:
    ```
    python3 generateSE.py -i '../data/Adj(Loop).txt' -o '../data/SE(Loop).txt' 
    ```