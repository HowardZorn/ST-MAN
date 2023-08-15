import node2vec
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import sys, getopt

is_directed = True
p = 2
q = 1
num_walks = 100
walk_length = 80
dimensions = 64
window_size = 10
iter = 1000
Adj_file = '../data/Adj.txt'
SE_file = '../data/SE.txt'

def in_out(argv):
    global Adj_file, SE_file, q, p, dimensions
    try:
        opts, args = getopt.getopt(argv,"hi:o:p:q:d:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('generateSE.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('generateSE.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            Adj_file = arg
        elif opt in ("-o", "--ofile"):
            SE_file = arg
        elif opt in ("-p"):
            p = float(arg)
        elif opt in ("-q"):
            q = float(arg)
        elif opt in ("-d"):
            dimensions = int(arg)

def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight', float),),
        create_using=nx.DiGraph())

    return G


def learn_embeddings(walks, dimensions, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, vector_size=dimensions, window=10, min_count=0, sg=1,
        workers=20, epochs=iter)
    model.wv.save_word2vec_format(output_file)

    return

in_out(sys.argv[1:])
nx_G = read_graph(Adj_file)
G = node2vec.Graph(nx_G, is_directed, p, q)
G.preprocess_transition_probs()
walks = G.simulate_walks(num_walks, walk_length)
learn_embeddings(walks, dimensions, SE_file)
