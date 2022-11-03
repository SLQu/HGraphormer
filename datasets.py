import inspect,os,pickle
import numpy as np
import  random
import scipy.sparse as sp
import torch
class parser(object):
    """
    an object for parsing data
    """

    def __init__(self, data, dataset):
        """
        initialises the data directory

        arguments:
        data: coauthorship/cocitation
        dataset: cora/dblp/acm for coauthorship and cora/citeseer/pubmed for cocitation
        """

        current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.d = os.path.join(current, 'data' ,data, dataset)
        self.data, self.dataset = data, dataset



    def parse(self):
        """
        returns a dataset specific function to parse
        """

        name = "_load_data"
        function = getattr(self, name, lambda: {})
        return function()



    def _load_data(self):
        """
        loads the coauthorship hypergraph, features, and labels of cora

        assumes the following files to be present in the dataset directory:
        hypergraph.pickle: coauthorship hypergraph
        features.pickle: bag of word features
        labels.pickle: labels of papers

        n: number of hypernodes
        returns: a dictionary with hypergraph, features, and labels as keys
        """

        with open(os.path.join(self.d, 'hypergraph.pickle'), 'rb') as handle:
            hypergraph = pickle.load(handle)

        with open(os.path.join(self.d, 'features.pickle'), 'rb') as handle:
            features = pickle.load(handle).todense()

        with open(os.path.join(self.d, 'labels.pickle'), 'rb') as handle:
            labels = self._1hot(pickle.load(handle))


        G = []
        for i ,k in enumerate(hypergraph):
            G.append(np.array(list(set(hypergraph[k])) ,dtype=int))


        return {'hypergraph': G, 'features': features, 'labels': labels, 'n': features.shape[0]}



    def _1hot(self, labels):
        """
        converts each positive integer (representing a unique class) into ints one-hot form

        Arguments:
        labels: a list of positive integers with eah integer representing a unique label
        """

        classes = set(labels)
        onehot = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        return np.array(list(map(onehot.get, labels)), dtype=np.int32)

def lable(Source2Id,args):
    current = os.path.abspath(inspect.getfile(inspect.currentframe()))
    Dir, _ = os.path.split(current)
    file = os.path.join(Dir, 'data' ,args.data, args.dataset, "splits", str(args.split) + ".pickle")
    if not os.path.isfile(file): print("split + ", str(args.split), "does not exist")
    with open(file, 'rb') as H:
        Splits = pickle.load(H)
        train, test = Splits['train'], Splits['test']

    train = set(Source2Id[train])
    test = set(Source2Id[test])

    return train, test

def loaddata(args):
    dataset = parser(args.data, args.dataset).parse()
    X, Y, G = dataset['features'], dataset['labels'], dataset['hypergraph']
    all_node = set()
    for e in G:
        all_node = set(e).union(all_node)

    temp_X = []
    temp_Y = []
    Source2Id = np.zeros((X.shape[0]), dtype=int)
    Id2Source = np.zeros((len(all_node)), dtype=int)

    for i, n in enumerate(all_node):
        temp_X.append(np.array(X[n, :]).reshape([-1]))
        temp_Y.append(Y[n])
        Source2Id[n] = i
        Id2Source[i] = n

    for i, e in enumerate(G):
        G[i] = Source2Id[e]
    dataset['features'], dataset['labels'], dataset['hypergraph'] = temp_X, temp_Y, G

    return dataset, Source2Id


def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M  
    where D is the diagonal node-degree matrix 
    """
    
    d = np.array(M.sum(1))
    
    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)    # D inverse i.e. D^{-1}
    
    return DI.dot(M)

def fetch_data(args):
    dataset,Source2Id = loaddata(args)
    X, Y, G = dataset['features'], dataset['labels'], dataset['hypergraph']
    
    # node features in sparse representation
    X = sp.csr_matrix(normalise(np.array(X)), dtype=np.float32)
    X = torch.FloatTensor(np.array(X.todense()))
    
    # labels
    Y = np.array(Y)
    Y = torch.LongTensor(np.where(Y)[1])
    return X, Y, G, Source2Id


def getNumpyH(G,node_num,edge_num):
    H = np.zeros((edge_num,node_num))
    for i,e in enumerate(G):
        # H[i,e.tolist()]=1
        H[i,e]=1

    # import model.laplace as laplace
    # Lap_M = laplace.generate_G_from_H(Lap_M.T)
    return H.T