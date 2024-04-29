import numpy as np
import csv
import networkx as nx

###################
# random baseline #
###################

# Load test samples 

def read_txt(filename):
    with open("data/" + filename + '.txt', "r") as f:
        reader = csv.reader(f)
        readen = list(reader)
    li_links = [element[0].split(" ") for element in readen]
    return li_links

# Make random predictions
def make_random_predictions(test_set,name = 'random_predictions'):
    random_predictions = np.random.choice([0, 1], size=len(test_set))
    random_pred = list(zip(np.array(range(len(test_set))), random_predictions)) #create ID column
    with open("data/" + name + '.csv',"w",newline='') as pred:
        csv_out = csv.writer(pred)
        csv_out.writerow(i for i in ["ID", "Predicted"])
        for row in list(random_pred):
            csv_out.writerow(list(row))
        pred.close()

def read_nodes():
    with open('data/node_information.csv', 'r') as f:
        reader = csv.reader(f)
    # Convertir le lecteur CSV en liste
        nodes = [[int(float(item)) for item in row] for row in reader]
        dic_nodes = [(elem[0], {'label': elem[1:]}) for elem in nodes] #liste des noeuds
        return dic_nodes

def create_training_graph():
    li_nodes = read_nodes()
    li_edges = read_txt('train')
    li_links = [(int(elem[0]), int(elem[1]),{'exist': int(elem[2])}) for elem in li_edges]
    G = nx.Graph()
    G.add_nodes_from(li_nodes)
    G.add_edges_from(li_links)
    return G





if __name__ == "__main__":
    #training_set = read_txt('train')
    #print('train set',training_set[:5])

    #test_set = read_txt('test')
    #print('test set',test_set[:5])

    #make_random_predictions(test_set)
    create_training_graph()
