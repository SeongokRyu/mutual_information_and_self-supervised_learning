import numpy as np
import random
from rdkit import Chem
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

def split_train_unlabel(smi_list, label_list, num_train, num_unlabel):
    smi_train = smi_list[0:num_train]
    label_train = label_list[0:num_train]
    smi_unlabel = smi_list[-num_unlabel:0]
    return smi_train, label_train, smi_unlabel

def np_sigmoid(x):
    return 1./(1.+np.exp(-x))

def calculate_mae_rmse(y_truth, y_pred):
    mae = np.mean(np.abs(y_truth - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_truth - y_pred)**2, axis=0))
    return mae, rmse

def calculate_accuracy_auroc(y_truth, y_pred):    
    auroc = 0.0
    try:
        auroc = roc_auc_score(y_truth, y_pred)
    except:
        auroc = 0.0    

    y_truth = np.around(y_truth)
    y_pred = np.around(y_pred).astype(int)

    accuracy = accuracy_score(y_truth, y_pred)
    precision = precision_score(y_truth, y_pred)
    recall = recall_score(y_truth, y_pred)
    f1_score = 2*(precision*recall)/(precision+recall+1e-5)
    return accuracy, auroc, precision, recall, f1_score

def shuffle_two_lists(list1, list2):
    list_total = list(zip(list1,list2))
    random.shuffle(list_total)
    list1, list2 = zip(*list_total)
    return list1, list2

def split_train_validation_test(input_list, train_ratio, test_ratio, validation_ratio):
    num_total = len(input_list)
    num_test = int(num_total*test_ratio)
    num_train = num_total-num_test
    num_validation = int(num_train*validation_ratio)
    num_train -= num_validation

    train_list = input_list[:num_train]
    validation_list = input_list[num_train:num_train+num_validation]
    test_list = input_list[num_train+num_validation:]
    return train_list, validation_list, test_list

def preprocess_inputs(smiles_list, label_list, max_atoms):
    adj = []
    features = []
    label = []
    n = len(smiles_list)
    for i in range(n):
        # Mol
        iMol = Chem.MolFromSmiles(smiles_list[i].strip())
        if iMol is not None:
            iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
            # Feature
            if( iAdjTmp.shape[0] <= max_atoms):
                # Feature-preprocessing
                iFeature = np.zeros((max_atoms, 58))
                iFeatureTmp = []
                for atom in iMol.GetAtoms():
                    iFeatureTmp.append( atom_feature(atom) ) ### atom features only
                iFeature[0:len(iFeatureTmp), 0:58] = iFeatureTmp ### 0 padding for feature-set
                features.append(iFeature)

                # Adj-preprocessing
                iAdj = np.zeros((max_atoms, max_atoms))
                iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
                adj.append(adj_k(np.asarray(iAdj), 1))
                label.append(label_list[i])

    features = np.asarray(features)
    adj = np.asarray(adj)
    label = np.asarray(label)
    return adj, features, label

def adj_k(adj, k):
    ret = adj
    for i in range(0, k-1):
        ret = np.dot(ret, adj)  
    return convert_adj(ret)

def convert_adj(adj):

    dim = len(adj)
    a = adj.flatten()
    b = np.zeros(dim*dim)
    c = (np.ones(dim*dim)-np.equal(a,b)).astype('float64')
    d = c.reshape((dim, dim))

    return d

def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                       'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                       'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                       'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                    one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])    # (40, 6, 5, 6, 1)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    #print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def print_metrics(y_truth, y_pred):
    accuracy, auroc, precision, recall, f1_score = \
        calculate_accuracy_auroc(y_truth, y_pred)
    print ("Accuracy:", round(accuracy, 5), 
           "AUROC:", round(auroc, 5),
           "Precision:", round(precision, 5),
           "Recall:", round(recall, 5),
           "F1-score:", round(f1_score, 5))
    return        

def print_metrics_regression(y_truth, y_pred):
    mae = np.mean(np.abs(y_truth - y_pred))
    print ("MAE", round(mae,5))
    return        

def get_sorted_idx(score_list):
    return sorted(range(score_list.shape[0]), key=lambda k:score_list[k])
