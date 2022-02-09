from build_dataset import create_data_train, create_data_test
import matplotlib.pyplot as plt
import matplotlib
from method import DeepFgwEstimator
import networkx as nx
import numpy as np
import torch
from time import time


# 1) Load data set
print("Loading dataset")
n_tr = 50
X_tr, Y_tr = create_data_train(n_tr)
X_te, Y_te = create_data_test()


# 2) Convert data to torch
print("Converting data to torch")
X_tr = np.array(X_tr)[:, None]
X_tr = torch.tensor(X_tr, dtype=torch.float32)
Y_tr = [[torch.tensor(y[0], dtype=torch.float32), torch.tensor(y[1], dtype=torch.float32)] for y in Y_tr]
X_te = np.array(X_te)[:, None]
X_te = torch.tensor(X_te, dtype=torch.float32)
Y_te = [[torch.tensor(y[0], dtype=torch.float32), torch.tensor(y[1], dtype=torch.float32)] for y in Y_te]

# 3) Instantiate the graph prediction model
print("Instantiate graph prediction model")
n_templates = 8
clf = DeepFgwEstimator(n_templates=n_templates)
clf.nb_node_template = 5

# 4) Choose training parameters
clf.max_iter = 5
clf.n_out = 40
clf.n_epochs = 600
clf.lr = 0.01
clf.alpha = 0.5


# 4) Training
print("Start training")
t0 = time()
loss_iter = clf.train(X_tr, Y_tr, dict_learning=True)
plt.plot(loss_iter)
plt.savefig(f"training_loss.pdf")
plt.close()
print("\nEnd training")
print(f'Training time {time() - t0}', flush=True)


# 5) Compute test score
print("Start test")
mean_loss_te = 0
t0 = time()
for i in range(len(X_te)):

    y_pred = clf.predict(X_te[i])
    loss_te = clf.loss(Y_te[i], y_pred)
    mean_loss_te += loss_te
print("End test")
print(f'Test time {time() - t0}', flush=True)
print(f'FGW mean test loss {mean_loss_te / len(X_te)}')


# 6) Plot the learned map
print("Plotting the true and learned map")
for i in range(len(X_te)):

    y_pred = clf.predict(X_te[i])
    C_pred = y_pred[0].cpu().detach().numpy()
    F_pred = y_pred[1].cpu().detach().numpy().ravel()
    n_nodes = C_pred.shape[0]

    # 1) remove self loop
    for m in range(n_nodes):
        C_pred[m, m] = 0

    # 2) Plot continuous predictions (display with continous edge widths and continuous colormap)
    fig, ax = plt.subplots()
    cmap = matplotlib.cm.get_cmap('plasma')
    node_color = [cmap(i/4) for i in F_pred]
    G = nx.from_numpy_matrix(C_pred)
    pos = nx.spring_layout(G)
    edges = G.edges()
    cmap_grey = matplotlib.cm.get_cmap('Greys')
    weights = [C_pred[u][v] for u, v in edges]
    weights = weights / np.max(weights)  # normalization
    alpha_max = 0.6
    weights = weights * alpha_max  # for clearer plot
    edge_color = [(0, 0, 0, w) for w in weights]
    nx.draw_networkx_nodes(G, pos, node_color=node_color)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color)
    ax.axis("off")
    plt.savefig(f"predicted_graph_{float(X_te[i])}.pdf")
    plt.close()

    # 3) Plot true map predictions
    C_true = Y_te[i][0].cpu().detach().numpy()
    F_true = Y_te[i][1].cpu().detach().numpy().ravel()
    n_nodes = C_pred.shape[0]

    fig, ax = plt.subplots()
    cmap = matplotlib.cm.get_cmap('plasma')
    node_color = [cmap(i / 4) for i in F_true]
    G = nx.from_numpy_matrix(C_true)
    pos = nx.spring_layout(G, k=0.2)
    nx.draw_networkx_nodes(G, pos, node_color=node_color)
    nx.draw_networkx_edges(G, pos, alpha=alpha_max)
    ax.axis("off")

    plt.savefig(f"true_graph_{float(X_te[i])}.pdf")
    plt.close()


# 7) Plot the learned templates
print("Plotting the learned templates")
n_templates = clf.n_templates
for i in range(n_templates):

    C = clf.C_templates[i].cpu().detach().numpy()
    F = clf.F_templates[i].cpu().detach().numpy()
    for j in range(C.shape[0]):
        C[j, j] = 0
    fig, ax = plt.subplots()
    cmap = matplotlib.cm.get_cmap('plasma')
    node_color = [cmap(i[0] / 4) for i in F]
    G = nx.from_numpy_matrix(C)
    pos = nx.spring_layout(G, k=0.2)
    edges = G.edges()
    weights = [C[u][v] for u, v in edges]
    weights = weights / np.max(weights)  # normalization
    weights = weights * 1.  # for clearer plot
    edge_color = [(0, 0, 0, w) for w in weights]
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=2000.)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=10)
    ax.axis("off")
    plt.savefig(f"learned_template_{i}.pdf")
    plt.close()
