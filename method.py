import ot
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.optim import Adam


class ModelAlpha(nn.Module):

    def __init__(self, n_template):
        super(ModelAlpha, self).__init__()
        self.fc1 = nn.Linear(1, 100)  # 5*5 from image dimension
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, n_template)

    def forward(self, x):
        x = fun.relu(self.fc1(x))
        x = fun.relu(self.fc2(x))
        x = fun.softmax(self.fc3(x), dim=-1)
        return x


class DeepFgwEstimator:

    def __init__(self, n_templates):

        # FGW parameter
        self.alpha = 1 / 2
        self.max_iter = 5

        # Model parameter
        self.n_templates = n_templates
        self.weights = ModelAlpha(n_template=self.n_templates)
        self.nb_node_template = 40
        self.feature_dim = 1
        self.C_templates = None
        self.F_templates = None
        self.w_templates = None
        self.params = None
        self.n_out = 40

        # Training parameter
        self.n_epochs = 5
        self.lr = 0.01

    def loss(self, Y_true, Y_pred):

        # Choose distribution over nodes: uniform
        C_true, F_true = Y_true
        C_pred, F_pred = Y_pred
        p1 = torch.ones(C_pred.shape[0], dtype=torch.float32) / C_pred.shape[0]
        p2 = torch.ones(C_true.shape[0], dtype=torch.float32) / C_true.shape[0]

        # Compute euclidean distance matrix between F_pred and F_true
        n_u = torch.linalg.norm(F_pred, axis=1).reshape((-1, 1)) ** 2
        n_v = torch.linalg.norm(F_true, axis=1).reshape((-1, 1)) ** 2
        n_uv = n_u + n_v.T
        B = torch.mm(F_pred, F_true.T)
        M = n_uv - 2 * B

        # Compute FGW distance
        fgw = ot.gromov.fused_gromov_wasserstein2(M, C_pred, C_true, p1, p2, loss_fun='square_loss', alpha=self.alpha,
                                                  log=False)
        return fgw

    def train(self, X, Y, dict_learning=False, Y_templates=None):

        # Initialize templates
        if dict_learning is False:

            # using given templates
            self.C_templates = [y[0] for y in Y_templates]
            self.F_templates = [y[1] for y in Y_templates]

        else:

            # using random templates
            if self.C_templates is None:
                self.C_templates = []
                self.F_templates = []
                for i in range(self.n_templates):
                    C = torch.rand(self.nb_node_template, self.nb_node_template, requires_grad=True)
                    F = torch.rand(self.nb_node_template, self.feature_dim, dtype=torch.float32, requires_grad=True)
                    self.C_templates.append(C)
                    self.F_templates.append(F)

        # Initialize templates distribution overs nodes: uniform
        self.w_templates = [torch.ones(t.shape[0], dtype=torch.float32) / t.shape[0] for t in self.C_templates]

        # Define model parameters for gradient descent
        if dict_learning:
            self.params = [*self.weights.parameters(), *self.C_templates, *self.F_templates]
        else:
            self.params = [*self.weights.parameters()]

        # Define torch optimizer
        optimizer = Adam(params=self.params, lr=self.lr)

        # Gradient descent
        N = X.shape[0]
        loss_iter = []
        for e in range(self.n_epochs):

            # One epoch
            loss_e = 0
            for i in range(N):
                pred = self.predict(X[i])
                loss_i = self.loss(pred, Y[i])
                loss_e = loss_i + loss_e

            loss_iter.append(float(loss_e.detach().cpu().numpy()) / N)
            print(str(loss_iter[-1]), end=' ')

            # Gradient step
            loss_e.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Clamping C in [0,1]
            with torch.no_grad():
                for C in self.C_templates:
                    C[:] = C.clamp(0, 1)

        return loss_iter

    def predict(self, x_te):

        # Predict weights
        lambdas = self.weights(x_te)[0]

        # Compute barycenter from weights and templates
        F_bary, C_bary = ot.gromov.fgw_barycenters(self.n_out, self.F_templates, self.C_templates, self.w_templates,
                                                   lambdas=lambdas, alpha=self.alpha, loss_fun='square_loss',
                                                   max_iter=self.max_iter, tol=1e-9)
        return C_bary, F_bary
