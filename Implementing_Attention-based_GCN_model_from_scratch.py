# Implementing Attention-based GCN model from scratch
# -------------------------------------------- #

#The GCN model consists of two linear layers followed by a non-linear activation function and an attention mechanism. The first linear layer transforms the input features of each node using a set of weight matrices and biases. The second linear layer transforms the output of the first linear layer using another set of weight matrices and biases. The output of the second linear layer is then passed through a non-linear activation function, such as ReLU (Rectified Linear Unit), to introduce non-linearity in the model.

#The attention mechanism is used to weight the output of the second linear layer based on the importance of each node in the graph. It consists of a linear layer that maps the output of the second linear layer to a set of attention weights, which are then normalized using a softmax function. The normalized attention weights are then multiplied element-wise with the output of the second linear layer to obtain the final output of the GCN model.

#During training, the GCN model is trained to minimize a loss function that measures the difference between the predicted output and the ground truth labels. The model parameters, such as the weight matrices and biases, are updated using an optimization algorithm, such as Adam, to minimize the loss.

import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_relations, dropout=0.5, weight_decay=0.01):
        super(GCN, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.dropout = dropout
        self.weight_decay = weight_decay

        # GCN layers
        self.gcn1 = nn.Linear(in_features, hidden_features)
        self.gcn2 = nn.Linear(hidden_features, hidden_features)
        self.gcn3 = nn.Linear(hidden_features, out_features)

        # Attention heads
        self.attention1 = nn.Linear(out_features, num_relations)
        self.attention2 = nn.Linear(out_features, num_relations)

        # Dropout and weight decay
        self.dropout = nn.Dropout(dropout)
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, x, adj):
        # Perform graph convolution
        x = self.gcn1(x)
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.gcn2(x)
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.gcn3(x)

        # Attention Mechanism
        attention_weights1 = self.attention1(x)
        attention_weights1 = torch.softmax(attention_weights1, dim=1)
        self.attention_weights1 = attention_weights1
        x = x * attention_weights1

        attention_weights2 = self.attention2(x)
        attention_weights2 = torch.softmax(attention_weights2, dim=1)
        self.attention_weights2 = attention_weights2
        x = x * attention_weights2

        return x

    def loss(self, predictions, targets):
        # Compute the attention loss
        attention_loss1 = self.l2_loss(self.attention_weights1, targets)
        attention_loss2 = self.l2_loss(self.attention_weights2, targets)
        attention_loss = attention_loss1 + attention_loss2

        # Compute the prediction loss
        prediction_loss = self.l2_loss(predictions, targets)

        # Compute the total loss
        total_loss = prediction_loss + self.weight_decay * attention_loss

        return total_loss




import torch.optim as optim


# Prepare the input data
# Sample input data
import numpy as np
beta = 1   # perturbation factor
batch_size = 1
x = np.array(([0.5*beta,-0.1*beta,0.3*beta],
                       [0.2,0.1,0.7],
                       [-0.5,0.7,-0.1],
                       [-0.1,-0.6,0.4],
                       [0.3,-0.5,-0.2],
                       [0.1,-0.1,-0.4],
                       [0.3,0.8,-0.1],
                       [0.1,-0.2,0.2]), dtype=float)

b10 = 0.1295
a11 = 517.0544
b11 = 115.5967
a12 = 4.2614
b12 = 4.6361
a13 = 1.3083
b13 = 1.4428
a14 = 2.7480
b14 = 3.1052
a21 = 0.9492
b20 = -0.5262
a22 = 2.6331
b21 = 3.7399
a41 = 239.0092
b22 = 9.6729
a42 = 1.7819
b40 = 0.1595
a43 = 3.6549
b41 = 26.8926
a44 = 5.2346
b42 = 1.9330
a31 = 191.8224
b43 = 4.0660
a32 = 49.4899
b44 = 5.9926
b31 = 0.9688
b30 = 3.3
b32 = 0.2043

# Generate random adjacency matrix
adj = np.array([[0,1,0,0,0,1,0,0],
              [0,0,1,0,0,0,1,0],
              [0,0,0,1,0,0,0,1],
              [-a14,a13,-a12,-a11,0,0,-a22,-a21],
              [0,1,0,0,0,1,0,0],
              [0,0,1,0,0,0,1,0],
              [0,0,0,1,0,0,0,1],
              [0,0,-a32,-a31,-a44,-a43,-a42,-a41]])

#adj = np.random.randint(0, 2, size=(num_nodes, num_nodes))

np.fill_diagonal(adj, 0)  # Set diagonal to 0

# Add batch size dimension to x and adj
x = np.expand_dims(x, 0)  # shape: (batch_size, num_nodes, num_features)
adj = np.expand_dims(adj, 0)  # shape: (batch_size, num_nodes, num_nodes)

# Repeat x and adj to match batch size
x = np.repeat(x, batch_size, axis=0)  # shape: (batch_size, num_nodes, num_features)
adj = np.repeat(adj, batch_size, axis=0)  # shape: (batch_size, num_nodes, num_nodes)

# Convert to PyTorch tensors
x = torch.from_numpy(x).float()
adj = torch.from_numpy(adj).float()

node_features = x
adjacency_matrix = adj
targets = torch.tensor(([0.01],[0.2],[0.2],[0.01],[0.01],[0.2],[0.2],[0.01]),dtype=torch.float)

# Define the GCN model
model = GCN(in_features=3, hidden_features=8, out_features=8, num_relations=8, dropout=0.1, weight_decay=0.01)

# Define the optimizer and the loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = model.loss

# Training loop
for epoch in range(1000):
    # Forward pass
    predictions = model(node_features, adjacency_matrix)
    loss = loss_fn(predictions, targets)

    # Backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss at every 50th epoch
    if epoch % 50 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item():.4f}')


attention_weights = model.attention_weights1.detach().cpu().numpy()
array1 = np.squeeze(attention_weights, axis=0)
attention_weights = model.attention_weights2.detach().cpu().numpy()
array2 = np.squeeze(attention_weights, axis=0)
array3 = np.add(array1,array2)
array = array2

adj2 = torch.squeeze(adj, dim=0).numpy()
degree_matrix = np.diag(np.sum(adj2, axis = 0))
degree_matrix_inv = np.linalg.inv(degree_matrix)
result = np.multiply(array, adj2)
result2 = np.sum(result, axis = 0).reshape([-1,1])
dddd = np.diag(degree_matrix_inv).reshape([-1,1])

node_scores = np.multiply(result2,dddd)


