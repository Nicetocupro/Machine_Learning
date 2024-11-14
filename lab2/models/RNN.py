import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # hidden to output
        self.bh = np.zeros((hidden_size, 1))  # hidden bias
        self.by = np.zeros((output_size, 1))  # output bias

    def forward(self, inputs):
        h_prev = np.zeros((self.hidden_size, 1))
        hs, ys = {}, {}
        hs[-1] = h_prev
        for t in range(len(inputs)):
            x = np.array(inputs[t]).reshape(-1, 1)
            hs[t] = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, hs[t-1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
        return ys, hs

    def loss(self, outputs, targets):
        loss = 0
        for t in range(len(outputs)):
            loss += np.sum((outputs[t] - targets[t]) ** 2)
        return loss / len(outputs)

    def backward(self, inputs, hs, outputs, targets, learning_rate=1e-3):
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])

        for t in reversed(range(len(inputs))):
            dy = outputs[t] - targets[t]
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dh_next
            dh_raw = (1 - hs[t] * hs[t]) * dh
            dbh += dh_raw
            dWxh += np.dot(dh_raw, inputs[t].reshape(1, -1))
            dWhh += np.dot(dh_raw, hs[t-1].T)
            dh_next = np.dot(self.Whh.T, dh_raw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

    def train(self, inputs, targets, epochs=100, learning_rate=1e-3):
        for epoch in range(epochs):
            outputs, hs = self.forward(inputs)
            loss = self.loss(outputs, targets)
            self.backward(inputs, hs, outputs, targets, learning_rate)
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')