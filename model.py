import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():
    """
    Random Neural Network
    """
    def __init__(self, X, y, eta=0.01, num_epochs=100, batch_size=32):
        self.X = X
        self.y = y
        self.w = np.random.randn(X.shape[1])  # vector of size equal to number of features
        self.b = np.random.randn()  # scalar bias term
        self.eta = eta  # learning rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.losses = []

    def compute_loss(self, y_true, y_pred) -> float:
      """
      Mean Squared Error
      """

      return np.mean((y_true - y_pred)**2)  # MSE

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters:
          X: Numpy Array of Features
        Returns:
          prediction: Numpy Array that results from the dot product of X and W, then adding b
        """
        return np.dot(X, self.w) + self.b

    def train(self):
        num_samples = self.X.shape[0]

        for epoch in range(self.num_epochs):
            # Random shuffle
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X = self.X[indices]
            y = self.y[indices]

            epoch_loss = 0

            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                X_batch = X[start:end]
                y_batch = y[start:end]

                # Forward pass
                y_pred = self.predict(X_batch)

                # Compute gradients
                error = y_pred - y_batch # L
                w_grad = np.dot(X_batch.T, error) / (end - start) # del w wrt L
                b_grad = np.mean(error) # del b wrt L

                # Update Step
                self.w -= self.eta * w_grad
                self.b -= self.eta * b_grad

                # Accumulate batch loss
                epoch_loss += self.compute_loss(y_batch, y_pred) * (end - start)

            # Compute average loss for the epoch
            epoch_loss /= num_samples
            self.losses.append(epoch_loss)

            print(f"Epoch: {epoch+1}, Loss: {epoch_loss:.6f}")

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        test_loss = self.compute_loss(y_test, y_pred)
        return test_loss

    def plot_loss(self):
        """
        Plot the loss over epochs
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.losses) + 1), self.losses)
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

    def get_params(self):
        return {'weights': self.w, 'bias': self.b}
