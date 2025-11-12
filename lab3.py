import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from typing import Tuple

class XORAndPerceptron:
    """Class for creating and training a perceptron for XOR and AND tasks"""
    
    def __init__(self, hidden_units: int = 4, learning_rate: float = 0.1):
        """
        Initialize the perceptron
        
        Args:
            hidden_units: number of neurons in the hidden layer
            learning_rate: learning rate for optimization
        """
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.model: keras.Model | None = None
        self.history = None
        
        # Set seed for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def create_model(self) -> None:
        """Create the model architecture with two outputs (XOR and AND)"""
        # Single output layer with 2 units: [xor_output, and_output]
        self.model = keras.Sequential([
            layers.Dense(
                self.hidden_units,
                activation='sigmoid',
                input_shape=(2,),
                name='hidden_layer'
            ),
            layers.Dense(
                2,
                activation='sigmoid',
                name='output_layer'
            )
        ], name='XOR_AND_Perceptron')
        
        # Compile the model
        # Using binary_crossentropy for the two binary targets
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model created successfully!")
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get training data for XOR and AND problems"""
        X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ], dtype=np.float32)
        
        # y columns: [xor, and]
        y = np.array([
            [0, 0],  # 0 XOR 0 = 0, 0 AND 0 = 0
            [1, 0],  # 0 XOR 1 = 1, 0 AND 1 = 0
            [1, 0],  # 1 XOR 0 = 1, 1 AND 0 = 0
            [0, 1]   # 1 XOR 1 = 0, 1 AND 1 = 1
        ], dtype=np.float32)
        
        return X, y
    
    def train(self, epochs: int = 500, verbose: int = 1) -> None:
        """
        Train the model
        
        Args:
            epochs: number of training epochs
            verbose: verbosity level (0, 1, 2)
        """
        if self.model is None:
            self.create_model()
        
        X, y = self.get_training_data()
        
        print(f"\nStarting training for {epochs} epochs...\n")
        
        class ProgressCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 100 == 0:
                    # logs['accuracy'] is the overall accuracy metric
                    print(f"Epoch {epoch:3d}/{epochs} - "
                          f"Loss: {logs['loss']:.4f} - "
                          f"Accuracy: {logs['accuracy']*100:.2f}%")
        
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            verbose=0,
            batch_size=4,
            callbacks=[ProgressCallback()] if verbose > 0 else []
        )
        
        final_loss = self.history.history['loss'][-1]
        final_accuracy = self.history.history['accuracy'][-1]
        print(f"Epoch {epochs-1:3d}/{epochs} - "
              f"Loss: {final_loss:.4f} - "
              f"Accuracy: {final_accuracy*100:.2f}%")
        print("\nTraining completed successfully!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for input data
        
        Args:
            X: input data
            
        Returns:
            model predictions (shape: [n_samples, 2]) where columns are [xor, and]
        """
        if self.model is None:
            raise ValueError("Model not created! Call train() first")
        
        return self.model.predict(X, verbose=0)
    
    def test(self) -> None:
        """Test the model on training data and print results for XOR and AND"""
        X, y = self.get_training_data()
        labels = ['XOR', 'AND']
        
        print("\nTesting the model:")
        print("=" * 60)
        
        predictions = self.predict(X)
        
        correct_counts = [0, 0]
        for i in range(len(X)):
            preds = predictions[i]      # two values
            rounded = [int(round(v)) for v in preds]
            expected = [int(v) for v in y[i]]
            
            print(f"\nInput: [{int(X[i][0])}, {int(X[i][1])}]")
            for j, label in enumerate(labels):
                is_correct = rounded[j] == expected[j]
                status = "CORRECT" if is_correct else "WRONG"
                print(f"  {label}: Prediction: {preds[j]:.6f} ~ {rounded[j]} | "
                      f"Expected: {expected[j]} [{status}]")
                if is_correct:
                    correct_counts[j] += 1
        
        print("=" * 60)
        for j, label in enumerate(labels):
            print(f"{label} correct: {correct_counts[j]}/{len(X)} "
                  f"({correct_counts[j]/len(X)*100:.0f}%)")
    
    def plot_training_history(self) -> None:
        """Visualize the training process"""
        if self.history is None:
            print("No data to visualize. Train the model first!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(self.history.history['loss'], linewidth=2, color='#667eea')
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.history.history['accuracy'], linewidth=2, color='#764ba2')
        ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved: training_history.png")
        plt.show()
    
    def save_model(self, filepath: str = 'xor_and_model.h5') -> None:
        """Save the model to disk"""
        if self.model is None:
            print("Model not created!")
            return
        
        self.model.save(filepath)
        print(f"Model saved: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a saved model from disk"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded: {filepath}")
    
    def summary(self) -> None:
        """Print model architecture"""
        if self.model is None:
            print("Model not created!")
            return
        
        print("\nModel Architecture:")
        self.model.summary()


def main():
    """Main function to demonstrate the perceptron for XOR and AND"""
    print("XOR and AND Perceptron - TensorFlow Keras Implementation")
    print("-" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Author: Ded0Ak(David Hrytsenok)")
    print("-" * 60)
    
    # Create perceptron instance
    perceptron = XORAndPerceptron(hidden_units=4, learning_rate=0.1)
    
    perceptron.create_model()
    perceptron.summary()
    
    perceptron.train(epochs=500, verbose=1)
    
    perceptron.test()
    
    try:
        perceptron.plot_training_history()
    except Exception as e:
        print(f"\nCould not generate plot: {e}")
    
    perceptron.save_model('xor_and_model.h5')
    
    print("\nProgram completed!")


if __name__ == "__main__":
    main()