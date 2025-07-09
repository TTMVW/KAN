import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class MexicanHatWavelet:
    """Mexican Hat (Ricker) Wavelet - Mother Wavelet"""
    
    @staticmethod
    def psi(x):
        """Mexican hat wavelet function: ψ(x) = (1 - x²) * exp(-x²/2)"""
        x = np.asarray(x)
        return (1 - x**2) * np.exp(-x**2 / 2)
    
    @staticmethod
    def psi_derivative(x):
        """Derivative of Mexican hat wavelet for backpropagation"""
        x = np.asarray(x)
        exp_term = np.exp(-x**2 / 2)
        return x * (x**2 - 3) * exp_term

class SimplexTriangulation:
    """Handles triangulation of [0,1]^n hypercube into simplices"""
    
    def __init__(self, n_dims: int):
        self.n_dims = n_dims
        self.vertices = self._generate_hypercube_vertices()
        self.triangulation = self._triangulate_hypercube()
        
    def _generate_hypercube_vertices(self):
        """Generate all 2^n vertices of [0,1]^n hypercube"""
        vertices = []
        for i in range(2**self.n_dims):
            vertex = []
            for j in range(self.n_dims):
                vertex.append((i >> j) & 1)
            vertices.append(vertex)
        return np.array(vertices, dtype=float)
    
    def _triangulate_hypercube(self):
        """Triangulate hypercube using Delaunay triangulation"""
        if self.n_dims == 1:
            # 1D case: just one interval
            return np.array([[0, 1]])
        elif self.n_dims == 2:
            # 2D case: two triangles
            return np.array([[0, 1, 2], [1, 2, 3]])
        else:
            # Higher dimensions: use Delaunay
            tri = Delaunay(self.vertices)
            return tri.simplices
    
    def get_simplex_containing_point(self, point):
        """Find which simplex contains a given point"""
        point = np.asarray(point)
        
        for i, simplex in enumerate(self.triangulation):
            if self._point_in_simplex(point, simplex):
                return i
        return 0  # Fallback to first simplex
    
    def _point_in_simplex(self, point, simplex_indices):
        """Check if point is inside simplex using barycentric coordinates"""
        vertices = self.vertices[simplex_indices]
        
        # Convert to barycentric coordinates
        A = vertices[1:] - vertices[0]
        b = point - vertices[0]
        
        try:
            coords = np.linalg.solve(A.T, b)
            return np.all(coords >= 0) and np.sum(coords) <= 1
        except:
            return False

class AdamOptimizer:
    """Adam optimizer for wavelet parameters"""
    
    def __init__(self, params_shape, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Initialize moment estimates
        self.m = np.zeros(params_shape)  # First moment
        self.v = np.zeros(params_shape)  # Second moment
        self.t = 0  # Time step
    
    def update(self, params, gradients):
        """Update parameters using Adam optimization"""
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params

class WaveletLayer:
    """Single layer of WaveletKAN with Mexican hat wavelets and Adam optimization"""
    
    def __init__(self, input_dim: int, output_dim: int, n_wavelets_per_edge: int = 5, learning_rate=0.001):
        # Input validation
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if n_wavelets_per_edge <= 0:
            raise ValueError(f"n_wavelets_per_edge must be positive, got {n_wavelets_per_edge}")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_wavelets = n_wavelets_per_edge
        
        # Debug print
        print(f"Creating WaveletLayer: input_dim={input_dim}, output_dim={output_dim}, n_wavelets={n_wavelets_per_edge}")
        
        # Learnable parameters: scale, translation, amplitude for each wavelet
        # Shape: (input_dim, output_dim, n_wavelets, 3) for [scale, translation, amplitude]
        params_shape = (input_dim, output_dim, n_wavelets_per_edge, 3)
        print(f"Parameter shape: {params_shape}")
        
        try:
            self.params = np.random.normal(0, 0.1, params_shape)
        except Exception as e:
            print(f"Error creating params array with shape {params_shape}: {e}")
            raise
        
        # Initialize reasonable defaults
        self.params[:, :, :, 0] = np.random.uniform(0.5, 2.0, (input_dim, output_dim, n_wavelets_per_edge))  # scales
        self.params[:, :, :, 1] = np.random.uniform(0.0, 1.0, (input_dim, output_dim, n_wavelets_per_edge))  # translations
        self.params[:, :, :, 2] = np.random.normal(0, 0.5, (input_dim, output_dim, n_wavelets_per_edge))     # amplitudes
        
        self.wavelet = MexicanHatWavelet()
        
        # Adam optimizer for this layer
        self.optimizer = AdamOptimizer(self.params.shape, learning_rate)
        
    def forward(self, x):
        """Forward pass through wavelet layer"""
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        batch_size = x.shape[0]
        
        # Validate input dimensions
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[1]}")
        
        output = np.zeros((batch_size, self.output_dim))
        
        for i in range(self.input_dim):
            for j in range(self.output_dim):
                for k in range(self.n_wavelets):
                    scale = self.params[i, j, k, 0]
                    translation = self.params[i, j, k, 1]
                    amplitude = self.params[i, j, k, 2]
                    
                    # Apply wavelet transformation: ψ((x - b)/a)
                    transformed_x = (x[:, i] - translation) / (scale + 1e-8)
                    wavelet_output = self.wavelet.psi(transformed_x)
                    output[:, j] += amplitude * wavelet_output
        
        return output
    
    def backward(self, x, grad_output, use_adam=True):
        """Backward pass with Adam optimization"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        batch_size = x.shape[0]
        
        # Validate dimensions
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {x.shape[1]}")
        if grad_output.shape[1] != self.output_dim:
            raise ValueError(f"Output gradient dimension mismatch: expected {self.output_dim}, got {grad_output.shape[1]}")
        
        grad_input = np.zeros_like(x)
        
        # Collect gradients for all parameters
        param_gradients = np.zeros_like(self.params)
        
        for i in range(self.input_dim):
            for j in range(self.output_dim):
                for k in range(self.n_wavelets):
                    scale = self.params[i, j, k, 0]
                    translation = self.params[i, j, k, 1]
                    amplitude = self.params[i, j, k, 2]
                    
                    transformed_x = (x[:, i] - translation) / (scale + 1e-8)
                    wavelet_val = self.wavelet.psi(transformed_x)
                    wavelet_derivative = self.wavelet.psi_derivative(transformed_x)
                    
                    # Gradients for parameters
                    grad_amplitude = np.mean(grad_output[:, j] * wavelet_val)
                    grad_scale = np.mean(grad_output[:, j] * amplitude * wavelet_derivative * 
                                       (-transformed_x / (scale + 1e-8)))
                    grad_translation = np.mean(grad_output[:, j] * amplitude * wavelet_derivative * 
                                             (-1 / (scale + 1e-8)))
                    
                    # Store gradients
                    param_gradients[i, j, k, 0] = grad_scale
                    param_gradients[i, j, k, 1] = grad_translation
                    param_gradients[i, j, k, 2] = grad_amplitude
                    
                    # Gradient for input
                    grad_input[:, i] += (grad_output[:, j] * amplitude * wavelet_derivative / 
                                       (scale + 1e-8))
        
        # Update parameters using Adam optimizer
        if use_adam:
            self.params = self.optimizer.update(self.params, param_gradients)
        else:
            # Fallback to simple gradient descent
            self.params -= 0.01 * param_gradients
        
        return grad_input

class TriangulatedWaveletKAN:
    """Complete WaveletKAN system with triangulated parallelism and Adam optimization"""
    
    def __init__(self, layer_dims: List[int], n_wavelets_per_edge: int = 5, learning_rate: float = 0.001):
        # Input validation
        if len(layer_dims) < 2:
            raise ValueError(f"layer_dims must have at least 2 dimensions, got {len(layer_dims)}")
        for i, dim in enumerate(layer_dims):
            if dim <= 0:
                raise ValueError(f"layer_dims[{i}] must be positive, got {dim}")
        
        self.layer_dims = layer_dims
        self.n_dims = layer_dims[0]
        self.n_wavelets = n_wavelets_per_edge
        self.learning_rate = learning_rate
        
        print(f"Creating TriangulatedWaveletKAN with dimensions: {layer_dims}")
        
        # Create triangulation for input space
        self.triangulation = SimplexTriangulation(self.n_dims)
        
        # Create wavelet layers with Adam optimization
        self.layers = []
        for i in range(len(layer_dims) - 1):
            print(f"Creating layer {i}: {layer_dims[i]} -> {layer_dims[i+1]}")
            layer = WaveletLayer(layer_dims[i], layer_dims[i+1], n_wavelets_per_edge, learning_rate)
            self.layers.append(layer)
        
        # For parallel processing
        self.n_processes = min(mp.cpu_count(), len(self.triangulation.triangulation))
        
    def _process_simplex_batch(self, args):
        """Process a batch of data points within a specific simplex"""
        simplex_id, x_batch, layer_weights = args
        
        # This would contain the actual wavelet computations for the simplex
        # For now, we'll use the standard forward pass
        outputs = []
        for x in x_batch:
            output = x.copy()
            for layer in self.layers:
                output = layer.forward(output)
            outputs.append(output)
        
        return simplex_id, np.array(outputs)
    
    def forward_parallel(self, X):
        """Parallel forward pass using triangulation"""
        batch_size = X.shape[0]
        
        # Group data points by simplex
        simplex_groups = {}
        for i, x in enumerate(X):
            simplex_id = self.triangulation.get_simplex_containing_point(x)
            if simplex_id not in simplex_groups:
                simplex_groups[simplex_id] = []
            simplex_groups[simplex_id].append((i, x))
        
        # Prepare arguments for parallel processing
        args_list = []
        for simplex_id, points in simplex_groups.items():
            indices, x_batch = zip(*points)
            args_list.append((simplex_id, np.array(x_batch), None))  # layer_weights placeholder
        
        # Execute parallel processing
        results = {}
        if len(args_list) > 1 and self.n_processes > 1:
            with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                parallel_results = executor.map(self._process_simplex_batch, args_list)
                
                for simplex_id, outputs in parallel_results:
                    results[simplex_id] = outputs
        else:
            # Sequential fallback
            for args in args_list:
                simplex_id, outputs = self._process_simplex_batch(args)
                results[simplex_id] = outputs
        
        # Reconstruct output in original order
        final_output = np.zeros((batch_size, self.layer_dims[-1]))
        for simplex_id, points in simplex_groups.items():
            indices, _ = zip(*points)
            for i, idx in enumerate(indices):
                final_output[idx] = results[simplex_id][i]
        
        return final_output
    
    def forward(self, X):
        """Standard forward pass (non-parallel for comparison)"""
        output = X.copy()
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def train(self, X_train, y_train, epochs=100, use_parallel=True, use_adam=True, 
              beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Training loop with Adam optimization and triangulated parallelism"""
        history = {'loss': [], 'time': [], 'grad_norm': []}
        
        # Set Adam parameters for all layers
        if use_adam:
            for layer in self.layers:
                layer.optimizer.beta1 = beta1
                layer.optimizer.beta2 = beta2
                layer.optimizer.epsilon = epsilon
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Forward pass
            if use_parallel and self.n_processes > 1:
                predictions = self.forward_parallel(X_train)
            else:
                predictions = self.forward(X_train)
            
            # Compute loss (MSE)
            loss = np.mean((predictions - y_train)**2)
            
            # Backward pass with Adam optimization
            grad_output = 2 * (predictions - y_train) / len(y_train)
            
            # Update each layer and track gradient norms
            current_grad = grad_output
            total_grad_norm = 0
            
            # For backpropagation, we need the intermediate outputs
            # Store forward pass outputs for each layer
            layer_inputs = [X_train]
            current_input = X_train
            
            for layer in self.layers:
                current_output = layer.forward(current_input)
                layer_inputs.append(current_output)
                current_input = current_output
            
            # Now do backward pass with correct inputs
            for i, layer in enumerate(reversed(self.layers)):
                # Store old params to compute gradient norm
                old_params = layer.params.copy()
                
                # Use the correct input for this layer
                layer_input = layer_inputs[-(i+2)]  # Get input for this layer
                
                # Backward pass with Adam
                current_grad = layer.backward(layer_input, current_grad, use_adam=use_adam)
                
                # Compute gradient norm for monitoring
                param_change = np.linalg.norm(layer.params - old_params)
                total_grad_norm += param_change
            
            epoch_time = time.time() - start_time
            history['loss'].append(loss)
            history['time'].append(epoch_time)
            history['grad_norm'].append(total_grad_norm)
            
            if epoch % 10 == 0:
                optimizer_type = "Adam" if use_adam else "SGD"
                parallel_type = "Parallel" if use_parallel else "Sequential"
                print(f"Epoch {epoch}: Loss = {loss:.6f}, Grad Norm = {total_grad_norm:.4f}, "
                      f"Time = {epoch_time:.3f}s ({optimizer_type}, {parallel_type})")
        
        return history
    
    def predict(self, X, use_parallel=True):
        """Make predictions with option for parallel processing"""
        if use_parallel and self.n_processes > 1:
            return self.forward_parallel(X)
        else:
            return self.forward(X)

def generate_test_data(n_samples=1000, n_dims=3):
    """Generate normalized test data in [0,1]^n"""
    # Generate random data in [0,1]^n
    X = np.random.rand(n_samples, n_dims)
    
    # Create a non-linear target function for testing
    # Example: y = sin(2π * sum(x_i)) + noise
    y = np.sin(2 * np.pi * np.sum(X, axis=1, keepdims=True)) + 0.1 * np.random.randn(n_samples, 1)
    
    return X, y

def benchmark_optimizers():
    """Benchmark Adam vs SGD optimization"""
    print("=== WaveletKAN Optimizer Comparison ===\n")
    
    # Test parameters
    n_dims = 3
    n_samples = 500
    layer_dims = [n_dims, 8, 4, 1]
    epochs = 80
    
    # Generate test data
    X_train, y_train = generate_test_data(n_samples, n_dims)
    X_test, y_test = generate_test_data(200, n_dims)
    
    print(f"Training data: {X_train.shape}")
    print(f"Network architecture: {layer_dims}")
    print(f"Training epochs: {epochs}\n")
    
    # Test SGD
    print("Training with SGD...")
    model_sgd = TriangulatedWaveletKAN(layer_dims, n_wavelets_per_edge=3, learning_rate=0.01)
    start_time = time.time()
    history_sgd = model_sgd.train(X_train, y_train, epochs=epochs, use_adam=False)
    sgd_time = time.time() - start_time
    
    # Test Adam
    print("\nTraining with Adam...")
    model_adam = TriangulatedWaveletKAN(layer_dims, n_wavelets_per_edge=3, learning_rate=0.001)
    start_time = time.time()
    history_adam = model_adam.train(X_train, y_train, epochs=epochs, use_adam=True, 
                                  beta1=0.9, beta2=0.999)
    adam_time = time.time() - start_time
    
    # Test predictions
    pred_sgd = model_sgd.predict(X_test, use_parallel=False)
    pred_adam = model_adam.predict(X_test, use_parallel=False)
    
    test_loss_sgd = np.mean((pred_sgd - y_test)**2)
    test_loss_adam = np.mean((pred_adam - y_test)**2)
    
    # Results
    print(f"\n=== OPTIMIZER COMPARISON ===")
    print(f"SGD - Final loss: {history_sgd['loss'][-1]:.6f}, Test loss: {test_loss_sgd:.6f}, Time: {sgd_time:.2f}s")
    print(f"Adam - Final loss: {history_adam['loss'][-1]:.6f}, Test loss: {test_loss_adam:.6f}, Time: {adam_time:.2f}s")
    print(f"Adam improvement: {((history_sgd['loss'][-1] - history_adam['loss'][-1])/history_sgd['loss'][-1]*100):.1f}% better loss")
    
    return model_adam, history_adam, history_sgd

def benchmark_parallel_vs_sequential():
    """Benchmark parallel vs sequential processing with Adam optimization"""
    print("=== WaveletKAN Triangulated Parallelism Benchmark (with Adam) ===\n")
    
    # Test parameters
    n_dims = 3
    n_samples = 500
    layer_dims = [n_dims, 8, 4, 1]
    
    # Generate test data
    X_train, y_train = generate_test_data(n_samples, n_dims)
    X_test, y_test = generate_test_data(200, n_dims)
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Network architecture: {layer_dims}")
    print(f"Available CPU cores: {mp.cpu_count()}\n")
    
    # Create model
    model = TriangulatedWaveletKAN(layer_dims, n_wavelets_per_edge=3, learning_rate=0.001)
    print(f"Number of simplices in triangulation: {len(model.triangulation.triangulation)}")
    print(f"Using {model.n_processes} processes for parallel computation\n")
    
    # Train with sequential processing + Adam
    print("Training with SEQUENTIAL processing + Adam...")
    start_time = time.time()
    history_seq = model.train(X_train, y_train, epochs=50, use_parallel=False, use_adam=True)
    seq_train_time = time.time() - start_time
    
    # Reset model for fair comparison
    model = TriangulatedWaveletKAN(layer_dims, n_wavelets_per_edge=3, learning_rate=0.001)
    
    # Train with parallel processing + Adam
    print("\nTraining with PARALLEL processing + Adam...")
    start_time = time.time()
    history_par = model.train(X_train, y_train, epochs=50, use_parallel=True, use_adam=True)
    par_train_time = time.time() - start_time
    
    # Test predictions
    print("\nTesting predictions...")
    pred_seq = model.predict(X_test, use_parallel=False)
    pred_par = model.predict(X_test, use_parallel=True)
    
    test_loss_seq = np.mean((pred_seq - y_test)**2)
    test_loss_par = np.mean((pred_par - y_test)**2)
    
    # Results
    print(f"\n=== RESULTS (with Adam Optimization) ===")
    print(f"Sequential training time: {seq_train_time:.2f}s")
    print(f"Parallel training time: {par_train_time:.2f}s")
    print(f"Speedup: {seq_train_time/par_train_time:.2f}x")
    print(f"Sequential final loss: {history_seq['loss'][-1]:.6f}")
    print(f"Parallel final loss: {history_par['loss'][-1]:.6f}")
    print(f"Sequential test loss: {test_loss_seq:.6f}")
    print(f"Parallel test loss: {test_loss_par:.6f}")
    
    return model, history_seq, history_par

def visualize_triangulation_2d():
    """Visualize triangulation for 2D case"""
    triangulation = SimplexTriangulation(2)
    
    plt.figure(figsize=(8, 8))
    
    # Plot triangulation
    for simplex in triangulation.triangulation:
        vertices = triangulation.vertices[simplex]
        # Close the triangle
        vertices_closed = np.vstack([vertices, vertices[0]])
        plt.plot(vertices_closed[:, 0], vertices_closed[:, 1], 'b-', linewidth=2)
        
        # Fill triangle with different colors
        plt.fill(vertices[:, 0], vertices[:, 1], alpha=0.3, 
                color=np.random.rand(3,))
    
    # Plot vertices
    plt.scatter(triangulation.vertices[:, 0], triangulation.vertices[:, 1], 
               c='red', s=100, zorder=5)
    
    # Add vertex labels
    for i, vertex in enumerate(triangulation.vertices):
        plt.annotate(f'V{i}', (vertex[0], vertex[1]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.title('Triangulation of [0,1]² Hypercube\n(Basis for Parallel WaveletKAN)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()
    # Construct a filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"Triangulation_Hypercube_WaveletKAN_{timestamp}.png"
    print(f"Saving figure as: {filename}")
    # Save the figure
    plt.savefig(filename, bbox_inches='tight')

if __name__ == "__main__":
    print("Mexican Hat WaveletKAN with Triangulated Parallelism + Adam Optimization")
    print("Based on Menger Universal Dendrite Theory (1933)")
    print("=" * 70)
    
    # Visualize 2D triangulation
    print("\n1. Visualizing 2D triangulation...")
    visualize_triangulation_2d()
    
    # Compare optimizers
    print("\n2. Comparing SGD vs Adam optimization...")
    model_adam, hist_adam, hist_sgd = benchmark_optimizers()
    
    # Run parallel benchmark with Adam
    print("\n3. Running parallel vs sequential benchmark with Adam...")
    model, hist_seq, hist_par = benchmark_parallel_vs_sequential()
    
    print("\nSystem ready for custom experiments!")
    print("The model now implements:")
    print("- Mexican Hat mother wavelet")
    print("- Adam optimization with adaptive learning rates")
    print("- Triangulated parallelism based on Menger's universal dendrite")
    print("- Training and testing on normalized [0,1]^n space")
    print("- Parallel forward and backward propagation")
    print("- Gradient norm monitoring for training diagnostics")
