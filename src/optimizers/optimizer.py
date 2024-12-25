import numpy as np

class OptimizerAdam:
    """
        Adam Optimizer for optimizing weights during training.

        Parameters:
        learning_rate (float): The learning rate for the optimizer.
        beta1 (float): Exponential decay rate for the first moment estimates.
        beta2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): Small constant to prevent division by zero.
    """

    def __init__(self, 
                 learning_rate: float = 0.001, 
                 beta1: float = 0.9, 
                 beta2: float = 0.999, 
                 epsilon: float = 1e-8):

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        
    def update(self, w: float, grad: np.ndarray) -> float:
        """
            Update weights using Adam optimization.

            Parameters:
            w (np.ndarray): Current weights.
            grad (np.ndarray): Gradient of the loss function w.r.t. weights.

            Returns:
            np.ndarray: Updated weights.
        """

        if self.m is None:
            self.m = np.zeros_like(grad)
            
        if self.v is None:
            self.v = np.zeros_like(grad)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update weights
        w = w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return w  
    
class OptimizerAdaDelta:
    def __init__(self, gamma=0.9, epsilon=1e-8):
        """
            Initialize the AdaDelta optimizer.

            Parameters:
            - gamma: Decay rate for the running average of the squared gradients.
            - epsilon: A small value to prevent division by zero.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Init accumulators for squared gradients and parameter updates
        self.accumulated_grad = 0
        self.accumulated_update = 0
        
    def update(self, w, grad):
        """
            Update the weight 'w' using the gradient 'grad' and AdaDelta optimization.

            Parameters:
            - w: Current weight.
            - grad: Gradient of the error with respect to the weight.

            Returns:
            - Updated weight 'w'.'
        """
        # Accumulate gradient squared
        self.accumulated_grad = self.gamma * self.accumulated_grad + (1-self.gamma) * grad**2
        
        # Compute the update (delta theta) using the root of the accumulated gradient
        update = - (np.sqrt(self.accumulated_update + self.epsilon) /
                    np.sqrt(self.accumulated_grad + self.epsilon)) * grad
                
        # Accumulate the squared updates
        self.accumulated_update = (
            self.gamma * self.accumulated_update + (1 - self.gamma) * update**2
        )
        
        # Apply the update to the weight
        w = w + update
        
        return w