# model.py
import numpy as np
from typing import Dict, List, Tuple

# Dimensiones del modelo
INPUT_SIZE = 784
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10

def relu(Z: np.ndarray) -> np.ndarray:
    """Función de activación ReLU."""
    return np.maximum(0, Z)

def softmax(Z: np.ndarray) -> np.ndarray:
    """Softmax estable numéricamente."""
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def forward(X: np.ndarray, weights: Dict[str, np.ndarray]) -> np.ndarray:
    """Forward pass de la red."""
    W1, b1 = weights["W1"], weights["b1"]
    W2, b2 = weights["W2"], weights["b2"]
    
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    
    return softmax(Z2)

def evaluate_model(X: np.ndarray, y: np.ndarray, weights: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """Evalúa el modelo: loss y accuracy."""
    m = X.shape[0]
    A2 = forward(X, weights)
    
    # Loss (cross-entropy)
    correct_logprobs = -np.log(A2[np.arange(m), y] + 1e-9)
    loss = np.sum(correct_logprobs) / m
    
    # Accuracy
    predictions = np.argmax(A2, axis=1)
    accuracy = np.mean(predictions == y)
    
    return loss, accuracy

def init_weights() -> Dict[str, np.ndarray]:
    """Inicializa pesos con He initialization."""
    W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * np.sqrt(2.0 / INPUT_SIZE)
    W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * np.sqrt(2.0 / HIDDEN_SIZE)
    
    return {
        "W1": W1,
        "b1": np.zeros(HIDDEN_SIZE),
        "W2": W2,
        "b2": np.zeros(OUTPUT_SIZE)
    }

def apply_gradients(weights: Dict[str, np.ndarray], 
                    grads: Dict[str, np.ndarray], 
                    lr: float) -> Dict[str, np.ndarray]:
    """Aplica gradientes con gradient clipping."""
    # Gradient clipping para estabilidad
    for key in grads:
        grads[key] = np.clip(grads[key], -5, 5)
    
    return {
        "W1": weights["W1"] - lr * grads["W1"],
        "b1": weights["b1"] - lr * grads["b1"],
        "W2": weights["W2"] - lr * grads["W2"],
        "b2": weights["b2"] - lr * grads["b2"]
    }

def average_gradients(all_grads: List[Dict[str, np.ndarray]], 
                      template: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Promedia los gradientes de todos los workers."""
    avg_grads = {key: np.zeros_like(template[key]) for key in template}
    n = len(all_grads)
    
    if n == 0:
        return avg_grads
    
    for grads in all_grads:
        for key in avg_grads:
            avg_grads[key] += np.array(grads[key]) / n
    
    return avg_grads