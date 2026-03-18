# dataset.py
import numpy as np
from torchvision import datasets, transforms
from typing import Tuple, List

def load_mnist_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carga MNIST usando torchvision."""
    print("Cargando MNIST con torchvision...")
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    X_train = train_dataset.data.numpy().reshape(-1, 784) / 255.0
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().reshape(-1, 784) / 255.0
    y_test = test_dataset.targets.numpy()
    
    print(f"✓ Dataset cargado: {X_train.shape[0]} train, {X_test.shape[0]} test")
    return X_train, y_train, X_test, y_test

def stratified_split(y: np.ndarray, n_workers: int) -> List[np.ndarray]:
    """Divide el dataset estratificadamente por clases."""
    class_indices = {label: np.where(y == label)[0] for label in range(10)}
    
    # Mezclar índices de cada clase
    for indices in class_indices.values():
        np.random.shuffle(indices)
    
    # Distribuir entre workers
    worker_chunks = [[] for _ in range(n_workers)]
    
    for indices in class_indices.values():
        splits = np.array_split(indices, n_workers)
        for i, split in enumerate(splits):
            worker_chunks[i].extend(split)
    
    # Mezclar cada chunk
    for i in range(n_workers):
        worker_chunks[i] = np.array(worker_chunks[i])
        np.random.shuffle(worker_chunks[i])
    
    return worker_chunks