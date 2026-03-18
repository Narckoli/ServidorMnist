# metrics.py
import matplotlib.pyplot as plt
import numpy as np
import time

from Config import state

def plot_metrics():
    """Genera gráficas de las métricas de entrenamiento."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Métricas de Entrenamiento Distribuido MNIST', 
                 fontsize=16, fontweight='bold')
    
    epochs = range(1, len(state.train_losses) + 1)
    
    # Gráfica 1: Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, state.train_losses, 'b-o', linewidth=2, label='Train Loss')
    ax1.plot(epochs, state.test_losses, 'r-s', linewidth=2, label='Test Loss')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss por Época')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, state.test_accuracies, 'g-o', linewidth=2, label='Test Accuracy')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Test Accuracy por Época')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Gráfica 3: Loss por worker
    ax3 = axes[1, 0]
    for worker_id, losses in state.worker_losses.items():
        ax3.plot(range(1, len(losses) + 1), losses, '-o', label=f'Worker {worker_id}')
    ax3.set_xlabel('Época')
    ax3.set_ylabel('Loss')
    ax3.set_title('Loss Individual por Worker')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfica 4: Tiempo por época
    ax4 = axes[1, 1]
    ax4.bar(epochs, state.epoch_times, color='skyblue', edgecolor='navy')
    ax4.set_xlabel('Época')
    ax4.set_ylabel('Tiempo (segundos)')
    ax4.set_title('Tiempo de Entrenamiento por Época')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = f"training_metrics_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n📊 Gráficas guardadas en: {filename}")
    plt.show()
    
    print_summary()

def print_summary():
    """Imprime un resumen detallado del entrenamiento."""
    total_time = sum(state.epoch_times)
    total_ms = total_time * 1000
    mins = int(total_time // 60)
    secs = int(total_time % 60)
    ms = int((total_time % 1) * 1000)
    
    print("\n" + "="*70)
    print("RESUMEN FINAL DE ENTRENAMIENTO")
    print("="*70)
    print(f"\nTIEMPO TOTAL DE ENTRENAMIENTO")
    print(f"  Milisegundos: {total_ms:,.0f} ms")
    print(f"  Formateado:   {mins:02d} min {secs:02d} s {ms:03d} ms")
    print(f"  Total:        {total_time:.3f} segundos")
    print(f"{'='*70}\n")
    
    print(f"{'Época':<8}{'Train Loss':<12}{'Test Loss':<12}{'Test Acc':<12}{'Tiempo(s)':<10}")
    print("-"*70)
    
    for i in range(len(state.train_losses)):
        print(f"{i+1:<8}{state.train_losses[i]:<12.4f}"
              f"{state.test_losses[i]:<12.4f}"
              f"{state.test_accuracies[i]:<12.4f}"
              f"{state.epoch_times[i]:<10.2f}")
    
    print("-"*70)
    best_epoch = np.argmax(state.test_accuracies)
    print(f"Mejor Test Accuracy: {max(state.test_accuracies):.4f} (Época {best_epoch+1})")
    print("="*70)