# export_metrics.py
import csv
import os
from datetime import datetime
from Config import state
import numpy as np

def save_epoch_metrics(epoch, avg_loss, test_loss, test_acc, epoch_time, worker_losses):
    """
    Guarda las métricas de cada época en un archivo CSV detallado.
    Llama esta función después de cada época en Training.py.
    """
    results_dir = "training_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Crear nombre de archivo con timestamp y configuración
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_file = f"{results_dir}/detailed_{state.dataset_name}_{state.expected_workers}w_{state.max_epochs}e_{timestamp}.csv"
    
    # Verificar si es la primera época para crear el archivo
    is_first_epoch = (epoch == 0)
    
    try:
        with open(detailed_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            if is_first_epoch:
                # ========== METADATOS DEL EXPERIMENTO ==========
                writer.writerow(["=== METADATOS DEL EXPERIMENTO ==="])
                writer.writerow(["timestamp", timestamp])
                writer.writerow(["dataset", state.dataset_name])
                writer.writerow(["input_size", state.input_size])
                writer.writerow(["workers", state.expected_workers])
                writer.writerow(["epochs", state.max_epochs])
                writer.writerow(["learning_rate", state.learning_rate])
                writer.writerow([])
                
                # ========== DATOS POR ÉPOCA ==========
                writer.writerow(["=== DATOS POR ÉPOCA ==="])
                writer.writerow(["epoch", "train_loss", "test_loss", "test_accuracy", "epoch_time_seconds"])
            
            # Escribir datos de esta época
            writer.writerow([
                epoch + 1,
                f"{avg_loss:.6f}",
                f"{test_loss:.6f}",
                f"{test_acc:.6f}",
                f"{epoch_time:.3f}"
            ])
            
            # ========== MÉTRICAS POR WORKER (si existen) ==========
            if worker_losses and is_first_epoch:
                # Escribir encabezado de pérdidas por worker solo una vez
                writer.writerow([])
                writer.writerow(["=== PÉRDIDAS POR WORKER (por época) ==="])
                writer.writerow(["epoch"] + [f"worker_{wid}_loss" for wid in sorted(worker_losses.keys())])
            
            if worker_losses:
                # Escribir pérdidas de los workers para esta época
                row = [epoch + 1]
                for wid in sorted(worker_losses.keys()):
                    row.append(f"{worker_losses[wid]:.6f}")
                writer.writerow(row)
        
        return detailed_file
        
    except Exception as e:
        print(f" Error guardando métricas: {e}")
        return None

def save_final_summary(final_loss, final_acc, total_time):
    """
    Guarda un resumen del experimento en un archivo acumulativo.
    Llama esta función al final del entrenamiento.
    """
    results_dir = "training_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    summary_file = f"{results_dir}/experiments_summary.csv"
    file_exists = os.path.isfile(summary_file)
    
    # Preparar datos del experimento
    experiment_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset': state.dataset_name,
        'workers': state.expected_workers,
        'epochs': state.max_epochs,
        'learning_rate': state.learning_rate,
        'input_size': state.input_size,
        'total_time_seconds': f"{total_time:.3f}",
        'total_time_minutes': f"{total_time/60:.2f}",
        'final_test_loss': f"{final_loss:.6f}",
        'final_test_accuracy': f"{final_acc:.6f}"
    }
    
    try:
        with open(summary_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=experiment_data.keys())
            
            if not file_exists:
                writer.writeheader()
                print(f"\n Creando archivo de resumen: {summary_file}")
            
            writer.writerow(experiment_data)
        
        print(f" Resumen agregado a: {summary_file}")
        return summary_file
        
    except Exception as e:
        print(f" Error exportando resumen: {e}")
        return None