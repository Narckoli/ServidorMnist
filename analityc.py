# visualize_experiments.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import re

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

def load_and_clean_data(csv_path='training_results/experiments_summary.csv'):
    """Carga y limpia el archivo CSV de experimentos manejando formatos inconsistentes."""
    
    # Intentar leer el CSV con manejo de errores
    try:
        # Primero intentar leer normalmente
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError:
        # Si falla, leer línea por línea para identificar el problema
        print("⚠️ Error de formato CSV. Intentando recuperar datos...")
        
        data_rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Obtener encabezados de la primera línea
        headers = lines[0].strip().split(',')
        expected_cols = len(headers)
        
        print(f"   Encabezados esperados ({expected_cols}): {headers}")
        
        # Procesar cada línea
        for i, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if not line:
                continue
            
            # Dividir respetando comillas
            values = []
            current = ''
            in_quotes = False
            
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    values.append(current)
                    current = ''
                else:
                    current += char
            values.append(current)  # Último valor
            
            # Si la línea tiene más columnas de las esperadas, truncar
            if len(values) > expected_cols:
                print(f"   Línea {i}: {len(values)} columnas, truncando a {expected_cols}")
                values = values[:expected_cols]
            # Si tiene menos, rellenar con NaN
            elif len(values) < expected_cols:
                print(f"   Línea {i}: {len(values)} columnas, rellenando con NaN")
                values.extend([''] * (expected_cols - len(values)))
            
            data_rows.append(values)
        
        # Crear DataFrame con los datos procesados
        df = pd.DataFrame(data_rows, columns=headers)
    
    # Limpiar nombres de columnas (remover espacios)
    df.columns = df.columns.str.strip()
    
    # Seleccionar solo las columnas que nos interesan
    expected_columns = ['timestamp', 'dataset', 'workers', 'epochs', 'learning_rate', 
                        'input_size', 'total_time_seconds', 'total_time_minutes', 
                        'final_test_loss', 'final_test_accuracy']
    
    # Verificar qué columnas existen
    existing_cols = [col for col in expected_columns if col in df.columns]
    missing_cols = [col for col in expected_columns if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️ Columnas faltantes: {missing_cols}")
    
    # Usar solo las columnas que existen
    if existing_cols:
        df = df[existing_cols].copy()
    else:
        raise ValueError("No se encontraron columnas válidas en el CSV")
    
    # Limpiar datos: eliminar filas con valores nulos en columnas clave
    df = df.dropna(subset=['final_test_accuracy', 'epochs', 'workers'], how='all')
    
    # Convertir tipos numéricos (manejando errores)
    numeric_cols = ['workers', 'epochs', 'learning_rate', 'input_size', 
                    'total_time_seconds', 'total_time_minutes', 
                    'final_test_loss', 'final_test_accuracy']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Limpiar dataset names
    if 'dataset' in df.columns:
        df['dataset'] = df['dataset'].str.lower().str.strip()
        # Reemplazar 'cnn' en columna dataset si es necesario
        df.loc[df['dataset'] == 'cnn', 'dataset'] = 'cifar10'
    
    # Eliminar filas con valores inválidos
    df = df.dropna(subset=['workers', 'epochs', 'final_test_accuracy'])
    
    # Filtrar valores razonables
    df = df[df['final_test_accuracy'] <= 1.0]  # Accuracy no puede ser > 1
    df = df[df['final_test_accuracy'] >= 0]     # Accuracy no puede ser < 0
    df = df[df['workers'] > 0]
    df = df[df['epochs'] > 0]
    
    # Separar por dataset
    df_mnist = df[df['dataset'] == 'mnist'].copy() if 'dataset' in df.columns else pd.DataFrame()
    df_cifar = df[df['dataset'] == 'cifar10'].copy() if 'dataset' in df.columns else pd.DataFrame()
    
    print(f"\n📊 Datos cargados:")
    print(f"   Total experimentos válidos: {len(df)}")
    print(f"   MNIST: {len(df_mnist)} experimentos")
    print(f"   CIFAR-10: {len(df_cifar)} experimentos")
    
    if len(df_mnist) > 0:
        print(f"\n   MNIST - Épocas: {df_mnist['epochs'].min():.0f}-{df_mnist['epochs'].max():.0f}, Workers: {df_mnist['workers'].min():.0f}-{df_mnist['workers'].max():.0f}")
        print(f"   MNIST - Accuracy: {df_mnist['final_test_accuracy'].min():.4f}-{df_mnist['final_test_accuracy'].max():.4f}")
    
    if len(df_cifar) > 0:
        print(f"\n   CIFAR - Épocas: {df_cifar['epochs'].min():.0f}-{df_cifar['epochs'].max():.0f}, Workers: {df_cifar['workers'].min():.0f}-{df_cifar['workers'].max():.0f}")
        print(f"   CIFAR - Accuracy: {df_cifar['final_test_accuracy'].min():.4f}-{df_cifar['final_test_accuracy'].max():.4f}")
    
    return df_mnist, df_cifar

def plot_accuracy_vs_epochs(df_mnist, df_cifar):
    """Gráfico: Precisión en función del número de épocas"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Precisión vs Número de Épocas', fontsize=16, fontweight='bold')
    
    # MNIST
    ax1 = axes[0]
    if len(df_mnist) > 0:
        for workers in sorted(df_mnist['workers'].unique()):
            subset = df_mnist[df_mnist['workers'] == workers]
            subset = subset.sort_values('epochs')
            ax1.plot(subset['epochs'], subset['final_test_accuracy'], 
                    'o-', linewidth=2, markersize=8, label=f'{int(workers)} worker(s)')
        
        ax1.set_xlabel('Número de Épocas')
        ax1.set_ylabel('Accuracy Final')
        ax1.set_title('MNIST')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
    else:
        ax1.text(0.5, 0.5, 'Sin datos de MNIST', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('MNIST')
    
    # CIFAR-10
    ax2 = axes[1]
    if len(df_cifar) > 0:
        for workers in sorted(df_cifar['workers'].unique()):
            subset = df_cifar[df_cifar['workers'] == workers]
            subset = subset.sort_values('epochs')
            ax2.plot(subset['epochs'], subset['final_test_accuracy'], 
                    's-', linewidth=2, markersize=8, label=f'{int(workers)} worker(s)')
        
        ax2.set_xlabel('Número de Épocas')
        ax2.set_ylabel('Accuracy Final')
        ax2.set_title('CIFAR-10')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
    else:
        ax2.text(0.5, 0.5, 'Sin datos de CIFAR-10', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('CIFAR-10')
    
    plt.tight_layout()
    plt.savefig('training_results/accuracy_vs_epochs.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico guardado: training_results/accuracy_vs_epochs.png")

def plot_accuracy_vs_workers(df_mnist, df_cifar):
    """Gráfico: Precisión en función del número de workers"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Precisión vs Número de Workers', fontsize=16, fontweight='bold')
    
    # MNIST
    ax1 = axes[0]
    if len(df_mnist) > 0:
        epoch_ranges = [(0, 100), (100, 300), (300, 1000)]
        colors = ['blue', 'green', 'red']
        labels = ['<100 épocas', '100-300 épocas', '>300 épocas']
        
        for i, (min_ep, max_ep) in enumerate(epoch_ranges):
            subset = df_mnist[(df_mnist['epochs'] >= min_ep) & (df_mnist['epochs'] < max_ep)]
            if not subset.empty:
                avg_by_workers = subset.groupby('workers')['final_test_accuracy'].agg(['mean', 'std']).reset_index()
                ax1.errorbar(avg_by_workers['workers'], avg_by_workers['mean'], 
                            yerr=avg_by_workers['std'], marker='o', capsize=5,
                            color=colors[i], label=labels[i], linewidth=2, markersize=8)
        
        ax1.set_xlabel('Número de Workers')
        ax1.set_ylabel('Accuracy Final')
        ax1.set_title('MNIST')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        ax1.set_xticks(sorted(df_mnist['workers'].unique()))
    else:
        ax1.text(0.5, 0.5, 'Sin datos de MNIST', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('MNIST')
    
    # CIFAR-10
    ax2 = axes[1]
    if len(df_cifar) > 0:
        epoch_ranges = [(0, 50), (50, 150), (150, 1000)]
        colors = ['blue', 'green', 'red']
        labels = ['<50 épocas', '50-150 épocas', '>150 épocas']
        
        for i, (min_ep, max_ep) in enumerate(epoch_ranges):
            subset = df_cifar[(df_cifar['epochs'] >= min_ep) & (df_cifar['epochs'] < max_ep)]
            if not subset.empty:
                avg_by_workers = subset.groupby('workers')['final_test_accuracy'].agg(['mean', 'std']).reset_index()
                ax2.errorbar(avg_by_workers['workers'], avg_by_workers['mean'], 
                            yerr=avg_by_workers['std'], marker='s', capsize=5,
                            color=colors[i], label=labels[i], linewidth=2, markersize=8)
        
        ax2.set_xlabel('Número de Workers')
        ax2.set_ylabel('Accuracy Final')
        ax2.set_title('CIFAR-10')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        ax2.set_xticks(sorted(df_cifar['workers'].unique()))
    else:
        ax2.text(0.5, 0.5, 'Sin datos de CIFAR-10', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('CIFAR-10')
    
    plt.tight_layout()
    plt.savefig('training_results/accuracy_vs_workers.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico guardado: training_results/accuracy_vs_workers.png")

def plot_time_vs_epochs(df_mnist, df_cifar):
    """Gráfico: Tiempo total de entrenamiento vs épocas"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Tiempo de Entrenamiento vs Número de Épocas', fontsize=16, fontweight='bold')
    
    # MNIST
    ax1 = axes[0]
    if len(df_mnist) > 0:
        for workers in sorted(df_mnist['workers'].unique()):
            subset = df_mnist[df_mnist['workers'] == workers]
            subset = subset.sort_values('epochs')
            ax1.plot(subset['epochs'], subset['total_time_minutes'], 
                    'o-', linewidth=2, markersize=8, label=f'{int(workers)} worker(s)')
        
        ax1.set_xlabel('Número de Épocas')
        ax1.set_ylabel('Tiempo Total (minutos)')
        ax1.set_title('MNIST')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Sin datos de MNIST', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('MNIST')
    
    # CIFAR-10
    ax2 = axes[1]
    if len(df_cifar) > 0:
        for workers in sorted(df_cifar['workers'].unique()):
            subset = df_cifar[df_cifar['workers'] == workers]
            subset = subset.sort_values('epochs')
            ax2.plot(subset['epochs'], subset['total_time_minutes'], 
                    's-', linewidth=2, markersize=8, label=f'{int(workers)} worker(s)')
        
        ax2.set_xlabel('Número de Épocas')
        ax2.set_ylabel('Tiempo Total (minutos)')
        ax2.set_title('CIFAR-10')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Sin datos de CIFAR-10', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('CIFAR-10')
    
    plt.tight_layout()
    plt.savefig('training_results/time_vs_epochs.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico guardado: training_results/time_vs_epochs.png")

def plot_efficiency_analysis(df_mnist, df_cifar):
    """Gráfico adicional: Eficiencia (accuracy por minuto)"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Eficiencia: Accuracy por Minuto de Entrenamiento', fontsize=16, fontweight='bold')
    
    # MNIST
    ax1 = axes[0]
    if len(df_mnist) > 0:
        df_mnist['accuracy_per_minute'] = df_mnist['final_test_accuracy'] / df_mnist['total_time_minutes']
        
        for workers in sorted(df_mnist['workers'].unique()):
            subset = df_mnist[df_mnist['workers'] == workers]
            ax1.scatter(subset['epochs'], subset['accuracy_per_minute'], 
                       s=100, alpha=0.7, label=f'{int(workers)} worker(s)')
        
        ax1.set_xlabel('Número de Épocas')
        ax1.set_ylabel('Accuracy por Minuto')
        ax1.set_title('MNIST')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Sin datos de MNIST', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('MNIST')
    
    # CIFAR-10
    ax2 = axes[1]
    if len(df_cifar) > 0:
        df_cifar['accuracy_per_minute'] = df_cifar['final_test_accuracy'] / df_cifar['total_time_minutes']
        
        for workers in sorted(df_cifar['workers'].unique()):
            subset = df_cifar[df_cifar['workers'] == workers]
            ax2.scatter(subset['epochs'], subset['accuracy_per_minute'], 
                       s=100, alpha=0.7, label=f'{int(workers)} worker(s)')
        
        ax2.set_xlabel('Número de Épocas')
        ax2.set_ylabel('Accuracy por Minuto')
        ax2.set_title('CIFAR-10')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Sin datos de CIFAR-10', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('CIFAR-10')
    
    plt.tight_layout()
    plt.savefig('training_results/efficiency_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico guardado: training_results/efficiency_analysis.png")

def print_statistics(df_mnist, df_cifar):
    """Imprime estadísticas resumidas"""
    
    print("\n" + "="*70)
    print("📊 ESTADÍSTICAS DE EXPERIMENTOS")
    print("="*70)
    
    if len(df_mnist) > 0:
        print("\n🔹 MNIST:")
        print(f"   Mejor accuracy: {df_mnist['final_test_accuracy'].max():.4f}")
        best_mnist = df_mnist.loc[df_mnist['final_test_accuracy'].idxmax()]
        print(f"     - Workers: {int(best_mnist['workers'])}, Épocas: {int(best_mnist['epochs'])}")
        print(f"     - Tiempo: {best_mnist['total_time_minutes']:.1f} min")
        
        df_mnist['acc_per_min'] = df_mnist['final_test_accuracy'] / df_mnist['total_time_minutes']
        best_efficiency = df_mnist.loc[df_mnist['acc_per_min'].idxmax()]
        print(f"\n   Mejor eficiencia (accuracy/min): {best_efficiency['acc_per_min']:.6f}")
        print(f"     - Workers: {int(best_efficiency['workers'])}, Épocas: {int(best_efficiency['epochs'])}")
    
    if len(df_cifar) > 0:
        print("\n🔹 CIFAR-10:")
        print(f"   Mejor accuracy: {df_cifar['final_test_accuracy'].max():.4f}")
        best_cifar = df_cifar.loc[df_cifar['final_test_accuracy'].idxmax()]
        print(f"     - Workers: {int(best_cifar['workers'])}, Épocas: {int(best_cifar['epochs'])}")
        print(f"     - Tiempo: {best_cifar['total_time_minutes']:.1f} min")
        
        df_cifar['acc_per_min'] = df_cifar['final_test_accuracy'] / df_cifar['total_time_minutes']
        best_efficiency = df_cifar.loc[df_cifar['acc_per_min'].idxmax()]
        print(f"\n   Mejor eficiencia (accuracy/min): {best_efficiency['acc_per_min']:.6f}")
        print(f"     - Workers: {int(best_efficiency['workers'])}, Épocas: {int(best_efficiency['epochs'])}")
    
    print("\n" + "="*70)

def main():
    """Función principal"""
    
    print("📈 Generando visualizaciones de experimentos...")
    print("="*70)
    
    # Crear directorio para resultados si no existe
    Path("training_results").mkdir(exist_ok=True)
    
    # Cargar datos
    df_mnist, df_cifar = load_and_clean_data()
    
    if len(df_mnist) == 0 and len(df_cifar) == 0:
        print("❌ No se encontraron datos válidos en el archivo CSV")
        print("   Verifica que el archivo training_results/experiments_summary.csv existe y tiene datos")
        return
    
    # Imprimir estadísticas
    print_statistics(df_mnist, df_cifar)
    
    # Generar gráficos
    print("\n📊 Generando gráficos...")
    plot_accuracy_vs_epochs(df_mnist, df_cifar)
    plot_accuracy_vs_workers(df_mnist, df_cifar)
    plot_time_vs_epochs(df_mnist, df_cifar)
    plot_efficiency_analysis(df_mnist, df_cifar)
    
    print("\n" + "="*70)
    print("✅ ¡Análisis completado!")
    print("📁 Todos los gráficos guardados en: training_results/")
    print("="*70)

if __name__ == "__main__":
    main()