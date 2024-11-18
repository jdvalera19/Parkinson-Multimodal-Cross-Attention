import matplotlib.pyplot as plt
import os
from joblib import load

# Cargar los espectrogramas desde el archivo `joblib`
train_data_X = load('spects/train_data_words_X.joblib')

# Crear un directorio para guardar las visualizaciones
output_dir = 'espectrogramas_words_DA'
os.makedirs(output_dir, exist_ok=True)

# Visualizar y guardar cada espectrograma en `train_data_C`
for index, spectrogram in enumerate(train_data_X):
    # Verifica si el espectrograma es un array de NumPy y tiene la forma esperada
    if spectrogram is not None and hasattr(spectrogram, 'shape'):
        
        # Seleccionar solo un canal (por ejemplo, el primer canal)
        if spectrogram.shape[0] == 2:
            spectrogram = spectrogram[0]  # Selecciona el primer canal

        # Configurar la visualización y guardar la imagen
        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='dB')  # Escala de colores en decibeles
        plt.xlabel('Tiempo')
        plt.ylabel('Frecuencia')
        #plt.title(f"Espectrograma de la muestra {index}")

        # Guardar la imagen
        plt.savefig(os.path.join(output_dir, f"espectrograma_{index}.png"))
        plt.close()  # Cerrar la figura para liberar memoria
    else:
        print(f"El espectrograma en el índice {index} está vacío o no es un array.")
