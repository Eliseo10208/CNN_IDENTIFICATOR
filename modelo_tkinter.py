import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

def load_images_by_folders(dataset_folder, img_size=(100, 100)):
    """
    Carga imágenes organizadas en carpetas donde cada carpeta representa una clase.
    
    Parámetros:
    - dataset_folder: Carpeta principal que contiene subcarpetas con imágenes
    - img_size: Tamaño al que se redimensionarán las imágenes (alto, ancho)
    
    Retorna:
    - X: Matrices de imágenes (muestras, alto, ancho, canales)
    - y: Etiquetas en formato one-hot
    - classes: Lista de nombres de clases
    """
    # Obtener las subcarpetas (cada una representa una clase)
    class_folders = [f for f in os.listdir(dataset_folder) 
                    if os.path.isdir(os.path.join(dataset_folder, f))]
    
    if not class_folders:
        raise ValueError("No se encontraron subcarpetas en el directorio del dataset.")
    
    # Ordenar las clases alfabéticamente para consistencia
    classes = sorted(class_folders)
    
    # Mapear cada clase a un índice
    class_to_index = {clase: i for i, clase in enumerate(classes)}
    
    X = []
    y_indices = []
    image_paths = []  # Para debugging/referencia
    
    # Recorrer cada carpeta (clase)
    for class_folder in classes:
        class_path = os.path.join(dataset_folder, class_folder)
        class_idx = class_to_index[class_folder]
        
        # Obtener todas las imágenes de esta clase
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"Advertencia: No se encontraron imágenes en la carpeta {class_folder}")
            continue
        
        # Procesar cada imagen
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            image_paths.append(img_path)
            
            # Cargar y preprocesar la imagen
            try:
                img = Image.open(img_path).convert("RGB")  # Convertir a RGB para CNN
                img = img.resize(img_size)
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalizar a [0,1]
                X.append(img_array)
                y_indices.append(class_idx)
            except Exception as e:
                print(f"Error al procesar imagen {img_path}: {e}")
    
    if not X:
        raise ValueError("No se pudieron cargar imágenes. Verifique el formato y las extensiones.")
    
    # Convertir a arrays numpy
    X = np.array(X)
    y_indices = np.array(y_indices)
    
    # Convertir índices a one-hot
    n_clases = len(classes)
    y = tf.keras.utils.to_categorical(y_indices, num_classes=n_clases)
    
    print(f"Cargadas {len(X)} imágenes de {len(classes)} clases.")
    for i, clase in enumerate(classes):
        count = np.sum(y_indices == i)
        print(f"  - Clase '{clase}': {count} imágenes")
    
    return X, y, classes

def create_cnn_model(input_shape, num_classes, dropout_rate=0.5):
    """
    Crea una CNN para clasificación de imágenes.
    
    Parámetros:
    - input_shape: Forma de la entrada (altura, anchura, canales)
    - num_classes: Número de clases para la clasificación
    - dropout_rate: Tasa de dropout para regularización
    
    Retorna:
    - Modelo secuencial de Keras
    """
    model = Sequential([
        # Primera capa convolucional
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Segunda capa convolucional
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Tercera capa convolucional
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Aplanar y capas densas
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compilar modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate_model(X, y, classes, batch_size=32, epochs=50, validation_split=0.2, log_callback=None):
    """
    Entrena y evalúa el modelo CNN.
    
    Parámetros:
    - X: Datos de imágenes (muestras, alto, ancho, canales)
    - y: Etiquetas en formato one-hot
    - classes: Lista de nombres de clases
    - batch_size: Tamaño del batch para entrenamiento
    - epochs: Número máximo de épocas
    - validation_split: Fracción de datos para validación
    - log_callback: Función para mostrar mensajes en la UI
    
    Retorna:
    - model: Modelo entrenado
    - history: Historial de entrenamiento
    - evaluation: Resultados de evaluación
    """
    # Dividir en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42, stratify=y)
    
    message = f"Datos divididos en:\n - Entrenamiento: {X_train.shape[0]} muestras\n - Validación: {X_val.shape[0]} muestras"
    print(message)
    if log_callback:
        log_callback(message)
    
    # Forma de entrada para el modelo
    input_shape = X_train[0].shape
    num_classes = len(classes)
    
    # Crear modelo
    model = create_cnn_model(input_shape, num_classes)
    
    # Resumen del modelo
    model_summary_lines = []
    model.summary(print_fn=lambda x: model_summary_lines.append(x))
    model_summary = "\n".join(model_summary_lines)
    
    print(model_summary)
    if log_callback:
        log_callback("Arquitectura del modelo CNN:")
        log_callback(model_summary)
    
    # Callbacks para entrenamiento
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001),
        ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
    ]
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Ajustar generador a los datos
    datagen.fit(X_train)
    
    # Entrenar modelo
    message = f"Iniciando entrenamiento con {epochs} épocas y batch size de {batch_size}..."
    print(message)
    if log_callback:
        log_callback(message)
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    # Evaluar modelo
    evaluation = model.evaluate(X_val, y_val)
    
    message = f"Evaluación del modelo:\n - Loss: {evaluation[0]:.4f}\n - Accuracy: {evaluation[1]:.4f}"
    print(message)
    if log_callback:
        log_callback(message)
    
    # Calcular matriz de confusión y reporte de clasificación
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    report = classification_report(y_true_classes, y_pred_classes, target_names=classes)
    
    print("Matriz de Confusión:")
    print(cm)
    print("\nReporte de Clasificación:")
    print(report)
    
    if log_callback:
        log_callback("Matriz de Confusión:")
        log_callback(str(cm))
        log_callback("\nReporte de Clasificación:")
        log_callback(report)
    
    return model, history, (cm, report)

def guardar_modelo_keras(model, classes, log_callback=None):
    """
    Guarda el modelo Keras y la información de clases.
    
    Parámetros:
    - model: Modelo keras entrenado
    - classes: Lista de nombres de clases
    - log_callback: Función para mostrar mensajes en la UI
    """
    # Crear directorio si no existe
    if not os.path.exists('modelo_cnn'):
        os.makedirs('modelo_cnn')
    
    # Guardar modelo
    model.save('modelo_cnn/modelo.h5')
    
    # Guardar clases y metadatos
    metadatos = {
        'clases': classes,
        'input_shape': model.input_shape[1:],
        'n_clases': len(classes)
    }
    
    with open('modelo_cnn/metadatos.json', 'w') as f:
        json.dump(metadatos, f)
    
    message = f"Modelo CNN guardado en la carpeta 'modelo_cnn'\nClases guardadas: {classes}"
    print(message)
    if log_callback:
        log_callback(message)

def cargar_modelo_keras():
    """
    Carga el modelo Keras y la información de clases.
    
    Retorna:
    - model: Modelo keras cargado
    - classes: Lista de nombres de clases
    - input_shape: Forma de entrada esperada
    """
    try:
        # Cargar modelo
        model = load_model('modelo_cnn/modelo.h5')
        
        # Cargar metadatos
        with open('modelo_cnn/metadatos.json', 'r') as f:
            metadatos = json.load(f)
        
        classes = metadatos['clases']
        input_shape = tuple(metadatos['input_shape'])
        
        return model, classes, input_shape
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None, None, None

def predecir_imagen_cnn(ruta_imagen, model, classes, input_shape):
    """
    Predice la clase de una nueva imagen usando el modelo CNN.
    
    Parámetros:
    - ruta_imagen: Ruta al archivo de imagen
    - model: Modelo CNN cargado
    - classes: Lista de clases
    - input_shape: Forma de entrada esperada (alto, ancho, canales)
    
    Retorna:
    - Diccionario con resultados de predicción
    """
    try:
        # Cargar y preprocesar la imagen
        img = Image.open(ruta_imagen).convert("RGB")
        img = img.resize((input_shape[0], input_shape[1]))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de lote
        
        # Predecir
        predicciones = model.predict(img_array)[0]
        
        # Obtener clase predicha
        clase_idx = np.argmax(predicciones)
        clase_predicha = classes[clase_idx]
        
        # Preparar resultados
        resultados = {
            'clase_predicha': clase_predicha,
            'confianza': float(predicciones[clase_idx]),
            'probabilidades': {clase: float(prob) for clase, prob in zip(classes, predicciones)}
        }
        
        return resultados
    except Exception as e:
        print(f"Error al predecir: {e}")
        return {'error': str(e)}

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clasificador CNN por Carpetas")
        self.geometry("1000x800")
        self.create_widgets()
        
    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame para controles
        control_frame = ttk.LabelFrame(main_frame, text="Configuración")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Botón para seleccionar carpeta
        ttk.Button(control_frame, text="Seleccionar carpeta de dataset", 
                   command=self.seleccionar_carpeta).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # Configuración de parámetros
        ttk.Label(control_frame, text="Tamaño del batch:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.input_batch = ttk.Entry(control_frame, width=10)
        self.input_batch.insert(0, "32")
        self.input_batch.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(control_frame, text="Épocas:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.input_epochs = ttk.Entry(control_frame, width=10)
        self.input_epochs.insert(0, "50")
        self.input_epochs.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(control_frame, text="División de validación (%):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.input_val_split = ttk.Entry(control_frame, width=10)
        self.input_val_split.insert(0, "20")
        self.input_val_split.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        # Botones de acción
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        self.btn_entrenar = ttk.Button(button_frame, text="Entrenar Modelo CNN", 
                                      command=self.entrenar_modelo)
        self.btn_entrenar.pack(side=tk.LEFT, padx=5)
        
        self.btn_guardar = ttk.Button(button_frame, text="Guardar Modelo", 
                                     command=self.guardar_modelo_actual, state=tk.DISABLED)
        self.btn_guardar.pack(side=tk.LEFT, padx=5)
        
        # Log de texto
        log_frame = ttk.LabelFrame(main_frame, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=80, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame para gráficas
        self.graph_frame = ttk.LabelFrame(main_frame, text="Gráficas")
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Crear figura para gráficas
        self.fig = plt.Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def log(self, message):
        """Añade un mensaje al log y hace scroll automático"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.update_idletasks()
        
    def seleccionar_carpeta(self):
        """Abre diálogo para seleccionar carpeta principal del dataset"""
        folder_path = filedialog.askdirectory(title="Seleccionar carpeta del dataset")
        if folder_path:
            self.folder_path = folder_path
            self.log(f"Carpeta seleccionada: {folder_path}")
            
            # Verificar si tiene subdirectorios (clases)
            subdirs = [d for d in os.listdir(folder_path) 
                      if os.path.isdir(os.path.join(folder_path, d))]
            
            if subdirs:
                self.log(f"Clases detectadas: {len(subdirs)}")
                self.log(f"Nombres de clases: {', '.join(subdirs)}")
            else:
                self.log("ADVERTENCIA: No se encontraron subcarpetas. " 
                        "Este modelo espera que cada clase esté en su propia subcarpeta.")
            
    def entrenar_modelo(self):
        """Entrena el modelo CNN con los parámetros configurados"""
        if not hasattr(self, 'folder_path'):
            self.log("Error: No se ha seleccionado ninguna carpeta.")
            return
            
        # Validar parámetros
        try:
            batch_size = int(self.input_batch.get())
            if batch_size < 1:
                self.log("Error: batch_size debe ser >= 1.")
                return
        except ValueError:
            self.log("Error: batch_size debe ser un entero válido.")
            return
            
        try:
            epochs = int(self.input_epochs.get())
            if epochs < 1:
                self.log("Error: epochs debe ser >= 1.")
                return
        except ValueError:
            self.log("Error: epochs debe ser un entero válido.")
            return
            
        try:
            val_split = float(self.input_val_split.get()) / 100.0  # Convertir de porcentaje a fracción
            if val_split <= 0 or val_split >= 1:
                self.log("Error: división de validación debe estar entre 1 y 99.")
                return
        except ValueError:
            self.log("Error: división de validación debe ser un número válido.")
            return
            
        # Cargar imágenes
        self.log("Cargando imágenes desde carpetas...")
        try:
            X, y, classes = load_images_by_folders(self.folder_path, img_size=(64, 64))
            self.log(f"Total de imágenes: {len(X)}")
            self.log(f"Forma de las imágenes: {X.shape}")
            self.log(f"Clases detectadas: {classes}")
        except Exception as e:
            self.log(f"Error al cargar imágenes: {e}")
            return
        
        if len(X) == 0:
            self.log("Error: No se encontraron imágenes en las subcarpetas.")
            return
            
        if len(classes) < 2:
            self.log("Error: Se necesitan al menos 2 clases para clasificación.")
            return
            
        self.log("Entrenando modelo CNN...")
        
        # Deshabilitar botones durante el entrenamiento
        self.btn_entrenar.config(state=tk.DISABLED)
        
        try:
            # Entrenar el modelo
            model, history, evaluation = train_and_evaluate_model(
                X, y, classes, 
                batch_size=batch_size, 
                epochs=epochs, 
                validation_split=val_split,
                log_callback=self.log
            )
            
            # Guardar resultados como atributos
            self.model = model
            self.classes = classes
            self.history = history
            self.cm = evaluation[0]
            
            # Graficar resultados
            self.fig.clear()
            
            # Gráfico de precisión
            ax1 = self.fig.add_subplot(221)
            ax1.plot(history.history['accuracy'], label='Entrenamiento', color='blue')
            ax1.plot(history.history['val_accuracy'], label='Validación', color='red')
            ax1.set_title("Evolución de la Precisión")
            ax1.set_xlabel("Épocas")
            ax1.set_ylabel("Precisión")
            ax1.legend()
            
            # Gráfico de pérdida
            ax2 = self.fig.add_subplot(222)
            ax2.plot(history.history['loss'], label='Entrenamiento', color='blue')
            ax2.plot(history.history['val_loss'], label='Validación', color='red')
            ax2.set_title("Evolución de la Pérdida")
            ax2.set_xlabel("Épocas")
            ax2.set_ylabel("Pérdida")
            ax2.legend()
            
            # Matriz de confusión
            ax3 = self.fig.add_subplot(212)
            sns.heatmap(self.cm, annot=True, fmt="d", cmap="Blues", ax=ax3, 
                        xticklabels=classes, yticklabels=classes)
            ax3.set_xlabel("Predicción")
            ax3.set_ylabel("Real")
            ax3.set_title("Matriz de Confusión")
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Habilitar botones
            self.btn_guardar.config(state=tk.NORMAL)
        except Exception as e:
            self.log(f"Error durante el entrenamiento: {e}")
        finally:
            self.btn_entrenar.config(state=tk.NORMAL)
        
    def guardar_modelo_actual(self):
        """Guarda el modelo CNN actual"""
        if hasattr(self, 'model') and hasattr(self, 'classes'):
            guardar_modelo_keras(self.model, self.classes, self.log)
        else:
            self.log("Error: No hay modelo para guardar.")

# ---- CÓDIGO PARA LA API ----

def crear_app_flask():
    """Crea y configura la aplicación Flask para la API"""
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)  # Permitir solicitudes cross-origin
    
    # Cargar el modelo al iniciar
    model, classes, input_shape = cargar_modelo_keras()
    
    if model is None:
        print("Error: No se pudo cargar el modelo CNN. Asegúrate de entrenar y guardar el modelo primero.")
        return None
        
    print(f"Modelo CNN cargado correctamente. Clases: {classes}")
    
    @app.route('/predict', methods=['POST'])
    def predict():
        if 'imagen' not in request.files:
            return jsonify({'error': 'No se envió ninguna imagen'}), 400
            
        archivo = request.files['imagen']
        
        if archivo.filename == '':
            return jsonify({'error': 'Nombre de archivo inválido'}), 400
            
        try:
            # Guardar temporalmente
            temp_path = 'temp_image.jpg'
            archivo.save(temp_path)
            
            # Predecir
            resultados = predecir_imagen_cnn(temp_path, model, classes, input_shape)
            
            # Limpiar
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return jsonify(resultados)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

# ---- PUNTO DE ENTRADA ----

if __name__ == "__main__":
    # Si se pasa --api como argumento, ejecutar la API
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        app = crear_app_flask()
        
        if app is not None:
            app.run(debug=True, host='0.0.0.0', port=5000)
        else:
            sys.exit(1)
    else:
        # Ejecutar la interfaz gráfica
        app = App()
        app.mainloop()