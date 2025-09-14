import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import sparse_top_k_categorical_accuracy
# from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import os

class ClassificadorBiomas:
    def __init__(self):
        self.img_size = (224, 224)
        self.batch_size = 16  # Reduzido para melhor generalização
        self.model = None
        self.class_names = None
        self.history = None
        
        # Data Augmentation intensivo
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1,
            fill_mode='nearest'
        )
        
        self.val_datagen = ImageDataGenerator(rescale=1./255)
        
    def treinar(self):
        """Treina o modelo com as imagens do dataset"""
        print("🌿 TREINANDO CLASSIFICADOR DE BIOMAS AVANÇADO")
        print("="*60)
        
        # Carregar datasets com data augmentation
        print("📁 Carregando imagens com data augmentation...")
        
        train_generator = self.train_datagen.flow_from_directory(
            "dataset/learn",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='sparse',
            shuffle=True
        )
        
        val_generator = self.val_datagen.flow_from_directory(
            "dataset/validation",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='sparse',
            shuffle=False
        )
        
        self.class_names = list(train_generator.class_indices.keys())
        print(f"✅ Classes encontradas: {self.class_names}")
        print(f"📊 Imagens de treino: {train_generator.samples}")
        print(f"📊 Imagens de validação: {val_generator.samples}")
        
        # Criar modelo mais sofisticado
        print("🏗️ Criando modelo avançado...")
        base_model = MobileNetV2(
            input_shape=self.img_size + (3,),
            include_top=False,
            weights="imagenet"
        )
        
        # Congelar as primeiras camadas, descongelar as últimas
        base_model.trainable = True
        fine_tune_at = len(base_model.layers) - 30  # Descongelar últimas 30 camadas
        
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Arquitetura mais robusta
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(len(self.class_names), activation="softmax")
        ])
        
        # Compilar com otimizador mais sofisticado
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', lambda y_true, y_pred: sparse_top_k_categorical_accuracy(y_true, y_pred, k=3)]
        )
        # Callbacks avançados
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'melhor_modelo.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Treinar
        print("🚀 Iniciando treinamento avançado...")
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=50,
            callbacks=callbacks,
            verbose=1
        )
        
        # Avaliar
        print("\n📊 AVALIAÇÃO FINAL:")
        print("="*30)
        val_loss, val_acc, val_top3_acc = self.model.evaluate(val_generator, verbose=0)
        print(f"🎯 Acurácia: {val_acc:.2%}")
        print(f"🎯 Top-3 Acurácia: {val_top3_acc:.2%}")
        
        # Salvar modelo
        self.model.save("modelo_biomas_avancado.h5")
        print("💾 Modelo salvo como 'modelo_biomas_avancado.h5'")
        
        # Análise detalhada
        self._analisar_resultados(val_generator)
        
        print("🎉 Treinamento concluído!")
        return True
    
    def _analisar_resultados(self, val_generator):
        """Análise detalhada dos resultados"""
        print("\n🔍 ANÁLISE DETALHADA:")
        print("="*40)
        
        # Gerar predições
        val_generator.reset()
        predictions = self.model.predict(val_generator, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Obter labels verdadeiros
        true_classes = val_generator.classes
        
        # Matriz de confusão (versão simplificada)
        num_classes = len(self.class_names)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for i in range(len(true_classes)):
            cm[true_classes[i], predicted_classes[i]] += 1
        
        # Plotar matriz de confusão
        plt.figure(figsize=(15, 12))
        
        # Subplot 1: Matriz de confusão
        plt.subplot(2, 3, 1)
        im = plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar(im)
        plt.title('Matriz de Confusão')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=45)
        plt.yticks(range(len(self.class_names)), self.class_names)
        
        # Adicionar valores nas células
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')
        
        # Subplot 2: Histórico de treinamento - Acurácia
        plt.subplot(2, 3, 2)
        plt.plot(self.history.history['accuracy'], label='Treino', linewidth=2)
        plt.plot(self.history.history['val_accuracy'], label='Validação', linewidth=2)
        plt.title('Acurácia por Época')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Histórico de treinamento - Loss
        plt.subplot(2, 3, 3)
        plt.plot(self.history.history['loss'], label='Treino', linewidth=2)
        plt.plot(self.history.history['val_loss'], label='Validação', linewidth=2)
        plt.title('Loss por Época')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Acurácia por classe
        plt.subplot(2, 3, 4)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        bars = plt.bar(range(len(self.class_names)), class_accuracy, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
        plt.title('Acurácia por Classe')
        plt.xlabel('Bioma')
        plt.ylabel('Acurácia')
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=45)
        plt.ylim(0, 1)
        
        # Adicionar valores nas barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Subplot 5: Distribuição de confiança
        plt.subplot(2, 3, 5)
        max_probs = np.max(predictions, axis=1)
        plt.hist(max_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribuição de Confiança')
        plt.xlabel('Confiança Máxima')
        plt.ylabel('Frequência')
        plt.grid(True, alpha=0.3)
        
        # Subplot 6: Top-3 Accuracy por época (se disponível)
        plt.subplot(2, 3, 6)
        if 'val_top_3_accuracy' in self.history.history:
            plt.plot(self.history.history['top_3_accuracy'], label='Treino Top-3', linewidth=2)
            plt.plot(self.history.history['val_top_3_accuracy'], label='Validação Top-3', linewidth=2)
            plt.title('Top-3 Acurácia')
        else:
            # Calcular top-3 accuracy manualmente
            top3_correct = 0
            for i, true_class in enumerate(true_classes):
                top3_preds = np.argsort(predictions[i])[-3:]
                if true_class in top3_preds:
                    top3_correct += 1
            top3_acc = top3_correct / len(true_classes)
            plt.text(0.5, 0.5, f'Top-3 Acurácia: {top3_acc:.2%}', 
                    ha='center', va='center', fontsize=14, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            plt.title('Top-3 Acurácia Final')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analise_detalhada.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Relatório de classificação (versão simplificada)
        print("\n📋 RELATÓRIO DE CLASSIFICAÇÃO:")
        print("-" * 50)
        for i, class_name in enumerate(self.class_names):
            true_positives = cm[i, i]
            false_positives = cm[:, i].sum() - true_positives
            false_negatives = cm[i, :].sum() - true_positives
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print(f"{class_name:15} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # Análise de erros mais comuns
        print("\n❌ ANÁLISE DE ERROS:")
        print("-" * 30)
        errors = predicted_classes != true_classes
        if np.any(errors):
            error_indices = np.where(errors)[0]
            print(f"Total de erros: {len(error_indices)}")
            
            # Mostrar alguns exemplos de erros
            print("\nExemplos de erros mais comuns:")
            error_pairs = []
            for idx in error_indices[:10]:  # Primeiros 10 erros
                true_class = self.class_names[true_classes[idx]]
                pred_class = self.class_names[predicted_classes[idx]]
                confidence = predictions[idx][predicted_classes[idx]]
                error_pairs.append((true_class, pred_class, confidence))
                print(f"  Real: {true_class} → Predito: {pred_class} (conf: {confidence:.2f})")
    
    def classificar_imagem(self, caminho_imagem, top_k=3):
        """Classifica uma imagem e retorna o bioma predito com análise detalhada"""
        if self.model is None:
            print("❌ Modelo não treinado! Execute o treinamento primeiro.")
            return None
        
        try:
            # Carregar e processar imagem
            img = load_img(caminho_imagem, target_size=self.img_size)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predição
            pred = self.model.predict(img_array, verbose=0)
            
            # Top-K predições
            top_k_indices = np.argsort(pred[0])[-top_k:][::-1]
            top_k_results = []
            
            for i, idx in enumerate(top_k_indices):
                top_k_results.append({
                    'bioma': self.class_names[idx],
                    'confianca': pred[0][idx],
                    'posicao': i + 1
                })
            
            return {
                'melhor_bioma': top_k_results[0]['bioma'],
                'melhor_confianca': top_k_results[0]['confianca'],
                'top_k': top_k_results,
                'todas_predicoes': pred[0]
            }
            
        except Exception as e:
            print(f"❌ Erro ao classificar imagem: {e}")
            return None
    
    def visualizar_ativacoes(self, caminho_imagem, layer_name=None):
        """Visualiza as ativações de uma camada específica"""
        if self.model is None:
            print("❌ Modelo não treinado!")
            return None
            
        try:
            # Carregar imagem
            img = load_img(caminho_imagem, target_size=self.img_size)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Criar modelo para extrair ativações
            if layer_name is None:
                # Usar a última camada convolucional do MobileNetV2
                layer_name = 'mobilenetv2_1.00_224'
            
            activation_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer(layer_name).output
            )
            
            # Obter ativações
            activations = activation_model.predict(img_array, verbose=0)
            
            # Visualizar algumas ativações
            plt.figure(figsize=(15, 10))
            
            # Imagem original
            plt.subplot(2, 3, 1)
            plt.imshow(img)
            plt.title('Imagem Original')
            plt.axis('off')
            
            # Primeiras 5 ativações
            for i in range(min(5, activations.shape[-1])):
                plt.subplot(2, 3, i + 2)
                plt.imshow(activations[0, :, :, i], cmap='viridis')
                plt.title(f'Ativação {i+1}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig('ativacoes_visualizacao.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            return activations
            
        except Exception as e:
            print(f"❌ Erro ao visualizar ativações: {e}")
            return None

def main():
    """Função principal"""
    classificador = ClassificadorBiomas()
    
    # Treinar modelo
    if classificador.treinar():
        print("\n🧪 TESTE AVANÇADO:")
        print("="*40)
        
        # Testar com uma imagem de exemplo
        imagem_teste = "dataset/validation/cerrado/cerrado21.jpg"
        if os.path.exists(imagem_teste):
            print(f"🔍 Testando: {os.path.basename(imagem_teste)}")
            resultado = classificador.classificar_imagem(imagem_teste, top_k=3)
            
            if resultado:
                print(f"\n🌿 MELHOR PREDIÇÃO:")
                print(f"   Bioma: {resultado['melhor_bioma'].upper()}")
                print(f"   Confiança: {resultado['melhor_confianca']:.1%}")
                
                print(f"\n📊 TOP-3 PREDIÇÕES:")
                for pred in resultado['top_k']:
                    print(f"   {pred['posicao']}. {pred['bioma']}: {pred['confianca']:.1%}")
                
                # Visualizar ativações
                print(f"\n🔬 Visualizando ativações...")
                classificador.visualizar_ativacoes(imagem_teste)
        
        print("\n💡 Para testar outras imagens:")
        print("   resultado = classificador.classificar_imagem('sua_imagem.jpg', top_k=3)")
        print("   classificador.visualizar_ativacoes('sua_imagem.jpg')")

if __name__ == "__main__":
    main()
