"""
Script simples para testar imagens com o modelo treinado
"""

import os
from modelo import ClassificadorBiomas

def testar_imagem(caminho_imagem):
    """Testa uma imagem específica"""
    if not os.path.exists(caminho_imagem):
        print(f"❌ Imagem não encontrada: {caminho_imagem}")
        return
    
    # Carregar modelo treinado
    modelo_path = "melhor_modelo.h5"
    if not os.path.exists(modelo_path):
        print("❌ Modelo não encontrado! Execute 'python modelo.py' primeiro.")
        return
    
    # Criar classificador e carregar modelo
    classificador = ClassificadorBiomas()
    classificador.model = tf.keras.models.load_model(modelo_path)
    classificador.class_names = ['amazonia', 'caatinga', 'cerrado', 'mata atlantica', 'pampa', 'pantanal']
        
        # Classificar imagem
    print(f"🔍 Classificando: {os.path.basename(caminho_imagem)}")
    resultado = classificador.classificar_imagem(caminho_imagem)
        
    if resultado:
        print(f"\n🎯 RESULTADO:")
        print(f"🌿 Bioma: {resultado['melhor_bioma'].upper()}")
        print(f"📊 Confiança: {resultado['melhor_confianca']:.1%}")
            
            # Mostrar todas as predições
        print(f"\n📈 TODAS AS PREDIÇÕES:")
        for i, (classe, conf) in enumerate(zip(classificador.class_names, resultado['todas_predicoes'])):
            print(f"   {classe.upper()}: {conf:.1%}")

def main():
    """Função principal"""
    print("🧪 TESTADOR DE IMAGENS - CLASSIFICADOR DE BIOMAS")
    print("="*50)
    
    # Verificar se o modelo existe
    if not os.path.exists("melhor_modelo.h5"):
        print("❌ Modelo não encontrado!")
        print("💡 Execute primeiro: python modelo.py")
        return
    
if __name__ == "__main__":
    try:
        import tensorflow as tf
        main()
    except ImportError:
        print("❌ TensorFlow não está instalado!")
        print("💡 Instale com: pip install tensorflow")

while True:

    caminho = input("\n📁 Digite o caminho da imagem para testar: ").strip()
    
    if caminho:
        testar_imagem(caminho)
    else:
        print("❌ Nenhum caminho fornecido!")