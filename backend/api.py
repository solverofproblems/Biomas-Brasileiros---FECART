"""
API Flask para integração Frontend-Backend
Classificador de Biomas Brasileiros
Desenvolvido por Carlos Eduardo de Lima - FECAP
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
from modelo import ClassificadorBiomas
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Permitir requisições do frontend

# Instância global do classificador
classificador = None

def inicializar_modelo():
    """Inicializa o modelo de classificação"""
    global classificador
    
    try:
        # Verificar se o arquivo do modelo existe
        modelo_path = "melhor_modelo.h5"
        if not os.path.exists(modelo_path):
            return False, f"Modelo não encontrado: {modelo_path}. Execute 'python modelo.py' primeiro."
        
        # Verificar se o dataset existe
        dataset_path = "dataset/validation"
        if not os.path.exists(dataset_path):
            return False, f"Dataset não encontrado: {dataset_path}"
        
        print("🔄 Carregando modelo TensorFlow...")
        classificador = ClassificadorBiomas()
        
        print("🔄 Carregando pesos do modelo...")
        classificador.model = tf.keras.models.load_model(modelo_path)
        classificador.class_names = ['amazonia', 'caatinga', 'cerrado', 'mata atlantica', 'pampa', 'pantanal']
        
        print("✅ Modelo inicializado com sucesso!")
        return True, "Modelo carregado com sucesso!"
        
    except ImportError as e:
        return False, f"Erro de importação: {str(e)}. Verifique se TensorFlow está instalado."
    except FileNotFoundError as e:
        return False, f"Arquivo não encontrado: {str(e)}"
    except Exception as e:
        return False, f"Erro ao carregar modelo: {str(e)}"

@app.route('/')
def index():
    """Servir o frontend"""
    return send_from_directory('../frontend/html', 'index.html')

@app.route('/css/<path:filename>')
def css_files(filename):
    """Servir arquivos CSS"""
    return send_from_directory('../frontend/css', filename)

@app.route('/js/<path:filename>')
def js_files(filename):
    """Servir arquivos JavaScript"""
    return send_from_directory('../frontend/js', filename)

@app.route('/dataset/<path:filename>')
def dataset_images(filename):
    """Servir imagens do dataset"""
    return send_from_directory('dataset', filename)

@app.route('/api/imagens', methods=['GET'])
def listar_imagens():
    """Lista todas as imagens disponíveis na base de dados"""
    try:
        imagens = []
        dataset_path = "dataset/validation"
        
        if not os.path.exists(dataset_path):
            return jsonify({"erro": "Dataset não encontrado"}), 404
        
        # Percorrer todas as pastas de biomas
        for bioma in os.listdir(dataset_path):
            bioma_path = os.path.join(dataset_path, bioma)
            if os.path.isdir(bioma_path):
                # Listar imagens do bioma
                for arquivo in os.listdir(bioma_path):
                    if arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                        imagens.append({
                            "nome": arquivo,
                            "bioma": bioma,
                            "caminho": f"validation/{bioma}/{arquivo}",
                            "bioma_formatado": bioma.replace("_", " ").title()
                        })
        
        return jsonify({
            "sucesso": True,
            "total": len(imagens),
            "imagens": imagens
        })
    
    except Exception as e:
        return jsonify({"erro": f"Erro ao listar imagens: {str(e)}"}), 500

@app.route('/api/classificar', methods=['POST'])
def classificar_imagem():
    """Classifica uma imagem usando o modelo treinado"""
    try:
        data = request.get_json()
        caminho_imagem = data.get('caminho')
        
        if not caminho_imagem:
            return jsonify({"erro": "Caminho da imagem não fornecido"}), 400
        
        # Construir caminho completo para verificação
        caminho_completo = os.path.join("dataset", caminho_imagem)
        if not os.path.exists(caminho_completo):
            return jsonify({"erro": "Imagem não encontrada"}), 404
        
        if classificador is None:
            return jsonify({"erro": "Modelo não inicializado"}), 500
        
        # Classificar imagem
        resultado = classificador.classificar_imagem(caminho_completo)
        
        if resultado is None:
            return jsonify({"erro": "Erro na classificação"}), 500
        
        # Formatar resultado para o frontend
        resposta = {
            "sucesso": True,
            "melhor_bioma": resultado['melhor_bioma'],
            "melhor_confianca": float(resultado['melhor_confianca']),
            "top_k": [],
            "todas_predicoes": {}
        }
        
        # Processar top-K
        for pred in resultado['top_k']:
            resposta["top_k"].append({
                "bioma": pred['bioma'],
                "confianca": float(pred['confianca']),
                "posicao": pred['posicao']
            })
        
        # Processar todas as predições (se disponível)
        if 'todas_predicoes' in resultado:
            for i, classe in enumerate(classificador.class_names):
                resposta["todas_predicoes"][classe] = float(resultado['todas_predicoes'][i])
        
        return jsonify(resposta)
    
    except Exception as e:
        return jsonify({"erro": f"Erro na classificação: {str(e)}"}), 500

@app.route('/api/status', methods=['GET'])
def status_modelo():
    """Verifica o status do modelo"""
    if classificador is None:
        return jsonify({
            "modelo_carregado": False,
            "mensagem": "Modelo não inicializado"
        })
    else:
        return jsonify({
            "modelo_carregado": True,
            "mensagem": "Modelo pronto para classificação",
            "classes": classificador.class_names
        })

if __name__ == '__main__':
    print("🌿 Iniciando API do Classificador de Biomas")
    print("=" * 50)
    
    # Verificar se estamos no diretório correto
    if not os.path.exists("modelo.py"):
        print("❌ Arquivo modelo.py não encontrado!")
        print("💡 Execute este script a partir da pasta backend/")
        exit(1)
    
    # Inicializar modelo
    print("🔄 Inicializando modelo de IA...")
    sucesso, mensagem = inicializar_modelo()
    
    if sucesso:
        print(f"✅ {mensagem}")
        print("🚀 Servidor iniciando em http://localhost:5000")
        print("📱 Frontend disponível em http://localhost:5000")
        print("🔗 API disponível em http://localhost:5000/api/")
        print("\n📋 Endpoints disponíveis:")
        print("   GET  /api/status      - Status do modelo")
        print("   GET  /api/imagens     - Lista de imagens")
        print("   POST /api/classificar - Classificar imagem")
        print("   GET  /dataset/<path>  - Servir imagens")
    else:
        print(f"❌ {mensagem}")
        print("💡 Execute primeiro: python modelo.py")
        print("⚠️  Servidor será iniciado sem o modelo carregado")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Erro ao iniciar servidor: {e}")
        print("💡 Verifique se a porta 5000 está disponível")
