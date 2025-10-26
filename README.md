# 🌿 Classificador de Biomas Brasileiros

Sistema completo de **Inteligência Artificial** para classificação de imagens de biomas brasileiros, desenvolvido com **Deep Learning** e interface web moderna.

## 🎯 Sobre o Projeto

Este projeto utiliza **Transfer Learning** com **MobileNetV2** para criar um classificador inteligente que identifica automaticamente biomas brasileiros a partir de imagens. O sistema combina:

- **Backend**: Modelo de IA treinado com TensorFlow/Keras
- **Frontend**: Interface web moderna e responsiva
- **API**: Comunicação em tempo real entre frontend e backend

## 🚀 Como Executar

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Treinar o Modelo de IA
```bash
cd backend
python modelo.py
```
> ⏱️ **Tempo estimado**: 15-30 minutos (dependendo do hardware)

### 3. Iniciar o Sistema Completo
```bash
cd backend
python api.py
```

### 4. Acessar a Interface Web
Abra seu navegador e acesse:
```
http://localhost:5000
```

## 📁 Estrutura do Projeto

```
Classificador de Biomas (finalizado)/
├── backend/                    # Backend da aplicação
│   ├── modelo.py              # Treinamento do modelo de IA
│   ├── testar.py              # Teste individual de imagens
│   ├── api.py                 # API Flask (servidor web)
│   ├── melhor_modelo.h5       # Modelo treinado
│   └── dataset/               # Base de dados de imagens
│       ├── learn/             # Imagens para treinamento
│       │   ├── amazonia/      # 30+ imagens
│       │   ├── caatinga/      # 30+ imagens
│       │   ├── cerrado/       # 30+ imagens
│       │   ├── mata atlantica/ # 20+ imagens
│       │   ├── pampa/         # 30+ imagens
│       │   └── pantanal/      # 20+ imagens
│       └── validation/        # Imagens para validação
│           ├── amazonia/      # 10 imagens
│           ├── caatinga/      # 10 imagens
│           ├── cerrado/       # 10 imagens
│           ├── mata atlantica/ # 10 imagens
│           ├── pampa/         # 10 imagens
│           └── pantanal/      # 10 imagens
├── frontend/                  # Interface web
│   ├── html/
│   │   └── index.html        # Página principal
│   ├── css/
│   │   └── style.css         # Estilos da interface
│   └── js/
│       └── javascript.js     # Lógica do frontend
├── requirements.txt          # Dependências Python
└── README.md               # Este arquivo
```

## 🎯 Biomas Suportados

| Bioma | Descrição | Características |
|-------|-----------|-----------------|
| **🌳 Amazônia** | Floresta tropical úmida | Vegetação densa, alta biodiversidade |
| **🌵 Caatinga** | Vegetação semiárida | Cactos, vegetação resistente à seca |
| **🌾 Cerrado** | Savana brasileira | Vegetação rasteira, árvores esparsas |
| **🌲 Mata Atlântica** | Floresta costeira | Vegetação densa, alta umidade |
| **🌱 Pampa** | Campos do sul | Vegetação herbácea, campos abertos |
| **🦆 Pantanal** | Planície alagada | Vegetação aquática, áreas alagadas |

## ✨ Funcionalidades

### 🤖 Inteligência Artificial
- ✅ **Transfer Learning** com MobileNetV2
- ✅ **Data Augmentation** para melhor generalização
- ✅ **Early Stopping** para evitar overfitting
- ✅ **Model Checkpointing** para salvar o melhor modelo
- ✅ **Acurácia**: ~85-90% nos dados de validação

### 🌐 Interface Web
- ✅ **Seleção de Imagens**: Escolha entre imagens da base de dados
- ✅ **Filtros por Bioma**: Visualize imagens por categoria
- ✅ **Classificação em Tempo Real**: Resultados instantâneos
- ✅ **Top-3 Predições**: Veja as 3 melhores classificações
- ✅ **Interface Responsiva**: Funciona em desktop e mobile
- ✅ **Feedback Visual**: Animações e indicadores de carregamento

### 🔧 API REST
- ✅ **GET /api/status**: Status do modelo
- ✅ **GET /api/imagens**: Lista de imagens disponíveis
- ✅ **POST /api/classificar**: Classificar imagem
- ✅ **GET /dataset/**: Servir imagens estáticas

## 💻 Tecnologias Utilizadas

### Backend
- **Python 3.8+**
- **TensorFlow/Keras**: Deep Learning
- **Flask**: API web
- **NumPy**: Computação numérica
- **Pillow**: Processamento de imagens
- **Matplotlib**: Visualização

### Frontend
- **HTML5**: Estrutura
- **CSS3**: Estilos e animações
- **JavaScript ES6+**: Lógica da interface
- **Axios**: Comunicação com API

## 🎮 Como Usar a Interface

1. **Acesse** `http://localhost:5000`
2. **Escolha** uma imagem da base de dados
3. **Clique** na imagem para classificá-la
4. **Veja** os resultados da IA em tempo real
5. **Use** os filtros para navegar por biomas

## 📊 Exemplo de Resultado

```
🎯 Resultado da Análise com IA

Bioma Classificado: Amazônia
Confiança: 94.2%

Top 3 Predições:
1. Amazônia - 94.2%
2. Mata Atlântica - 3.1%
3. Cerrado - 1.8%
```

## 🔧 Configurações Avançadas

### Modificar Parâmetros do Modelo
Edite `backend/modelo.py`:
```python
# Ajustar tamanho da imagem
IMG_SIZE = (224, 224)

# Modificar batch size
BATCH_SIZE = 32

# Alterar épocas de treinamento
EPOCHS = 50
```

### Adicionar Novos Biomas
1. Crie pastas em `dataset/learn/` e `dataset/validation/`
2. Adicione imagens nas pastas
3. Atualize `class_names` em `modelo.py` e `api.py`

## 🐛 Solução de Problemas

### ❌ "Modelo não encontrado"
```bash
cd backend
python modelo.py  # Treinar primeiro
```

### ❌ "Backend não conecta"
- Verifique se está rodando na porta 5000
- Confirme se `python api.py` foi executado
- Verifique firewall/antivírus

### ❌ "Imagens não aparecem"
- Verifique se o dataset existe
- Confirme se as imagens estão nas pastas corretas
- Verifique console do navegador (F12)

### ❌ "Erro de memória"
- Reduza `BATCH_SIZE` em `modelo.py`
- Feche outros programas
- Use GPU se disponível

## 📈 Performance

- **Treinamento**: 15-30 minutos
- **Classificação**: < 1 segundo
- **Acurácia**: 85-90%
- **Modelo**: ~15MB (MobileNetV2)

## 🤝 Contribuição

Para contribuir com o projeto:
1. Fork o repositório
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto foi desenvolvido por **Carlos Eduardo de Lima** para a **FECAP**.

## 🎓 Agradecimentos

- **FECAP** - Faculdade de Economia, Administração e Contabilidade
- **TensorFlow Team** - Framework de Deep Learning
- **Comunidade Python** - Bibliotecas e recursos

---

**🌿 Sistema completo de classificação de biomas brasileiros com IA! 🇧🇷**