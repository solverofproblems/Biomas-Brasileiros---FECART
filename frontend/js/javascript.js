// Classificador de Biomas Brasileiros - JavaScript
// Desenvolvido por Carlos Eduardo de Lima

// Configuração da API - Backend Independente
const API_BASE_URL = 'http://localhost:5000/api';

// Configuração do Axios
axios.defaults.timeout = 30000; // 30 segundos de timeout
axios.defaults.headers.common['Content-Type'] = 'application/json';

// Variáveis globais
let imagensDisponiveis = [];
let filtroAtual = 'todos';

document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling para links de navegação
    const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                const headerHeight = document.querySelector('.header').offsetHeight;
                const targetPosition = targetSection.offsetTop - headerHeight;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Variáveis globais para resultados
    const resultsArea = document.querySelector('.results-placeholder');
    
    // Inicializar funcionalidades
    inicializarFiltros();
    carregarImagens();
    verificarStatusAPI();
    
    console.log('🌱 Classificador de Biomas Brasileiros carregado com sucesso!');
    console.log('Desenvolvido por Carlos Eduardo de Lima - FECAP');
});

// Função para mostrar mensagens
    function showMessage(message, type = 'info') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${type}`;
        messageDiv.textContent = message;
        messageDiv.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            background: ${type === 'error' ? '#ff4444' : '#48b340'};
            color: white;
            padding: 1rem 2rem;
            border-radius: 10px;
            z-index: 10000;
            animation: slideInRight 0.3s ease;
        `;
        
        document.body.appendChild(messageDiv);
        
        setTimeout(() => {
            messageDiv.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => {
                document.body.removeChild(messageDiv);
            }, 300);
        }, 3000);
    }
    
// Funções da API usando Axios
async function verificarStatusAPI() {
    try {
        const response = await axios.get(`${API_BASE_URL}/status`);
        const data = response.data;
        
        if (data.modelo_carregado) {
            console.log('✅ Modelo carregado e pronto para classificação');
            console.log(`🖥️ Servidor: ${data.servidor}`);
        } else {
            console.warn('⚠️ Modelo não carregado:', data.mensagem);
            showMessage('Modelo não está carregado. Execute o backend primeiro.', 'error');
        }
    } catch (error) {
        console.error('❌ Erro ao verificar status da API:', error);
        if (error.code === 'ECONNREFUSED') {
            showMessage('Não foi possível conectar com o backend. Verifique se está rodando na porta 5000.', 'error');
        } else {
            showMessage('Erro de conexão com o servidor backend.', 'error');
        }
    }
}

async function carregarImagens() {
    try {
        const response = await axios.get(`${API_BASE_URL}/imagens`);
        const data = response.data;
        
        if (data.sucesso) {
            imagensDisponiveis = data.imagens;
            console.log(`📁 ${data.total} imagens carregadas da base de dados`);
            console.log(`🖥️ Servidor: ${data.servidor}`);
            renderizarImagens();
        } else {
            console.error('❌ Erro ao carregar imagens:', data.erro);
            showMessage('Erro ao carregar imagens da base de dados', 'error');
        }
    } catch (error) {
        console.error('❌ Erro na requisição:', error);
        if (error.code === 'ECONNREFUSED') {
            showMessage('Backend não está rodando. Execute: python backend/servidor_backend.py', 'error');
        } else {
            showMessage('Erro de conexão com o servidor', 'error');
        }
    }
}

function renderizarImagens() {
    const imageGrid = document.getElementById('imageGrid');
    
    if (!imageGrid) return;
    
    // Filtrar imagens baseado no filtro atual
    let imagensFiltradas = imagensDisponiveis;
    if (filtroAtual !== 'todos') {
        imagensFiltradas = imagensDisponiveis.filter(img => 
            img.bioma.toLowerCase() === filtroAtual.toLowerCase()
        );
    }
    
    if (imagensFiltradas.length === 0) {
        imageGrid.innerHTML = `
            <div class="loading-images">
                <p>Nenhuma imagem encontrada para este filtro.</p>
            </div>
        `;
        return;
    }
    
    // Renderizar imagens
    imageGrid.innerHTML = imagensFiltradas.map(imagem => `
        <div class="image-item" onclick="classificarImagem('${imagem.caminho}', '${imagem.nome}')">
            <img src="/dataset/${imagem.caminho}" alt="${imagem.nome}" loading="lazy" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIwIiBoZWlnaHQ9IjEyMCIgdmlld0JveD0iMCAwIDEyMCAxMjAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIxMjAiIGhlaWdodD0iMTIwIiBmaWxsPSIjMzMzIi8+CjxwYXRoIGQ9Ik00MCA0MEg4MFY4MEg0MFY0MFoiIGZpbGw9IiM2NjYiLz4KPHN2Zz4K'">
            <div class="bioma-label">${imagem.bioma_formatado}</div>
        </div>
    `).join('');
}

function inicializarFiltros() {
    const filterBtns = document.querySelectorAll('.filter-btn');
    
    filterBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            // Remover classe active de todos os botões
            filterBtns.forEach(b => b.classList.remove('active'));
            
            // Adicionar classe active ao botão clicado
            this.classList.add('active');
            
            // Atualizar filtro atual
            filtroAtual = this.getAttribute('data-bioma');
            
            // Re-renderizar imagens
            renderizarImagens();
        });
    });
}

async function classificarImagem(caminhoImagem, nomeImagem) {
    const resultsArea = document.querySelector('.results-placeholder');
    
    if (!resultsArea) return;
    
    // Mostrar loading
    resultsArea.innerHTML = `
        <div class="image-preview">
            <img src="/dataset/${caminhoImagem}" alt="${nomeImagem}" style="max-width: 100%; border-radius: 10px; margin-bottom: 1rem;" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjMzMzIi8+CjxwYXRoIGQ9Ik02MCA2MEgxNDBWMTRIMTYwVjE0MEg2MFY2MFoiIGZpbGw9IiM2NjYiLz4KPHN2Zz4K'">
            <p><strong>Arquivo:</strong> ${nomeImagem}</p>
            <div class="analysis-status">
                <div class="loading-spinner"></div>
                <p>🧠 Analisando com IA...</p>
            </div>
        </div>
    `;
    
    try {
        const response = await axios.post(`${API_BASE_URL}/classificar`, {
            caminho: caminhoImagem
        });
        
        const data = response.data;
        
        if (data.sucesso) {
            mostrarResultadoReal(data, nomeImagem, caminhoImagem);
        } else {
            throw new Error(data.erro || 'Erro na classificação');
        }
        
    } catch (error) {
        console.error('❌ Erro na classificação:', error);
        if (error.code === 'ECONNREFUSED') {
            resultsArea.innerHTML = `
                <div class="error-message">
                    <h4>❌ Backend Não Conectado</h4>
                    <p>O servidor backend não está rodando.</p>
                    <p><em>Execute: python backend/servidor_backend.py</em></p>
                </div>
            `;
        } else {
            resultsArea.innerHTML = `
                <div class="error-message">
                    <h4>❌ Erro na Classificação</h4>
                    <p>${error.response?.data?.erro || error.message}</p>
                    <p><em>Verifique se o backend está rodando e o modelo está carregado.</em></p>
                </div>
            `;
        }
    }
}

function mostrarResultadoReal(data, nomeImagem, caminhoImagem) {
    const resultsArea = document.querySelector('.results-placeholder');
    
    // Formatar confiança como porcentagem
    const confiancaFormatada = (data.melhor_confianca * 100).toFixed(1);
    
    resultsArea.innerHTML = `
        <div class="analysis-results">
            <div class="image-preview">
                <img src="/dataset/${caminhoImagem}" alt="${nomeImagem}" style="max-width: 100%; border-radius: 10px; margin-bottom: 1rem;" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjMzMzIi8+CjxwYXRoIGQ9Ik02MCA2MEgxNDBWMTRIMTYwVjE0MEg2MFY2MFoiIGZpbGw9IiM2NjYiLz4KPHN2Zz4K'">
                <p><strong>Arquivo:</strong> ${nomeImagem}</p>
            </div>
            
            <h4>🎯 Resultado da Análise com IA</h4>
            <div class="result-card">
                <div class="bioma-result">
                    <span class="bioma-name">${data.melhor_bioma.toUpperCase()}</span>
                    <span class="confidence">${confiancaFormatada}% de confiança</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${confiancaFormatada}%"></div>
                </div>
            </div>
            
            <div class="top-predictions">
                <h5>📊 Top-3 Predições:</h5>
                <div class="predictions-list">
                    ${data.top_k.map((pred, index) => `
                        <div class="prediction-item ${index === 0 ? 'best' : ''}">
                            <span class="position">${pred.posicao}º</span>
                            <span class="bioma">${pred.bioma.toUpperCase()}</span>
                            <span class="confidence">${(pred.confianca * 100).toFixed(1)}%</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            
            <p class="note"><em>✅ Resultado real gerado pelo modelo de IA treinado</em></p>
        </div>
    `;
}
