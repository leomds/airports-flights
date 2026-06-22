import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_folium import st_folium
from modules.data_loader import get_data
from modules.analysis import (
    compute_degree_centrality, compute_betweenness_centrality,
    compute_closeness_centrality, compute_pagerank,
    compute_in_degree_centrality, compute_out_degree_centrality,
    top_nodes, connectivity_summary,
    get_bridges, get_articulation_points,
    shortest_path, shortest_path_avoiding,
    path_edges, simulate_node_removal, robustness_curve,
    degree_distribution, average_clustering_coefficient,
    average_shortest_path_sample,
)
from modules.geo import classify_graph, CONTINENTS, BR_REGIONS
from modules.map_viz import (
    build_base_map, add_airports, add_airports_scaled, add_route,
    add_background_routes, add_centrality_layer,
    add_removed_nodes, map_to_html,
)


# Configuração geral da página do Streamlit.
# Título que aparece na aba do navegador, ícone, layout largo e sidebar expandida.
st.set_page_config(
    page_title="Malha Aérea Mundial",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Bloco gigante de CSS + HTML + JavaScript injetado diretamente na página.
# unsafe_allow_html=True é necessário para o Streamlit não bloquear o HTML puro.
st.markdown("""
<style>
  /* Fundo escuro na sidebar */
  [data-testid="stSidebar"] { background: #0d1117; }

  /* Estilo dos cards de métricas (os quadradinhos com número grande) */
  .metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 1rem; text-align: center;
  }
  /* Número grande azul dentro do card */
  .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
  /* Texto pequeno embaixo do número */
  .metric-card .label { font-size: 0.78rem; color: #8b949e; margin-top: 4px; }

  /* Força cor clara nos títulos h1, h2, h3 */
  h1, h2, h3 { color: #f0f6fc !important; }

  /* Estilo das abas (tabs) do Streamlit */
  .stTabs [data-baseweb="tab"] { color: #8b949e; }
  /* Aba selecionada fica azul com sublinhado */
  .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff; }

  /* Overlay de loading com animacao de aviao.
     Fica invisivel por padrao, so aparece quando o Streamlit esta processando. */
  #plane-overlay {
    display: none;
    position: fixed;
    inset: 0;                          /* ocupa a tela toda */
    z-index: 99999;                    /* fica na frente de tudo */
    background: rgba(13, 17, 23, 0.82); /* fundo semitransparente escuro */
    backdrop-filter: blur(3px);        /* desfoca o conteudo atras */
    flex-direction: column;
    align-items: center;
    justify-content: center;
    pointer-events: none;              /* nao bloqueia cliques do usuario */
  }
  /* Quando a classe "active" é adicionada, o overlay aparece como flex */
  #plane-overlay.active { display: flex; }

  /* Container da cena de animacao do aviao */
  .plane-scene {
    position: relative;
    width: 420px;
    height: 90px;
    overflow: visible;
  }

  /* Trilha tracejada horizontal que o aviao voa por cima */
  .plane-trail {
    position: absolute;
    top: 50%;
    left: 0;
    width: 100%;
    height: 2px;
    /* Gradiente repetido cria o efeito de tracejado */
    background: repeating-linear-gradient(
      90deg,
      #58a6ff 0px, #58a6ff 14px,
      transparent 14px, transparent 26px
    );
    opacity: 0.35;
    transform: translateY(-50%);
  }

  /* O emoji de aviao que voa da esquerda pra direita */
  .plane-icon {
    position: absolute;
    top: 50%;
    left: -60px;            /* começa fora da tela, à esquerda */
    transform: translateY(-50%);
    font-size: 2.6rem;
    animation: fly-across 2.2s cubic-bezier(0.4, 0, 0.2, 1) infinite;
    filter: drop-shadow(0 0 8px #58a6ff88); /* brilho azul ao redor */
  }

  /* Keyframes da animacao do aviao: entra pela esquerda, cruza e sai pela direita */
  @keyframes fly-across {
    0%   { left: -60px;  opacity: 0;   transform: translateY(-50%) scale(0.85); }
    8%   { opacity: 1; }
    50%  { left: 50%;    opacity: 1;   transform: translateY(-58%) scale(1); }
    92%  { opacity: 1; }
    100% { left: calc(100% + 60px); opacity: 0; transform: translateY(-50%) scale(0.85); }
  }

  /* Nuvenszinhas decorativas espalhadas pela cena */
  .plane-cloud {
    position: absolute;
    font-size: 1.5rem;
    opacity: 0;
    animation: cloud-fade 2.2s ease-in-out infinite;
    color: #8b949e;
  }
  /* Cada nuvem tem posicao e delay de animacao diferentes para nao ficarem sincronizadas */
  .plane-cloud.c1 { top: 10px; left: 18%;  animation-delay: 0.3s;  font-size: 1.1rem; }
  .plane-cloud.c2 { top: 55px; left: 55%;  animation-delay: 0.9s;  font-size: 1.8rem; }
  .plane-cloud.c3 { top: 5px;  left: 75%;  animation-delay: 0.0s;  font-size: 0.9rem; }

  /* Animacao das nuvens: aparecem e somem suavemente */
  @keyframes cloud-fade {
    0%,100% { opacity: 0; }
    40%,60%  { opacity: 0.45; }
  }

  /* Texto "CARREGANDO..." embaixo do aviao */
  .plane-label {
    margin-top: 22px;
    color: #58a6ff;
    font-size: 0.85rem;
    font-family: monospace;
    letter-spacing: 0.12em;
    animation: blink-text 1.1s step-start infinite; /* pisca feio igual terminal antigo */
  }
  @keyframes blink-text {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.3; }
  }

  /* Faz o iframe do mapa Folium ser quadrado e responsivo */
  iframe[title="st_folium"] {
    aspect-ratio: 1 / 1 !important;
    width: 100% !important;
    height: auto !important;
    min-height: 400px;
  }
</style>

<!-- O overlay propriamente dito, inicialmente oculto -->
<div id="plane-overlay">
  <div class="plane-scene">
    <div class="plane-trail"></div>
    <div class="plane-cloud c1">☁</div>
    <div class="plane-cloud c2">☁</div>
    <div class="plane-cloud c3">☁</div>
    <div class="plane-icon">✈️</div>
  </div>
  <div class="plane-label">CARREGANDO...</div>
</div>

<script>
(function() {
  // Fica observando o DOM pra detectar quando o Streamlit esta processando algo.
  function watchSpinner() {
    const overlay = document.getElementById('plane-overlay');
    if (!overlay) return;

    // MutationObserver é a API do browser que avisa quando o DOM muda.
    const observer = new MutationObserver(() => {
      // Verifica se o Streamlit esta em modo "running" por diferentes sinais:
      // spinner SVG, elemento .stSpinner, ou classe no body.
      const running =
        document.querySelector('[data-testid="stStatusWidget"] svg') ||
        document.querySelector('.stSpinner') ||
        document.body.classList.contains('stAppRunning');

      // Liga ou desliga o overlay dependendo do estado.
      overlay.classList.toggle('active', !!running);
    });

    // Observa mudancas em todo o body: filhos, atributos, arvore inteira.
    observer.observe(document.body, {
      childList: true, subtree: true,
      attributes: true, attributeFilter: ['class']
    });
  }

  // Espera o DOM estar pronto antes de comecar a observar.
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', watchSpinner);
  } else {
    watchSpinner();
  }
})();
</script>
""", unsafe_allow_html=True)


# cache_resource guarda o resultado em memória entre reruns do Streamlit.
# Diferente de cache_data, é pra objetos grandes que não mudam (como o grafo).
# O spinner aparece durante o primeiro carregamento.
@st.cache_resource(show_spinner="Carregando malha aérea mundial...")
def load():
    # Chama o módulo de dados que lê os CSVs e monta o grafo dirigido.
    airports, G = get_data()
    return airports, G

# Executa o carregamento. Na segunda visita, vai do cache sem reprocessar.
airports, G = load()

# Cria versão não dirigida do grafo (arestas sem direção).
# Necessário pra calcular pontes, pontos de articulação, etc.
U = G.to_undirected()

# Lista de todos os códigos IATA ordenados alfabeticamente.
all_iata = sorted(G.nodes())


# Constrói um dicionário de labels amigáveis pra cada aeroporto.
# Ex: "GRU - Guarulhos, Brazil" ao invés de só "GRU".
# O underscore em _G é convencao do Streamlit pra dizer "não hash esse argumento".
@st.cache_data
def build_labels(_G):
    labels = {}
    for iata in sorted(_G.nodes()):
        d = _G.nodes[iata]
        city = d.get("city", "").strip()
        country = d.get("country", "").strip()
        # Monta o label dependendo do que tem disponível nos dados.
        if city and country:
            labels[iata] = f"{iata} — {city}, {country}"
        elif city:
            labels[iata] = f"{iata} — {city}"
        else:
            labels[iata] = iata  # fallback: só o código IATA mesmo
    return labels

# Gera os labels e cria o mapeamento inverso (label -> IATA) pra recuperar o código
# quando o usuario seleciona um item no selectbox.
iata_labels = build_labels(G)
label_to_iata = {v: k for k, v in iata_labels.items()}
all_labels = [iata_labels[i] for i in all_iata]

# Retorna o índice de um IATA na lista all_labels, pra pre-selecionar no selectbox.
def label_index(iata):
    lbl = iata_labels.get(iata, iata)
    return all_labels.index(lbl) if lbl in all_labels else 0

# Extrai o código IATA de um label completo usando o dicionário inverso.
def selected_iata(label):
    return label_to_iata.get(label, label.split(" — ")[0])


# Bloco da sidebar. Tudo dentro desse "with" aparece no painel lateral.
with st.sidebar:
    st.markdown("## ✈️ Malha Aérea Mundial")
    # Mostra o tamanho do grafo logo de cara.
    st.markdown(f"**{G.number_of_nodes():,}** aeroportos · **{G.number_of_edges():,}** rotas")
    st.divider()
    # Radio button de navegação entre as secoes do app.
    # A variável "section" controla qual painel o usuario está vendo.
    section = st.radio("Navegação", [
        "🌍 Visão Geral",
        "📊 Centralidade",
        "🔗 Conectividade",
        "🛣️ Rotas & Caminhos",
        "⚠️ Vulnerabilidade",
        "📖 Sobre",
    ])


# Função pesada que calcula tudo de uma vez e guarda em cache.
# O argumento _G_hash é um número que representa o estado do grafo,
# assim o cache é invalidado se o grafo mudar (mesmo com o underscore).
@st.cache_data(show_spinner="Calculando estrutura global da rede...")
def _precompute(_G_hash):

    import networkx as _nx
    _U = G.to_undirected()

    # Pontes: arestas cuja remoção desconecta partes da rede.
    bridges = get_bridges(G)

    # Pontos de articulação: nós cuja remoção desconecta a rede.
    artics = get_articulation_points(G)

    # Centralidades calculadas pro grafo todo.
    deg_cent = compute_degree_centrality(G)
    indeg_cent = compute_in_degree_centrality(G)
    outdeg_cent = compute_out_degree_centrality(G)
    pr_cent = compute_pagerank(G)

    # Resumo de conectividade: WCC, SCC, densidade, grau médio, etc.
    conn_stats = connectivity_summary(G)

    # Distribuição de grau com ajuste de lei de potência (scale-free).
    dd = degree_distribution(G)

    # Coeficiente de agrupamento médio (small-world).
    cc = average_clustering_coefficient(G)

    # Caminho médio estimado por amostragem de 500 pares (calcular tudo seria lento demais).
    apl = average_shortest_path_sample(G, sample=500)

    # Classificacao geografica dos nos por continente, pais e regiao do Brasil.
    geo = classify_graph(G)

    return {
        "bridges": bridges,
        "artics": artics,
        "deg_cent": deg_cent,
        "indeg_cent": indeg_cent,
        "outdeg_cent": outdeg_cent,
        "pr_cent": pr_cent,
        "conn_stats": conn_stats,
        "dd": dd,
        "cc": cc,
        "apl": apl,
        "geo": geo,
        # Mapeia cada nó ao seu país, usado pra filtros por país na seção de conectividade.
        "node_country": {n: G.nodes[n].get("country", "") for n in G.nodes},
    }


# Cria uma chave única pra identificar essa versão do grafo no cache.
# Se o número de nós ou arestas mudar, o cache é refeito.
_cache_key = G.number_of_nodes() * 100000 + G.number_of_edges()
PRECOMP = _precompute(_cache_key)


# Pega os resultados do pré-processamento em variáveis de fácil acesso.
ALL_BRIDGES = PRECOMP["bridges"]
ALL_ARTICS = PRECOMP["artics"]
NODE_COUNTRY = PRECOMP["node_country"]


# Constrói um dicionário invertido: país -> conjunto de aeroportos naquele país.
# defaultdict(set) cria um set vazio automaticamente pra chaves novas.
from collections import defaultdict as _dd
COUNTRY_NODES = _dd(set)
for _n, _c in NODE_COUNTRY.items():
    if _c:
        COUNTRY_NODES[_c].add(_n)

# Calcula o centro geográfico (média de lat/lon) de um conjunto de aeroportos.
# Usado pra centralizar o mapa quando o usuario filtra por país.
def _country_center(iata_iter):
    lats = [G.nodes[n]["lat"] for n in iata_iter if G.nodes[n].get("lat") is not None]
    lons = [G.nodes[n]["lon"] for n in iata_iter if G.nodes[n].get("lon") is not None]
    return (sum(lats) / len(lats), sum(lons) / len(lons)) if lats else (20, 0)


# ============================================================
# SECAO: VISAO GERAL
# ============================================================
if section == "🌍 Visão Geral":
    st.title("🌍 Malha Aérea Mundial - Visão Geral")

    # Pega as estatísticas globais já calculadas no pré-processamento.
    stats = PRECOMP["conn_stats"]

    # Cria 4 colunas lado a lado para os cards de métricas principais.
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Aeroportos", f"{stats['nodes']:,}"),
        ("Rotas Diretas", f"{stats['edges']:,}"),
        ("Densidade", f"{stats['density']:.4f}"),
        ("Grau Médio", f"{stats['avg_degree']:.1f}"),
    ]
    # Renderiza cada card com HTML customizado (o CSS .metric-card definido lá em cima).
    for col, (label, val) in zip([col1, col2, col3, col4], metrics):
        col.markdown(f"""<div class="metric-card">
            <div class="value">{val}</div>
            <div class="label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📐 Propriedades de Rede Complexa (Scale-Free & Small-World)")

    # Pega os resultados de distribuição de grau, agrupamento e caminho médio.
    dd = PRECOMP["dd"]
    cc = PRECOMP["cc"]
    apl = PRECOMP["apl"]

    # Três métricas lado a lado mostrando as propriedades da rede complexa.
    sw_col1, sw_col2, sw_col3 = st.columns(3)

    # Expoente alpha da lei de potência. Redes scale-free tipicamente ficam entre 2 e 3.
    sw_col1.metric("Expoente α (lei de potência)", f"{dd['alpha']:.3f}",
                   help="P(k) ~ k^{-α}. Scale-free típico: α ∈ [2, 3]")
    sw_col1.caption(f"R² do ajuste log-log: {dd['r_squared']:.3f}")

    # Coeficiente de agrupamento: o quanto os vizinhos de um nó também se conhecem entre si.
    sw_col2.metric("Coef. de Agrupamento Médio", f"{cc:.4f}",
                   help="Fração de triângulos fechados. Alto = comunidades densas.")

    # Diâmetro médio: quantos hops em média pra chegar de um aeroporto a outro qualquer.
    sw_col3.metric("Diâmetro Médio estimado (hops)", f"{apl:.2f}",
                   help="Comprimento médio do menor caminho dentro do maior WCC (500 pares aleatórios).")

    import numpy as _np

    # Gráfico de distribuição de grau em escala log-log.
    # Se a reta for aproximadamente linear em log-log, confirma a lei de potência.
    fig_dd = go.Figure()

    # Pontos empíricos: grau k no eixo X, probabilidade P(k) no eixo Y.
    fig_dd.add_trace(go.Scatter(
        x=dd["bins"], y=dd["pk"],
        mode="markers",
        marker=dict(color="#58a6ff", size=5, opacity=0.7),
        name="P(k) empírico",
    ))

    # Linha de ajuste da lei de potência em cima dos pontos.
    # Faz regressão linear no espaço log-log e converte de volta.
    k_arr = _np.array([b for b in dd["bins"] if b > 0], dtype=float)
    pk_arr = _np.array([dd["pk"][i] for i, b in enumerate(dd["bins"]) if b > 0])
    if len(k_arr) > 1:
        c0 = _np.polyfit(_np.log10(k_arr), _np.log10(pk_arr + 1e-12), 1)
        pk_fit = 10 ** _np.polyval(c0, _np.log10(k_arr))
        fig_dd.add_trace(go.Scatter(
            x=k_arr.tolist(), y=pk_fit.tolist(),
            mode="lines",
            line=dict(color="#ff6b35", width=2, dash="dash"),
            name=f"Ajuste lei de potência: α={dd['alpha']:.2f}, R²={dd['r_squared']:.3f}",
        ))

    fig_dd.update_layout(
        title="Distribuição de grau - escala log-log (indício de rede scale-free)",
        xaxis=dict(title="Grau k", type="log"),
        yaxis=dict(title="P(k)", type="log"),
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        legend=dict(bgcolor="#161b22"),
        margin=dict(t=50, b=20),
    )
    st.plotly_chart(fig_dd, use_container_width=True)
    st.caption(
        f"**Interpretação:** Reta em escala log-log confirma lei de potência (α ≈ {dd['alpha']:.2f}). "
        f"Diâmetro médio de apenas ~{apl:.1f} hops confirma a propriedade small-world."
    )

    st.markdown("---")

    # Pega a classificação geográfica e a lista de países disponíveis nos dados.
    geo = PRECOMP["geo"]
    all_countries = sorted(geo["country"].keys())

    # Expander de filtros: fica recolhido por padrão pra não poluir a tela.
    with st.expander("🔍 Filtros de visualização", expanded=False):
        filter_mode = st.radio(
            "Modo de visualização",
            ["🌍 Rede completa", "✈️ Nó único e conexões",
             "🌎 Por continente", "🗺️ Por país", "🇧🇷 Brasil por região"],
            horizontal=True,
        )

        # Essas variáveis vão conter o subconjunto de nós/arestas a mostrar.
        # None significa "mostra tudo".
        filter_nodes = None
        filter_edges = None
        map_title = "Rede global de voos - nós escalados por grau"

        if filter_mode == "✈️ Nó único e conexões":
            # Usuario escolhe um aeroporto e vê só ele e seus vizinhos diretos.
            chosen_label = st.selectbox("Aeroporto", all_labels)
            chosen = selected_iata(chosen_label)
            # Pega todos os vizinhos de entrada e saída (grafo dirigido).
            neighbors = list(set(
                list(G.successors(chosen)) + list(G.predecessors(chosen))
            ))
            filter_nodes = [chosen] + neighbors
            filter_edges = (
                [(chosen, v) for v in G.successors(chosen)] +
                [(u, chosen) for u in G.predecessors(chosen)]
            )
            map_title = f"Conexões de {chosen} — {G.nodes[chosen].get('city', '')} ({G.degree(chosen)} rotas)"

        elif filter_mode == "🌎 Por continente":
            # Filtra pelo continente escolhido usando o dicionário de geo.
            chosen_cont = st.selectbox("Continente", CONTINENTS)
            filter_nodes = geo["continent"].get(chosen_cont, [])
            filter_edges = [(u, v) for u, v in G.edges()
                            if u in filter_nodes and v in filter_nodes]
            map_title = f"Malha aérea — {chosen_cont} ({len(filter_nodes)} aeroportos)"

        elif filter_mode == "🗺️ Por país":
            # Filtra por país. Já começa selecionando o Brasil.
            chosen_country = st.selectbox("País", all_countries,
                                          index=all_countries.index("Brazil") if "Brazil" in all_countries else 0)
            filter_nodes = geo["country"].get(chosen_country, [])
            filter_edges = [(u, v) for u, v in G.edges()
                            if u in filter_nodes and v in filter_nodes]
            map_title = f"Malha aérea — {chosen_country} ({len(filter_nodes)} aeroportos)"

        elif filter_mode == "🇧🇷 Brasil por região":
            # Permite selecionar múltiplas regiões do Brasil ao mesmo tempo.
            chosen_regions = st.multiselect(
                "Regiões",
                BR_REGIONS,
                default=["Norte"],
            )
            if chosen_regions:
                filter_nodes = []
                for r in chosen_regions:
                    filter_nodes += geo["br_region"].get(r, [])
                # Remove duplicatas que podem acontecer se um aeroporto aparecer em mais de uma região.
                filter_nodes = list(set(filter_nodes))
                filter_edges = [(u, v) for u, v in G.edges()
                                if u in filter_nodes and v in filter_nodes]
                map_title = f"Brasil — {' + '.join(chosen_regions)} ({len(filter_nodes)} aeroportos)"

    st.subheader(map_title)

    # Mapa completo: fica em cache como recurso pra não renderizar de novo a cada interação.
    @st.cache_resource(show_spinner="Renderizando mapa...")
    def _full_map():
        m = build_base_map()
        add_background_routes(m, G)  # linhas finas de todas as rotas
        add_airports_scaled(m, G)    # circulos escalados pelo grau de cada aeroporto
        return m

    # Mapa filtrado: recebe tuplas como chave pro cache funcionar (listas não são hashable).
    def _filtered_map(nodes_key, edges_key):
        nodes = list(nodes_key)
        edges = list(edges_key)
        nodes_set = set(nodes)
        m = build_base_map()
        import folium as _f
        # Desenha as rotas internas ao filtro.
        for u, v in edges:
            if u not in G.nodes or v not in G.nodes:
                continue
            n1, n2 = G.nodes[u], G.nodes[v]
            _f.PolyLine(
                [[n1["lat"], n1["lon"]], [n2["lat"], n2["lon"]]],
                color="#3a6bbf", weight=0.8, opacity=0.4,
            ).add_to(m)
        add_airports_scaled(m, G, nodes=nodes)
        return m

    # Decide qual mapa renderizar baseado no filtro ativo.
    if filter_nodes is None:
        map_obj = _full_map()
        map_key = "map_overview_full"
    elif filter_mode == "✈️ Nó único e conexões":
        # Caso especial: destaca as rotas do aeroporto selecionado em azul mais vivo.
        import folium as _f
        m = build_base_map()
        for u, v in filter_edges:
            if u in G.nodes and v in G.nodes:
                n1, n2 = G.nodes[u], G.nodes[v]
                _f.PolyLine(
                    [[n1["lat"], n1["lon"]], [n2["lat"], n2["lon"]]],
                    color="#58a6ff", weight=1.5, opacity=0.7,
                ).add_to(m)
        # highlight=[chosen] faz o aeroporto selecionado aparecer diferente dos vizinhos.
        add_airports_scaled(m, G, nodes=filter_nodes, highlight=[chosen])
        map_obj = m
        map_key = f"map_node_{chosen}"
    else:
        # Converte pra tupla pra ser hashable e poder usar no cache.
        map_obj = _filtered_map(tuple(sorted(filter_nodes)), tuple(sorted(filter_edges)))
        map_key = f"map_filter_{hash(map_title)}"

    # Renderiza o mapa no Streamlit. returned_objects=[] evita que o componente
    # retorne dados ao Python a cada clique no mapa, o que causaria reruns desnecessários.
    st_folium(map_obj, height=800, use_container_width=True,
              key=map_key, returned_objects=[])

    if filter_nodes is not None:
        st.caption(f"Exibindo **{len(filter_nodes)}** aeroportos e "
                   f"**{len(filter_edges)}** rotas internas ao filtro selecionado.")
    else:
        st.caption("Tamanho e brilho dos pontos proporcional ao número de rotas (grau). ")

    # Monta o subgrafo correspondente ao filtro ativo (ou o grafo inteiro se nenhum filtro).
    if filter_nodes is None:
        active_nodes = list(G.nodes())
        active_edges = list(G.edges())
    else:
        active_nodes = filter_nodes
        active_edges = filter_edges if filter_edges else []

    # G.subgraph cria uma visão do grafo com só esses nós (não copia, é uma view).
    SG = G.subgraph(active_nodes)

    st.markdown("---")
    st.subheader("📈 Análise do subgrafo atual")

    # Calcula estatísticas básicas do subgrafo atual.
    sg_degrees = [d for _, d in SG.degree()]
    sg_n = SG.number_of_nodes()
    sg_e = SG.number_of_edges()
    sg_avg_deg = np.mean(sg_degrees) if sg_degrees else 0
    sg_max_deg = max(sg_degrees) if sg_degrees else 0
    sg_density = nx.density(SG) if sg_n > 1 else 0

    # Exibe as métricas do subgrafo em 4 colunas.
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Aeroportos", f"{sg_n:,}")
    m2.metric("Rotas internas", f"{sg_e:,}")
    m3.metric("Grau médio", f"{sg_avg_deg:.1f}")
    m4.metric("Grau máximo", f"{sg_max_deg}")

    col_left, col_right = st.columns(2)

    with col_left:
        # Gráfico de distribuição de grau do subgrafo em log-log.
        # value_counts conta quantos nós têm cada grau, sort_index ordena pelo grau.
        deg_count = pd.Series(sg_degrees).value_counts().sort_index()
        fig_deg = px.scatter(
            x=deg_count.index, y=deg_count.values,
            log_x=True, log_y=True,
            labels={"x": "Grau (k)", "y": "Número de nós"},
            title="Distribuição de grau (log-log)",
            template="plotly_dark",
        )
        fig_deg.update_traces(marker=dict(color="#58a6ff", size=6))
        fig_deg.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            title_font_color="#f0f6fc", margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_deg, use_container_width=True)

    with col_right:
        # Gráfico de barras com os 15 aeroportos de maior grau no subgrafo.
        top15 = sorted(SG.degree(), key=lambda x: x[1], reverse=True)[:15]
        top15_iata = [x[0] for x in top15]
        top15_deg = [x[1] for x in top15]
        # Pega o nome da cidade pra mostrar embaixo do código IATA no eixo X.
        top15_city = [G.nodes[i].get("city", i) for i in top15_iata]
        fig_top = go.Figure(go.Bar(
            x=[f"{i}<br><sub>{c}</sub>" for i, c in zip(top15_iata, top15_city)],
            y=top15_deg,
            marker=dict(
                color=top15_deg,
                colorscale="Blues",
                showscale=False,
            ),
            text=top15_deg,
            textposition="outside",
        ))
        fig_top.update_layout(
            title="Top 15 hubs por grau",
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            title_font_color="#f0f6fc",
            xaxis_title="", yaxis_title="Grau",
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_top, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        # PageRank do subgrafo. Só faz sentido calcular se tiver mais de 2 nós.
        # alpha=0.85 é o amortecimento padrão do PageRank original do Google.
        # weight="weight" usa a distância como peso das arestas se existir.
        # max_iter=200 pra garantir convergência em grafos densos.
        if sg_n > 2:
            pr = nx.pagerank(SG, alpha=0.85, weight="weight", max_iter=200)
            pr_top = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:15]
            pr_iata = [x[0] for x in pr_top]
            pr_vals = [x[1] for x in pr_top]
            pr_city = [G.nodes[i].get("city", i) for i in pr_iata]
            fig_pr = go.Figure(go.Bar(
                x=[f"{i}<br><sub>{c}</sub>" for i, c in zip(pr_iata, pr_city)],
                y=pr_vals,
                marker=dict(color="#ff6b35"),
                text=[f"{v:.4f}" for v in pr_vals],
                textposition="outside",
            ))
            fig_pr.update_layout(
                title="Top 15 por PageRank",
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                title_font_color="#f0f6fc",
                xaxis_title="", yaxis_title="PageRank",
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_pr, use_container_width=True)

    with col_d:
        # Histograma de grau: distribuição contínua (40 bins) dos graus do subgrafo.
        # Complementa o scatter log-log mostrando a cauda longa de forma mais intuitiva.
        fig_hist = px.histogram(
            x=sg_degrees, nbins=40,
            labels={"x": "Grau", "y": "Frequência"},
            title="Histograma de grau",
            template="plotly_dark",
            color_discrete_sequence=["#58a6ff"],
        )
        fig_hist.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            title_font_color="#f0f6fc", margin=dict(t=40, b=20),
            bargap=0.05,
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ============================================================
# SECAO: CENTRALIDADE
# ============================================================
elif section == "📊 Centralidade":
    st.title("📊 Análise de Centralidade")

    # Usuario escolhe qual métrica de centralidade quer visualizar.
    metric_choice = st.selectbox("Métrica de Centralidade", [
        "Degree", "In-Degree", "Out-Degree",
        "Betweenness", "Closeness", "PageRank"
    ])
    top_n = st.slider("Top N aeroportos", 10, 100, 30)

    # Retorna a centralidade correspondente à métrica escolhida.
    # Betweenness e Closeness são calculados aqui por serem lentos (não estão no precompute global).
    # k=300 no betweenness significa amostragem de 300 nós (aproximação pra ser mais rápido).
    @st.cache_data(show_spinner="Calculando centralidade...")
    def get_centrality(metric):
        if metric == "Degree":
            return PRECOMP["deg_cent"]
        elif metric == "In-Degree":
            return PRECOMP["indeg_cent"]
        elif metric == "Out-Degree":
            return PRECOMP["outdeg_cent"]
        elif metric == "PageRank":
            return PRECOMP["pr_cent"]
        elif metric == "Betweenness":
            return compute_betweenness_centrality(G, k=300)
        elif metric == "Closeness":
            return compute_closeness_centrality(G)

    centrality = get_centrality(metric_choice)
    # Pega os top N aeroportos ordenados por score decrescente.
    top = top_nodes(centrality, top_n)

    # Divide a tela: 2/3 pro mapa, 1/3 pra tabela.
    col_map, col_table = st.columns([2, 1])

    with col_map:
        st.subheader("Mapa de centralidade")

        # Renderiza o mapa com os nós coloridos/escalados pela centralidade escolhida.
        @st.cache_resource(show_spinner="Renderizando...")
        def _centrality_map(metric_key, n):
            cent = get_centrality(metric_key)
            m = build_base_map()
            add_centrality_layer(m, G, cent, top_n=n)
            return m

        st_folium(_centrality_map(metric_choice, top_n),
                  height=800, use_container_width=True,
                  key=f"map_cent_{metric_choice}_{top_n}", returned_objects=[])

    with col_table:
        st.subheader(f"Top {top_n} aeroportos")
        # Monta DataFrame com IATA, cidade, país e score formatado.
        df_top = pd.DataFrame(top, columns=["IATA", "Score"])
        df_top["Cidade"] = df_top["IATA"].map(
            lambda x: G.nodes[x].get("city", "") if x in G.nodes else ""
        )
        df_top["País"] = df_top["IATA"].map(
            lambda x: G.nodes[x].get("country", "") if x in G.nodes else ""
        )
        df_top["Score"] = df_top["Score"].map(lambda x: f"{x:.5f}")
        df_top.index = range(1, len(df_top) + 1)  # índice começa em 1 (ranking)
        st.dataframe(df_top[["IATA", "Cidade", "País", "Score"]],
                     use_container_width=True, height=420)

    st.subheader("Ranking visual")
    # Gráfico de barras com os top 20 aeroportos da métrica selecionada.
    labels_bar = [f"{iata}" for iata, _ in top[:20]]
    values_bar = [v for _, v in top[:20]]
    fig = go.Figure(go.Bar(
        x=labels_bar, y=values_bar,
        marker_color="#58a6ff",
        text=[f"{v:.4f}" for v in values_bar],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        xaxis_title="Aeroporto (IATA)", yaxis_title="Score",
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# SECAO: CONECTIVIDADE
# ============================================================
elif section == "🔗 Conectividade":
    st.title("🔗 Análise de Conectividade")

    # Filtro de país pra restringir a análise de pontes e articulações.
    all_countries_conn = sorted(COUNTRY_NODES.keys())
    country_conn = st.selectbox(
        "Filtrar por país (ou ver rede completa)",
        ["Todos"] + all_countries_conn,
        key="conn_country_filter"
    )

    # Define quais pontes/articulações mostrar e como centrar o mapa.
    if country_conn == "Todos":
        # Usa os valores pré-calculados do grafo inteiro.
        bridges_show = ALL_BRIDGES
        artics_show = ALL_ARTICS
        map_center = (20, 0)   # centro aproximado do mundo
        map_zoom = 2
        stats_show = PRECOMP["conn_stats"]
    else:
        # Filtra pontes que tocam algum nó do país selecionado.
        conn_nodes = COUNTRY_NODES[country_conn]
        bridges_show = [(u, v) for u, v in ALL_BRIDGES
                        if u in conn_nodes or v in conn_nodes]
        artics_show = [a for a in ALL_ARTICS if a in conn_nodes]
        map_center = _country_center(conn_nodes)
        map_zoom = 4

        # Calcula estatísticas só pro subgrafo do país selecionado.
        @st.cache_data
        def _stats_country(country):
            sub = G.subgraph(COUNTRY_NODES[country]).copy()
            return connectivity_summary(sub)

        stats_show = _stats_country(country_conn)

    # Métricas de conectividade: WCC, SCC, pontes, pontos de articulação.
    c1, c2, c3 = st.columns(3)
    c1.metric("Componentes Fracamente Conexos", stats_show["weakly_connected_components"])
    c1.metric("Maior WCC (nós)", stats_show["largest_wcc_size"])
    c2.metric("Componentes Fortemente Conexos", stats_show["strongly_connected_components"])
    c2.metric("Maior SCC (nós)", stats_show["largest_scc_size"])
    c3.metric("Pontes (bridges)", len(bridges_show))
    c3.metric("Pontos de Articulação", len(artics_show))

    st.markdown("---")

    # Três abas: pontes, pontos de articulação, distribuição dos SCCs.
    tab1, tab2, tab3 = st.tabs(["Pontes", "Pontos de Articulação", "Distribuição SCC"])

    with tab1:
        st.subheader("Rotas críticas (bridges)")
        st.info(
            "Uma **ponte (bridge)** é uma aresta cuja remoção aumenta o número de componentes conexos. "
            "Na malha aérea, representa uma rota que é o único elo entre dois grupos de aeroportos."
        )

        if bridges_show:
            # Categoriza os extremos de cada ponte como "hub" (maior grau) ou "dependente".
            hub_nodes = set()
            dep_nodes = set()
            for u, v in bridges_show:
                if G.degree(u) >= G.degree(v):
                    hub_nodes.add(u); dep_nodes.add(v)
                else:
                    hub_nodes.add(v); dep_nodes.add(u)
            # Remove da lista de dependentes os que também são hubs em outras pontes.
            dep_nodes -= hub_nodes

            # Tabela de pontes com nome completo dos aeroportos e papel de cada um.
            bridge_df = pd.DataFrame(bridges_show, columns=["De", "Para"])
            bridge_df["De (nome)"] = bridge_df["De"].map(lambda x: G.nodes[x].get("name", "") if x in G.nodes else "")
            bridge_df["Para (nome)"] = bridge_df["Para"].map(lambda x: G.nodes[x].get("name", "") if x in G.nodes else "")
            bridge_df["Papel De"] = bridge_df["De"].map(lambda x: "🔴 Hub" if x in hub_nodes else "🔵 Dependente")
            bridge_df["Papel Para"] = bridge_df["Para"].map(lambda x: "🔴 Hub" if x in hub_nodes else "🔵 Dependente")
            st.dataframe(bridge_df, use_container_width=True, height=280)

            import folium as _folium
            m_b = build_base_map(center=map_center, zoom=map_zoom)

            # Plota os hubs em vermelho.
            for iata in hub_nodes:
                if iata not in G.nodes: continue
                d = G.nodes[iata]
                if d.get("lat") is None: continue
                _folium.CircleMarker(
                    location=[d["lat"], d["lon"]], radius=5,
                    color="#e53935", fill=True, fill_color="#e53935",
                    fill_opacity=0.95, weight=0,
                    tooltip=f"🔴 HUB: <b>{iata}</b> — {d.get('city', '')} ({d.get('country', '')})<br>"
                            f"Grau: {G.degree(iata)} | ancora {sum(1 for u, v in bridges_show if u == iata or v == iata)} bridge(s)",
                ).add_to(m_b)

            # Plota os dependentes em ciano.
            for iata in dep_nodes:
                if iata not in G.nodes: continue
                d = G.nodes[iata]
                if d.get("lat") is None: continue
                _folium.CircleMarker(
                    location=[d["lat"], d["lon"]], radius=4,
                    color="#00e5ff", fill=True, fill_color="#00e5ff",
                    fill_opacity=0.9, weight=0,
                    tooltip=f"🔵 DEPENDENTE: <b>{iata}</b> — {d.get('city', '')} ({d.get('country', '')})<br>"
                            f"Grau: {G.degree(iata)} | isolado se a bridge cair",
                ).add_to(m_b)

            # Plota as linhas das pontes em vermelho. Limita a 400 pra não travar o navegador.
            for u, v in bridges_show[:400]:
                if u not in G.nodes or v not in G.nodes: continue
                n1, n2 = G.nodes[u], G.nodes[v]
                _folium.PolyLine(
                    [[n1["lat"], n1["lon"]], [n2["lat"], n2["lon"]]],
                    color="#ff4444", weight=1.5, opacity=0.75,
                    tooltip=f"Bridge: {u} ↔ {v}",
                ).add_to(m_b)

            st_folium(m_b, height=800, use_container_width=True,
                      key=f"map_bridges_{country_conn}", returned_objects=[])

            # Legenda textual abaixo do mapa.
            leg1, leg2, leg3 = st.columns(3)
            leg1.markdown("🔴 **Hub crítico** - ancora a bridge, maior grau")
            leg2.markdown("🔵 **Dependente** - isolado se a bridge cair")
            leg3.markdown("🔴 **Linha vermelha** - a bridge (aresta crítica)")
        else:
            st.success("Nenhuma ponte encontrada no subgrafo selecionado.")

    with tab2:
        st.subheader("Aeroportos críticos (pontos de articulação)")
        st.info(
            "Um **ponto de articulação** é um nó cuja remoção aumenta o número de componentes. "
            "Na prática, é um aeroporto que conecta regiões que ficariam isoladas sem ele."
        )

        if artics_show:
            artics_set = set(artics_show)
            artic_df = pd.DataFrame(artics_show, columns=["IATA"])
            artic_df["Cidade"] = artic_df["IATA"].map(lambda x: G.nodes[x].get("city", "") if x in G.nodes else "")
            artic_df["País"] = artic_df["IATA"].map(lambda x: G.nodes[x].get("country", "") if x in G.nodes else "")
            artic_df["Grau"] = artic_df["IATA"].map(lambda x: G.degree(x) if x in G.nodes else 0)
            # Ordena pelos mais conectados primeiro, pois tendem a ser os mais críticos.
            artic_df = artic_df.sort_values("Grau", ascending=False)
            st.dataframe(artic_df, use_container_width=True, height=280)

            import folium as _folium
            _U = G.to_undirected()  # precisa do grafo não dirigido pra encontrar vizinhos
            m_a = build_base_map(center=map_center, zoom=map_zoom)

            # Vizinhos dos pontos de articulação que não são eles mesmos articulações.
            # São os aeroportos que dependem deles pra se conectar ao resto da rede.
            dep_neighbors = set()
            for ap in artics_set:
                for nb in _U.neighbors(ap):
                    if nb not in artics_set:
                        dep_neighbors.add(nb)

            # Plota os vizinhos dependentes em roxo claro.
            for iata in dep_neighbors:
                if iata not in G.nodes: continue
                d = G.nodes[iata]
                if d.get("lat") is None: continue
                _folium.CircleMarker(
                    location=[d["lat"], d["lon"]], radius=3,
                    color="#ce93d8", fill=True, fill_color="#ce93d8",
                    fill_opacity=0.7, weight=0,
                    tooltip=f"🟣 <b>{iata}</b> — {d.get('city', '')} ({d.get('country', '')})<br>Vizinho de ponto de articulação",
                ).add_to(m_a)

            # Plota os pontos de articulação em laranja.
            for iata in artics_set:
                if iata not in G.nodes: continue
                d = G.nodes[iata]
                if d.get("lat") is None: continue
                _folium.CircleMarker(
                    location=[d["lat"], d["lon"]], radius=6,
                    color="#ff6d00", fill=True, fill_color="#ff6d00",
                    fill_opacity=0.95, weight=0,
                    tooltip=f"🟠 ARTICULAÇÃO: <b>{iata}</b> — {d.get('city', '')} ({d.get('country', '')})<br>"
                            f"Grau: {G.degree(iata)}",
                ).add_to(m_a)

            st_folium(m_a, height=800, use_container_width=True,
                      key=f"map_artic_{country_conn}", returned_objects=[])
            la1, la2 = st.columns(2)
            la1.markdown("🟠 **Ponto de articulação** - remoção desconecta a rede")
            la2.markdown("🟣 **Vizinho dependente** - alcança a rede via esse hub")
        else:
            st.info("Nenhum ponto de articulação no subgrafo selecionado.")

    with tab3:
        st.subheader("Tamanho dos componentes (SCC)")
        from modules.analysis import get_strongly_connected_components
        sccs = get_strongly_connected_components(G)
        # Ordena os SCCs do maior pro menor e pega só os 30 maiores pra mostrar.
        sizes = sorted([len(s) for s in sccs], reverse=True)
        fig = px.bar(
            x=list(range(1, min(30, len(sizes)) + 1)),
            y=sizes[:30],
            labels={"x": "SCC rank", "y": "Tamanho"},
            template="plotly_dark",
        )
        fig.update_traces(marker_color="#58a6ff")
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
        st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# SECAO: ROTAS E CAMINHOS
# ============================================================
elif section == "🛣️ Rotas & Caminhos":
    st.title("🛣️ Rotas & Menor Caminho")

    col1, col2 = st.columns(2)
    with col1:
        # Selectbox de origem, começa em MAO (Manaus) se existir.
        src_label = st.selectbox(
            "Origem",
            all_labels,
            index=label_index("MAO") if "MAO" in all_iata else 0,
        )
    with col2:
        # Selectbox de destino, começa em LHR (Heathrow) se existir.
        dst_label = st.selectbox(
            "Destino",
            all_labels,
            index=label_index("LHR") if "LHR" in all_iata else 1,
        )

    # Converte os labels de volta pra código IATA.
    src = selected_iata(src_label)
    dst = selected_iata(dst_label)

    # Campo pra inserir aeroportos a evitar, separados por vírgula.
    avoid_input = st.text_input(
        "Aeroportos a evitar (opcional)",
        placeholder="ex: CDG, FRA",
    )
    # Processa o input: separa por vírgula, remove espaços, converte pra maiúsculo.
    avoid_nodes = [x.strip().upper() for x in avoid_input.split(",") if x.strip()] if avoid_input else []

    # Guarda o resultado do cálculo na session_state pra não recalcular em cada rerun.
    if "route_result" not in st.session_state:
        st.session_state.route_result = None

    if st.button("🔍 Calcular Rota", type="primary"):
        # Calcula o caminho mais curto entre origem e destino.
        path, dist = shortest_path(G, src, dst)
        path_alt, dist_alt = (None, None)
        # Se tiver aeroportos a evitar, calcula a rota alternativa.
        if avoid_nodes:
            path_alt, dist_alt = shortest_path_avoiding(G, src, dst, avoid_nodes=avoid_nodes)
        # Salva o resultado completo no state.
        st.session_state.route_result = dict(
            src=src, dst=dst,
            path=path, dist=dist,
            path_alt=path_alt, dist_alt=dist_alt,
            avoid_nodes=avoid_nodes,
        )

    # Só renderiza o resultado se existir e se ainda bater com a origem/destino atuais.
    res = st.session_state.route_result
    if res and res["src"] == src and res["dst"] == dst:
        path = res["path"]
        dist = res["dist"]
        path_alt = res["path_alt"]
        dist_alt = res["dist_alt"]
        av_nodes = res["avoid_nodes"]

        # Importa funções internas do módulo de visualização pra checar cruzamento do antimeridiano.
        from modules.map_viz import _arc_points, _split_antimeridian

        # Verifica se um trecho de voo cruza o antimeridiano (linha de data internacional).
        # Isso causa arcos "quebrados" no mapa porque o Folium não lida bem com isso por padrão.
        def _crosses_antimeridian(p1, p2):
            arc = _arc_points(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
            return len(_split_antimeridian(arc)) > 1

        if path:
            # Lista os trechos que cruzam o antimeridiano pra avisar o usuario.
            crossing_legs = [
                f"{u}→{v}"
                for u, v in zip(path[:-1], path[1:])
                if u in G.nodes and v in G.nodes
                   and _crosses_antimeridian(G.nodes[u], G.nodes[v])
            ]
            if crossing_legs:
                legs_str = ', '.join(crossing_legs)
                st.warning(
                    "Atenção: um ou mais trechos desta rota cruzam o antimeridiano (linha de data). "
                    "A visualização no mapa pode parecer invertida, mas a rota está correta. "
                    f"Trecho(s) afetado(s): **{legs_str}**"
                )

        import folium as _f
        m = build_base_map()
        add_background_routes(m, G)  # fundo de todas as rotas em cinza escuro

        if path:
            # Desenha a rota principal em vermelho, mais grossa.
            add_route(m, G, path, color="#e53935", weight=4)

            # Marca as escalas intermediárias em verde.
            stopovers = path[1:-1]
            for iata in stopovers:
                if iata not in G.nodes: continue
                d = G.nodes[iata]
                _f.CircleMarker(
                    location=[d["lat"], d["lon"]],
                    radius=9,
                    color="#00e676", fill=True, fill_color="#00e676",
                    fill_opacity=0.95, weight=0,
                    tooltip=f"🟢 <b>{iata}</b> — {d.get('city', '?')}, {d.get('country', '?')}<br>Escala",
                ).add_to(m)

            # Marca origem e destino em laranja, maiores que as escalas.
            for iata in [src, dst]:
                if iata not in G.nodes: continue
                d = G.nodes[iata]
                label = "🛫 Origem" if iata == src else "🛬 Destino"
                _f.CircleMarker(
                    location=[d["lat"], d["lon"]],
                    radius=12,
                    color="#ff6b35", fill=True, fill_color="#ff6b35",
                    fill_opacity=1.0, weight=0,
                    tooltip=f"{label}: <b>{iata}</b> — {d.get('city', '?')}, {d.get('country', '?')}",
                ).add_to(m)

        if av_nodes:
            # Marca os aeroportos bloqueados com X no mapa.
            add_removed_nodes(m, G, av_nodes)
            if path_alt:
                # Rota alternativa em laranja, um pouco mais fina.
                add_route(m, G, path_alt, color="#ff6b35", weight=3)

        st_folium(m, height=800, use_container_width=True,
                  key="map_routes", returned_objects=[])

        # Exibe os detalhes das duas rotas lado a lado.
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if path:
                st.success(f"✅ **Rota ótima** — {dist:,.0f} km")
                # Formata o caminho como "GRU (Guarulhos) → LHR (Heathrow)"
                readable = " → ".join(
                    f"{n} ({G.nodes[n].get('city', '?')})" for n in path
                )
                st.code(readable)
                st.caption(f"{len(path) - 1} escala(s)")
            else:
                st.error("❌ Sem rota disponível entre esses aeroportos.")

        with res_col2:
            if av_nodes:
                if path_alt:
                    # Calcula quanto quilômetros a mais a rota alternativa tem.
                    delta = dist_alt - (dist or 0)
                    st.warning(
                        f"🔄 **Rota alternativa** (evitando {', '.join(av_nodes)}) "
                        f"— {dist_alt:,.0f} km (+{delta:,.0f} km)"
                    )
                    readable_alt = " → ".join(
                        f"{n} ({G.nodes[n].get('city', '?')})" for n in path_alt
                    )
                    st.code(readable_alt)
                    st.caption(f"{len(path_alt) - 1} escala(s)")
                else:
                    st.error(f"❌ Sem rota alternativa evitando {', '.join(av_nodes)}.")

        # Tabela detalhada de cada trecho: de onde pra onde, cidades, distância.
        if path:
            st.subheader("Detalhes do caminho")
            rows = []
            for u, v in path_edges(path):
                d = G[u][v].get("weight", 0)
                rows.append({
                    "Trecho": f"{u} → {v}",
                    "De": f"{G.nodes[u].get('city', '?')} ({u})",
                    "Para": f"{G.nodes[v].get('city', '?')} ({v})",
                    "Distância (km)": f"{d:,.0f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ===========================================================
# SECAO: VULNERABILIDADE
# =======================================================
elif section == "⚠️ Vulnerabilidade":
    st.title("⚠️ Análise de Vulnerabilidade & Robustez")
    st.markdown(
        "Simule falhas e ataques na malha aérea para entender quais aeroportos são críticos "
        "e como a rede se fragmenta sob diferentes estratégias de remoção."
    )
    st.divider()

    # Duas abas: simulação pontual de remoção, e curva de robustez progressiva.
    tab1, tab2 = st.tabs(["💥 Simulação de Falha", "📉 Curva de Robustez"])

    with tab1:
        st.subheader("Simular remoção de aeroportos")
        st.markdown(
            "Selecione aeroportos manualmente ou deixe o sistema escolher automaticamente "
            "os N maiores hubs por alguma métrica de centralidade."
        )

        col_sel, col_quick = st.columns([1, 1])
        with col_sel:
            # Seleção manual: multiselect com todos os aeroportos disponíveis.
            nodes_to_remove = st.multiselect(
                "Aeroportos a remover",
                all_labels,
                # Começa com ATL, ORD e LHR pre-selecionados (se existirem).
                default=[iata_labels[x] for x in ["ATL", "ORD", "LHR"]
                         if x in iata_labels] or [],
            )
            nodes_to_remove = [selected_iata(l) for l in nodes_to_remove]

        with col_quick:
            # Seleção rápida: escolhe os N maiores hubs por centralidade.
            quick = st.radio(
                "Seleção automática",
                ["(manual)", "Degree", "Betweenness", "PageRank"],
            )
            quick_n = st.slider("N hubs", 1, 20, 5)
            if quick != "(manual)":
                @st.cache_data
                def get_quick_centrality(metric):
                    if metric == "Degree": return compute_degree_centrality(G)
                    elif metric == "Betweenness": return compute_betweenness_centrality(G, k=200)
                    elif metric == "PageRank": return compute_pagerank(G)

                cent = get_quick_centrality(quick)
                nodes_to_remove = [n for n, _ in top_nodes(cent, quick_n)]
                st.info(f"Selecionados: {', '.join(nodes_to_remove)}")

        # Guarda o resultado da simulação no session_state.
        if "vuln_result" not in st.session_state:
            st.session_state.vuln_result = None

        if nodes_to_remove and st.button("💥 Simular Falha", type="primary"):
            st.session_state.vuln_result = (
                simulate_node_removal(G, nodes_to_remove),
                list(nodes_to_remove),
            )

        if st.session_state.vuln_result:
            result, removed = st.session_state.vuln_result

            st.markdown("#### Impacto da remoção")
            c1, c2, c3, c4 = st.columns(4)
            edges_lost = result["original_edges"] - result["remaining_edges"]

            # Cards de impacto: nós removidos, rotas perdidas, tamanho do componente antes e depois.
            c1.metric("Aeroportos removidos", result["removed_nodes"],
                      delta=f"-{result['removed_nodes']} nós", delta_color="inverse")
            c2.metric("Rotas perdidas", f"{edges_lost:,}",
                      delta=f"-{result['edges_lost_pct']:.1f}%", delta_color="inverse")
            c3.metric("Maior componente (antes)", f"{result['original_largest_wcc']:,}")
            c4.metric("Maior componente (depois)", f"{result['new_largest_wcc']:,}",
                      delta=f"-{result['wcc_reduction_pct']:.1f}%", delta_color="inverse")

            # Gráfico de barras comparando antes e depois em 3 dimensões.
            fig_impact = go.Figure()
            cats = ["Aeroportos", "Rotas", "Maior componente (nós)"]
            befores = [result["original_nodes"], result["original_edges"], result["original_largest_wcc"]]
            afters = [result["remaining_nodes"], result["remaining_edges"], result["new_largest_wcc"]]
            fig_impact.add_trace(go.Bar(
                name="Antes", x=cats, y=befores,
                marker_color="#58a6ff", opacity=0.85,
            ))
            fig_impact.add_trace(go.Bar(
                name="Depois", x=cats, y=afters,
                marker_color="#ff6b35", opacity=0.85,
            ))
            fig_impact.update_layout(
                barmode="group",
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                title=f"Impacto da remoção de {len(removed)} aeroporto(s): {', '.join(removed[:5])}{'...' if len(removed) > 5 else ''}",
                title_font_color="#f0f6fc",
                legend=dict(bgcolor="#161b22"),
                margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_impact, use_container_width=True)

            # Gráfico do grau de cada aeroporto removido (pra mostrar o peso do que foi tirado).
            removed_deg = [(iata, G.degree(iata),
                            G.nodes[iata].get("city", ""),
                            G.nodes[iata].get("country", ""))
                           for iata in removed if iata in G.nodes]
            removed_deg.sort(key=lambda x: x[1], reverse=True)
            df_rem = pd.DataFrame(removed_deg, columns=["IATA", "Grau", "Cidade", "País"])

            fig_rem = go.Figure(go.Bar(
                x=df_rem["IATA"], y=df_rem["Grau"],
                marker=dict(color=df_rem["Grau"], colorscale="Reds", showscale=False),
                text=df_rem["Grau"], textposition="outside",
                customdata=df_rem[["Cidade", "País"]].values,
                hovertemplate="<b>%{x}</b> — %{customdata[0]}, %{customdata[1]}<br>Grau: %{y}<extra></extra>",
            ))
            fig_rem.update_layout(
                title="Grau dos aeroportos removidos",
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                title_font_color="#f0f6fc",
                xaxis_title="", yaxis_title="Grau (nº de rotas)",
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_rem, use_container_width=True)

            st.markdown("#### Mapa - aeroportos removidos (✕) e rotas perdidas (vermelho)")
            removed_set = set(removed)

            # Arestas que tocam algum nó removido: essas rotas somem com ele.
            lost_edges = [
                (u, v) for u, v in G.edges()
                if u in removed_set or v in removed_set
            ]

            import folium as _fv
            m = build_base_map()

            # Plota as rotas perdidas em vermelho.
            for u, v in lost_edges:
                if u not in G.nodes or v not in G.nodes: continue
                n1, n2 = G.nodes[u], G.nodes[v]
                if n1.get("lat") is None or n2.get("lat") is None: continue
                _fv.PolyLine(
                    [[n1["lat"], n1["lon"]], [n2["lat"], n2["lon"]]],
                    color="#ff4444", weight=1.2, opacity=0.7,
                ).add_to(m)

            # Plota as rotas restantes em azul bem escuro (quase invisível, só pra contexto).
            for u, v in G.edges():
                if u in removed_set or v in removed_set: continue
                if u not in G.nodes or v not in G.nodes: continue
                n1, n2 = G.nodes[u], G.nodes[v]
                if n1.get("lat") is None or n2.get("lat") is None: continue
                _fv.PolyLine(
                    [[n1["lat"], n1["lon"]], [n2["lat"], n2["lon"]]],
                    color="#1a3a6b", weight=0.4, opacity=0.3,
                ).add_to(m)

            add_airports_scaled(m, G)
            add_removed_nodes(m, G, removed)  # marca os removidos com X
            st_folium(m, height=800, use_container_width=True,
                      key="map_vuln", returned_objects=[])

            # Calcula o grafo após a remoção pra encontrar quem ficou isolado.
            G_after = G.copy()
            G_after.remove_nodes_from(removed)

            # Nós sem nenhuma aresta restante.
            isolated_after = [
                n for n in G_after.nodes()
                if G_after.degree(n) == 0
            ]

            # Nós que ficaram fora do maior componente conexo.
            if G_after.number_of_nodes() > 0:
                wcc_after = list(nx.weakly_connected_components(G_after))
                largest_wcc = max(wcc_after, key=len)
                disconnected = [
                    n for n in G_after.nodes()
                    if n not in largest_wcc
                ]
            else:
                disconnected = []

            # Junta isolados e desconectados, ordena pelos de maior grau original.
            all_affected = sorted(set(isolated_after + disconnected),
                                  key=lambda x: G.degree(x), reverse=True)

            if all_affected:
                st.markdown(f"#### ✈️ Aeroportos incomunicáveis após a remoção ({len(all_affected)})")
                st.caption(
                    "Aeroportos que perderam toda conectividade com o componente principal da rede."
                )
                aff_df = pd.DataFrame([{
                    "IATA": n,
                    "Nome": G.nodes[n].get("name", ""),
                    "Cidade": G.nodes[n].get("city", ""),
                    "País": G.nodes[n].get("country", ""),
                    "Grau original": G.degree(n),
                } for n in all_affected])
                # Altura dinâmica: 35px por linha, máximo de 400px.
                st.dataframe(aff_df, use_container_width=True,
                             height=min(400, 35 * len(aff_df) + 38))
            else:
                st.success("Nenhum aeroporto ficou completamente incomunicável.")

            # Parágrafo de interpretação automática dos resultados.
            st.markdown(
                f"> **Interpretação:** remover **{len(removed)}** aeroporto(s) com grau médio de "
                f"**{df_rem['Grau'].mean():.0f} rotas** cada causou a perda de "
                f"**{result['edges_lost_pct']:.1f}%** de todas as rotas, desconectou "
                f"**{len(all_affected)}** aeroportos do componente principal e reduziu o maior "
                f"componente conexo em **{result['wcc_reduction_pct']:.1f}%** — de "
                f"{result['original_largest_wcc']:,} para {result['new_largest_wcc']:,} aeroportos."
            )

    with tab2:
        st.subheader("Curva de robustez")
        st.markdown("""
A **curva de robustez** mede como a rede se fragmenta à medida que nós são removidos progressivamente.
O eixo Y mostra a fração do maior componente conexo restante (1.0 = rede intacta, 0.0 = rede completamente fragmentada).

Três estratégias são comparadas:
- 🔴 **Ataque por Grau** - remove os hubs com mais conexões primeiro. Simula um ataque direcionado ou crise nos maiores aeroportos.
- 🟡 **Ataque por Betweenness** - remove os nós com maior betweenness (intermediadores críticos). Simula bloqueio de rotas de passagem.
- 🔵 **Falha Aleatória** - remove nós aleatoriamente. Simula falhas naturais, pandemias, desastres localizados.

Uma rede **livre de escala** como a malha aérea é resistente a falhas aleatórias mas colapsa rapidamente sob ataque direcionado - o que você verá no gráfico.
        """)

        col_strat, col_steps = st.columns([2, 1])
        with col_strat:
            # Usuario escolhe quais estratégias incluir na simulação.
            strategies = st.multiselect(
                "Estratégias",
                ["degree", "betweenness", "random"],
                default=["degree", "betweenness", "random"],
            )
        with col_steps:
            # Mais passos = curva mais suave, mas demora mais pra calcular.
            steps = st.slider("Passos da simulação", 10, 50, 25)

        # Guarda a figura e os AUCs no session_state pra não recalcular a cada rerun.
        if "robustness_fig" not in st.session_state:
            st.session_state.robustness_fig = None
        if "robustness_auc" not in st.session_state:
            st.session_state.robustness_auc = {}

        if strategies and st.button("▶ Calcular Robustez", type="primary"):
            fig = go.Figure()
            colors = {"degree": "#ff4444", "betweenness": "#ffd700", "random": "#58a6ff"}
            names = {"degree": "🔴 Ataque por Grau", "betweenness": "🟡 Ataque por Betweenness", "random": "🔵 Falha Aleatória"}
            auc_vals = {}

            for strat in strategies:
                with st.spinner(f"Simulando: {names[strat]}..."):
                    # Roda a simulação: remove nós progressivamente e mede o tamanho do maior componente.
                    curve = robustness_curve(G, strategy=strat, steps=steps)

                xs = [p[0] for p in curve]  # fração de nós removidos
                ys = [p[1] for p in curve]  # fração do maior componente restante

                # Calcula a área sob a curva (AUC) usando a regra do trapézio.
                # AUC alto = rede aguenta bem. AUC baixo = rede colapsa rápido.
                auc = sum((xs[i] - xs[i-1]) * (ys[i] + ys[i-1]) / 2 for i in range(1, len(xs)))
                auc_vals[strat] = auc

                # Linha com preenchimento abaixo pra visualizar a área que representa a AUC.
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    name=names[strat],
                    mode="lines+markers",
                    line=dict(color=colors[strat], width=2.5),
                    fill="tozeroy",
                    fillcolor={
                        "degree": "rgba(255,68,68,0.07)",
                        "betweenness": "rgba(255,215,0,0.07)",
                        "random": "rgba(88,166,255,0.07)",
                    }[strat],
                ))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                title="Curva de robustez da malha aérea mundial",
                title_font_color="#f0f6fc",
                xaxis_title="Fração de nós removidos",
                yaxis_title="Fração da maior componente",
                yaxis=dict(range=[0, 1.05]),
                legend=dict(bgcolor="#161b22", x=0.65, y=0.95),
                margin=dict(t=50),
            )
            st.session_state.robustness_fig = fig
            st.session_state.robustness_auc = auc_vals

        if st.session_state.robustness_fig:
            st.plotly_chart(st.session_state.robustness_fig, use_container_width=True)

            auc = st.session_state.robustness_auc
            if auc:
                st.markdown("#### Índice de robustez (área sob a curva)")
                auc_cols = st.columns(len(auc))
                labels_auc = {"degree": "Ataque por Grau", "betweenness": "Ataque por Betweenness", "random": "Falha Aleatória"}
                colors_auc = {"degree": "🔴", "betweenness": "🟡", "random": "🔵"}
                # Ordena do menos robusto pro mais robusto (AUC crescente).
                for col, (strat, val) in zip(auc_cols, sorted(auc.items(), key=lambda x: x[1])):
                    col.metric(f"{colors_auc[strat]} {labels_auc[strat]}", f"{val:.3f}")

            st.markdown("""
> **Como ler o gráfico:**
> - Curva que cai mais rápido = rede mais vulnerável àquela estratégia
> - A diferença entre 🔴 Ataque por Grau e 🔵 Falha Aleatória é a "assinatura" de redes livre de escala
> - **Índice de robustez** (AUC): quanto maior, mais a rede aguenta aquela estratégia de ataque - valores próximos de 0.5 indicam rede robusta, próximos de 0 indicam colapso rápido
            """)