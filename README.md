# Grafos Aplicados à Malha Aérea Mundial

Trabalho Final | ICC041 Introdução à Teoria dos Grafos, 2026/01
UFAM, Instituto de Computação | Profa. Rosiane de Freitas

Aplicação Streamlit que modela a malha aérea comercial mundial como um grafo dirigido ponderado e aplica algoritmos de teoria dos grafos para análise de centralidade, conectividade e vulnerabilidade da rede.

**Grafo final:** 3.257 aeroportos (vértices) e 37.042 rotas diretas (arestas), com peso = distância geodésica em km.

---

## Requisitos

- Python 3.10+
- As dependências estão todas em `requirements.txt`

```
streamlit>=1.35.0
networkx>=3.3
folium>=0.16.0
streamlit-folium>=0.21.0
pandas>=2.1.0
numpy>=1.26.0
plotly>=5.22.0
geopandas>=0.14.0
shapely>=2.0.0
scipy
```

---

## Como rodar

```bash
# 1. Clonar o repositório
git clone https://github.com/leomelo-dev/airport-graph
cd airport-graph

# 2. Criar e ativar um ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Rodar
streamlit run app.py
```

Na primeira execução, a classificação geográfica por point-in-polygon roda sobre os 3.257 aeroportos e leva cerca de 2 minutos. O resultado fica salvo em `data/_geo_cache.pkl` e é reutilizado nas execuções seguintes. Todas as outras métricas pesadas (betweenness, clustering, comprimento médio de caminho) também são cacheadas pelo Streamlit na primeira execução de cada aba.

---

## Estrutura do projeto

```
airports-flights_graph/
├── app.py                          # interface Streamlit, 5 seções de navegação
├── modules/
│   ├── __init__.py
│   ├── data_loader.py              # leitura do OpenFlights, filtros, construção do grafo
│   ├── analysis.py                 # todos os algoritmos: centralidade, conectividade, Dijkstra, robustez
│   ├── geo.py                      # classificação geográfica por point-in-polygon
│   └── map_viz.py                  # visualização Folium: arcos geodésicos, layers de centralidade
├── data/
│   ├── airports.dat                # 7.698 aeroportos brutos (OpenFlights)
│   ├── routes.dat                  # 67.663 rotas brutas (OpenFlights)
│   ├── countries.geojson           # Natural Earth: polígonos de países para classificar continente
│   └── brazil_states.geojson       # IBGE: estados brasileiros para classificar região
└── requirements.txt
```

---

## Dataset e pipeline de filtragem

Os dados vêm do [OpenFlights](https://openflights.org/data.html), um dataset público com aeroportos e rotas comerciais ao redor do mundo. Os dois arquivos relevantes são `airports.dat` e `routes.dat`, ambos sem cabeçalho e com campos separados por vírgula.

**airports.dat** tem 7.698 registros com: `airport_id, name, city, country, iata, icao, lat, lon, altitude, timezone, dst, tz, type, source`. Campos nulos usam `\N`.

**routes.dat** tem 67.663 registros com: `airline, airline_id, src_iata, src_id, dst_iata, dst_id, codeshare, stops, equipment`.

O pipeline de filtragem aplicado em `data_loader.py`:

1. **Filtro de IATA**: mantém apenas aeroportos com código IATA de exatamente 3 letras e com coordenadas lat/lon válidas. Isso exclui bases militares, helipads e pistas privadas que existem no dataset mas não operam voos comerciais regulares. Remove aproximadamente 1.600 registros.

2. **Filtro de rotas diretas**: mantém apenas linhas com `stops == 0`. Rotas com escala não representam uma conexão direta entre origem e destino e não deveriam ser modeladas como aresta direta no grafo. Algumas rotas também usam ID numérico em vez de código IATA nos campos de origem/destino; essas são descartadas porque o lookup pelo nó quebraria.

3. **Remoção de nós isolados**: depois de construir o grafo, aeroportos com grau total zero são removidos. Isso acontece quando um aeroporto sobrevive ao filtro de IATA mas nenhuma de suas rotas sobreviveu ao filtro anterior (ex: base militar com código IATA válido mas só rotas não-comerciais). Sem essa etapa o grafo teria centenas de nós isolados que distorceriam métricas como densidade e grau médio.

Resultado final: **3.257 aeroportos** e **37.042 rotas diretas**.

---

## Modelo formal

O grafo é modelado como um **dígrafo ponderado** G = (V, E, w), onde:

- **V** = conjunto de aeroportos (vértices), indexados pelo código IATA
- **E ⊆ V × V** = conjunto de rotas diretas (arestas dirigidas)
- **w(u, v)** = distância geodésica entre u e v em km, calculada pela fórmula de Haversine

O grafo é **dirigido** porque a existência de GRU→JFK não garante JFK→GRU. Na prática a maioria das rotas é bidirecional, mas o dataset registra cada direção separadamente e cerca de 41 aeroportos têm cobertura assimétrica (chegam voos mas não saem, ou vice-versa). O dígrafo captura isso corretamente.

O peso usa **Haversine** em vez de distância euclidiana porque a Terra é esférica e a diferença é relevante em rotas longas. A fórmula calcula a distância sobre a superfície esférica a partir de lat/lon em radianos, usando o raio médio da Terra de 6.371 km.

Para análises que requerem grafo não-dirigido (pontes, pontos de articulação, coeficiente de clustering), o código usa `G.to_undirected()` localmente dentro de cada função de análise.

---

## Arquitetura do código

### `data_loader.py`

Responsável por toda a fase de carregamento e construção do grafo. Expõe uma única função pública, `get_data()`, que retorna `(airports_df, G)` e mantém um cache em memória no nível do módulo para não reconstruir o grafo a cada chamada.

Funções internas:
- `load_airports()`: lê airports.dat, aplica os filtros de IATA e coordenadas, retorna DataFrame indexado por IATA
- `load_routes()`: lê routes.dat, filtra stops=0 e IATAs válidos
- `haversine(lat1, lon1, lat2, lon2)`: calcula distância geodésica em km
- `build_graph(airports, routes)`: constrói o DiGraph, adiciona nós com atributos geográficos, adiciona arestas com peso Haversine, remove isolados

### `analysis.py`

Contém todos os algoritmos de análise, organizados em quatro grupos:

**Centralidade**

Seis métricas implementadas, todas retornando `dict {iata: score}`:

- `compute_degree_centrality(G)`: grau normalizado `deg(v) / (|V|-1)`, via `nx.degree_centrality`
- `compute_in_degree_centrality(G)` e `compute_out_degree_centrality(G)`: separação de rotas que chegam vs. que saem
- `compute_betweenness_centrality(G, k=500)`: fração dos caminhos mínimos que passam por cada nó. O cálculo exato é O(V·E) e levaria aproximadamente 7 minutos neste grafo; com `k=500` amostras aleatórias a estimativa roda em cerca de 8 segundos com erro O(1/√k). O parâmetro `weight="weight"` faz o algoritmo usar distância km como custo, então "caminho mínimo" significa o de menor distância total, não o de menos escalas
- `compute_closeness_centrality(G)`: `(|V|-1) / Σ d(v,u)`. Usa `distance="weight"` para medir em km
- `compute_pagerank(G, alpha=0.85)`: probabilidade de um passeio aleatório longo chegar ao nó. `alpha=0.85` é o fator de amortecimento padrão da literatura. `max_iter=200` garante convergência

Função auxiliar `top_nodes(metric, n)` ordena e retorna os n melhores.

**Conectividade**

- `get_weakly_connected_components(G)`: componentes onde existe caminho entre qualquer par ignorando direção
- `get_strongly_connected_components(G)`: componentes onde existe caminho de ida **e** volta entre qualquer par
- `get_bridges(G)`: arestas cuja remoção aumenta o número de componentes. Roda no grafo não-dirigido usando DFS com valores low-link
- `get_articulation_points(G)`: vértices cuja remoção desconecta o grafo. Mesmo algoritmo DFS
- `connectivity_summary(G)`: agrega todas as métricas acima num dict, inclui densidade, grau médio, máximo e mediano

**Distribuição de grau e propriedades de rede complexa**

- `degree_distribution(G)`: calcula P(k) para cada grau k e ajusta lei de potência P(k) ~ k^{-α} por regressão linear no espaço log-log. Retorna o expoente α e o R² do ajuste
- `average_clustering_coefficient(G)`: coeficiente de Watts-Strogatz médio, no grafo não-dirigido
- `average_shortest_path_sample(G, sample=500)`: estima L̄ em hops por amostragem de 500 pares aleatórios dentro do maior WCC. Mede hops (sem peso) porque o critério small-world é L̄ ~ O(log N) em número de saltos. Restrito ao maior WCC para não descartar pares sem caminho e distorcer a média

**Caminho mínimo e roteamento**

- `shortest_path(G, src, dst)`: Dijkstra direto, retorna `(path, dist_km)` ou `(None, None)` se inalcançável
- `shortest_path_avoiding(G, src, dst, avoid_nodes, avoid_edges)`: cria cópia do grafo, remove os nós/arestas proibidos (exceto src e dst), roda Dijkstra no subgrafo. Simula roteamento alternativo quando aeroportos estão fechados
- `path_edges(path)` e `path_total_distance(G, path)`: helpers para processar o resultado do Dijkstra

**Vulnerabilidade**

- `simulate_node_removal(G, nodes_to_remove)`: remove os nós em cópia do grafo, retorna dict com rotas perdidas (%), redução no maior WCC (%) e grau médio restante
- `robustness_curve(G, strategy, steps)`: remove nós iterativamente em `steps` batches e registra o tamanho do maior WCC a cada passo. O ranking é recalculado no subgrafo restante a cada iteração (não no grafo original), porque quando um hub cai o próximo hub crítico muda. Estratégias: `'degree'` (ataque por grau), `'betweenness'` (intermediadores críticos, k=100 por rodar `steps` vezes), `'random'` (falha aleatória). Retorna lista de `(fração_removida, fração_maior_WCC)`.

### `geo.py`

Classifica cada aeroporto por continente, país e (para aeroportos brasileiros) região, usando point-in-polygon sobre dois datasets geoespaciais:

- **countries.geojson** (Natural Earth): polígonos de países com atributo `CONTINENT` em inglês e `SUBREGION`. O código mapeia os nomes para português e usa `SUBREGION` para separar "América Central e Caribe" de "América do Norte", já que o Natural Earth não distingue os dois no campo `CONTINENT`.
- **brazil_states.geojson** (IBGE): polígonos dos estados brasileiros com campo `regiao_id` (1=Sul, 2=Sudeste, 3=Norte, 4=Nordeste, 5=Centro-Oeste).

A função `_point_in_gdf(lon, lat, gdf, col)` itera pelos polígonos e retorna o campo `col` do primeiro que contém o ponto. Se nenhum contiver (aeroportos em ilhas pequenas ou muito próximos de fronteiras), usa o polígono mais próximo por distância geométrica como fallback.

A operação completa leva aproximadamente 2 minutos. O resultado é cacheado em `data/_geo_cache.pkl` com chave `"{n_nodes}_{n_edges}"`. Se o grafo não mudar, o cache é reutilizado sem recalcular.

O retorno é um dict com:
- `continent`: `{nome_continente: [lista de IATAs]}`
- `country`: `{nome_pais: [lista de IATAs]}`
- `br_region`: `{nome_regiao: [lista de IATAs]}`
- `node_continent`: `{iata: nome_continente}` (lookup O(1) por nó)
- `node_br_region`: `{iata: nome_regiao}` (só para aeroportos brasileiros)

### `map_viz.py`

Constrói os mapas Folium usados em todas as abas da aplicação.

`build_base_map(center, zoom)`: cria o mapa com tema CartoDB Dark Matter. Usa `prefer_canvas=True` para trocar o renderer SVG por Canvas; com 37.000 arestas o SVG fica lento demais. `no_wrap=True` no TileLayer evita que o mapa se repita horizontalmente ao dar zoom out.

`_arc_points(lat1, lon1, lat2, lon2, n=30)`: interpola n+1 pontos ao longo do arco de grande círculo entre dois pontos usando interpolação esférica (SLERP). Voos reais seguem geodésicas esféricas, não linhas retas no plano 2D; a diferença é visível em rotas longas como transpacíficas. O algoritmo converte as coordenadas para radianos, calcula a distância angular `d` entre os pontos usando Haversine, e para cada fração `f = i/n` do caminho interpola em coordenadas cartesianas 3D com os pesos `A = sin((1-f)*d)/sin(d)` e `B = sin(f*d)/sin(d)`, convertendo de volta para lat/lon no final. `n=30` dá uma curva suave sem encher o DOM de pontos.

`_split_antimeridian(points)`: rotas transpacíficas cruzam a linha de data internacional (antimeridiano, ±180°). Sem tratamento, a longitude salta de +180 para -180 e o Folium desenha uma linha horizontal atravessando o mapa inteiro. A função detecta saltos de longitude maiores que 180°, interpola o ponto exato de cruzamento na borda ±180° e divide o segmento em dois. Cada segmento é desenhado separadamente como uma `PolyLine` no Folium, eliminando a linha indesejada.

`add_airports_scaled(m, G, nodes, highlight)`: plota cada aeroporto com raio e cor proporcionais ao grau (2px, azul frio para pequenos; 11px, ciano brilhante para hubs). Nós em `highlight` aparecem em laranja com raio fixo de 8px.

`add_background_routes(m, G)`: desenha todas as rotas como fundo. Usa **linhas retas**, não arcos, porque com 37k arestas e arcos de 30 pontos cada seriam 1,1 milhão de pontos no DOM. Deduplica pares A→B e B→A para não desenhar linhas sobrepostas.

`add_route(m, G, path)`: desenha uma rota específica (resultado do Dijkstra) como arcos geodésicos com tratamento do antimeridiano.

`add_centrality_layer(m, G, centrality, top_n)`: plota os top_n aeroportos mais centrais com tamanho (4px a 22px) e cor (azul/roxo a vermelho) proporcionais ao score relativo dentro do top-N.

`add_removed_nodes(m, G, nodes)`: marca aeroportos removidos na simulação com ícone de X vermelho.

### `app.py`

Interface Streamlit com 5 seções de navegação na sidebar. O pré-cálculo pesado roda uma vez ao iniciar via `_precompute()` com `@st.cache_data`, que calcula e armazena: bridges, articulation points, degree/in-degree/out-degree/pagerank centrality, connectivity summary, degree distribution, clustering coefficient, comprimento médio de caminho, e classificação geográfica. Betweenness e closeness são calculados sob demanda na aba de Centralidade porque são mais lentos.

O app usa três níveis de cache:
- `@st.cache_resource`: para o grafo e para mapas Folium pesados (são objetos Python grandes e imutáveis; ficam na memória entre reruns)
- `@st.cache_data`: para métricas calculadas e DataFrames (são serializados e comparados por hash dos argumentos)
- `pickle` em disco: para a classificação geográfica (persiste entre sessões)

---

## Funcionalidades

### Visão Geral

Exibe métricas globais da rede (aeroportos, rotas, densidade, grau médio) e as propriedades de rede complexa:

- **Distribuição de grau (log-log)**: scatter plot de P(k) vs k em escala log-log com reta de ajuste de lei de potência. Uma reta nesse espaço confirma P(k) ~ k^{-α}. O expoente α ≈ 1,06 e o R² = 0,75 estão exibidos.
- **Small-world**: coeficiente de clustering médio C = 0,49 e comprimento médio de caminho L̄ ≈ 3,97 hops.

O mapa principal é filtrável por:
- Rede completa (todos os 3.257 aeroportos e 37.042 rotas)
- Nó único e suas conexões diretas (vizinhos de entrada e saída)
- Continente
- País
- Região brasileira (Norte, Nordeste, Centro-Oeste, Sudeste, Sul)

Abaixo do mapa, um subgráfico do filtro ativo exibe distribuição de grau, top-15 hubs por grau, top-15 por PageRank e histograma de grau.

### Centralidade

Seletor de métrica (Degree, In-Degree, Out-Degree, Betweenness, Closeness, PageRank) e slider de top-N (10 a 100). Exibe mapa com a camada de centralidade e tabela com IATA, cidade, país e score. Inclui gráfico de barras do top-20.

Betweenness e Closeness são calculados sob demanda com `@st.cache_data` por serem mais lentos que as outras métricas.

### Conectividade

Filtro por país no topo. Métricas exibidas: número de WCC e SCC, tamanho do maior de cada um, número de pontes e pontos de articulação.

Três tabs:
- **Pontes**: tabela com as rotas críticas classificando cada aeroporto como "hub" (maior grau na ponte) ou "dependente" (menor grau). Mapa com hubs em vermelho, dependentes em ciano e as arestas de ponte em vermelho.
- **Pontos de articulação**: tabela ordenada por grau, mapa com articulações em laranja e seus vizinhos dependentes em roxo.
- **Distribuição SCC**: gráfico de barras com o tamanho dos 30 maiores componentes fortemente conexos.

### Rotas e Caminhos

Seletores de origem e destino (todos os aeroportos da rede, com label "IATA: Cidade, País"). Campo de texto para aeroportos a evitar (separados por vírgula).

Ao clicar em "Calcular Rota", o resultado é salvo em `st.session_state` para não sumir no próximo rerun. O mapa exibe as rotas sobre o fundo completo de rotas; a rota ótima aparece em vermelho, escalas intermediárias em verde, origem/destino em laranja; a rota alternativa (se informados aeroportos a evitar) aparece em laranja mais claro. Rotas que cruzam o antimeridiano exibem um aviso explicando a quebra visual.

Abaixo do mapa, tabela com cada trecho da rota e sua distância em km.

### Vulnerabilidade

Duas tabs:

**Simulação de falha**: seletor de aeroportos a remover (multiselect manual ou seleção automática por top-N de uma métrica). Ao confirmar, exibe métricas de impacto (aeroportos removidos, rotas perdidas, variação no maior WCC), gráfico de barras antes/depois, gráfico de grau dos aeroportos removidos, mapa com rotas perdidas em vermelho e rotas restantes em azul escuro, e tabela de aeroportos que ficaram fora do maior componente conexo após a remoção.

**Curva de robustez**: seletor de estratégias (degree, betweenness, random) e número de passos (10 a 50). Plota as curvas de fração do maior WCC em função da fração de nós removidos, com área preenchida. Exibe o AUC (área sob a curva) de cada estratégia como índice de robustez.

---

## Detalhes de implementação

### Distribuição de grau e ajuste de lei de potência

O histograma normalizado P(k) é calculado a partir da lista de graus de todos os nós. Para ajustar a lei de potência P(k) ~ k^{-α}, o código trabalha no espaço log-log: uma lei de potência vira uma reta nesse espaço, então o coeficiente angular da reta é -α. O ajuste é feito por regressão linear simples via `numpy.polyfit`. Pontos com k=0 ou P(k)=0 são excluídos antes de tomar o log porque log(0) é indefinido.

O R² é calculado como `1 - SS_res / SS_tot` no espaço log-log, onde SS_res é a soma dos quadrados dos resíduos e SS_tot é a variância total dos valores de log(P(k)). Um R² próximo de 1 indica que a distribuição segue bem uma lei de potência.

```python
def degree_distribution(G):
    degrees = [d for _, d in G.degree()]
    unique_k, counts = np.unique(degrees, return_counts=True)
    pk = counts / counts.sum()
    mask = (unique_k > 0) & (pk > 0)
    log_k  = np.log10(unique_k[mask].astype(float))
    log_pk = np.log10(pk[mask])
    coeffs = np.polyfit(log_k, log_pk, 1)
    alpha  = -coeffs[0]
    fitted = np.polyval(coeffs, log_k)
    ss_res = np.sum((log_pk - fitted)**2)
    ss_tot = np.sum((log_pk - log_pk.mean())**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    return {"bins": unique_k.tolist(), "pk": pk.tolist(),
            "alpha": float(alpha), "r_squared": float(r2)}
```

### Comprimento médio de caminho por amostragem

O cálculo exato do comprimento médio de caminho requer rodar BFS ou Dijkstra de todos os |V| nós, o que dá O(|V|²) e seria inviável para 3.257 nós. A estimativa por amostragem seleciona 500 pares aleatórios dentro do maior WCC e calcula o comprimento em hops (sem peso) para cada um. Restringir ao maior WCC é importante: pares em componentes diferentes não têm caminho, e incluí-los como "infinito" ou simplesmente descartá-los distorceria a média dependendo de quantos existem.

A métrica usa hops, não km, porque o critério small-world de Watts e Strogatz é definido em número de saltos: L̄ ~ O(log N) para uma rede com N nós. Para N = 3.257, O(log N) ≈ 3,8 em base 10, e o valor medido foi L̄ ≈ 3,97.

```python
def average_shortest_path_sample(G, sample=500, seed=42):
    rng  = random.Random(seed)
    wcc  = max(nx.weakly_connected_components(G), key=len)
    subG = G.subgraph(wcc).copy()
    nodes = list(subG.nodes())
    total, count = 0.0, 0
    for _ in range(sample):
        src, dst = rng.sample(nodes, 2)
        try:
            total += nx.shortest_path_length(subG, src, dst)
            count += 1
        except nx.NetworkXNoPath:
            pass
    return total / count if count else 0.0
```

### Roteamento alternativo com aeroportos bloqueados

A função `shortest_path_avoiding` cria uma cópia do grafo, remove os nós e arestas proibidos e roda Dijkstra no subgrafo resultante. A cópia é necessária para não modificar o grafo original, que é compartilhado entre todas as abas da aplicação. Os nós `src` e `dst` nunca são removidos, mesmo que apareçam na lista de bloqueados, porque sem eles não haveria rota possível.

```python
def shortest_path_avoiding(G, src, dst,
                            avoid_nodes=None, avoid_edges=None,
                            weight="weight"):
    H = G.copy()
    if avoid_nodes:
        H.remove_nodes_from([
            n for n in avoid_nodes
            if n in H and n != src and n != dst
        ])
    if avoid_edges:
        H.remove_edges_from([e for e in avoid_edges if H.has_edge(*e)])
    return shortest_path(H, src, dst, weight)
```

### Visualização geográfica: arcos de grande círculo

Voos reais não seguem linhas retas no mapa plano; eles seguem geodésicas esféricas (arcos de grande círculo). A diferença é visível principalmente em rotas longas: um voo de São Paulo para Tóquio no mapa plano parece desviar "para cima" em direção ao Polo Norte porque esse arco é o caminho mais curto na superfície da Terra.

A função `_arc_points` interpola 31 pontos (n=30 intervalos) ao longo do arco usando interpolação esférica linear (SLERP). O algoritmo:

1. Converte lat/lon de graus para radianos
2. Calcula a distância angular `d` entre os dois pontos com Haversine
3. Para cada passo `i` de 0 a n, calcula a fração `f = i/n` do caminho e os pesos SLERP: `A = sin((1-f)*d)/sin(d)` e `B = sin(f*d)/sin(d)`
4. Calcula as coordenadas cartesianas 3D do ponto interpolado: `x`, `y`, `z`
5. Converte de volta para lat/lon com `atan2`

```python
def _arc_points(lat1, lon1, lat2, lon2, n=30):
    la1, lo1 = math.radians(lat1), math.radians(lon1)
    la2, lo2 = math.radians(lat2), math.radians(lon2)
    d = 2 * math.asin(math.sqrt(
        math.sin((la2 - la1) / 2)**2 +
        math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2)**2
    ))
    if d < 1e-10:
        return [(lat1, lon1), (lat2, lon2)]
    points = []
    for i in range(n + 1):
        f = i / n
        A = math.sin((1 - f) * d) / math.sin(d)
        B = math.sin(f * d) / math.sin(d)
        x = A*math.cos(la1)*math.cos(lo1) + B*math.cos(la2)*math.cos(lo2)
        y = A*math.cos(la1)*math.sin(lo1) + B*math.cos(la2)*math.sin(lo2)
        z = A*math.sin(la1) + B*math.sin(la2)
        lat = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))
        lon = math.degrees(math.atan2(y, x))
        points.append((lat, lon))
    return points
```

O fundo do mapa com todas as 37.042 rotas usa linhas retas em vez de arcos por uma razão prática: arcos de 30 pontos cada dariam 1,1 milhão de pontos no DOM, o que travaria qualquer browser. Para o fundo, a perda de precisão visual é aceitável porque o objetivo é mostrar a densidade de rotas, não cada trajetória com precisão. O `prefer_canvas=True` no Folium troca o renderer SVG por Canvas, que suporta muito mais elementos simultâneos.

### Tratamento do antimeridiano

Rotas que cruzam o antimeridiano (a linha dos ±180°, que passa pelo Pacífico) têm um problema no Folium: a longitude salta de +180 para -180 de um ponto para o próximo, e o Folium interpreta isso como uma linha horizontal atravessando o mundo inteiro.

A solução é detectar o salto (diferença de longitude maior que 180°), calcular onde exatamente a rota cruza a borda e dividir o segmento em dois. Cada pedaço vai até a borda ±180° e começa no lado oposto. O Folium então desenha cada pedaço separadamente e o resultado visual fica correto.

```python
def _split_antimeridian(points):
    segments, current = [], [points[0]]
    for i in range(1, len(points)):
        prev_lon = points[i-1][1]
        curr_lon = points[i][1]
        if abs(curr_lon - prev_lon) > 180:
            frac = (180 - abs(prev_lon)) / abs(curr_lon - prev_lon)
            lat_c = points[i-1][0] + frac * (points[i][0] - points[i-1][0])
            border = 180.0 if prev_lon > 0 else -180.0
            current.append((lat_c, border))
            segments.append(current)
            current = [(lat_c, -border), points[i]]
        else:
            current.append(points[i])
    segments.append(current)
    return segments
```

### Dijkstra e suas limitações no contexto aéreo

O Dijkstra implementado minimiza a distância física total em km (peso Haversine), produzindo o caminho geograficamente mais curto entre dois aeroportos dentro das rotas disponíveis no dataset. Isso nem sempre coincide com o itinerário operacional por três razões:

**Frequência de voos não considerada.** Uma rota operada duas vezes por semana tem o mesmo peso que uma com 20 voos diários. Conexões raras ou sazonais aparecem como opções igualmente válidas, o que pode gerar itinerários que existem no papel mas não têm voos conectando nos horários certos.

**Tempo de espera em escalas ignorado.** Um caminho com três escalas curtas pode ser menor em km mas impraticável se as conexões não coincidem. O modelo não sabe quanto tempo um passageiro espera em cada aeroporto intermediário.

**Padrões operacionais comerciais ausentes.** Companhias aéreas estruturam hubs por razões comerciais, não geométricas. Manaus se conecta ao mundo principalmente via São Paulo porque é o hub doméstico dominante com mais frequências e conexões. O modelo geodésico escolhe Fortaleza para rotas transatlânticas porque está 8° mais a leste, encurtando o trecho sobre o Atlântico, mesmo que na prática não exista itinerário comercial combinando aquelas rotas como conexões contínuas.

Para construir um modelo mais fiel ao roteamento real seriam necessários: grafos temporais com horários de voo, pesos multidimensionais (distância + tempo de espera + frequência de voos) e dados de capacidade de assentos por trecho.

---

## Decisões de implementação relevantes

**Betweenness por amostragem (k=500)**: o cálculo exato requer O(V·E) operações e leva aproximadamente 7 minutos no grafo completo. Com k=500 pares de nós como fonte, a estimativa tem variância O(1/k) e roda em cerca de 8 segundos. O ranking dos top aeroportos fica estável com esse valor de k.

**Robustez adaptativa**: em `robustness_curve`, o ranking de nós é recalculado a cada iteração no subgrafo restante, não no grafo original. Quando FRA é removido, o próximo nó mais crítico não é necessariamente o segundo no ranking original do grafo completo. Isso modela um ataque real mais fielmente do que usar o ranking fixo do início.

**Cache em três níveis**: `@st.cache_resource` para o grafo e mapas pesados (objetos Python grandes que não precisam ser rehashados), `@st.cache_data` para métricas (serializados por hash dos argumentos de entrada), e pickle em disco para a classificação geográfica (persiste entre sessões e reinicializações).

**Arcos vs. linhas no mapa**: rotas de caminho mínimo são desenhadas como arcos geodésicos (interpolação SLERP, 30 pontos por segmento) para fidelidade visual. O fundo do mapa com todas as rotas usa linhas retas porque 37.000 arestas × 30 pontos = 1,1 milhão de pontos no DOM. `prefer_canvas=True` no Folium troca o renderer SVG por Canvas, que suporta muito mais elementos.

**Grafo dirigido**: mesmo que a maioria das rotas seja bidirecional no OpenFlights, aproximadamente 41 aeroportos têm cobertura assimétrica. O dígrafo captura essa assimetria. Análises que precisam do grafo não-dirigido (pontes, articulações, clustering) convertem localmente com `G.to_undirected()`.

---

## Dataset

[OpenFlights](https://openflights.org/data.html), dados públicos de aeroportos e rotas aéreas.

Os dados são de aproximadamente 2014 e podem não refletir a malha atual com precisão. Algumas rotas existem no dataset mas não são mais operadas; algumas rotas novas não estão incluídas. Para os fins de análise estrutural da rede, isso não compromete as conclusões sobre propriedades topológicas.

---

## Uso de IA

Modalidade **III** e **IV**: IA usada para sugestões de arquitetura, refatoração e revisão de código, com análise crítica e parcialmente para geração de código (trechos feitos por IA estão sinalizados e explicados no código).
