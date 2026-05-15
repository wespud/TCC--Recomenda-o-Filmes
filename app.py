# app.py — Protótipo de Feed de Recomendações (versão completa — melhorias 1 a 5)
# TCC: Análise e Funcionamento de Algoritmos de Recomendação

import pandas as pd
import numpy as np
import requests
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
import altair as alt
import streamlit as st

# ══════════════════════════════════════════════════════════════
# CONFIGURAÇÕES GLOBAIS
# ══════════════════════════════════════════════════════════════

DATA_PATH = './ml-1m/ml-1m/'
SEED      = 42
np.random.seed(SEED)

# Cole aqui sua chave gratuita do TMDB
# Obtenha em: https://www.themoviedb.org → Configurações → API
TMDB_API_KEY  = ""  # Adicione sua chave TMDB aqui
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w185"

# Dicionário de tradução EN → PT
GENRE_TRANSLATION = {
    'unknown':     'Desconhecido',
    'Action':      'Ação',
    'Adventure':   'Aventura',
    'Animation':   'Animação',
    'Children':    'Infantil',
    'Comedy':      'Comédia',
    'Crime':       'Crime',
    'Documentary': 'Documentário',
    'Drama':       'Drama',
    'Fantasy':     'Fantasia',
    'Film-Noir':   'Film Noir',
    'Horror':      'Terror',
    'Musical':     'Musical',
    'Mystery':     'Mistério',
    'Romance':     'Romance',
    'Sci-Fi':      'Ficção Científica',
    'Thriller':    'Suspense',
    'War':         'Guerra',
    'Western':     'Faroeste',
}

# Lista de todos os gêneros em PT (usada em multiselect)
GENRES_PT = sorted(GENRE_TRANSLATION.values())

# Gêneros bloqueados para menores de 18
ADULT_ONLY_GENRES = {'Horror', 'Crime', 'Film-Noir', 'Thriller'}

# Mapeamento de ocupação (u.user) para português
OCCUPATION_PT = {
    'administrator': 'Administrador',  'artist': 'Artista',
    'doctor':        'Médico',         'educator': 'Educador',
    'engineer':      'Engenheiro',     'entertainment': 'Entretenimento',
    'executive':     'Executivo',      'healthcare': 'Saúde',
    'homemaker':     'Dona de casa',   'lawyer': 'Advogado',
    'librarian':     'Bibliotecário',  'marketing': 'Marketing',
    'none':          'Sem ocupação',   'other': 'Outro',
    'programmer':    'Programador',    'retired': 'Aposentado',
    'salesman':      'Vendedor',       'scientist': 'Cientista',
    'student':       'Estudante',      'technician': 'Técnico',
    'writer':        'Escritor',
}

# Ocupações numéricas do ML-1M → chaves do OCCUPATION_PT
ML1M_OCCUPATION = {
    0: 'other',         1: 'educator',    2: 'artist',       3: 'administrator',
    4: 'student',       5: 'other',       6: 'doctor',       7: 'executive',
    8: 'other',         9: 'homemaker',  10: 'student',      11: 'lawyer',
   12: 'programmer',   13: 'retired',    14: 'salesman',     15: 'scientist',
   16: 'other',        17: 'technician', 18: 'other',        19: 'none',
   20: 'writer',
}


# ══════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════

def traduzir_generos(genres_str: str) -> str:
    """Converte 'Action Comedy Drama' → 'Ação · Comédia · Drama'."""
    if not genres_str or not isinstance(genres_str, str):
        return "Sem gênero"
    termos = genres_str.split()
    traduzidos = [GENRE_TRANSLATION.get(t, t) for t in termos]
    return ' · '.join(traduzidos)


def get_user_age(user_id, users_df) -> int | None:
    """Retorna a idade do usuário (dataset ou criado na sessão), ou None se desconhecida."""
    novos = st.session_state.get('novos_usuarios', {})
    if user_id in novos:
        return novos[user_id].get('idade')
    row = users_df[users_df['user_id'] == user_id]
    if not row.empty:
        return int(row.iloc[0]['age'])
    return None


def filtrar_por_idade(rec_df: pd.DataFrame, user_age: int | None) -> pd.DataFrame:
    """Remove Horror/Crime/Film-Noir/Thriller para usuários menores de 18 anos."""
    if user_age is None or user_age >= 18 or rec_df.empty:
        return rec_df

    def inapropriado(genres_str):
        return bool(set(genres_str.split()) & ADULT_ONLY_GENRES)

    return rec_df[~rec_df['genres_str'].apply(inapropriado)].reset_index(drop=True)


def init_session_state():
    """
    Inicializa todas as chaves do session_state na primeira execução.
    Centralizar aqui evita KeyError em qualquer parte do código.
    """
    defaults = {
        'novos_usuarios':         {},   # {user_id: {nome, idade, genero, genres_pt}}
        'ratings_ampliado':       None,
        'svd_retreinado':         None,
        # Melhoria 5 — estruturas de interação por sessão
        'user_watchlist':         {},   # {user_id: [movie_id, ...]}
        'user_ratings_cache':     {},   # {user_id: {movie_id: nota}}
        # Cache do feed — persiste recomendações entre reruns de botões
        'rec_mode':               None,
        'rec_collab_df':          None,
        'rec_content_df':         None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ══════════════════════════════════════════════════════════════
# CARREGAMENTO DE DADOS
# ══════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    """Lê avaliações (ratings.dat), filmes (movies.dat) e usuários (users.dat) do ML-1M."""
    ratings = pd.read_csv(
        DATA_PATH + 'ratings.dat', sep='::', engine='python',
        names=['user_id', 'movie_id', 'rating', 'timestamp'], encoding='latin-1'
    )
    movies_raw = pd.read_csv(
        DATA_PATH + 'movies.dat', sep='::', engine='python',
        names=['movie_id', 'title', 'genres'], encoding='latin-1'
    )
    movies_raw['genres_str'] = movies_raw['genres'].apply(
        lambda g: g.replace("Children's", 'Children').replace('|', ' ')
    )
    movies = movies_raw[['movie_id', 'title', 'genres_str']].copy()
    users = pd.read_csv(
        DATA_PATH + 'users.dat', sep='::', engine='python',
        names=['user_id', 'gender', 'age', 'occupation', 'zip_code'], encoding='latin-1'
    )
    users['occupation'] = users['occupation'].map(ML1M_OCCUPATION).fillna('other')
    return ratings, movies, users


# ══════════════════════════════════════════════════════════════
# MELHORIA 4 — GÊNEROS FAVORITOS
# ══════════════════════════════════════════════════════════════

def get_user_favorite_genres(user_id, ratings_df, movies_df, top_n=5) -> list:
    """
    Calcula os gêneros mais frequentes entre os filmes bem avaliados
    (nota >= 4) pelo usuário. Retorna lista de strings em português.
    """
    liked_ids = ratings_df[
        (ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)
    ]['movie_id'].tolist()

    if not liked_ids:
        return []

    liked_movies  = movies_df[movies_df['movie_id'].isin(liked_ids)]
    genre_counter = Counter()
    for genres_str in liked_movies['genres_str']:
        for g in genres_str.split():
            genre_counter[g] += 1

    top_genres_en = [g for g, _ in genre_counter.most_common(top_n)]
    return [GENRE_TRANSLATION.get(g, g) for g in top_genres_en]


def get_secondary_genres(user_id, primary_genres_pt: list, movies_df) -> list:
    """
    Retorna gêneros descobertos nas avaliações feitas na sessão (nota >= 4)
    que ainda não estão na lista de gêneros primários do usuário.
    """
    rated_cache = st.session_state['user_ratings_cache'].get(user_id, {})
    if not rated_cache:
        return []

    primary_set = set(primary_genres_pt)
    counter = Counter()
    for movie_id, nota in rated_cache.items():
        if nota >= 4:
            row = movies_df[movies_df['movie_id'] == movie_id]
            if not row.empty:
                for g in row.iloc[0]['genres_str'].split():
                    g_pt = GENRE_TRANSLATION.get(g, g)
                    if g_pt not in primary_set:
                        counter[g_pt] += 1

    return [g for g, _ in counter.most_common()]


def _render_genre_badges(primary: list, secondary: list):
    """Renderiza badges de gênero: primários em cinza-escuro, secundários em vermelho."""
    parts = [f"`{g}`" for g in primary]
    if parts:
        st.markdown(" ".join(parts))

    if secondary:
        badges = " ".join(
            f'<span style="background-color:#4a1515;color:#ff6b6b;'
            f'padding:2px 8px;border-radius:4px;font-family:monospace;'
            f'font-size:0.85em;margin:2px 1px;display:inline-block">{g}</span>'
            for g in secondary
        )
        st.markdown(badges, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MELHORIA 4 — CARD DE PERFIL DO USUÁRIO
# ══════════════════════════════════════════════════════════════

def exibir_perfil(selected_user, users_df, ratings_df, movies_df):
    """
    Exibe o card de perfil completo do usuário selecionado.
    - Usuários do dataset: idade, gênero e ocupação do users.dat.
    - Novo usuário: dados do session_state.
    Também exibe watchlist e avaliações feitas na sessão (Melhoria 5).
    """
    with st.container(border=True):
        st.markdown("#### 👤 Perfil do Usuário")

        novos_usuarios = st.session_state['novos_usuarios']
        eh_novo = selected_user in novos_usuarios

        if eh_novo:
            # ── Novo usuário criado na sessão ──
            dados     = novos_usuarios[selected_user]
            nome      = dados.get('nome', '?')
            idade     = dados.get('idade')
            genero    = dados.get('genero', '?')
            genres_pt = dados.get('genres_pt', [])

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Nome:** {nome}")
                st.markdown(f"**Gênero:** {genero}")
                if idade:
                    st.markdown(f"**Idade:** {idade} anos")
            with col_b:
                st.markdown(f"**ID:** `{selected_user}` (novo)")
                st.markdown("**Tipo:** Criado na sessão")

            if genres_pt:
                st.markdown("**Gêneros preferidos:**")
                secondary = get_secondary_genres(selected_user, genres_pt, movies_df)
                _render_genre_badges(genres_pt, secondary)

        else:
            # ── Usuário existente do dataset ──
            col_a, col_b = st.columns(2)
            user_row = users_df[users_df['user_id'] == selected_user]

            with col_a:
                st.markdown(f"**Nome:** Usuário {selected_user}")
                if not user_row.empty:
                    row = user_row.iloc[0]
                    genero_txt = "Masculino" if row['gender'] == 'M' else "Feminino"
                    st.markdown(f"**Gênero:** {genero_txt}")
                    st.markdown(f"**Idade:** {row['age']} anos")
            with col_b:
                st.markdown(f"**ID:** `{selected_user}`")
                if not user_row.empty:
                    row       = user_row.iloc[0]
                    ocupacao  = OCCUPATION_PT.get(row['occupation'], row['occupation'])
                    st.markdown(f"**Ocupação:** {ocupacao}")

            # Gêneros favoritos inferidos das avaliações
            fav = get_user_favorite_genres(selected_user, ratings_df, movies_df)
            if fav:
                st.markdown("**Gêneros favoritos (top 5):**")
                secondary = get_secondary_genres(selected_user, fav, movies_df)
                _render_genre_badges(fav, secondary)

        st.markdown("---")

        # ── Watchlist (Melhoria 5) ──
        watchlist = st.session_state['user_watchlist'].get(selected_user, [])
        with st.expander(f"🔖 Minha Lista ({len(watchlist)} filmes)", expanded=False):
            if watchlist:
                wl_df = movies_df[movies_df['movie_id'].isin(watchlist)]
                for _, r in wl_df.iterrows():
                    c1, c2 = st.columns([5, 1])
                    c1.markdown(f"• {r['title']}")
                    if c2.button("✕", key=f"rm_wl_{selected_user}_{r['movie_id']}"):
                        st.session_state['user_watchlist'][selected_user].remove(r['movie_id'])
                        st.rerun()
            else:
                st.caption("Nenhum filme salvo. Use o botão 🔖 nos cards.")

        # ── Filmes avaliados na sessão (Melhoria 5) ──
        rated_cache = st.session_state['user_ratings_cache'].get(selected_user, {})
        with st.expander(f"⭐ Avaliados na sessão ({len(rated_cache)})", expanded=False):
            if rated_cache:
                for mid, nota in rated_cache.items():
                    titulo_s = movies_df[movies_df['movie_id'] == mid]['title']
                    titulo   = titulo_s.values[0] if not titulo_s.empty else f"Filme {mid}"
                    st.markdown(f"• {titulo} — {'⭐' * int(nota)} ({nota:.0f}/5)")
            else:
                st.caption("Nenhum filme avaliado ainda. Use o botão 👍 nos cards.")


# ══════════════════════════════════════════════════════════════
# MELHORIA 2 — POSTERS VIA TMDB
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def buscar_poster(titulo: str):
    """
    Busca o URL do poster no TMDB pelo título do filme.
    Cache: cada título só é consultado uma vez por sessão.
    Retorna None se a chave não estiver configurada ou o filme não for encontrado.
    """
    if not TMDB_API_KEY:
        return None
    titulo_limpo = titulo.split('(')[0].strip()
    try:
        resp = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_API_KEY, "query": titulo_limpo, "language": "pt-BR"},
            timeout=5
        )
        resp.raise_for_status()
        resultados = resp.json().get('results', [])
        if resultados and resultados[0].get('poster_path'):
            return TMDB_IMG_BASE + resultados[0]['poster_path']
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════
# MODELOS DE RECOMENDAÇÃO
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_content_model(_movies):
    """TF-IDF nos gêneros + cosine similarity. Cache para não recalcular."""
    tfidf      = TfidfVectorizer()
    tfidf_mat  = tfidf.fit_transform(_movies['genres_str'])
    cosine_sim = cosine_similarity(tfidf_mat, tfidf_mat)
    return cosine_sim


def treinar_svd(ratings_df: pd.DataFrame) -> SVD:
    """
    Treina SVD sem cache — necessário para retreinar quando
    um novo usuário é adicionado (Melhoria 1).
    """
    reader   = Reader(rating_scale=(1, 5))
    data     = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    model    = SVD(random_state=SEED)
    model.fit(trainset)
    return model


@st.cache_resource
def load_collaborative_model(_ratings):
    """Carrega e treina o SVD com os dados originais (apenas uma vez)."""
    return treinar_svd(_ratings)


# ══════════════════════════════════════════════════════════════
# FUNÇÕES DE RECOMENDAÇÃO
# ══════════════════════════════════════════════════════════════

def get_popular_movies(ratings_df, movies_df, top_n=20):
    """Filmes mais bem avaliados com pelo menos 20 avaliações (fallback cold start)."""
    stats   = ratings_df.groupby('movie_id')['rating'].agg(['mean', 'count'])
    popular = stats[stats['count'] >= 20].sort_values('mean', ascending=False)
    top_ids = popular.head(top_n).index.tolist()
    result  = movies_df[movies_df['movie_id'].isin(top_ids)].copy()
    result['rank'] = result['movie_id'].apply(lambda x: top_ids.index(x))
    return result.sort_values('rank')[['movie_id', 'title', 'genres_str']].reset_index(drop=True)


def get_content_recommendations(user_id, ratings_df, movies_df, cosine_sim, top_n=10):
    """Recomendação Baseada em Conteúdo: TF-IDF nos gêneros + cosine similarity."""
    liked = ratings_df[
        (ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)
    ]
    if liked.empty:
        return get_popular_movies(ratings_df, movies_df, top_n)

    scores = {}
    for movie_id in liked['movie_id']:
        idx_list = movies_df.index[movies_df['movie_id'] == movie_id].tolist()
        if not idx_list:
            continue
        for i, score in enumerate(cosine_sim[idx_list[0]]):
            mid = int(movies_df.iloc[i]['movie_id'])
            scores[mid] = scores.get(mid, 0) + float(score)

    watched  = set(ratings_df[ratings_df['user_id'] == user_id]['movie_id'])
    sorted_m = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_ids  = [mid for mid, _ in sorted_m if mid not in watched][:top_n]

    result = movies_df[movies_df['movie_id'].isin(top_ids)].copy()
    result['rank'] = result['movie_id'].apply(lambda x: top_ids.index(x))
    return result.sort_values('rank')[['movie_id', 'title', 'genres_str']].reset_index(drop=True)


def get_collaborative_recommendations(user_id, model, ratings_df, movies_df, top_n=10):
    """Recomendação Colaborativa (SVD): prevê notas e ordena por maior predição."""
    all_ids    = movies_df['movie_id'].unique()
    watched    = set(ratings_df[ratings_df['user_id'] == user_id]['movie_id'])
    candidates = [mid for mid in all_ids if mid not in watched]

    predictions = [(mid, model.predict(uid=user_id, iid=mid).est) for mid in candidates]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_ids = [mid for mid, _ in predictions[:top_n]]

    result = movies_df[movies_df['movie_id'].isin(top_ids)].copy()
    result['rank'] = result['movie_id'].apply(lambda x: top_ids.index(x))
    return result.sort_values('rank')[['movie_id', 'title', 'genres_str']].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# MÉTRICAS DE AVALIAÇÃO
# ══════════════════════════════════════════════════════════════

def precision_at_k(rec, rel, k=10):
    r = rec[:k]; s = set(rel)
    return len([x for x in r if x in s]) / len(r) if r else 0.0

def recall_at_k(rec, rel, k=10):
    r = rec[:k]; s = set(rel)
    return len([x for x in r if x in s]) / len(s) if s else 0.0

def ndcg_at_k(rec, rel, k=10):
    r    = rec[:k]; s = set(rel)
    dcg  = sum(1 / np.log2(i + 2) for i, x in enumerate(r) if x in s)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(s), k)))
    return dcg / idcg if idcg > 0 else 0.0

def diversity_at_k(rec, movies_df, k=10):
    sub = movies_df[movies_df['movie_id'].isin(rec[:k])].copy()
    if len(sub) < 2:
        return 0.0
    try:
        tfidf = TfidfVectorizer()
        mat   = tfidf.fit_transform(sub['genres_str'])
        sim   = cosine_similarity(mat)
        n     = len(sim)
        total = sum(1 - sim[i][j] for i in range(n) for j in range(i + 1, n))
        pairs = n * (n - 1) / 2
        return total / pairs if pairs > 0 else 0.0
    except Exception:
        return 0.0

def evaluate_user(user_id, svd_model, ratings_df, movies_df, cosine_sim):
    """
    Avaliação offline leave-k-out:
    - 30% dos filmes favoritos são reservados como ground truth (test_movies).
    - As recomendações são geradas sem esses filmes no histórico, para que
      possam aparecer no resultado e serem encontrados pelas métricas.
    """
    liked = ratings_df[
        (ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)
    ]['movie_id'].tolist()
    if len(liked) < 5:
        return None

    rng         = np.random.RandomState(SEED)
    n_test      = max(1, int(len(liked) * 0.3))
    test_movies = rng.choice(liked, size=n_test, replace=False).tolist()

    # Remove os filmes de teste do histórico para que possam ser recomendados
    ratings_train = ratings_df[
        ~((ratings_df['user_id'] == user_id) &
          (ratings_df['movie_id'].isin(test_movies)))
    ]

    collab_df  = get_collaborative_recommendations(user_id, svd_model, ratings_train, movies_df)
    content_df = get_content_recommendations(user_id, ratings_train, movies_df, cosine_sim)

    def m(ids):
        return {
            'precision':   precision_at_k(ids, test_movies),
            'recall':      recall_at_k(ids, test_movies),
            'ndcg':        ndcg_at_k(ids, test_movies),
            'diversidade': diversity_at_k(ids, movies_df),
        }
    c = collab_df['movie_id'].tolist()
    b = content_df['movie_id'].tolist()
    return m(c), m(b), collab_df, content_df


# ══════════════════════════════════════════════════════════════
# MELHORIA 5 — FUNÇÕES AUXILIARES DE INTERAÇÃO
# ══════════════════════════════════════════════════════════════

def salvar_filme(user_id, movie_id):
    """Adiciona filme à watchlist (sem duplicatas). Filmes já avaliados são ignorados."""
    if ja_avaliou(user_id, movie_id):
        return
    wl = st.session_state['user_watchlist']
    if user_id not in wl:
        wl[user_id] = []
    if movie_id not in wl[user_id]:
        wl[user_id].append(movie_id)

def remover_watchlist(user_id, movie_id):
    """Remove filme da watchlist se estiver nela."""
    wl = st.session_state['user_watchlist']
    if user_id in wl and movie_id in wl[user_id]:
        wl[user_id].remove(movie_id)

def registrar_avaliacao(user_id, movie_id, nota):
    """
    Registra avaliação feita via botão Curtir.
    Remove automaticamente da watchlist se estiver lá.
    """
    rc = st.session_state['user_ratings_cache']
    if user_id not in rc:
        rc[user_id] = {}
    rc[user_id][movie_id] = nota
    remover_watchlist(user_id, movie_id)

def ja_avaliou(user_id, movie_id) -> bool:
    return movie_id in st.session_state['user_ratings_cache'].get(user_id, {})

def esta_na_watchlist(user_id, movie_id) -> bool:
    return movie_id in st.session_state['user_watchlist'].get(user_id, [])


# ══════════════════════════════════════════════════════════════
# CARD DE FILME — MELHORIAS 2, 3 E 5 INTEGRADAS
# ══════════════════════════════════════════════════════════════

def render_movie_card(i: int, row, selected_user: int, ratings_df):
    """
    Renderiza card de filme com:
    - Poster (TMDB) — Melhoria 2
    - Gêneros em português — Melhoria 3
    - Nota média geral do filme (todos os usuários)
    - Botão 👍 Curtir com popover de nota — Melhoria 5
    - Botão 🔖 Salvar com lógica de watchlist — Melhoria 5
    """
    movie_id   = int(row.movie_id)
    poster_url = buscar_poster(row.title)
    generos_pt = traduzir_generos(row.genres_str)

    badge_av   = " ✅" if ja_avaliou(selected_user, movie_id)       else ""
    badge_sal  = " 🔖" if esta_na_watchlist(selected_user, movie_id) else ""

    movie_ratings = ratings_df[ratings_df['movie_id'] == movie_id]['rating']
    if not movie_ratings.empty:
        avg_rating = movie_ratings.mean()
        n_ratings  = len(movie_ratings)
        rating_str = f"⭐ {avg_rating:.1f} · {n_ratings} avaliações"
    else:
        rating_str = "⭐ Sem avaliações"

    with st.container(border=True):
        col_poster, col_info, col_btns = st.columns([2, 7, 2])

        with col_poster:
            if poster_url:
                st.image(poster_url, width=100)
            else:
                st.markdown(
                    "<div style='width:100px;height:140px;background:#2a2a2a;"
                    "border-radius:6px;display:flex;align-items:center;"
                    "justify-content:center;color:#888;font-size:11px;"
                    "text-align:center;padding:4px'>Sem<br>imagem</div>",
                    unsafe_allow_html=True
                )

        with col_info:
            st.markdown(f"**#{i + 1} — {row.title}**{badge_av}{badge_sal}")
            st.caption(f"🎭 {generos_pt}")
            st.caption(rating_str)

        with col_btns:
            # ── Botão Curtir com popover (Streamlit >= 1.29) ──
            label_curtir = "✅ Curtido" if ja_avaliou(selected_user, movie_id) else "👍 Curtir"
            with st.popover(label_curtir, use_container_width=True):
                st.markdown(f"**{row.title}**")
                st.markdown("Qual nota você dá para este filme?")
                nota = st.slider(
                    "Nota", min_value=1, max_value=5, value=4,
                    key=f"slider_{i}_{movie_id}_{selected_user}"
                )
                st.markdown("⭐" * nota)
                if st.button(
                    "✔ Confirmar",
                    key=f"confirm_{i}_{movie_id}_{selected_user}",
                    type="primary"
                ):
                    registrar_avaliacao(selected_user, movie_id, nota)
                    st.rerun()

            # ── Botão Salvar / Remover (oculto se o filme já foi avaliado) ──
            if not ja_avaliou(selected_user, movie_id):
                if esta_na_watchlist(selected_user, movie_id):
                    if st.button(
                        "🔖 Salvo", key=f"unsave_{i}_{movie_id}_{selected_user}",
                        help="Clique para remover da lista", use_container_width=True
                    ):
                        remover_watchlist(selected_user, movie_id)
                        st.rerun()
                else:
                    if st.button(
                        "🔖 Salvar", key=f"save_{i}_{movie_id}_{selected_user}",
                        help="Salvar para ver depois", use_container_width=True
                    ):
                        salvar_filme(selected_user, movie_id)
                        st.rerun()


# ══════════════════════════════════════════════════════════════
# MELHORIA 1 — PAINEL DE NOVO USUÁRIO (sidebar)
# ══════════════════════════════════════════════════════════════

def painel_novo_usuario(ratings_df, movies_df):
    """
    Formulário na sidebar para criar um novo usuário.
    Coleta nome, idade, gênero, gêneros preferidos e avaliações de filmes.
    Com >= 3 avaliações, retreina o SVD. Abaixo disso, demonstra cold start.
    """
    with st.sidebar.expander("➕ Criar novo usuário", expanded=False):
        st.markdown("**Crie seu perfil e receba recomendações personalizadas!**")
        st.caption(
            "Cold start: com menos de 3 filmes avaliados, "
            "o sistema usa popularidade como fallback."
        )

        populares      = get_popular_movies(ratings_df, movies_df, top_n=20)
        opcoes_titulos = populares['title'].tolist()

        with st.form("form_novo_usuario", clear_on_submit=True):

            st.markdown("**Dados pessoais**")
            nome = st.text_input("Nome *", placeholder="Ex: Prof. João da Silva")

            col_idade, col_genero = st.columns(2)
            with col_idade:
                idade = st.number_input("Idade", min_value=5, max_value=110, value=25, step=1)
            with col_genero:
                genero = st.selectbox("Gênero", ["Masculino", "Feminino", "Prefiro não dizer"])

            # Melhoria 4 — gêneros preferidos
            st.markdown("**Gêneros preferidos (3 a 5)**")
            genres_escolhidos = st.multiselect(
                "Escolha os gêneros que você mais gosta",
                GENRES_PT, max_selections=5
            )

            # Avaliações de filmes
            st.markdown("**Avalie de 3 a 5 filmes populares**")
            filmes_escolhidos = st.multiselect(
                "Filmes disponíveis", opcoes_titulos, max_selections=5
            )

            notas = {}
            if filmes_escolhidos:
                st.markdown("**Notas para cada filme:**")
                for titulo in filmes_escolhidos:
                    notas[titulo] = st.slider(
                        titulo, min_value=1, max_value=5, value=4,
                        key=f"nota_{titulo}"
                    )

            enviado = st.form_submit_button("🚀 Criar perfil", type="primary")

        if enviado:
            if not nome.strip():
                st.sidebar.error("Por favor, informe seu nome.")
                return
            if len(genres_escolhidos) < 3:
                st.sidebar.warning("Escolha pelo menos 3 gêneros preferidos.")
                return

            novo_id = int(ratings_df['user_id'].max()) + 1

            # Registra o novo usuário no dicionário da sessão
            st.session_state['novos_usuarios'][novo_id] = {
                'nome':      nome.strip(),
                'idade':     int(idade),
                'genero':    genero,
                'genres_pt': genres_escolhidos,
            }

            if len(filmes_escolhidos) >= 3:
                novas_linhas = []
                for titulo in filmes_escolhidos:
                    mid_s = populares[populares['title'] == titulo]['movie_id']
                    if not mid_s.empty:
                        novas_linhas.append({
                            'user_id':   novo_id,
                            'movie_id':  int(mid_s.values[0]),
                            'rating':    float(notas.get(titulo, 4)),
                            'timestamp': 0
                        })

                ratings_amp = pd.concat(
                    [ratings_df, pd.DataFrame(novas_linhas)], ignore_index=True
                )
                with st.sidebar.status("🔄 Retreinando o SVD...", expanded=True) as s:
                    novo_modelo = treinar_svd(ratings_amp)
                    s.update(label="✅ Modelo atualizado!", state="complete")

                st.session_state['ratings_ampliado'] = ratings_amp
                st.session_state['svd_retreinado']   = novo_modelo
            else:
                # Cold start: não altera ratings nem SVD já existentes
                st.sidebar.info(
                    "⚠️ Cold start: avaliações insuficientes. "
                    "Usando popularidade como fallback."
                )

            st.sidebar.success(
                f"✅ Perfil de **{nome}** criado!\n\n"
                f"ID: **{novo_id}** · Filmes: **{len(filmes_escolhidos)}**"
            )


# ══════════════════════════════════════════════════════════════
# INTERFACE PRINCIPAL
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Feed de Recomendações — TCC",
    page_icon="🎬",
    layout="wide"
)

# CSS customizado — carrega de arquivo externo
def carregar_css(caminho: str):
    with open(caminho, encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

carregar_css("style.css")

# Inicializa session_state
init_session_state()

st.title("🎬 Protótipo de Feed de Recomendações")
st.caption("TCC — Comparando Filtragem Colaborativa (SVD) vs. Baseada em Conteúdo (TF-IDF)")

# ── Sidebar ──
st.sidebar.header("⚙️ Configurações")

# ── Carregamento inicial ──
with st.spinner("Carregando dados e treinando modelos (só na primeira vez)..."):
    ratings, movies, users = load_data()
    cosine_sim = load_content_model(movies)
    svd_model  = load_collaborative_model(ratings)

# ── Dados ativos (originais ou ampliados com novo usuário) ──
_r = st.session_state.get('ratings_ampliado')
ratings_ativos = _r if _r is not None else ratings
_s = st.session_state.get('svd_retreinado')
svd_ativo      = _s if _s is not None else svd_model

# ── Sidebar — painel de novo usuário ──
painel_novo_usuario(ratings_ativos, movies)

# ── Seletor de usuário ──
novos_usuarios = st.session_state['novos_usuarios']
user_list = sorted(ratings['user_id'].unique().tolist())
if novos_usuarios:
    novos_ids = sorted(novos_usuarios.keys())
    user_list = novos_ids + user_list
    def display_fn(i):
        uid = user_list[i]
        if uid in novos_usuarios:
            return f"🆕 {novos_usuarios[uid]['nome']} (ID {uid})"
        return str(uid)
else:
    display_fn = lambda i: str(user_list[i])

selected_idx  = st.sidebar.selectbox(
    "👤 Selecione um Usuário",
    range(len(user_list)),
    format_func=display_fn
)
selected_user = user_list[selected_idx]

n_ratings = len(ratings_ativos[ratings_ativos['user_id'] == selected_user])
n_liked   = len(ratings_ativos[
    (ratings_ativos['user_id'] == selected_user) & (ratings_ativos['rating'] >= 4)
])
st.sidebar.info(f"📋 **{n_ratings}** avaliações · **{n_liked}** positivas")

model_choice = st.sidebar.radio(
    "🤖 Modo de Exibição",
    [
        "Filtragem Colaborativa (SVD)",
        "Baseada em Conteúdo (TF-IDF)",
        "⚖️ Comparar os dois lado a lado",
    ]
)
show_metrics = st.sidebar.checkbox("📊 Mostrar métricas de avaliação", value=False)
gerar        = st.sidebar.button("🚀 Gerar Recomendações", type="primary")

# ── Layout principal: perfil (esq) | feed (dir) ──
col_perfil, col_feed = st.columns([1, 3])

with col_perfil:
    # Perfil sempre visível, atualizado ao trocar de usuário
    exibir_perfil(selected_user, users, ratings_ativos, movies)

with col_feed:
    # Aviso de cold start para novo usuário
    eh_novo = selected_user in st.session_state['novos_usuarios']
    if eh_novo:
        nome_novo = st.session_state['novos_usuarios'][selected_user]['nome']
        if n_liked < 3:
            st.warning(
                "⚠️ **Cold Start demonstrado:** usuário com poucos dados. "
                "O sistema usa popularidade como fallback — conforme descrito no TCC."
            )
        else:
            st.info(
                f"✅ Perfil de **{nome_novo}** ativo. "
                "SVD retreinado com suas avaliações."
            )

    if gerar:
        # Calcula e persiste no session_state para sobreviver a reruns de botões
        user_age = get_user_age(selected_user, users)
        st.session_state['rec_mode'] = model_choice
        if "Comparar" in model_choice:
            st.session_state['rec_collab_df'] = filtrar_por_idade(
                get_collaborative_recommendations(selected_user, svd_ativo, ratings_ativos, movies, top_n=20),
                user_age
            )
            st.session_state['rec_content_df'] = filtrar_por_idade(
                get_content_recommendations(selected_user, ratings_ativos, movies, cosine_sim, top_n=20),
                user_age
            )
        elif "Colaborativa" in model_choice:
            st.session_state['rec_collab_df'] = filtrar_por_idade(
                get_collaborative_recommendations(selected_user, svd_ativo, ratings_ativos, movies, top_n=20),
                user_age
            )
            st.session_state['rec_content_df'] = None
        else:
            st.session_state['rec_collab_df'] = None
            st.session_state['rec_content_df'] = filtrar_por_idade(
                get_content_recommendations(selected_user, ratings_ativos, movies, cosine_sim, top_n=20),
                user_age
            )

    rec_mode   = st.session_state.get('rec_mode')
    collab_df  = st.session_state.get('rec_collab_df')
    content_df = st.session_state.get('rec_content_df')

    # Remove filmes já avaliados na sessão e limita a 10 para exibição
    def _feed(df, uid, n=10):
        if df is None:
            return None
        rated = set(st.session_state['user_ratings_cache'].get(uid, {}).keys())
        return df[~df['movie_id'].isin(rated)].head(n).reset_index(drop=True)

    if rec_mode is not None:
        if "Comparar" in rec_mode:
            st.subheader(f"Usuário {selected_user} — Comparação lado a lado")
            st.markdown("---")
            cc, cb = st.columns(2)
            with cc:
                st.markdown("### 🤝 Filtragem Colaborativa (SVD)")
                st.caption("Baseada no comportamento de usuários similares")
                for i, row in enumerate(_feed(collab_df, selected_user).itertuples()):
                    render_movie_card(i, row, selected_user, ratings_ativos)
            with cb:
                st.markdown("### 📄 Baseada em Conteúdo (TF-IDF)")
                st.caption("Baseada nos gêneros do histórico do usuário")
                for i, row in enumerate(_feed(content_df, selected_user).itertuples()):
                    render_movie_card(i, row, selected_user, ratings_ativos)

        else:
            if "Colaborativa" in rec_mode:
                rec_df  = _feed(collab_df, selected_user)
                label   = "🤝 Filtragem Colaborativa (SVD)"
                caption = "Baseada no comportamento de usuários similares"
            else:
                rec_df  = _feed(content_df, selected_user)
                label   = "📄 Baseada em Conteúdo (TF-IDF)"
                caption = "Baseada nos gêneros do histórico do usuário"

            st.subheader(f"📱 Feed — Usuário {selected_user} — {label}")
            st.caption(caption)
            st.markdown("---")
            for i, row in enumerate(rec_df.itertuples()):
                render_movie_card(i, row, selected_user, ratings_ativos)

        # ── Seção de métricas ──
        if show_metrics:
            st.markdown("---")
            st.subheader("📊 Avaliação Comparativa dos Modelos")
            st.markdown(
                "> **Como funciona:** 30% dos filmes favoritos são escondidos como "
                "*ground truth*. Os modelos recomendam sem saber desses filmes."
            )
            result = evaluate_user(selected_user, svd_ativo, ratings_ativos, movies, cosine_sim)
            if result is None:
                st.warning(
                    "Usuário com dados insuficientes (menos de 5 filmes favoritos). "
                    "Escolha outro usuário ou avalie mais filmes."
                )
            else:
                cm, bm, _, _ = result
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🤝 Filtragem Colaborativa")
                    st.metric("Precision@10",  f"{cm['precision']:.3f}",
                              help="Proporção de acertos entre os 10 recomendados")
                    st.metric("Recall@10",     f"{cm['recall']:.3f}",
                              help="Cobertura dos filmes favoritos")
                    st.metric("NDCG@10",       f"{cm['ndcg']:.3f}",
                              help="Precisão ponderada pela posição no ranking")
                    st.metric("Diversidade",   f"{cm['diversidade']:.3f}",
                              help="Variedade de gêneros (maior = mais diverso)")
                with c2:
                    st.markdown("#### 📄 Baseada em Conteúdo")
                    st.metric("Precision@10",  f"{bm['precision']:.3f}")
                    st.metric("Recall@10",     f"{bm['recall']:.3f}")
                    st.metric("NDCG@10",       f"{bm['ndcg']:.3f}")
                    st.metric("Diversidade",   f"{bm['diversidade']:.3f}")

                metricas = ['Precision@10', 'Recall@10', 'NDCG@10', 'Diversidade']
                chaves   = ['precision',    'recall',    'ndcg',    'diversidade']
                chart_data = pd.DataFrame([
                    {'Métrica': m, 'Modelo': 'CF (SVD)',          'Valor': cm[k]}
                    for m, k in zip(metricas, chaves)
                ] + [
                    {'Métrica': m, 'Modelo': 'Conteúdo (TF-IDF)', 'Valor': bm[k]}
                    for m, k in zip(metricas, chaves)
                ])
                chart = (
                    alt.Chart(chart_data)
                    .mark_bar()
                    .encode(
                        x=alt.X('Modelo:N', title=None, axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('Valor:Q', scale=alt.Scale(domain=[0, 1]), title='Valor'),
                        color=alt.Color(
                            'Modelo:N',
                            scale=alt.Scale(
                                domain=['CF (SVD)', 'Conteúdo (TF-IDF)'],
                                range=['#8aa9ff', '#c896f0']
                            )
                        ),
                        column=alt.Column(
                            'Métrica:N',
                            sort=metricas,
                            spacing=12,
                            header=alt.Header(titleOrient='bottom', labelOrient='bottom')
                        ),
                        tooltip=['Modelo', 'Métrica', alt.Tooltip('Valor:Q', format='.3f')]
                    )
                    .properties(width=110, height=260)
                )
                st.altair_chart(chart)
                st.info(
                    "💡 **O que observar:** CF tende a ter maior precisão. "
                    "BC tende a ter maior diversidade. "
                    "Esse é o **trade-off precisão vs. bolha de filtro** — tema central do TCC."
                )
    else:
        st.info("👈 Configure as opções na barra lateral e clique em **Gerar Recomendações**.")
