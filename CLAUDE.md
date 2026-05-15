# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```powershell
# Activate the virtual environment (Windows)
.\venv\Scripts\Activate.ps1

# Run the Streamlit app
streamlit run app.py
# or directly via venv:
.\venv\Scripts\streamlit.exe run app.py
```

The app runs on http://localhost:8501 by default.

## Dataset setup

Dataset: **MovieLens 1M** — 1.000.209 avaliações · 3.883 filmes · 6.040 usuários · nota média 3,582.

Files located at `./ml-1m/ml-1m/` (nested folder from the original zip):

| File | Contents |
|------|----------|
| `ratings.dat` | Ratings — `user_id`, `movie_id`, `rating`, `timestamp` (separator `::`) |
| `movies.dat` | Movies — `movie_id`, `title`, `genres` (separator `::`, genres as `"Action\|Comedy\|Drama"`) |
| `users.dat` | Users — `user_id`, `gender`, `age`, `occupation`, `zip_code` (separator `::`) |

`DATA_PATH = './ml-1m/ml-1m/'` is the single constant controlling the path.

**ML-1M format notes:**
- Genres are a pipe-separated string per film; `load_data()` converts `"Children's"` → `"Children"` and replaces `|` with spaces to produce `genres_str`.
- Occupation column is a numeric code (0–20); `ML1M_OCCUPATION` maps it to the string keys used by `OCCUPATION_PT`.
- Age column stores the lower bound of an age range (1, 18, 25, 35, 45, 50, 56). The age filter (`filtrar_por_idade`) works correctly since users coded as 1 ("Under 18") have `age < 18`.

## Dependencies

```
streamlit==1.56.0   pandas==3.0.2      numpy==1.26.4
scikit-learn==1.8.0 scikit-surprise==1.1.4
altair==6.0.0       requests==2.33.1
```

## Architecture

The entire application is a single file (`app.py`, ~990 lines). There are no modules, packages, or test files.

### Data flow

```
load_data()             → (ratings, movies, users) DataFrames — @st.cache_data
load_content_model()    → cosine_sim matrix (TF-IDF on genres) — @st.cache_resource
load_collaborative_model() → trained SVD model — @st.cache_resource
treinar_svd()           → re-trains SVD without cache (used when a new user is added)
```

When a new user is created via the sidebar form, `ratings_ampliado` and `svd_retreinado` replace the originals in session_state for the rest of the session. All recommendation functions receive the active versions (`ratings_ativos`, `svd_ativo`).

### Session state keys

| Key | Purpose |
|-----|---------|
| `novos_usuarios` | `{user_id: {nome, idade, genero, genres_pt}}` — users created this session |
| `ratings_ampliado` | Extended ratings DataFrame after a new user is added |
| `svd_retreinado` | SVD model retrained with the new user's ratings |
| `user_watchlist` | `{user_id: [movie_id, ...]}` — per-user saved films |
| `user_ratings_cache` | `{user_id: {movie_id: nota}}` — in-session ratings |
| `rec_mode` | Last selected model mode (persists recs across button reruns) |
| `rec_collab_df` | Last collaborative recommendation results |
| `rec_content_df` | Last content-based recommendation results |

All keys are initialized in `init_session_state()` at startup to avoid `KeyError`.

### SVD hyperparameters

`SVD(random_state=42)` — todos os demais parâmetros são defaults do scikit-surprise:

| Parâmetro | Valor |
|-----------|------:|
| `n_factors` | 100 |
| `n_epochs` | 20 |
| `biased` | True |
| `lr_bu = lr_bi = lr_pu = lr_qi` | 0.005 |
| `reg_bu = reg_bi = reg_pu = reg_qi` | 0.02 |
| `init_mean` | 0 |
| `init_std_dev` | 0.1 |

`lr_all=0.005` e `reg_all=0.02` são copiados para cada componente (bu, bi, pu, qi) internamente pelo Surprise quando não sobrescritos individualmente.

### TF-IDF hyperparameters

`TfidfVectorizer()` — todos os defaults do scikit-learn. Parâmetros relevantes:

| Parâmetro | Valor | Nota |
|-----------|-------|------|
| `stop_words` | None | nenhuma palavra removida |
| `ngram_range` | (1, 1) | apenas unigramas |
| `max_features` | None | sem limite de vocabulário |
| `norm` | `'l2'` | vetores normalizados |
| `use_idf` | True | penaliza gêneros muito frequentes (Drama, Comedy) |
| `smooth_idf` | True | adiciona 1 ao denominador do IDF |
| `sublinear_tf` | False | TF linear |
| `binary` | False | conta frequência, não só presença |

Com `genres_str` como corpus (ex: `"Action Comedy Drama"`), o vectorizer funciona como **bag-of-genres** ponderado por IDF. A matriz `cosine_sim` (3883 × 3883) é calculada **entre todos os pares de filmes** uma única vez no boot e fica em cache.

### Performance (benchmark medido no ML-100K; valores ML-1M serão maiores)

| Operação | ML-100K (referência) |
|----------|--------------------:|
| Treino SVD (`model.fit`) | ~735 ms |
| Build conteúdo (TF-IDF + cosine_sim) | ~9,5 ms |
| Rec SVD Top-10 para 1 usuário | ~2,6 ms |
| Rec conteúdo Top-10 para 1 usuário (loop Python) | ~7.366 ms |

O gargalo da recomendação por conteúdo é o loop Python puro em `get_content_recommendations`. A substituição por `cosine_sim[liked_indices].sum(axis=0)` (NumPy vetorizado) resolveria o problema, mas não foi aplicada.

### Recommendation logic

- **Collaborative (SVD)** — `get_collaborative_recommendations()`: predicts ratings for all unwatched films, returns top-N by predicted score.
- **Content-based (TF-IDF)** — `get_content_recommendations()`: aggregates cosine similarity scores from all liked films (rating ≥ 4), returns top-N unwatched.
- **Cold start fallback**: `get_content_recommendations()` falls back to `get_popular_movies()` when `liked.empty` (no liked films at all). For new users, the SVD is only retrained when ≥ 3 films are rated; below that threshold the UI shows a cold-start warning and popularity is used instead.
- **`get_popular_movies()`**: filters movies with `count >= 20` ratings, then sorts by mean rating descending.
- **Age filter** — `filtrar_por_idade()`: strips Horror/Crime/Film-Noir/Thriller for users under 18.

### Evaluation

`evaluate_user()` runs a leave-30%-out offline evaluation: holds out 30% of liked films as ground truth, generates recommendations from the remaining history, and computes Precision@10, Recall@10, NDCG@10, and Diversity (mean pairwise genre dissimilarity). Returns `None` if the user has fewer than 5 liked films. Displayed only when the "Mostrar métricas" checkbox is on.

**Pre-computed evaluation snapshots** (generated externally, not by `app.py`):

| File | Dataset | Users | Columns |
|------|---------|------:|---------|
| `evaluation_results.csv` | ML-100K (referência) | 22 | `Usuário, Avaliações, SVD_Precision@10, TFIDF_Precision@10, SVD_Recall@10, TFIDF_Recall@10, SVD_NDCG@10, TFIDF_NDCG@10, SVD_Diversidade, TFIDF_Diversidade` |
| `evaluation_1m.csv` | ML-1M (produção) | 22 | idem |

### TMDB integration

`buscar_poster(titulo)` calls the TMDB search API (`@st.cache_data`) using the key stored in `TMDB_API_KEY` at the top of the file. Strip the year from the title before querying. Returns `None` gracefully on failure — the card renders a placeholder div instead.

### Profile card helpers

- `get_secondary_genres(user_id, primary_genres_pt, movies_df)` — returns genres discovered from in-session ratings (nota ≥ 4) that are not already in the user's primary genre list.
- `_render_genre_badges(primary, secondary)` — renders primary genres as normal backtick badges and secondary genres as red HTML badges (`#ff6b6b` on `#4a1515`).

### Movie card

`render_movie_card(i, row, selected_user, ratings_df)` — displays poster, title, genre badges, and the **global average rating** of the film (mean + count from `ratings_df`). Also renders two interactive buttons:
- **👍 Curtir** — opens a `st.popover` with a 1–5 star slider; confirming calls `registrar_avaliacao()` and triggers `st.rerun()`.
- **🔖 Salvar / Salvo** — toggles the film in the user's watchlist via `salvar_filme()` / `remover_watchlist()`. **Hidden entirely when the film has already been rated** (`ja_avaliou()` returns `True`).

Watchlist invariant: a rated film can never enter the watchlist. `registrar_avaliacao()` removes it on rating; `salvar_filme()` guards against re-adding by checking `ja_avaliou()` before inserting.

### New user panel

`painel_novo_usuario(ratings_df, movies_df)` — sidebar form (`st.expander`) that collects name, age, gender, preferred genres (≥ 3), and ratings for up to 5 popular films. On submit:
- If ≥ 3 films rated: appends new rows to ratings, calls `treinar_svd()`, and stores results in `ratings_ampliado` / `svd_retreinado`.
- If < 3 films rated: cold-start path — no retraining, popularity fallback is used.

### Feed rendering

Recommendations are generated with `top_n=20` and stored in `rec_collab_df` / `rec_content_df`. At render time the helper `_feed(df, uid, n=10)` filters out any movie already in `user_ratings_cache` for the current user and returns the first `n` remaining — so rating a film causes it to disappear from the feed on the next `st.rerun()`, replaced by the next item from the pool of 20.

### UI layout

Sidebar → new-user form (`painel_novo_usuario`, collapsed expander) + user selector + ratings info box + model radio + metrics checkbox + "Gerar Recomendações" button.  
Main area → two columns: `col_perfil` (1/4 width, always visible) | `col_feed` (3/4 width, rendered after button press and persisted in session_state).

> **Note:** the data-loading `st.spinner` is a main-area element (not sidebar), since it is called in global scope rather than inside `st.sidebar`.

Genre strings are stored internally in English (space-separated, e.g. `"Action Comedy Drama"`) and translated to Portuguese only at render time via `GENRE_TRANSLATION` and `traduzir_generos()`.
