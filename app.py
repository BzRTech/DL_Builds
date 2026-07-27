"""Interface web (local) do DL_Builds — Streamlit.

Roda na máquina com GPU + dados; acessível pelo navegador em localhost e, na
rede local, pelo IP da máquina (ex.: http://SEU_IP:8501). Aponta arquivos no
disco (sem upload de ortofotos gigantes), executa os modelos e organiza a saída
em outputs/<cidade>/. Também navega/visualiza os resultados já gerados.

Uso:
    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import streamlit as st

from src.inference.predict import predict_image
from src.postprocess.vectorize import vectorize
from src.road_surface.predict import classify_roads
from src.utils import ensure_dir, load_config

st.set_page_config(page_title="DL_Builds — Extração", layout="wide")
st.title("🛰️ DL_Builds — Extração automática de feições")
st.caption("Edificações · Quadras · Lotes · Pavimento das vias — a partir da ortofoto")

# Produtos de segmentação: (config, nome do arquivo de saída)
SEG_PRODUCTS = {
    "Edificações": ("configs/buildings.yaml", "edificacoes"),
    "Quadras": ("configs/quadras.yaml", "quadras"),
    "Lotes": ("configs/lotes.yaml", "lotes"),
}
OUTPUTS = Path("outputs")


def _preview(gpkg: str, title: str, max_feats: int = 3000) -> None:
    gdf = gpd.read_file(gpkg)
    st.write(f"**{title}** — {len(gdf)} feições · `{gpkg}`")
    if len(gdf) == 0:
        return
    plot_gdf = gdf.sample(max_feats, random_state=0) if len(gdf) > max_feats else gdf
    fig, ax = plt.subplots(figsize=(7, 7))
    if "PAV_PRED" in plot_gdf.columns:
        plot_gdf.plot(ax=ax, column="PAV_PRED", legend=True, linewidth=1.0)
    else:
        plot_gdf.boundary.plot(ax=ax, linewidth=0.4, color="crimson")
    ax.set_axis_off()
    ax.set_aspect("equal")
    st.pyplot(fig)
    plt.close(fig)
    data = Path(gpkg).read_bytes()
    if len(data) < 200_000_000:
        st.download_button(f"⬇️ Baixar {Path(gpkg).name}", data, key=f"dl_{gpkg}",
                           file_name=Path(gpkg).name, mime="application/geopackage+sqlite3")
    else:
        st.caption("Arquivo grande — abra direto no QGIS pelo caminho acima.")


# ============================================================ barra lateral
mode = st.sidebar.radio("Modo", ["▶️ Executar extração", "📂 Ver resultados prontos"])


# ============================================================ MODO: RESULTADOS
def results_view() -> None:
    st.subheader("📂 Resultados já gerados")
    cities = sorted(p.name for p in OUTPUTS.iterdir() if p.is_dir()) if OUTPUTS.exists() else []
    cities = [c for c in cities if not c.startswith("_")]
    if not cities:
        st.info("Ainda não há resultados em `outputs/`. Rode uma extração no modo "
                "**Executar**.")
        return
    city = st.selectbox("Cidade", cities)
    gpkgs = sorted(p for p in (OUTPUTS / city).glob("*.gpkg"))
    if not gpkgs:
        st.warning(f"Nenhuma camada .gpkg em `outputs/{city}/`.")
        return
    st.write(f"{len(gpkgs)} camada(s) em `outputs/{city}/`:")
    for g in gpkgs:
        with st.expander(g.stem, expanded=(len(gpkgs) == 1)):
            try:
                _preview(str(g), g.stem)
            except Exception as e:  # noqa: BLE001
                st.exception(e)


# ============================================================ MODO: EXECUTAR
def run_view() -> None:
    st.sidebar.header("Entrada")
    city = st.sidebar.text_input("Nome da cidade", "NovaCidade").strip() or "NovaCidade"

    processed = sorted(str(p) for p in Path("data/processed").glob("*.tif"))
    if processed:
        image = st.sidebar.selectbox("Ortofoto (COG em data/processed)", processed)
    else:
        image = st.sidebar.text_input("Caminho da ortofoto (COG .tif)",
                                      "data/processed/NovaCidade.tif")

    products = st.sidebar.multiselect(
        "O que extrair", list(SEG_PRODUCTS) + ["Vias (pavimento)"],
        default=["Edificações"])

    roads = ""
    if "Vias (pavimento)" in products:
        roads = st.sidebar.text_input("Shapefile de logradouros (linhas)",
                                      f"data/raw/{city}/logradouros.shp")

    run = st.sidebar.button("🚀 Executar", type="primary")
    st.sidebar.info("O progresso detalhado (janelas processadas) aparece no terminal "
                    "onde você rodou `streamlit run app.py`.")

    if not run:
        st.write("Configure a entrada na barra lateral e clique em **Executar**.")
        st.markdown(
            "- **Ortofoto**: use um COG já em `data/processed/` "
            "(gere com `ecw_to_cog` se necessário).\n"
            "- **Vias**: precisa do shapefile de logradouros (linhas).\n"
            "- A saída de cada cidade fica em `outputs/<cidade>/`.\n"
            "- Para ver o que já foi gerado, use o modo **Ver resultados prontos**.")
        return

    if not Path(image).exists():
        st.error(f"Ortofoto não encontrada: {image}")
        return
    out_dir = ensure_dir(OUTPUTS / city)
    tmp = ensure_dir(out_dir / "_tmp")
    st.write(f"Saída em **`outputs/{city}/`**")

    for prod in products:
        with st.status(f"Processando: {prod}", expanded=True) as status:
            try:
                if prod == "Vias (pavimento)":
                    if not Path(roads).exists():
                        raise FileNotFoundError(f"Logradouros não encontrados: {roads}")
                    cfg = load_config("configs/road_surface.yaml")
                    out = str(out_dir / "vias.gpkg")
                    st.write("Classificando o pavimento de cada trecho…")
                    classify_roads(image, roads, out, cfg)
                else:
                    cfg_path, name = SEG_PRODUCTS[prod]
                    cfg = load_config(cfg_path)
                    prob = str(tmp / f"{name}_prob.tif")
                    out = str(out_dir / f"{name}.gpkg")
                    st.write("Inferência (janela deslizante)… pode demorar em imagens grandes.")
                    predict_image(image, prob, cfg)
                    st.write("Vetorizando…")
                    vectorize(prob, out, cfg)
                status.update(label=f"{prod} ✓ concluído", state="complete")
                _preview(out, prod)
            except Exception as e:  # noqa: BLE001
                status.update(label=f"{prod} — erro", state="error")
                st.exception(e)

    st.success(f"Concluído! Camadas em outputs/{city}/ — veja também no modo "
               "**Ver resultados prontos**.")


if mode.startswith("📂"):
    results_view()
else:
    run_view()
