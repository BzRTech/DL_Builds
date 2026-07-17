# DL_Builds — Extração de Edificações de Ortofotos com Deep Learning

Sistema de deep learning para extrair automaticamente **edificações** (e, em fases
futuras, **quadras** e **lotes**) a partir de **ortofotos**, gerando **camadas
vetoriais GIS** (GeoPackage/Shapefile) prontas para QGIS/ArcGIS.

Este repositório é o **piloto** focado em edificações. Ele monta um pipeline
completo e reutilizável para processar novas cidades de forma automatizada.

## Por que começar por edificações

Os três alvos têm dificuldades muito diferentes para extração por DL:

| Alvo         | Visível na imagem? | Abordagem                                            |
|--------------|--------------------|-----------------------------------------------------|
| Edificações  | Sim (telhados)     | Segmentação semântica direta ✅ (este piloto)        |
| Quadras      | Indireto           | Extrair rede viária → quadras = "negativo" das ruas |
| Lotes        | Muitas vezes não   | Subdividir quadras + edificações + dados auxiliares  |

Edificações entregam valor rápido e o pipeline serve de base para os demais alvos.

## Pipeline

```
ECW ortofoto ─┐
              ├─► [1] Preparo ─► tiles (img+máscara) ─► [2] Treino ─► modelo (.pt)
Shapefile ────┘                                                          │
                                                                         ▼
Nova ortofoto ─────────────────► [3] Inferência (sliding window) ─► máscara georref.
                                                                         │
                                                                         ▼
                                          [4] Vetorização/regularização ─► [5] Avaliação
                                                                         │
                                                                         ▼
                                                        GeoPackage/Shapefile (edificações)
```

## Instalação

Requer **conda/mamba** (por causa do GDAL) e uma **GPU NVIDIA** para o treino.

```bash
mamba env create -f environment.yml      # ou: conda env create -f environment.yml
conda activate dl_builds
```

> ⚠️ **Leitura de ECW:** o driver de leitura de ECW no GDAL depende do SDK
> proprietário da ERDAS, que **não vem** no GDAL do conda-forge. Verifique com
> `python -c "from osgeo import gdal; print('ECW' in [gdal.GetDriver(i).ShortName for i in range(gdal.GetDriverCount())])"`.
> Se retornar `False`, converta os ECW para GeoTIFF/COG usando o **QGIS** (que
> empacota o driver) antes de rodar o pipeline, ou instale um build do GDAL com
> suporte a ECW. Os scripts abaixo aceitam tanto ECW (se o driver existir) quanto
> GeoTIFF/COG.

## Uso (ponta a ponta)

Organize os dados assim (a pasta `data/` é ignorada pelo git):

```
data/raw/
├── cidade_a/  (ortofoto.ecw  edificacoes.shp ...)
└── cidade_b/  (ortofoto.ecw  edificacoes.shp ...)
```

Ajuste caminhos e CRS em `configs/buildings.yaml` e rode:

```bash
bash scripts/run_pilot.sh
```

Ou execute estágio a estágio — ver `configs/buildings.yaml` e os módulos em `src/`.

## Estrutura

```
src/
├── data_prep/   ecw_to_cog.py · rasterize_labels.py · tile.py · split.py
├── train/       dataset.py · model.py · train.py
├── inference/   predict.py   (sliding window + blending)
├── postprocess/ vectorize.py (máscara → polígono, regularização)
└── eval/        evaluate.py
```

## Usar o modelo em uma cidade nova (só inferência)

Quando chegar uma **ortofoto nova**, você **não** precisa de rótulos nem re-treinar —
o modelo já treinado (`runs/buildings/best.pt`) extrai as edificações direto. Um comando:

```bash
python -m src.predict_city --image data/raw/NovaCidade/ortofoto.tif --out outputs/NovaCidade.gpkg
```

No Windows há também um atalho (duplo-clique ou linha de comando):

```bat
scripts\prever_nova_cidade.bat "data\raw\NovaCidade\ortofoto.tif" "outputs\NovaCidade.gpkg"
```

> Se a ortofoto nova for **ECW**, converta antes para GeoTIFF no QGIS (mesmo passo do piloto).
> O resultado é um `GeoPackage` com os polígonos das edificações, pronto para o QGIS.

## Interface (fase futura)

Está previsto um **app web local** (Streamlit/Gradio) para: escolher/enviar a ortofoto,
disparar a extração com **barra de progresso**, e **visualizar os polígonos sobre a imagem
em um mapa**, com botão de download do GeoPackage. Observação: ortofotos têm vários GB, então
o fluxo prático é apontar para o arquivo no disco/servidor (em vez de "upload" pelo navegador)
e renderizar o resultado num mapa (Leaflet/folium). Enquanto isso, o **QGIS** já serve como
interface para abrir e revisar as camadas geradas.

## Roadmap

- [x] Fase 0 — Setup e inspeção de dados
- [ ] Fase 1 — Preparo de dados (conversão, rasterização, tiling, split)
- [ ] Fase 2 — Treino do modelo de edificações
- [ ] Fase 3 — Inferência + vetorização
- [ ] Fase 4 — Avaliação (IoU/F1, leave-one-city-out)
- [x] Inferência em cidade nova (`src/predict_city.py`)
- [x] Suporte a múltiplos alvos (`configs/quadras.yaml`, `configs/lotes.yaml`)
- [ ] Interface web (upload + progresso + mapa) — Streamlit/Gradio

## Múltiplos alvos: quadras e lotes

O mesmo pipeline atende os três alvos — muda-se só o rótulo e o alvo (`project.target`
namespeia máscara/tiles/modelo, e a ortofoto COG é reutilizada). Rode os mesmos comandos
trocando o `--config`:

```bash
# Quadras (segmentação + componentes conectados; separadas pelas ruas)
python -m src.data_prep.rasterize_labels --config configs/quadras.yaml
python -m src.data_prep.tile            --config configs/quadras.yaml
python -m src.data_prep.split           --config configs/quadras.yaml
python -m src.train.train               --config configs/quadras.yaml
python -m src.inference.predict         --config configs/quadras.yaml --city Malta
python -m src.postprocess.vectorize     --config configs/quadras.yaml --city Malta
```

> **Lotes** (`configs/lotes.yaml`) são experimentais: muitos limites são jurídicos/invisíveis
> na imagem, então a extração por DL tende a ser limitada. O caminho robusto é subdividir as
> quadras com edificações + limites visíveis + dado cadastral.
