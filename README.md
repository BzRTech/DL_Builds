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

## Roadmap

- [x] Fase 0 — Setup e inspeção de dados
- [ ] Fase 1 — Preparo de dados (conversão, rasterização, tiling, split)
- [ ] Fase 2 — Treino do modelo de edificações
- [ ] Fase 3 — Inferência + vetorização
- [ ] Fase 4 — Avaliação (IoU/F1, leave-one-city-out)
- [ ] Futuro — Quadras (rede viária) e Lotes (subdivisão)
