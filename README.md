# DL_Builds — Extração automática de feições urbanas de ortofotos

Sistema de deep learning que, a partir de **ortofotos**, extrai automaticamente:

- **Edificações** — segmentação (footprint dos telhados)
- **Quadras** — segmentação (blocos)
- **Lotes** — segmentação (parcelas)
- **Pavimento das vias** — classificação do tipo de piso de cada trecho de logradouro

Saída em **camadas vetoriais GIS** (GeoPackage), prontas para QGIS/ArcGIS, organizadas por
cidade em `outputs/<cidade>/`. Treina em cidades rotuladas e **aplica em cidades novas só com a
ortofoto**.

## Resultados do piloto

Treinado em 3 cidades (**Malta, Catolé, Tabira**) e validado em uma **cidade cega**
(**Itapororoca**, fora do treino):

| Produto | Método | Validação (treino) | Teste cego (cidade nova) |
|---|---|---|---|
| **Edificações** | Segmentação | IoU **0,90** | razão de área **1,00** ✅ |
| **Pavimento das vias** | Classificação | acurácia **92%** | acurácia **91%** ✅ |
| **Quadras** | Segmentação | IoU **0,77** | footprint bom; divisão de blocos aproximada |
| **Lotes** | Segmentação | IoU **0,77** | footprint razoável; limites finos são o limite do método |

> Edificações e vias generalizam muito bem para cidades novas. Quadras/lotes capturam bem a
> **área**, mas a **contagem/divisão** exata é mais difícil por segmentação pura (ver
> *Limitações*).

## Instalação

Requer **conda/mamba** (por causa do GDAL) e uma **GPU NVIDIA** para treino/inferência.

```bash
conda env create -f environment.yml      # ou: mamba env create -f environment.yml
conda activate dl_builds
```

> ⚠️ **ECW:** o driver de leitura de ECW não vem no GDAL do conda-forge. Verifique com
> `python -c "from src.utils import has_ecw_driver; print(has_ecw_driver())"`. Se `False`,
> converta a ortofoto para **GeoTIFF** no **QGIS** antes (o QGIS embute o driver).

## Organização dos dados

A pasta `data/` (e `outputs/`, `runs/`) é ignorada pelo git — os rasters são grandes demais.

```
data/
├── raw/<cidade>/         ortofoto (.tif/.ecw) + shapefiles de rótulo
└── processed/<cidade>.tif   COG gerado (compartilhado por todos os alvos)
outputs/<cidade>/         camadas previstas (edificacoes.gpkg, quadras.gpkg, ...)
runs/<alvo>/best.pt       modelos treinados (edificações/quadras/lotes/road_surface)
```

## Operação

### 1. Interface web (recomendado)

App Streamlit que roda **na máquina com GPU + dados** (sem upload de ortofotos gigantes):

```bash
streamlit run app.py
```

Abre em `http://localhost:8501`. Aponte um COG de `data/processed/`, escolha os produtos,
execute e baixe as camadas — tudo salvo em `outputs/<cidade>/`. Para a equipe acessar pela
rede: `streamlit run app.py --server.address 0.0.0.0` → `http://SEU_IP:8501`.

### 2. Cidade nova por linha de comando (só inferência, sem rótulos)

```bash
# 1) se for ECW, converta no QGIS para .tif; depois gere o COG:
python -m src.data_prep.ecw_to_cog --input data/raw/<cidade>/ortofoto.tif --output data/processed/<cidade>.tif

# 2) edificações:
python -m src.predict_city --image data/processed/<cidade>.tif --out outputs/<cidade>/edificacoes.gpkg

# 3) pavimento das vias (precisa da geometria dos logradouros):
python -m src.road_surface.predict --config configs/road_surface.yaml \
    --roads data/raw/<cidade>/logradouros.shp --image data/processed/<cidade>.tif \
    --out outputs/<cidade>/vias.gpkg
```

### 3. Treinar / adicionar uma cidade ao treino

Cada alvo tem sua config (`configs/buildings.yaml`, `quadras.yaml`, `lotes.yaml`). Para incluir
uma cidade, adicione-a na lista `cities` da config e rode (o `--only <cidade>` processa só a
cidade nova e **preserva** os tiles já feitos das outras — evita reprocessar ortofotos gigantes):

```bash
python -m src.data_prep.ecw_to_cog       --input data/raw/<cidade>/ortofoto.tif --output data/processed/<cidade>.tif
python -m src.data_prep.rasterize_labels --config configs/buildings.yaml --only <cidade>
python -m src.data_prep.tile             --config configs/buildings.yaml --only <cidade>
python -m src.data_prep.split            --config configs/buildings.yaml
python -m src.train.train                --config configs/buildings.yaml
```

Troque `--config` para `quadras.yaml` / `lotes.yaml` para os outros alvos.

### 4. Pavimento das vias — treino

Classificação (não segmentação): amostra patches ao longo de cada trecho → CNN (ResNet) de 3
classes (`STATUS`: não pavimentada / pavimentada / asfáltico) → votação por trecho. Rótulos com
grafias diferentes (acento/caixa) são normalizados automaticamente.

```bash
python -m src.road_surface.sample_patches --config configs/road_surface.yaml
python -m src.road_surface.train          --config configs/road_surface.yaml
python -m src.road_surface.predict        --config configs/road_surface.yaml --city <cidade>
```

### Utilitários de avaliação

```bash
# métricas por objeto (IoU) previsto x verdade
python -m src.eval.evaluate      --config configs/buildings.yaml --pred outputs/<cidade>/edificacoes.gpkg --truth <rotulo>.shp
# panorama rápido: nº e área, recortando à área rotulada (comparação justa)
python -m src.compare_counts     --pred outputs/<cidade>/edificacoes.gpkg --truth <rotulo>.shp --clip-to-truth --buffer-m 20
# juntar as camadas de uma cidade num único GeoPackage
python -m src.combine_layers     --out outputs/<cidade>/resultado.gpkg --layer edificacoes=outputs/<cidade>/edificacoes.gpkg ...
```

## Estrutura do projeto

```
app.py                        interface Streamlit
configs/                      buildings · quadras · lotes · road_surface (YAML)
src/
├── data_prep/  ecw_to_cog · rasterize_labels · tile · split   (--only p/ incremental)
├── train/      dataset · model · train                        (U-Net/DeepLabV3+, Dice+BCE)
├── inference/  predict                                        (sliding window + blending, memmap)
├── postprocess/vectorize                                      (polígonos, watershed por instância)
├── eval/       evaluate                                       (IoU por objeto)
├── road_surface/ sample_patches · dataset · train · predict · labels  (classificação de piso)
├── predict_city.py           edificações em cidade nova (1 comando)
├── compare_counts.py         nº/área previsto x verdade
└── combine_layers.py         junta camadas num .gpkg
```

Pontos técnicos que permitem escala (ortofotos de dezenas de gigapixels):
- **Inferência** acumula em `memmap` (disco) e grava em blocos — não carrega a imagem na RAM.
- **Rasterização** e **vetorização** operam em blocos/tiles.
- **Separação por instância** (watershed) roda em tiles; `min_peak_distance_m` (metros) mantém a
  separação consistente entre cidades com resoluções (GSD) diferentes.

## Limitações e próximos passos

- **Quadras/Lotes:** a segmentação acerta a área, mas fundir/dividir blocos e lotes com fidelidade
  precisa de outra abordagem — **cortar pela rede viária** (usando os logradouros) e, para lotes,
  subdividir com edificações + dado cadastral. É a evolução natural.
- **Edifícios/prédios encostados:** o watershed separa a maioria; casos densos podem exigir
  segmentação por instância dedicada (Mask R-CNN).
- **Generalização:** validada em 1 cidade cega; mais cidades no treino tende a melhorar ainda mais.
