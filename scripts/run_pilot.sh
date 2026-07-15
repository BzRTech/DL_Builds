#!/usr/bin/env bash
# Executa o piloto de extração de edificações ponta a ponta.
# Pré-requisitos: ambiente 'dl_builds' ativo e dados em data/raw/ (ver README).
#
# Uso:  bash scripts/run_pilot.sh [caminho/para/config.yaml]
set -euo pipefail

CONFIG="${1:-configs/buildings.yaml}"
cd "$(dirname "$0")/.."

echo "=============================================="
echo " Piloto de edificações  |  config: ${CONFIG}"
echo "=============================================="

echo; echo ">> [0/6] Verificando leitura de ECW no GDAL..."
python -c "from src.utils import has_ecw_driver; \
print('  driver ECW disponível:', has_ecw_driver()); \
print('  (se False, converta os ECW para GeoTIFF no QGIS antes — ver README)')"

echo; echo ">> [1/6] Conversão ECW -> COG"
python -m src.data_prep.ecw_to_cog --config "${CONFIG}"

echo; echo ">> [2/6] Rasterização dos rótulos"
python -m src.data_prep.rasterize_labels --config "${CONFIG}"

echo; echo ">> [3/6] Tiling (imagem + máscara)"
python -m src.data_prep.tile --config "${CONFIG}"

echo; echo ">> [3b] Split treino/val/teste + normalização"
python -m src.data_prep.split --config "${CONFIG}"

echo; echo ">> [4/6] Treino do modelo"
python -m src.train.train --config "${CONFIG}"

echo; echo ">> [5/6] Inferência (cidades de teste)"
python -m src.inference.predict --config "${CONFIG}"

echo; echo ">> [5b] Vetorização"
python -m src.postprocess.vectorize --config "${CONFIG}"

echo; echo ">> [6/6] Avaliação"
python -m src.eval.evaluate --config "${CONFIG}"

echo; echo "Piloto concluído. Saídas em outputs/ (máscaras, vetores, relatórios)."
