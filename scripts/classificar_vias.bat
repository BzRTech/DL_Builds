@echo off
REM Classifica o pavimento das vias de uma cidade nova, usando o modelo ja treinado.
REM
REM Uso (caminhos entre aspas):
REM   classificar_vias.bat "NovaCidade" "data\raw\NovaCidade\ortofoto.tif" "data\raw\NovaCidade\logradouros.shp"
REM
REM Gera o COG (se ainda nao existir) e salva outputs\NovaCidade\vias.gpkg.
REM Se a ortofoto for ECW, exporte antes para GeoTIFF no QGIS.

if "%~3"=="" (
  echo Uso: classificar_vias.bat "Cidade" "caminho\ortofoto.tif" "caminho\logradouros.shp"
  exit /b 1
)

call conda activate dl_builds
python -m src.classify_roads_city --city "%~1" --image "%~2" --roads "%~3"
