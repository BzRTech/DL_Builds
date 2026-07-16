@echo off
REM Extrai edificacoes de uma nova ortofoto usando o modelo ja treinado.
REM
REM Uso (arraste ou informe os caminhos entre aspas):
REM   prever_nova_cidade.bat "data\raw\NovaCidade\ortofoto.tif" "outputs\NovaCidade.gpkg"
REM
REM Se a ortofoto for ECW, converta antes para GeoTIFF no QGIS.

if "%~2"=="" (
  echo Uso: prever_nova_cidade.bat "caminho\ortofoto.tif" "saida\edificacoes.gpkg"
  exit /b 1
)

call conda activate dl_builds
python -m src.predict_city --image "%~1" --out "%~2"
