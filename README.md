# Patente Engine 🚗

Sistema de reconocimiento de placas de vehículos chilenas basado en OpenCV y Tesseract OCR.

## Características

- 🎯 Detección específica para placas chilenas (HCJH72, ABC123, etc.)
- 🔧 Algoritmo basado en contornos y threshold binario
- 🚀 API REST simple con Flask
- 📊 Logging detallado del proceso
- ✅ Validación de formatos chilenos

## Algoritmo

1. **Conversión a escala de grises**
2. **Threshold binario** con múltiples valores
3. **Detección de contornos**
4. **Filtrado por aspect ratio** (2-6) y tamaño
5. **Recorte de región** de la placa
6. **Limpieza de bordes**
7. **OCR con Tesseract**
8. **Validación de formato chileno**

## Instalación

### Requisitos del sistema

```bash
# MacOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr

# Windows
# Descargar desde: https://github.com/UB-Mannheim/tesseract/wiki
```

### Dependencias Python

```bash
pip install flask opencv-python pytesseract scikit-image numpy
```

## Uso

### Ejecutar el servicio

```bash
python lpr_app.py
```

El servidor se ejecutará en `http://localhost:5001`

### Endpoints

#### Health Check
```bash
GET /health
```

#### Detectar Placa
```bash
POST /detect-license-plate
POST /detect-license-plate/v1

Content-Type: multipart/form-data
Body: image=@path/to/image.jpg
```

### Ejemplos

```bash
# Health check
curl http://localhost:5001/health

# Detectar placa
curl -X POST \
  http://localhost:5001/detect-license-plate/v1 \
  -F "image=@mi_imagen.jpg"
```

### Respuesta exitosa
```json
{
  "placa": "HCJH72"
}
```

### Sin placa detectada
```json
{
  "placa": null
}
```

## Formatos de placas soportados

- `AAAA##` - 4 letras, 2 números (ej: HCJH72)
- `AA####` - 2 letras, 4 números (ej: AB1234)
- `AAA###` - 3 letras, 3 números (ej: ABC123)

## Estructura del proyecto

```
patente_engine/
├── lpr_app.py          # Aplicación principal
├── simple_app.py       # Versión alternativa
├── app.py             # Implementación con EasyOCR
├── Dockerfile         # Para containerización
├── requirements.txt   # Dependencias
└── README.md         # Este archivo
```

## Créditos

Algoritmo basado en el trabajo de [santifiorino/license-plate-recognition](https://github.com/santifiorino/license-plate-recognition) y adaptado para placas chilenas.

## Licencia

MIT License