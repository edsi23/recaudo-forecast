# Capturas de pantalla del tablero para el libro (cap. 21). Requiere el tablero corriendo en URL.
# Uso:  python capturas_dashboard.py http://127.0.0.1:8050
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8050"
SALIDA = Path(__file__).parent / "imagenes"

with sync_playwright() as p:
    nav = p.chromium.launch()
    pag = nav.new_page(viewport={"width": 1400, "height": 900})
    for ruta, nombre in [("/", "contexto"), ("/eda", "eda"), ("/modelos", "modelos")]:
        pag.goto(URL + ruta)
        pag.wait_for_load_state("networkidle")
        pag.wait_for_timeout(4000)
        pag.screenshot(path=SALIDA / f"dashboard_{nombre}.png", full_page=True)
        print("captura", nombre, flush=True)
    # predicción en vivo con un ejemplo real y el modelo original
    pag.goto(URL + "/modelos")
    pag.wait_for_load_state("networkidle")
    pag.wait_for_timeout(3000)
    pag.click("#sel-modelo")
    pag.keyboard.type("ACS")
    pag.keyboard.press("Enter")
    pag.click("#ejemplo")
    pag.keyboard.press("ArrowDown")
    pag.keyboard.press("Enter")
    pag.wait_for_timeout(1500)
    pag.click("#btn-predecir")
    pag.wait_for_timeout(6000)
    seccion = pag.locator("text=Predicción en vivo").locator("xpath=ancestor::div[contains(@class,'card')][1]")
    seccion.screenshot(path=SALIDA / "dashboard_prediccion.png")
    # entrada inválida: el formulario debe mostrar el error sin caerse
    pag.fill("input[id*='lag1']", "999")
    pag.click("#btn-predecir")
    pag.wait_for_timeout(2000)
    pag.locator("#alerta-pred").screenshot(path=SALIDA / "dashboard_validacion.png")
    # vista de tableta
    pag.set_viewport_size({"width": 820, "height": 1100})
    pag.goto(URL + "/eda")
    pag.wait_for_load_state("networkidle")
    pag.wait_for_timeout(4000)
    pag.screenshot(path=SALIDA / "dashboard_tableta.png")
    print("listo", flush=True)
    nav.close()
