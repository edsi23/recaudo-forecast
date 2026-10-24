# Entrega 3: búsqueda sistemática de literatura para la evidencia de novedad del modelo original.
# Consulta 4 bases con API pública (OpenAlex, Semantic Scholar, arXiv, Crossref) con las mismas cadenas,
# guarda el número de resultados y los 10 primeros de cada búsqueda en resultados/literatura_*.csv.
# Uso:  python revision_literatura.py
import datetime as dt
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

SALIDA = Path(__file__).parent / "resultados"
# (i) familia del modelo, (ii) problema de aplicación, (iii) sinónimos en inglés y español
CADENAS = [
    ("C1", '"forecast combination" regime tree "sales"'),
    ("C2", '"forecast combination" simplex weights decision tree'),
    ("C3", '"model-based recursive partitioning" forecast combination'),
    ("C4", '"dynamic ensemble selection" "sales forecasting"'),
    ("C5", '"gas station" OR "fuel station" daily sales forecasting machine learning'),
    ("C6", '"convenience store" daily sales forecasting ensemble'),
    ("C7", '"combinación de pronósticos" ventas diarias'),
    ("C8", '"pronóstico de ventas" "estaciones de servicio"'),
    ("C9", 'meta-learning forecast combination weights features FFORMA'),
    ("C10", 'stacking non-negative weights partition "forecast combination" retail'),
]


def _get(url, cabeceras=None):
    req = urllib.request.Request(url, headers={"User-Agent": "recaudo-forecast/1.0 (proyecto academico)", **(cabeceras or {})})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read()


def openalex(q):
    d = json.loads(_get("https://api.openalex.org/works?per-page=10&search=" + urllib.parse.quote(q)))
    return d["meta"]["count"], [(w.get("title"), w.get("publication_year"), w.get("doi")) for w in d["results"]]


def semantic_scholar(q):
    for intento in range(4):
        try:
            d = json.loads(_get("https://api.semanticscholar.org/graph/v1/paper/search?limit=10&fields=title,year,externalIds&query="
                                + urllib.parse.quote(q)))
            return d.get("total", 0), [(p.get("title"), p.get("year"), (p.get("externalIds") or {}).get("DOI")) for p in d.get("data", [])]
        except Exception:
            time.sleep(30)  # límite de peticiones del API público (429)
    return None, []


def arxiv(q):
    # arXiv no soporta OR/comillas igual: se buscan los términos en todos los campos
    terminos = " AND ".join(f"all:{t}" for t in q.replace('"', "").replace(" OR ", " ").split() if len(t) > 2)
    x = ET.fromstring(_get("http://export.arxiv.org/api/query?max_results=10&search_query=" + urllib.parse.quote(terminos)))
    ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
    total = int(x.find("o:totalResults", ns).text)
    return total, [(e.find("a:title", ns).text.strip().replace("\n", " "), e.find("a:published", ns).text[:4],
                    e.find("a:id", ns).text) for e in x.findall("a:entry", ns)]


def crossref(q):
    # Crossref ordena por relevancia pero cuenta cualquier coincidencia parcial: el total no es comparable con
    # las otras bases; se usa para revisar los 10 primeros resultados
    d = json.loads(_get("https://api.crossref.org/works?rows=10&select=title,issued,DOI&query.bibliographic=" + urllib.parse.quote(q)))
    m = d["message"]
    return m["total-results"], [((w.get("title") or [""])[0], (w.get("issued", {}).get("date-parts") or [[None]])[0][0], w.get("DOI"))
                                for w in m["items"]]


BASES = {"OpenAlex": openalex, "Semantic Scholar": semantic_scholar, "arXiv": arxiv, "Crossref": crossref}

if __name__ == "__main__":
    SALIDA.mkdir(exist_ok=True)
    hoy = dt.date.today().isoformat()
    conteos, primeros = [], []
    for cid, q in CADENAS:
        for base, f in BASES.items():
            try:
                n, top = f(q)
            except Exception as e:  # se registra el fallo en vez de ocultarlo
                n, top = None, []
                print(f"{base} {cid}: error {e}")
            conteos.append({"fecha": hoy, "id": cid, "cadena": q, "base": base, "resultados": n, "revisados": len(top)})
            primeros += [{"id": cid, "base": base, "rango": i + 1, "titulo": t, "anio": a, "doi_o_url": u}
                         for i, (t, a, u) in enumerate(top)]
            time.sleep(12.0 if base == "Semantic Scholar" else 1.2)
        print(cid, "listo", flush=True)
    pd.DataFrame(conteos).to_csv(SALIDA / "literatura_conteos.csv", index=False)
    pd.DataFrame(primeros).to_csv(SALIDA / "literatura_resultados.csv", index=False)
    print(pd.DataFrame(conteos).pivot(index="id", columns="base", values="resultados"))
