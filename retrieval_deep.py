# -*- coding: utf-8 -*-
"""Djup retrieval for AIPrototype v7: HAMTA-OCH-LAS + lokal embedding-
reranker (och byggstenar for agent-loopen).

Idag laser huvudpipelinen bara Braves korta snippets. Denna modul hamtar
SJALVA sidorna for topptraffarna, rensar bort menyer/reklam (trafilatura),
delar upp brodtexten i stycken och rankar dem efter semantisk likhet mot
fragan med en liten lokal GGUF-embeddingmodell (bge-small, kors pa CPU sa
den inte konkurrerar med 9B:an pa GPU:t). Da hittas fakta som ligger i
tabeller/brodtext men inte i snippet (medaljligor, styrrantor, vinnare).

Allt ar defensivt: varje fel -> None, sa den vanliga snippet-vagen tar
over som fallback. Ingen ny hard beroende (trafilatura/numpy finns redan).
"""
import os
import re
import numpy as np
import requests

try:
    import trafilatura
except Exception:  # pragma: no cover
    trafilatura = None

from llama_cpp import Llama

_BASE = os.path.dirname(os.path.abspath(__file__))
_EMB_SOKVAG = os.path.join(_BASE, "models", "bge-small-en-v1.5-f16.gguf")
_EMB = None
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _logg(msg):
    # Anvand AIPrototype:s debug-logg BARA om modulen redan ar laddad (dvs.
    # nar vi kors inifran appen). Att importera den fristaende skulle starta
    # hela GUI:t (root.mainloop) och hanga - sa vi ror den aldrig sjalva.
    import sys as _sys
    _m = _sys.modules.get("AIPrototype")
    if _m is not None and hasattr(_m, "_debug_logg"):
        try:
            _m._debug_logg("[deep] " + msg)
        except Exception:
            pass


def _ladda_emb():
    """Lazy-laddar embeddingmodellen (CPU). None om filen saknas/fel."""
    global _EMB
    if _EMB is None:
        if not os.path.isfile(_EMB_SOKVAG):
            _logg("embeddingmodell saknas: " + _EMB_SOKVAG)
            return None
        try:
            _EMB = Llama(model_path=_EMB_SOKVAG, embedding=True,
                         n_gpu_layers=0, n_ctx=512, n_batch=512, verbose=False)
            _logg("embeddingmodell laddad (CPU)")
        except Exception as e:
            _logg("kunde inte ladda embeddingmodell: %r" % e)
            _EMB = None
    return _EMB


def _embed(texter):
    """Returnerar en (n, d)-array med L2-normaliserade embeddingar, eller
    None vid fel."""
    m = _ladda_emb()
    if m is None or not texter:
        return None
    try:
        # OBS: m.embed(LISTA) hanger sig i denna Vulkan-build - embedda
        # en strang i taget (encode-vagen, ~3 ms/styck) och stapla.
        vecs = []
        for t in texter:
            v = m.embed(t if isinstance(t, str) else str(t))
            vecs.append(np.asarray(v, dtype=np.float32).reshape(-1))
        arr = np.vstack(vecs)
        n = np.linalg.norm(arr, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return arr / n
    except Exception as e:
        _logg("embed-fel: %r" % e)
        return None


def hamta_sidtext(url, timeout=8, maxbytes=2_000_000):
    """Hamtar en sida och returnerar ren brodtext (trafilatura), inkl.
    tabeller. None vid fel/icke-HTML/tom text."""
    if not url or trafilatura is None:
        return None
    try:
        r = requests.get(url, timeout=timeout,
                         headers={"User-Agent": _UA, "Accept-Language": "en,sv"},
                         stream=True)
        if r.status_code != 200:
            return None
        ct = (r.headers.get("Content-Type") or "").lower()
        if "html" not in ct and "text" not in ct and ct:
            return None
        rader = r.raw.read(maxbytes, decode_content=True)
        html = rader.decode(r.encoding or "utf-8", "ignore")
        txt = trafilatura.extract(
            html, include_comments=False, include_tables=True,
            favor_recall=True, no_fallback=False)
        if txt and len(txt) > 120:
            return txt
    except Exception as e:
        _logg("hamta_sidtext-fel %s: %r" % (url[:60], e))
    return None


_STYCKE = re.compile(r"\n{2,}")


def _chunka(text, mal_tecken=700, min_tecken=120):
    """Delar text i stycken och slar ihop till ~mal_tecken-bitar (under
    bge:s 512-token-tak). Behaller radbrytningar sa tabellrader halls ihop."""
    bitar = []
    buff = ""
    for st in _STYCKE.split(text):
        st = st.strip()
        if not st:
            continue
        if len(buff) + len(st) + 1 <= mal_tecken:
            buff = (buff + "\n" + st) if buff else st
        else:
            if len(buff) >= min_tecken:
                bitar.append(buff)
            # ett langt stycke: hugg det i mal_tecken-langder
            if len(st) > mal_tecken:
                for i in range(0, len(st), mal_tecken):
                    bitar.append(st[i:i + mal_tecken])
                buff = ""
            else:
                buff = st
    if len(buff) >= min_tecken:
        bitar.append(buff)
    return bitar


def rerank_passager(fraga_en, texter_med_url, topp=6, max_chunks=120):
    """Chunkar alla sidtexter, embeddar fraga + chunks, returnerar de
    topp mest lika (text, url, score). [] vid fel."""
    chunks = []
    for text, url in texter_med_url:
        for c in _chunka(text):
            chunks.append((c, url))
            if len(chunks) >= max_chunks:
                break
        if len(chunks) >= max_chunks:
            break
    if not chunks:
        return []
    qv = _embed([fraga_en])
    cv = _embed([c for c, _ in chunks])
    if qv is None or cv is None:
        return []
    sims = cv @ qv[0]
    ordning = np.argsort(-sims)[:topp]
    return [(chunks[i][0], chunks[i][1], float(sims[i])) for i in ordning]


from concurrent.futures import ThreadPoolExecutor


def djup_kontext(fraga_en, traffar, n_sidor=4, topp=6):
    """HAMTA-OCH-LAS: hamtar topp n_sidor URL:er ur Brave-traffarna
    (parallellt), rankar styckena mot fragan och bygger en sok-kontext av
    de mest relevanta passagerna. Returnerar en textkontext (samma format
    som snippet-kontexten: '- <text> (Kalla: <url>)') eller None -> da
    faller anroparen tillbaka pa snippets.

    traffar: lista av dict med minst 'url' (och garna 'titel'/'body')."""
    urls = []
    for t in (traffar or []):
        u = (t.get("url") or "").strip()
        # hoppa over PDF/media - trafilatura ar for HTML.
        if u and not re.search(r"\.(pdf|jpg|jpeg|png|mp4|zip)(\?|$)", u, re.I):
            urls.append(u)
        if len(urls) >= n_sidor:
            break
    if not urls:
        return None
    try:
        with ThreadPoolExecutor(max_workers=min(4, len(urls))) as ex:
            texter = list(ex.map(hamta_sidtext, urls))
    except Exception as e:
        _logg("parallell hamtning fel: %r" % e)
        texter = [hamta_sidtext(u) for u in urls]
    tmu = [(txt, url) for url, txt in zip(urls, texter) if txt]
    _logg("hamtade %d/%d sidor" % (len(tmu), len(urls)))
    if not tmu:
        return None
    topp_chunks = rerank_passager(fraga_en, tmu, topp=topp)
    if not topp_chunks:
        return None
    rader = []
    for c, url, score in topp_chunks:
        bit = re.sub(r"\s+", " ", c).strip()[:600]
        rader.append("- %s (Kalla: %s)" % (bit, url))
    return "\n".join(rader)


if __name__ == "__main__":
    # Fristaende test: hamta en sida och ranka passager mot en fraga.
    import json
    import sys
    prov = [{"url": "https://en.wikipedia.org/wiki/2025_Ballon_d%27Or"}]
    fraga = "Who won the 2025 men's Ballon d'Or?"
    ut = djup_kontext(fraga, prov, n_sidor=1, topp=3)
    print("=== FRAGA:", fraga)
    print(ut)
    with open(os.path.join(_BASE, "_deep_test_out.txt"), "w",
              encoding="utf-8") as f:
        f.write(str(ut))
