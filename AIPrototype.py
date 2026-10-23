import tkinter as tk
from tkinter import filedialog
import threading
import os
import sys
import re
import time
import datetime
import difflib
import gc
import requests
import certifi
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_cpp import Llama
from PIL import Image, ImageDraw, ImageTk

# Sakerstall att modulen sprak.py hittas oavsett hur AIPrototype.py
# koras (som script, som .exe, eller via exec i regressionstest).
_MODUL_MAPP = os.path.dirname(os.path.abspath(__file__))
if _MODUL_MAPP not in sys.path:
    sys.path.insert(0, _MODUL_MAPP)

# DJUP RETRIEVAL (hamta-och-las + lokal embedding-reranker). Valfri: om
# modulen eller embeddingmodellen saknas -> None och appen kor precis som
# forr (snippet-vagen). Importeras defensivt.
try:
    import retrieval_deep
except Exception as _e_rd:
    retrieval_deep = None
from sprak import _str  # noqa: E402

# PyInstaller-fix: 'requests' hittar ibland inte SSL-certifikaten i en
# paketerad .exe, vilket gor att HTTPS-anrop (t.ex. Wikipedia) tyst
# misslyckas. Peka explicit pa certifi:s certifikat-paket.
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

_DATUM_IDAG = datetime.date.today().isoformat()

# ============ BRAVE SEARCH API (betald) ============
# Ersatter DDGS/DuckDuckGo med Brave Search API. Skaffa en nyckel pa
# https://api-dashboard.search.brave.com/ (t.ex. planen "Data for Search").
#
# HUR DU LAGGER IN DIN NYCKEL - valj ETT av tva satt:
#   1) Satt miljovariabeln BRAVE_API_KEY (Windows: setx BRAVE_API_KEY "din-nyckel")
#   2) ELLER byt raden nedan mot:  _BRAVE_API_KEY = "din-nyckel"
_BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "").strip() or ""

_BRAVE_WEB_URL = "https://api.search.brave.com/res/v1/web/search"
_BRAVE_NEWS_URL = "https://api.search.brave.com/res/v1/news/search"
_HTML_TAGG = re.compile(r"<[^>]+>")
# Sokning ar tillganglig endast om en API-nyckel finns.
_SOKNING_TILLGANGLIG = bool(_BRAVE_API_KEY)

# A4: enkel per-session-cache for Brave-svar (nyckel: query+count+nyheter).
# Sparar API-kostnad + latens nar samma/overlappande fraga sok:s igen
# (t.ex. re-fragor i samma session). Cachar bara lyckade traffar - tomma/
# felande svar cachas EJ, sa transienta natverksfel kan forsokas igen.
_BRAVE_CACHE = {}

# ============ FILVAGAR ============
if getattr(sys, "frozen", False):
    BASKATALOG = os.path.dirname(sys.executable)
else:
    BASKATALOG = os.path.dirname(os.path.abspath(__file__))

MODELL_SOKVAG = os.path.join(BASKATALOG, "models",
                             "Qwen3.5-9B-Q4_K_M.gguf")
SOK_MODELL_SOKVAG = os.path.join(BASKATALOG, "models",
                                 "qwen2.5-1.5b-instruct-q4_k_m.gguf")
# KOD-LAGE: kodfragor besvaras av 9B-modellen (MODELL_SOKVAG) med en
# kod-inriktad systemprompt (_KOD_SYSTEM). Tidigare fanns en dedikerad
# Coder-7B som vaxlades in pa GPU:t; den togs bort - 9B:an klarar kod bra
# och slipper ~15-25 s modellvaxling.

# STEG 3: Qwen borttagen. Denna byggare (som fortfarande heter
# bygg_qwen_prompt av bakatkompatibilitet) producerar nu LLAMA-format,
# sa att alla gamla "sok-hjalp"-anrop (klassificering, oversattning,
# titelval, entitetsextraktion, sprakdetektering) kors pa Llama-modellen
# istallet for en separat Qwen-modell.
def bygg_qwen_prompt(system_msg, user_msg):
    # Qwen3.5 ar en "tankande" modell. /no_think stanger av <think>-blocken
    # sa svaren inte fylls av intern resonemangstext (och sa att appens
    # korta klassificerar-/extraktions-svar blir rena JA/NO/text).
    return (
        "<|im_start|>system\n"
        f"{system_msg}\n/no_think<|im_end|>\n<|im_start|>user\n"
        f"{user_msg}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


# Qwen3 ChatML-stopp.
_QWEN_STOPP = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]


# Qwen3 anvander ChatML-promptformat (<|im_start|>/<|im_end|>).
def bygg_llama_prompt(system_msg, user_msg):
    return (
        "<|im_start|>system\n"
        f"{system_msg}\n/no_think<|im_end|>\n<|im_start|>user\n"
        f"{user_msg}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _bygg_rullande_sammanfattning(slappta_varv):
    # A3: extraktiv (INGEN modell) rullande sammanfattning av de aldsta
    # turer som inte langre ryms i teckenbudgeten. Bevarar tradens borjan
    # (tur 1:s fraga + kort svarsutdrag) och listar amnena for ovriga
    # bortslappta turer, sa langa samtal inte "tappar tur 1".
    try:
        if not slappta_varv:
            return ""
        rader = ["[Tidigare i SAMMA konversation (behall som kontext):]"]
        _v0 = slappta_varv[0]
        _q0 = (_v0.get("fraga", "") or "").strip()
        _a0 = (_v0.get("svar", "") or "").strip()
        if _q0:
            rader.append("- Samtalet inleddes med: " + _q0[:400])
        if _a0:
            _a0s = _a0[:240].rsplit(" ", 1)[0]
            rader.append("  (ditt svar slog fast: " + _a0s + " …)")
        _ovriga = slappta_varv[1:]
        _amnen = []
        for _v in _ovriga:
            _q = (_v.get("fraga", "") or "").strip()
            if _q:
                _amnen.append(_q[:90])
        if _amnen:
            rader.append("- Anvandaren fragade sedan aven om: "
                         + " | ".join(_amnen[:8]))
        return "\n".join(rader)
    except Exception:
        return ""


def bygg_llama_prompt_med_historik(system_msg, historik, ny_fraga):
    """Som bygg_llama_prompt, men inkluderar tidigare varv i SAMMA
    konversation (finns bara i minnet under appens korning - forsvinner
    vid omstart, sparas aldrig till disk)."""
    # GREPP 3: begransa historiken sa prompten aldrig spranger kontextfonstret
    # (n_ctx=4096). Behall de SENASTE varven som ryms inom en teckenbudget,
    # slapp de aldsta. Utan detta kraschar langa samtal med
    # "Requested tokens (4141) exceed context window of 4096" pa varv 6-7.
    _HIST_BUDGET = 4500  # tecken (~1300 tokens) - lamnar gott om plats for svar
    _hela = historik or []
    _trim, _summa = [], 0
    for _v in reversed(_hela):
        _bit = len(_v.get("fraga", "")) + len(_v.get("svar", ""))
        if _summa + _bit > _HIST_BUDGET and _trim:
            break
        _trim.append(_v); _summa += _bit
    _trim.reverse()
    # A3: turer som inte fick plats slapps inte tyst - vav in en kort
    # rullande sammanfattning av dem i systemmeddelandet sa tradens
    # borjan bevaras aven i mycket langa samtal.
    _slappt = _hela[:len(_hela) - len(_trim)]
    if _slappt:
        _rull = _bygg_rullande_sammanfattning(_slappt)
        if _rull:
            system_msg = system_msg + "\n\n" + _rull
    historik = _trim
    delar = ["<|im_start|>system\n",
             system_msg, "\n/no_think", "<|im_end|>\n"]
    for varv in historik:
        delar.append("<|im_start|>user\n")
        delar.append(varv["fraga"])
        delar.append("<|im_end|>\n")
        delar.append("<|im_start|>assistant\n")
        delar.append(varv["svar"])
        delar.append("<|im_end|>\n")
    delar.append("<|im_start|>user\n")
    delar.append(ny_fraga)
    delar.append("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(delar)


_LLAMA_STOPP = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]

# ============ SPRAKDETEKTOR ============
# Starka signalord: finns ENDAST pa svenska, aldrig som vanliga engelska ord.
# Racker med ett enda for att klassificera som svenska.
_SVENSKA_ORD_STARKA = {
    "ar", "har", "hade", "blir", "blev", "borde", "kunde", "skulle",
    "kommer", "kom", "gor", "gjort", "gick", "gar",
    "kan", "ska", "vill", "maste", "far",
    "vad", "vem", "vilka", "vilken", "vilket", "hur", "nar", "varfor",
    "jag", "mig", "min", "mitt", "mina", "du", "dig", "din", "ditt", "dina",
    "han", "hon", "honom", "henne", "vi", "oss", "vart", "vara",
    "och", "eller", "men", "att", "till", "med", "pa",
    "ett", "det", "den", "de", "detta", "dessa",
    "inte", "aven", "ocksa", "bara", "redan", "annu", "sen",
    "hej", "hejsan", "tack", "nasta", "senaste", "beratta",
}
# Svaga/tvetydiga ord: finns aven som vanliga engelska ord ("for", "en",
# "om" etc). Kravs minst 2 av dessa (eller 1 stark) for att bli svenska,
# sa en enstaka engelsk mening som rakar innehalla "for" inte lurar oss.
_SVENSKA_ORD_SVAGA = {"om", "for", "av", "en", "som"}


def detektera_sprak(fraga):
    text = fraga.lower()
    if any(c in text for c in "åäö"):
        return "sv"
    ord_i_fraga = set(re.findall(r"[a-zåäö]+", text))
    if ord_i_fraga & _SVENSKA_ORD_STARKA:
        return "sv"
    if len(ord_i_fraga & _SVENSKA_ORD_SVAGA) >= 2:
        return "sv"
    return "en"


# ============ GLITCH-DETEKTOR ============
_CJK_PATTERN = re.compile(
    r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_LACKTA_TAGGAR = ["<|im_end|>", "<|im_start|>", "<|endoftext|>",
                   "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>",
                   "<|begin_of_text|>", "\nuser\n", "\nsystem\n",
                   "\nUser\n", "\nSystem\n",
                   "SKRIV INTE OM", "Do NOT rewrite", "hitta ALDRIG pa",
                   "NEVER invent facts", "Svara PA SVENSKA", "Reply IN ENGLISH",
                   "Uppdragstaken"]


_HEDGE_FRASER = [
    "saknar realtidsinformation", "saknar aktuell information",
    "vet inte vem", "kan inte verifiera", "traningsdata",
    "min trainingsdata", "har inte tillgang till realtid",
    "i lack real-time", "don't have access to real-time",
    "my training data", "i cannot verify",
    # Utokade svenska hedge-triggers: fangar upp fler vaga/osaker-fraser
    # som Llama ofta anvander pa svenska nar den egentligen borde salt
    # 'vet inte' hart istallet for att blanda in hallucinerad extra info.
    "kunde inte hitta", "saknar information", "har ingen aktuell",
    "kan inte garantera", "min kunskap ar fran", "min kunskap ar forlagd",
    "aktuella uppgifter", "utan tillgang till internet",
    "i don't have current", "i'm not aware of",
    "i couldn't find any", "no reliable information",
]


def ar_svaret_trasigt(svar, hade_fakta=False, sprak=None):
    if not svar or len(svar.strip()) < 2:
        return True
    if _CJK_PATTERN.search(svar):
        return True
    for tagg in _LACKTA_TAGGAR:
        if tagg.lower() in svar.lower():
            return True
    rader = [r.strip() for r in svar.split("\n") if r.strip()]
    for i in range(len(rader) - 2):
        if rader[i] == rader[i + 1] == rader[i + 2]:
            return True
    # Om vi FAKTISKT gav modellen en verifierad fakta, men den anda
    # hedgar/pastar att den saknar info - det ar ett fel, inte arlighet.
    if hade_fakta:
        svar_lower = svar.lower()
        for fras in _HEDGE_FRASER:
            if fras in svar_lower:
                return True
    # Sprakkonsistens: Llama 3B tenderar att flippa till svenska nar
    # kontexten (kallor/fakta) innehaller svenska ord. Om vi bett om
    # engelska men fick svenska - trasigt, prova igen. OBS: kolla BARA
    # 'a-ring' (unikt svensk bokstav) - INTE ao/oo, som ocksa finns pa
    # tyska m.fl. sprak som vi nu speglar tillbaka till anvandaren.
    if sprak == "en" and any(c in svar for c in "åÅ"):
        return True
    # Ovantat skript: ett sv/en-svar ska aldrig vara pa kyrilliska (t.ex.
    # 'Россия' istallet for 'Russia') - da har modellen glidit over till
    # fel sprak fran en kalla, prova igen sa vi far ratt skrift.
    if sprak in ("sv", "en") and re.search(r"[Ѐ-ӿ]", svar):
        return True
    return False


# ============ MODELL ============
llm = None
llm_sok = None


# Qwen3.5 kan anda emellanat skriva ut ett (ibland tomt) <think>...</think>
# -block trots /no_think. Strippa bort det centralt sa varken slutsvar
# eller interna klassificerar-/extraktions-svar innehaller taggarna.
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _strip_think(text):
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


class _NoThinkLlama:
    """Tunn omslagsklass runt Llama: rensar <think>-block ur alla
    ICKE-strommande completions (klassificering, extraktion, oversattning,
    m.m.). Strommande anrop (stream=True) slapps igenom oforandrade -
    de rensas istallet i _generera nar texten satts ihop."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __call__(self, *args, **kwargs):
        if kwargs.get("stream"):
            return object.__getattribute__(self, "_inner")(*args, **kwargs)
        out = object.__getattribute__(self, "_inner")(*args, **kwargs)
        try:
            for ch in out["choices"]:
                if ch.get("text"):
                    ch["text"] = _strip_think(ch["text"])
        except Exception:
            pass
        return out

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)


def ladda_modell(status_label):
    global llm, llm_sok

    def _satt_text(t):
        root.after(0, lambda: status_label.config(text=t))

    _satt_text("Laddar Qwen3.5 9B-modellen... / Loading the Qwen3.5 9B model...")
    try:
        if not os.path.isfile(MODELL_SOKVAG):
            raise FileNotFoundError(f"Hittar ingen modellfil pa {MODELL_SOKVAG}")
        # Qwen3.5 9B (Q4_K_M) ar ~5.5-6 GB och far inte alltid plats HELT i
        # 6 GB VRAM (RTX 2060) tillsammans med KV-cachen. Lagg darfor sa
        # manga lager som mojligt pa GPU och lat resten ligga pa CPU (16 GB
        # RAM finns). Vi provar en fallande lista med n_gpu_layers och tar
        # forsta som laddar utan att VRAM tar slut - robust mot exakta
        # storlekar och kan finjusteras uppat om det finns VRAM-marginal.
        # Modellen har 32 lager. Tidigare lastes bara 30 pa GPU, sa 2 lager
        # + KV-cachen lag pa CPU = markant langsammare token/sek. Med flash
        # attention + q8-kvantiserad KV-cache krymper VRAM-behovet sa att
        # HELA modellen (alla 32 lager) far plats i 6 GB -> allt pa GPU =
        # snabbare svar. n_ctx haller vi kvar pa 4096 sa minnesdjupet (6
        # historikvarv) inte forsamras. Faller automatiskt tillbaka till den
        # gamla, sakra konfigen om det snabba laget inte gar att ladda pa
        # detta GPU/denna llama.cpp-build.
        _Q8 = 8  # GGML_TYPE_Q8_0 - halverar KV-cachens VRAM mot f16
        _konfigar = (
            ("snabb",
             dict(n_ctx=4096, flash_attn=True, type_k=_Q8, type_v=_Q8),
             (-1, 33, 32, 31, 30, 24, 18, 12, 6, 0)),
            ("saker",
             dict(n_ctx=4096),
             (30, 24, 18, 12, 6, 0)),
        )
        llm = None
        _sista_fel = None
        for _knamn, _kargs, _lagerlista in _konfigar:
            for _lager in _lagerlista:
                try:
                    llm = Llama(
                        model_path=MODELL_SOKVAG,
                        n_threads=max(1, (os.cpu_count() or 4) - 1),
                        n_batch=1024,
                        n_gpu_layers=_lager,
                        verbose=False,
                        **_kargs,
                    )
                    _debug_logg(
                        f"Modell laddad ({_knamn}) med n_gpu_layers={_lager}")
                    break
                except Exception as _lfel:
                    _sista_fel = _lfel
                    _debug_logg(
                        f"[{_knamn}] n_gpu_layers={_lager} misslyckades: "
                        f"{_lfel!r}")
                    llm = None
            if llm is not None:
                break
        if llm is None:
            raise _sista_fel or RuntimeError("Kunde inte ladda modellen")
        # Slå in modellen så alla completions får <think>-block bortrensade.
        llm = _NoThinkLlama(llm)
        # STEG 3: ingen separat Qwen sok-modell langre. Llama skoter nu
        # BADE konversationen OCH alla sok-hjalpuppgifter (klassificering,
        # extraktion, oversattning, titelval, entitet, sprakdetektering).
        # Sparar ~1 GB VRAM extra + snabbare start, och Llama 3B ar
        # starkare an gamla Qwen 1.5B pa just dessa uppgifter.
        llm_sok = llm
        _debug_logg(f"retrieval_deep (djup-vag) aktiv: "
                    f"{retrieval_deep is not None}")
        _satt_text("")
    except Exception as e:
        _satt_text(f"Fel vid laddning / Loading error: {e}")


# ============ BEHOVER-AKTUELL-INFO-CHECK ============
_ROLLER = (
    r"president|statsminister|premiarminister|kung|drottning|monark|"
    r"\bvd\b|\bceo\b|ordforande|ledare|borgmastare|kansler|"
    r"utrikesminister|finansminister|guvernor|partiledare"
)
_KRAVER_AKTUELL = re.compile(
    r"\b(senaste?|nyheter|aktuell[at]?|"
    r"vader|priset?|kostar|krig(et)?|konflikt(en)?|attack(en)?|"
    r"angrepp|strid(en|erna)?|invasion(en)?|"
    r"respons(en)?|recension(er|erna)?|mottagande(t)?|betyg(et)?|"
    r"review(s)?|reception|"
    r"latest|current(ly)?|news|weather|war\b|conflict|"
    r"invasion|"
    r"vm\b|em\b|match(en|erna)?|vann|forlorade|spelade oavgjort|"
    r"hur gick( det)?|hur gar( det)?|"
    r"who won|match result|game result|how did .* go|"
    r"slappt|lanserat|lansering(en)?|gatt med i|blivit medlem|"
    r"medlem(skap)? i|anslutit sig|released|launched|joined|membership|"
    r"pandemi(n)?|utbrott(et)?|lagligt|olagligt|legal(ized)?|"
    r"what.?s happening|score|resultat(et)?|"
    # Utmarkelser: fanga aven fragor med 'fick', 'tilldelades' istallet
    r"nobelpris|oscar|academy award|fields.?medalj(en)?|abel.?priset?|"
    r"pulitzer)\b|"
    r"vem (ar|leder|styr|blev)\b.*(" + _ROLLER + r")|"
    r"who (is|are|leads)\b.*(" + _ROLLER + r")|"
    r"(" + _ROLLER + r").*vem (ar|blev)",
    re.IGNORECASE,
)
_KRAVER_INTE_AKTUELL = re.compile(
    r"(funktion|algoritm|kod\b|dikt|berattelse|forklara|betyder|"
    r"hur fungerar|rakna ut|sammanfatta|primtal|klass\b|variabel|"
    r"explain|function|algorithm|\bcode\b|poem|story|calculate|"
    r"summarize|what does .* mean|"
    r"bra jobbat|bra gjort|tack|grattis|snallt|schysst|"
    r"good job|well done|thanks|thank you|nice work|great job)",
    re.IGNORECASE,
)


_DIAKRITIK_TABELL = str.maketrans({
    "å": "a", "ä": "a", "ö": "o", "Å": "a", "Ä": "a", "Ö": "o", "ü": "u",
    "Ü": "u",
    "á": "a", "à": "a", "â": "a", "ã": "a", "Á": "a", "À": "a", "Â": "a",
    "Ã": "a",
    "é": "e", "è": "e", "ê": "e", "ë": "e", "É": "e", "È": "e", "Ê": "e",
    "Ë": "e",
    "í": "i", "ì": "i", "î": "i", "ï": "i", "Í": "i", "Ì": "i", "Î": "i",
    "Ï": "i",
    "ó": "o", "ò": "o", "ô": "o", "õ": "o", "ø": "o", "Ó": "o", "Ò": "o",
    "Ô": "o", "Õ": "o", "Ø": "o",
    "ú": "u", "ù": "u", "û": "u", "Ú": "u", "Ù": "u", "Û": "u",
    "ñ": "n", "Ñ": "n", "ç": "c", "Ç": "c", "ý": "y", "ÿ": "y", "Ý": "y"})


def _normalisera_for_matchning(text):
    """Tar bort svenska prickar (a, o -> a, o) sa regex-monster skrivna
    utan diakritik (t.ex. 'vem ar') matchar aven riktig svensk text med
    prickar (t.ex. 'vem ar' fran 'vem ar')."""
    return text.translate(_DIAKRITIK_TABELL).lower()


# Framtids-/slappdatum- och attributionsfragor: verifierbara faktafragor
# (nar slapps/lanseras X, vem uppfann/upptackte/bevisade/grundade Y) som
# annars glider forbi grinden och da fritt-genererar sjalvsakra
# gissningar. Route in i verifierings-pipelinen sa att samma regel
# "hittade vi ingen kalla -> var arlig istallet for att gissa" galler
# aven har. Kategorioberoende monster, inte en lista over enskilda fragor.
_FRAMTID_ELLER_ATTRIBUTION = re.compile(
    r"nar (kommer|slapps|slapptes|lanseras|lanserades|releasas|kom ut|kommer ut)\b|"
    r"slapp(s|datum|te|tes)\b|lanserings?datum|slappdatum|release date|"
    r"when (will|is|was|does|did)\b.*(release|come out|launch|out\b)|"
    r"vem (uppfann|upptackte|bevisade|grundade|skapade|utvecklade|"
    r"komponerade|forfattade|regisserade|uppforde)\b|"
    r"who (invented|discovered|proved|founded|created|developed|"
    r"composed|wrote|directed|designed)\b",
    re.IGNORECASE)


# Signalord som nastan alltid betyder att AKTUELL info kravs - fangar
# fragor som klassificeraren annars missar ("what is tesla worth now",
# "borsvarde idag", "how much is X", "last month's job report").
_AKTUELL_SIGNALORD = re.compile(
    r"\b(now|nu|currently|today|as of now|latest|up.?to.?date|"
    r"worth|valuation|market cap|stock price|share price|net worth|"
    r"how much (is|are|does)|"
    r"just nu|idag|for narvarande|"
    r"vard|varde|borsvarde|aktiekurs|marknadsvarde|vad kostar|"
    r"last (month|week|year)|this (month|year)|"
    r"senaste (manaden|veckan|aret)|forra (manaden|veckan)|i ar)\b",
    re.IGNORECASE)


# A6: specifika PAGAENDE/KOMMANDE evenemang pa en plats/tid ("vad hander i
# Berlin i helgen", "which concerts are on in London tonight"). Sadana
# ogrundade fragor far 9B:an att HITTA PA konkreta event/datum/platser.
# Route dem till retrieval - hittas inget svarar appen arligt i stallet for
# att fabricera ett event. Hogprecist: kraver en evenemang-FRAMING
# (inte bara ordet "event") + en plats/tid-markor inom kort avstand.
_EVENEMANG_AKTUELL = re.compile(
    r"(what.?s (on|happening|going on)"
    r"|what.?s there to do|things to do"
    r"|which (concerts|matches|games|shows|festivals|events)"
    r"|any (concerts|events|festivals|shows|matches)"
    r"|vad (hander|ar pa gang|pagar)"
    r"|vad finns (det )?att gora"
    r"|vilka (evenemang|konserter|matcher|shower|festivaler))"
    r".{0,40}\b(in|at|near|this|tonight|weekend|today|near me"
    r"|i|pa|ikvall|i\s?helgen|denna|nasta|har)\b",
    re.IGNORECASE)


# Flersprakiga "aktuell/senaste/nuvarande"-signaler (IT/ES/FR/DE/PT) som
# annars glider forbi de sv/en-tunga grindarna och da fritt-genererar ur
# minnet ("chi ha vinto l'ultimo...", "quien gano el ultimo..."). OBS:
# _normalisera_for_matchning tar bara bort a-ring/prickar, sa accenter pa
# e/i/o/u finns kvar -> vi matchar bade med och utan accent. Engelska
# falska vanner ("actual" = verklig, ej aktuell) UNDVIKS medvetet.
_AKTUELL_FLERSPRAK = re.compile(
    r"\b(attuale|attuali|"
    r"[uú]ltim[oa]s?|"
    r"actuel(le|les|s)?|"
    r"dernier|derni[eè]re|derniers|derni[eè]res|"
    r"recent[ei]|r[eé]cente?|"
    r"aktuell\w*|letzt\w*|neueste[rs]?|"
    r"pi[uù]\s+recente|m[aá]s\s+reciente|plus\s+r[eé]cent)\b",
    re.IGNORECASE)


def snabbcheck_aktuell_info(fraga):
    fraga_norm = _normalisera_for_matchning(fraga)
    if _KRAVER_AKTUELL.search(fraga_norm):
        return True
    if _AKTUELL_SIGNALORD.search(fraga_norm):
        return True
    if _AKTUELL_FLERSPRAK.search(fraga_norm):
        return True
    if _EVENEMANG_AKTUELL.search(fraga_norm):  # A6: konkreta event -> grunda
        return True
    if _FRAMTID_ELLER_ATTRIBUTION.search(fraga_norm):
        return True
    # Ar-trigger: en fraga som namner innevarande eller nyligt/framtida
    # artal (t.ex. "draft pick 2026", "jobbrapport 2025") handlar nastan
    # alltid om aktuell info - tvinga fram en sokning aven om ovanstaende
    # klassificerare skulle missa den. Gamla artal (2010 osv) ignoreras.
    _idag_ar = datetime.date.today().year
    if any(int(_a) >= _idag_ar - 1
           for _a in re.findall(r"\b20\d\d\b", fraga_norm)):
        return True
    # Fuzzy roll+land-koll (tolererar stavfel som "stadsnminister",
    # "sverigee") - samma logik som Wikipedia-uppslaget anvander, men
    # kord har sa aven grindfragan ("behovs sokning alls?") fangar den.
    if _hitta_roll_sv_nyckel(fraga_norm) and _hitta_land_sv_nyckel(fraga_norm):
        return True
    # Ledarfragor ("VD for X", "generalsekreterare for Y") - hanteras av
    # den generella Wikidata-uppslagningen, men grindfragan behover veta
    # att detta racker for att trigga sokning/uppslagning.
    if _ar_faktafraga(fraga_norm):
        return True
    if _KRAVER_INTE_AKTUELL.search(fraga_norm):
        return False
    return None


def mikrocheck_aktuell_info(fraga):
    prompt = bygg_qwen_prompt(
        "Svara ENDAST med JA eller NEJ, inget annat.",
        "Kraver denna fraga aktuell/realtidsinformation, eller information "
        "om en specifik handelse/prestation/status som kan ha andrats "
        "eller instraffat (t.ex. 'har X gjort Y an?', 'ar X fortfarande "
        "sant?', nyheter, priser, vader, sportresultat, produktlanseringar) "
        "for att besvaras korrekt och sakert? Om du ar OSAKER, svara JA - "
        "det ar mycket battre att soka i onodan an att missa en fraga som "
        f"faktiskt behovde aktuell info.\n\nFraga: {fraga}"
    )
    out = llm_sok(prompt, max_tokens=3, temperature=0.0, stop=_QWEN_STOPP)
    svar = out["choices"][0]["text"].strip().upper()
    return svar.startswith("JA")


# ============ TVA-STEGS SOKBESLUT ============
# STEG 1: en LITEN, STABIL lista med UPPENBART lokala fragor (matte, kod,
# oversattning, kreativt, smaprat, begreppsforklaringar). Dessa svaras
# direkt utan AI-anrop och utan sokning -> mycket snabbt. Listan VAXER
# INTE for varje ny fragetyp; den tacker fasta kategorier.
# STEG 2: allt annat lamnas till Llama som few-shot-klassificerare (den
# "AI:n avgor"). Vid osakerhet -> JA (hellre soka i onodan an missa).
# Vi slutar alltsa mata pa ord i "kravs-sokning"-listor; AI:n forstar
# meningen istallet.
_UPPENBART_LOKAL = re.compile(
    # Matte / rakning
    r"\brakna ut\b|\bberakna\b|\bcalculate\b|\bkvadratroten\b|"
    r"\bsquare root\b|\bprimtal\b|\bprime number\b|"
    r"\d+\s*%\s*(av|of)\b|\d+\s*[+\-*/x]\s*\d+|"
    r"\d+\s*(ganger|delat|plus|minus|times|divided|multiplied|percent|procent)\b|"
    r"\bpercent (of|av)\b|\bprocent (av|of)\b|\bdivided by\b|\bmultiplied by\b|"
    r"\bconvert\b.{0,30}\b(to|till|into)\b|\bhow many\b.{0,25}\bin (a|an|one)\b|"
    # Kod-generering
    r"\bskriv\b.{0,25}\b(funktion|kod|script|klass|program|metod)\b|"
    r"\bwrite\b.{0,25}\b(function|code|script|class|program|method|snippet)\b|"
    r"\bregex\b|\balgoritm\b|\balgorithm\b|"
    # Oversattning
    r"\boversatt\b|\btranslate\b|\bhur sager man\b|\bhow do you say\b|"
    r"\bvad heter\b.{0,30}\bpa (engelska|svenska|tyska|franska|spanska)\b|"
    r"\bwhat is\b.{0,30}\bin (english|swedish|german|french|spanish|italian)\b|"
    # Kreativt
    r"\bskriv\b.{0,25}\b(dikt|historia|berattelse|saga|sang|mejl|brev|limerick|haiku)\b|"
    r"\bwrite\b.{0,25}\b(poem|story|essay|song|email|letter|joke|rhyme|limerick|haiku)\b|"
    r"\bberatta\b.{0,15}\b(historia|saga|skamt|story|joke)\b|\bcompose a\b|"
    # Smaprat / meta / beroem
    r"\bhej\b|\bhejsan\b|\bhallo\b|\bhello\b|\btack\b|\bthanks\b|\bthank you\b|"
    r"\bvem ar du\b|\bwho are you\b|\bhur mar du\b|\bhow are you\b|"
    r"\bvad kan du\b|\bwhat can you do\b|\bbra jobbat\b|\bgood job\b|"
    r"\bwell done\b|\bnice work\b|"
    # Definitioner / begrepp (stabil kunskap)
    r"\bvad betyder\b|\bwhat does\b.{0,40}\bmean\b|\bskillnaden mellan\b|"
    r"\bdifference between\b|\bdefine\b|\bvad ar (en|ett) \w|"
    r"\bwhat is an? \w|\bhow do (i|you|we)\b|\bhow can i\b|\bhur gor man\b|"
    r"\bhow does\b.{0,45}\bwork\b|\bhur fungerar\b|\bexplain\b|\bforklara\b",
    re.IGNORECASE)


def _ar_uppenbart_lokal(fraga_norm):
    return bool(_UPPENBART_LOKAL.search(fraga_norm))


# Few-shot-klassificerare: kors pa Llama. Blandade sv/en-exempel som
# kontrasterar STABIL kunskap (NO) mot FORANDERLIG/aktuell info (YES).
_SOK_BESLUT_PROMPT = """You decide whether a question needs up-to-date information from a web search. Reply with ONLY one word: YES or NO.

Answer YES if the answer changes over time or is recent/current: who currently holds a role, prices, a company's or stock's value, latest software versions, sports results, news, release dates, records, or anything about "now/today/this year", or if you are unsure.

Answer NO only if it is stable knowledge that does not change: arithmetic, writing code, translation, definitions of concepts, history, geography, science facts, or creative writing.

Q: Vad ar 15% av 850?
A: NO
Q: Write a Python function to reverse a string.
A: NO
Q: What is the capital of France?
A: NO
Q: Vem skrev Hamlet?
A: NO
Q: When did World War 2 end?
A: NO
Q: What is a mutex?
A: NO
Q: Vem ar president i USA?
A: YES
Q: What is Tesla worth now?
A: YES
Q: Who won the last Super Bowl?
A: YES
Q: What is the latest version of Node.js?
A: YES
Q: how was last month's jobs report?
A: YES
Q: When will GTA 6 be released?
A: YES
Q: Vad hande i varlden idag?
A: YES
Q: {fraga}
A:"""


def _ai_beslut_sok(fraga):
    """STEG 2: lat Llama avgora om fragan behover webbsokning. Vid fel
    eller osaker output -> defaulta till JA (hellre soka an missa)."""
    try:
        prompt = _SOK_BESLUT_PROMPT.replace("{fraga}", fraga.strip())
        out = llm(prompt, max_tokens=3, temperature=0.0,
                  stop=_LLAMA_STOPP + ["\nQ:", "\n\n"])
        svar = out["choices"][0]["text"].strip().upper()
        beslut = not svar.startswith("NO") and not svar.startswith("NEJ")
        _debug_logg(f"AI-sokbeslut: {svar!r} -> {beslut}")
        return beslut
    except Exception as e:
        _debug_logg(f"AI-sokbeslut fel: {e} -> defaultar till JA")
        return True


# Rollinnehavar-fragor i NUTID pa flera sprak ("who is the prime minister
# of X", "vem ar USA:s president") - inneboende aktuella (vem HAR rollen
# just nu). Present tens kravs sa historiska "who WAS the first..." inte
# tvingas till ett (fel) incumbent-uppslag.
_ROLL_REALTID = re.compile(
    r"\b(who\s+(is|are)|vem\s+(ar|leder|styr)|wer\s+ist|qui\s+est|"
    r"quien\s+es|quem\s+e|chi\s+e|wie\s+is|kto\s+jest|kdo\s+je|"
    r"hvem\s+er|kuka\s+on)\b.{0,45}?\b("
    r"president(e|a)?|prezydent\w*|prezident\w*|prime\s+minister|"
    r"premier\b|chancellor|bundeskanzler|"
    r"statsminister|premiarminister|kansler|monarch|\bking\b|\bqueen\b|"
    r"\bkung\b|drottning|\bpope\b|\bpave\b|mayor|borgmastare|governor|"
    r"guvernor|\bceo\b|\bvd\b|chairman|ordforande|"
    r"premier\s+ministre|primer\s+ministro|primeiro\s+ministro)\b",
    re.IGNORECASE)


# ---- Historisk/temporal fraga: handlar INTE om NUvarande innehavare.
# Ex: "vem var FORSTA presidenten" (Washington, ej dagens), "vilket AR
# blev Gustav Vasa kung" (1523). Sadana far INTE tvingas till aktuellt
# roll-uppslag (som annars hamtar DAGENS president/kung - buggen vi sett).
_HISTORISK_FRAGA = re.compile(
    r"\b(first|former|previous|"
    r"forsta|forste|tidigare|foregaende|"
    r"erste|erster|ehemalige|ehemaliger|vorherige|"
    r"premiere?|ancien|ancienne|"
    r"primer|primero|primera|anterior|"
    r"primo|prima|precedente)\b"
    r"|\b(vilket ar|what year|which year|in what year|"
    r"in welchem jahr|en quelle annee|en que ano|in che anno)\b"
    r"|\b(nar|when|wann|quand|cuando|quando)\b.{0,30}\b(blev|became|"
    r"wurde|devint|fue|divento|kront|crowned|founded|grundad|"
    r"gegrundet|born|fodd)\b"
    r"|\b(blev|became|wurde)\b.{0,20}\b(kung|queen|king|drottning|"
    r"president|kejsare|emperor|kansler|konig)\b")


def _ar_historisk_fraga(fraga_norm):
    return bool(_HISTORISK_FRAGA.search(fraga_norm))


# Spekulativa/asikts-/prognosfragor ("vem tror du vinner", "if you had to
# guess", "vem ar troligast"). Har vill anvandaren ha en KVALIFICERAD
# gissning - namn pa den mest sannolika + kort reservation - INTE tre
# vagringar i rad ("det gar inte att forutsaga"). Behandlas med en egen vag.
_SPEKULATIV = re.compile(
    r"who do (you|yuo|u|ya|yo) think|what do (you|yuo|u) think will|"
    r"if (you|yuo|u) had to guess|your (best )?guess|best guess|"
    r"take a guess|your prediction|most likely|front.?runner|favou?rite|"
    r"likely winner|who (will|would|is going to|s going to) "
    r"(probably |likely )?(win|be|become)|who is the next|"
    r"the next (president|prime minister|\bpm\b|leader|chancellor|"
    r"nominee|winner|champion|government|monarch|king|queen|pope)|"
    r"vem tror du|vad tror du|vem blir|vem kommer (att )?bli|"
    r"om (du|ni) (maste|skulle) gissa|din gissning|gissa vem|"
    r"vem ar (mest sannolik|troligast|favorit)|vem ligger bast till|"
    r"vem tippar du|vem satsar du pa|vad ar din gissning|"
    r"vem skulle du gissa|troligast att (vinna|bli)|"
    r"nasta (regering|president|statsminister|ledare|kung|drottning)",
    re.IGNORECASE)


# FLERSPRAKIG spekulativ-detektor (FR/ES/DE/IT/PT/NL). Den engelska/svenska
# _SPEKULATIV missade t.ex. franska "a ton avis, qui sera le prochain
# president" -> fragan foll igenom till den hardare grinden och vagrade.
# OBS: _normalisera_for_matchning tar bara bort a-ring/prickar (a,a,o), sa
# franska/spanska/tyska accenter (e,a,o,i,u) finns kvar -> vi matchar bade
# med och utan accent dar det behovs. Tyska a/o-omljud blir dock a/o.
_SPEKULATIV_FLER = re.compile(
    # Franska
    r"[aà] ton avis|[aà] votre avis|selon (toi|vous)|d.?apr[eè]s (toi|vous)|"
    r"penses?-?\s?tu|qui (sera|serait|va|gagnera|remportera|l.?emportera)|"
    r"le\s+prochaine?\s+pr[eé]sident|prochaine?\s+(pr[eé]sident|premier"
    r"\s+ministre|vainqueur|gagnant|champion)|qui gagnera|qui va gagner|"
    # Spanska
    r"qui[eé]n crees|en tu opini[oó]n|a tu juicio|qui[eé]n (ganar[aá]|"
    r"ser[aá]|va a ganar)|crees que .{0,20}(ganar|gan[eé])|"
    r"pr[oó]xim[oa]\s+(presidente|mundial|campe[oó]n|elecci[oó]n|ganador)|"
    # Tyska
    r"deiner meinung nach|was denkst du|wer wird .{0,25}(gewinnen|"
    r"n[aä]chste)|wer gewinnt|wer wird der n[aä]chste|wen tippst du|"
    r"was tippst du|wer schafft es|n[aä]chste[rn]? (pr[aä]sident|kanzler|"
    r"bundeskanzler|sieger|gewinner)|"
    # Italienska
    r"secondo te|a tuo parere|chi (vincer[aà]|sar[aà]|pensi)|"
    r"chi vince|prossimo\s+(presidente|campione|vincitore|campionato)|"
    # Portugisiska
    r"na tua opini[aã]o|quem achas|quem (vai ganhar|ganhar[aá]|ser[aá])|"
    r"pr[oó]xim[oa]\s+(presidente|vencedor|mundial|campe[aã]o)|"
    # Nederlandska
    r"wie denk je|wat denk je|wie (wint|wordt de volgende)|"
    r"volgende\s+(president|premier|winnaar|kampioen)",
    re.IGNORECASE)


def _ar_spekulativ_fraga(fraga_norm):
    return bool(_SPEKULATIV.search(fraga_norm)
                or _SPEKULATIV_FLER.search(fraga_norm))


def behover_aktuell_info(fraga):
    """Tva-stegs-beslut om fragan behover aktuell webbinfo."""
    fraga_norm = _normalisera_for_matchning(fraga)
    # ORDNING (LATENS-OPTIMERAD 2026-07-24): realtidssignaler FORST, sedan
    # "uppenbart lokal", sist AI-beslutet. Sa kan bada deterministiska
    # grindarna ta beslutet utan ett enda LLM-anrop for de allra flesta
    # fragor (varje LLM-klassificering kostar ~0.8 s pga few-shot-prefill).
    # Realtid vinner alltid over lokal (t.ex. "what is the LATEST X").
    #
    # 0) Historisk/temporal fraga (forsta/forre/vilket ar/blev kung ...)?
    #    Da handlar det INTE om dagens innehavare - sla av de rollbaserade
    #    "aktuellt"-triggrarna sa vi inte hamtar DAGENS president/kung.
    _hist = _ar_historisk_fraga(fraga_norm)
    # 1) Roll + land (president i USA osv) - aktuellt Wikipedia-uppslag.
    if (not _hist and _hitta_roll_sv_nyckel(fraga_norm)
            and _hitta_land_sv_nyckel(fraga_norm)):
        return True
    # 2) Innevarande/nyligt/framtida artal (2025, 2026 ...) -> aktuellt.
    _idag_ar = datetime.date.today().year
    if any(int(_a) >= _idag_ar - 1
           for _a in re.findall(r"\b20\d\d\b", fraga_norm)):
        return True
    # 3) Starka realtidssignaler (current/latest/who won/who is the PM ...).
    #    Roll-realtidssignalen ('vem ar president') slas av for historiska
    #    fragor ('vem VAR forsta presidenten').
    if (_KRAVER_AKTUELL.search(fraga_norm)
            or _AKTUELL_SIGNALORD.search(fraga_norm)
            or _AKTUELL_FLERSPRAK.search(fraga_norm)
            or (_EVENEMANG_AKTUELL.search(fraga_norm) and not _hist)  # A6
            or _ar_spekulativ_fraga(fraga_norm)
            or _FRAMTID_ELLER_ATTRIBUTION.search(fraga_norm)
            or (_ROLL_REALTID.search(fraga_norm) and not _hist)):
        _debug_logg("realtidssignal (regex) -> kravs_aktuell=True")
        return True
    # 4) Uppenbart lokal (matte/kod/oversattning/kreativt/definition/how-to)
    #    -> svara direkt, HOPPA alla klassificerar-anrop (spar ~2 s). Sakert
    #    nu nar realtid redan uteslutits ovan.
    if _ar_uppenbart_lokal(fraga_norm):
        return False
    # 5) Allt annat -> lat AI:n avgora.
    return _ai_beslut_sok(fraga)


# ============ WIKIPEDIA DIREKTUPPSLAG (algoritmisk, ingen AI) ============
# For fragor om roller/positioner (president, statsminister, kung, osv)
# kombinerat med ett kant land - fraga Wikipedia direkt istallet for att
# soka brett och lata AI:n tolka rorig text. Snabbare OCH mer palitligt.
# Om det inte matchar exakt: returnera None sa den befintliga sok+AI-
# pipelinen tar over som vanligt (garanterad fallback, aldrig ett hinder).
_LAND_MAP = {
    "usa": "the United States", "forenta staterna": "the United States",
    "amerika": "the United States", "united states": "the United States",
    "america": "the United States",
    "sverige": "Sweden", "sweden": "Sweden",
    "norge": "Norway", "norway": "Norway",
    "danmark": "Denmark", "denmark": "Denmark",
    "finland": "Finland",
    "tyskland": "Germany", "germany": "Germany",
    "frankrike": "France", "france": "France",
    "storbritannien": "the United Kingdom", "england": "the United Kingdom",
    "uk": "the United Kingdom", "kungariket": "the United Kingdom",
    "britain": "the United Kingdom", "united kingdom": "the United Kingdom",
    "kanada": "Canada", "canada": "Canada",
    "italien": "Italy", "italy": "Italy",
    "spanien": "Spain", "spain": "Spain",
    "ryssland": "Russia", "russia": "Russia",
    "kina": "China", "china": "China",
    "indien": "India", "india": "India",
    "japan": "Japan",
    "brasilien": "Brazil", "brazil": "Brazil",
    "australien": "Australia", "australia": "Australia",
    "nederlanderna": "the Netherlands", "holland": "the Netherlands",
    "netherlands": "the Netherlands",
    "polen": "Poland", "poland": "Poland",
    "portugal": "Portugal",
    "irland": "Ireland", "ireland": "Ireland",
    "schweiz": "Switzerland", "switzerland": "Switzerland",
    "osterrike": "Austria", "austria": "Austria",
    "belgien": "Belgium", "belgium": "Belgium",
    "grekland": "Greece", "greece": "Greece",
    "turkiet": "Turkey", "turkey": "Turkey",
    "mexiko": "Mexico", "mexico": "Mexico",
    "sydkorea": "South Korea", "south korea": "South Korea",
    "sydafrika": "South Africa", "south africa": "South Africa",
    "argentina": "Argentina",
    "egypten": "Egypt", "egypt": "Egypt",
    "israel": "Israel",
    "ukraina": "Ukraine", "ukraine": "Ukraine",
    "island": "Iceland", "iceland": "Iceland",
    "nya zeeland": "New Zealand", "new zealand": "New Zealand",
    "singapore": "Singapore",
    "indonesien": "Indonesia", "indonesia": "Indonesia",
    "saudiarabien": "Saudi Arabia", "saudi arabia": "Saudi Arabia",
    "forenade arabemiraten": "the United Arab Emirates",
}
_LAND_MAP_SV_VISNING = {
    "usa": "USA", "forenta staterna": "USA", "amerika": "USA",
    "sverige": "Sverige", "norge": "Norge", "danmark": "Danmark",
    "finland": "Finland", "tyskland": "Tyskland", "frankrike": "Frankrike",
    "storbritannien": "Storbritannien", "england": "Storbritannien",
    "uk": "Storbritannien", "kungariket": "Storbritannien",
    "kanada": "Kanada", "italien": "Italien", "spanien": "Spanien",
    "ryssland": "Ryssland", "kina": "Kina", "indien": "Indien",
    "japan": "Japan", "brasilien": "Brasilien", "australien": "Australien",
    "nederlanderna": "Nederlanderna", "holland": "Nederlanderna",
    "polen": "Polen", "portugal": "Portugal", "irland": "Irland",
    "schweiz": "Schweiz", "osterrike": "Osterrike", "belgien": "Belgien",
    "grekland": "Grekland", "turkiet": "Turkiet", "mexiko": "Mexiko",
    "sydkorea": "Sydkorea", "sydafrika": "Sydafrika",
    "argentina": "Argentina", "egypten": "Egypten", "israel": "Israel",
    "ukraina": "Ukraina", "island": "Island", "nya zeeland": "Nya Zeeland",
    "singapore": "Singapore", "indonesien": "Indonesien",
    "saudiarabien": "Saudiarabien", "forenade arabemiraten": "Forenade Arabemiraten",
}
_ROLL_TILL_ENGELSKA = {
    "president": "President", "statsminister": "Prime Minister",
    "premiarminister": "Prime Minister", "kung": "King",
    "drottning": "Queen", "kansler": "Chancellor",
    "utrikesminister": "Foreign Minister", "finansminister": "Finance Minister",
    "prime minister": "Prime Minister", "king": "King", "queen": "Queen",
    "chancellor": "Chancellor", "foreign minister": "Foreign Minister",
    "finance minister": "Finance Minister",
}
_ROLL_VISNING_SV = {
    "president": "President", "statsminister": "Statsminister",
    "premiarminister": "Premiarminister", "kung": "Kung",
    "drottning": "Drottning", "kansler": "Kansler",
    "utrikesminister": "Utrikesminister", "finansminister": "Finansminister",
}


# Byggd fran _LAND_MAP.values() - engelska landsnamn som Llama 3B ibland
# hallucinerat lagger som klarifikation i slutet av engelska svar
# ("Charles III is the king of Storbritannien (Sweden).") nar fragan
# innehaller ett svenskt landsnamn. Anvands av _stada_hallucinerad_
# landsparentes for att strippa dessa utan att paverka legitima
# parenteser som "(2024)" eller "(prime minister)".
_ENGELSKA_LANDNAMN_LOWER = set()
for _v in _LAND_MAP.values():
    _ENGELSKA_LANDNAMN_LOWER.add(_v.lower())
    if _v.lower().startswith("the "):
        _ENGELSKA_LANDNAMN_LOWER.add(_v.lower()[4:])
_SLUTPARENTES_PATTERN = re.compile(r"\s*\(([^)]+)\)\s*([.!?])?\s*$")


def _stada_hallucinerad_landsparentes(svar, sprak):
    """Llama 3B tenderar att lagga till hallucinerade landsklarifikationer
    som '(Sweden)' i slutet av engelska svar - speciellt nar fragan
    innehaller svenska landsnamn. Ta bort dessa fangster utan att paverka
    legitima parenteser (t.ex. '(2024)', '(prime minister)')."""
    if not svar or sprak != "en":
        return svar
    match = _SLUTPARENTES_PATTERN.search(svar)
    if not match:
        return svar
    innehall = match.group(1).strip().lower()
    if innehall in _ENGELSKA_LANDNAMN_LOWER:
        # Ta bort parentes + eventuell efterfoljande punkt; behall
        # den avslutande punkten fran svaret om det fanns en.
        efterskrift = match.group(2) or ""
        return svar[:match.start()].rstrip().rstrip(".") + (efterskrift or ".")
    return svar


def _hitta_nyckel_fuzzy(fraga_norm, nyckellista):
    """Exakt match forst; om inget hittas, tolerera mindre stavfel
    (t.ex. 'sverire' istallet for 'sverige') via likhetsjamforelse.
    Provar aven hopslagna par av angransande ord (t.ex. 'premiar
    minister' skrivet med mellanslag istallet for ihopskrivet)."""
    for nyckel in nyckellista:
        if re.search(r"\b" + re.escape(nyckel) + r"\b", fraga_norm):
            return nyckel
    ord_lista = re.findall(r"[a-z]+", fraga_norm)
    kandidater = list(ord_lista)
    for i in range(len(ord_lista) - 1):
        kandidater.append(ord_lista[i] + ord_lista[i + 1])
    for kandidat in kandidater:
        if len(kandidat) < 4:
            continue
        # "Vice-"-sammansatta ord (t.ex. "vicepresident") ska INTE
        # fuzzy-matcha mot "president" - det ar en helt annan roll vi
        # inte har stod for, aven om orden liknar varandra bokstavligt.
        if kandidat.startswith("vice") and len(kandidat) > 4:
            continue
        traff = difflib.get_close_matches(kandidat, nyckellista, n=1, cutoff=0.8)
        if traff:
            return traff[0]
    return None


def _hitta_land(fraga_norm):
    nyckel = _hitta_land_sv_nyckel(fraga_norm)
    return _LAND_MAP.get(nyckel) if nyckel else None


def _hitta_land_sv_nyckel(fraga_norm):
    return _hitta_nyckel_fuzzy(fraga_norm, list(_LAND_MAP.keys()))


def _hitta_roll_engelska(fraga_norm):
    nyckel = _hitta_roll_sv_nyckel(fraga_norm)
    return _ROLL_TILL_ENGELSKA.get(nyckel) if nyckel else None


def _hitta_roll_sv_nyckel(fraga_norm):
    return _hitta_nyckel_fuzzy(fraga_norm, list(_ROLL_TILL_ENGELSKA.keys()))


_HISTORIK_FRASER = re.compile(
    r"historik|historia|genom tiderna|genom historien|senaste \d+ ?ar|"
    r"under de senaste|tidigare (innehavare|personer)|alla som|lista over|"
    r"history|historically|over the years|throughout history|past \d+ ?years|"
    r"list of|all (the )?people who",
    re.IGNORECASE,
)
_VARFOR_FRASER = re.compile(
    r"^varfor\b|\bvarfor\b|hur kom det sig|why (did|does|is|was)|"
    r"how come",
    re.IGNORECASE,
)


def _ar_historikfraga(fraga_norm):
    return bool(_HISTORIK_FRASER.search(fraga_norm))


def _ar_flerlandsfraga(fraga_norm):
    antal = sum(
        1 for nyckel in _LAND_MAP
        if re.search(r"\b" + re.escape(nyckel) + r"\b", fraga_norm)
    )
    return antal >= 2


_JA_NEJ_FRAGA = re.compile(
    r"^(har|ar|kan|kommer|gjorde|blev|har inte|finns|existerar|"
    r"has|is|are|can|will|did|does|was|were)\b",
    re.IGNORECASE,
)


def _ar_ja_nej_fraga(fraga_norm):
    """Kanner igen klassiska ja/nej-fragor ('Har X hant Y an?', 'Ar X
    sant?'). For dessa vill vi ha en explicit JA/NEJ/OKLART-analys av
    sokresultaten istallet for att lata modellen fritt formulera ett
    svar, eftersom fri formulering visat sig kunna dra fel slutsats
    aven nar sokresultaten faktiskt ger ratt svar."""
    return bool(_JA_NEJ_FRAGA.search(fraga_norm.strip()))


def _ar_varforfraga(fraga_norm):
    return bool(_VARFOR_FRASER.search(fraga_norm))


_ATTRIBUT_FRAGA = re.compile(
    r"\b(vilken|vilket|vilka|hur manga|hur mycket|nar|"
    r"which|how many|how much|when|what year|"
    r"welche|welcher|welches|wie viele|wie viel|wann|"
    r"quel|quelle|quels|quelles|combien|quand|"
    r"quale|quali|quanti|quante|quando|"
    r"cual|cuales|cuantos|cuantas|cuando|"
    r"qual|quais|quantos|quantas)\b",
    re.IGNORECASE)


def _ar_attributfraga(fraga_norm):
    """True om fragan efterfragar ett SPECIFIKT attribut (vilken liga/
    division, nar, hur manga ...) snarare an en definition ('vad ar X').
    For sadana ar Wikipedias inledande summary ofta otillracklig (den
    saknar t.ex. aktuell divisionstillhorighet) medan Braves snuttar har
    svaret - sa i nisch-grenen gar attributfragor direkt till Brave
    istallet for wiki-uppslaget."""
    return bool(_ATTRIBUT_FRAGA.search(fraga_norm))


def wikipedia_direktuppslag(fraga, sprak="sv"):
    """Forsoker svara direkt via Wikipedias infobox-data (deterministiskt,
    ingen AI-tolkning). Hamtar 'incumbent'-faltet ur artikelns infobox,
    som Wikipedia-redaktorer haller uppdaterat i princip omedelbart nar
    nagon tilltrader en roll. Returnerar None om nagot inte stammer - da
    tar den befintliga sok+AI-pipelinen over automatiskt."""
    try:
        fraga_norm = _normalisera_for_matchning(fraga)
        # Historik-fragor ("senaste 10 aren", "genom tiderna") vill ha
        # FLERA personer over tid, inte bara den nuvarande innehavaren -
        # vart uppslag klarar bara "vem ar det just nu", sa hoppa over
        # och lat sok+AI-pipelinen hantera det istallet.
        if _ar_historikfraga(fraga_norm):
            return None
        # Om fragan namner FLERA olika lander, klarar vart enkla
        # roll+land-monster inte att para ihop ratt - hoppa over for
        # att undvika att blanda ihop t.ex. "president i USA" med
        # "statsminister i Sverige" till en felaktig kombination.
        if _ar_flerlandsfraga(fraga_norm):
            return None
        roll = _hitta_roll_engelska(fraga_norm)
        land = _hitta_land(fraga_norm)
        if not roll or not land:
            return None
        titel = f"{roll} of {land}"
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query", "prop": "revisions", "rvprop": "content",
            "rvslots": "main", "format": "json", "titles": titel,
            "formatversion": "2", "redirects": "1",
        }
        resp = None
        # Flera forsok med okande vantetid (exponentiell backoff) -
        # tillfalliga natverksblippar/hastighetsbegransningar ska inte
        # tvinga fram den mindre palitliga sok+extraktions-vagen i onodan.
        for forsok in range(4):
            try:
                resp = requests.get(url, params=params, timeout=4,
                                    headers={"User-Agent": "AIPrototype/1.0"})
                if resp.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                resp = None
            time.sleep(0.3 * (2 ** forsok))
        if resp is None or resp.status_code != 200:
            return None
        data = resp.json()
        sidor = data.get("query", {}).get("pages", [])
        if not sidor or sidor[0].get("missing"):
            return None
        wikitext = (sidor[0].get("revisions", [{}])[0]
                    .get("slots", {}).get("main", {}).get("content", ""))
        # Anchora sokningen till sjalva ambets-infoboxen ({{Infobox
        # official/political post}}) sa att en ev. sidebar/navbox for
        # ETT ANNAT ambete hogre upp i wikitexten inte kan "lacka" fel
        # incumbent. Universell harding - inte knuten till nagot
        # specifikt land eller roll. Faller tillbaka pa hela texten om
        # ingen sadan infobox hittas.
        ib = re.search(r"\{\{\s*Infobox\s+(?:official|political)\s+post",
                       wikitext, re.IGNORECASE)
        sokstart = ib.start() if ib else 0
        match = re.search(r"\|\s*incumbent\s*=\s*\[?\[?([^\]|\n]+)",
                          wikitext[sokstart:], re.IGNORECASE)
        if not match:
            return None
        namn = match.group(1).strip()
        if not namn or len(namn) > 100:
            return None
        # KORSVALIDERING: Wikipedias infobox och Wikidata (P6 regeringschef /
        # P35 statschef) ska vara ENSE om innehavaren. Skiljer de sig klart
        # (en kalla kan vara inaktuell/vandaliserad, t.ex. "Andy Burnham"
        # som UK-PM) litar vi inte pa nagondera utan faller igenom till sok
        # som laser flera farska kallor. Ense (eller Wikidata saknas) ->
        # behall det snabba deterministiska svaret.
        try:
            _pid = ("P6" if _ROLL_REGERINGSCHEF.search(fraga_norm)
                    else "P35" if _ROLL_STATSCHEF.search(fraga_norm)
                    else None)
            if _pid:
                _land_sok = land[4:] if land.lower().startswith("the ") else land
                _wd = None
                for _qid in _wikidata_sok_qid(_land_sok, "en")[:2]:
                    _kand = _wikidata_hamta_varde(_qid, _pid)
                    if _kand and 2 < len(_kand) <= 80 and not re.fullmatch(
                            r"Q\d+", _kand):
                        _wd = _kand
                        break
                if _wd and not _namn_overlappar(namn, _wd):
                    _debug_logg(f"wiki_direkt KROCK wiki={namn!r} "
                                f"wd={_wd!r} -> None (faller till sok)")
                    return None
        except Exception:
            pass
        if sprak == "sv":
            roll_sv = _ROLL_VISNING_SV.get(_hitta_roll_sv_nyckel(fraga_norm), roll)
            land_sv = _LAND_MAP_SV_VISNING.get(_hitta_land_sv_nyckel(fraga_norm), land)
            return _str("wiki_direkt_svar", sprak,
                         roll=roll_sv, land=land_sv, namn=namn)
        return _str("wiki_direkt_svar", sprak, titel=titel, namn=namn,
                     roll=roll, land=land)
    except Exception as e:
        # Natverksfel, timeout, ovantat format, etc. - garantera att
        # detta ALDRIG kraschar appen, bara faller tillbaka pa sokningen.
        try:
            with open(os.path.join(BASKATALOG, "wikipedia_fel.log"), "a",
                      encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now()}: {fraga!r} -> {e!r}\n")
        except Exception:
            pass
        return None


# ============ WIKIDATA-UPPSLAG (algoritmisk, for pris/utmarkelser) ============
# Samma princip som Wikipedia-uppslaget - strukturerad data istallet for
# att lata AI:n tolka rorig sokresultat-text for kategorier dar det ofta
# gar fel (Nobelpris, Oscar). Verifierade Wikidata-ID:n, inga gissningar.
_UTMARKELSE_FRAGOR = [
    (re.compile(r"nobelpris.*litteratur|nobel.*prize.*literature", re.IGNORECASE),
     "Q37922", "ar"),
    (re.compile(r"oscar.*(basta film|best picture)|(basta film|best picture).*oscar",
                re.IGNORECASE),
     "Q102427", "film"),
]


def _wikidata_query(sparql):
    url = "https://query.wikidata.org/sparql"
    for forsok in range(3):
        try:
            resp = requests.get(url, params={"query": sparql, "format": "json"},
                                headers={"User-Agent": "AIPrototype/1.0"}, timeout=6)
            if resp.status_code == 200:
                return resp.json()
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.3)
    return None


def wikidata_direktuppslag(fraga, sprak="sv"):
    """Fragar Wikidata direkt for kategorier dar sok+AI ofta hallucinerat
    (Nobelpris, Oscar) - strukturerad data istallet for AI-tolkning av
    rorig text. Returnerar None om inget matchar - da tar befintlig
    sok+AI-pipeline over som vanligt (sakert, aldrig ett hinder)."""
    for monster, award_id, typ in _UTMARKELSE_FRAGOR:
        if not monster.search(_normalisera_for_matchning(fraga)):
            continue
        try:
            if typ == "ar":
                sparql = f"""
                SELECT ?itemLabel ?date WHERE {{
                  ?item wdt:P166 wd:{award_id}.
                  ?item p:P166 ?statement.
                  ?statement pq:P585 ?date.
                  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "sv,en". }}
                }}
                ORDER BY DESC(?date)
                LIMIT 1
                """
            else:
                sparql = f"""
                SELECT DISTINCT ?itemLabel ?date WHERE {{
                  ?item wdt:P166 wd:{award_id}.
                  ?item wdt:P31 wd:Q11424.
                  ?item p:P166 ?statement.
                  ?statement pq:P585 ?date.
                  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "sv,en". }}
                }}
                ORDER BY DESC(?date)
                LIMIT 1
                """
            data = _wikidata_query(sparql)
            if not data:
                return None
            bindings = data.get("results", {}).get("bindings", [])
            if not bindings:
                return None
            namn = bindings[0]["itemLabel"]["value"]
            ar = bindings[0]["date"]["value"][:4]
            if not namn or len(namn) > 150:
                return None
            return _str("wikidata_utmarkelse_legacy", sprak,
                         ar=ar, namn=namn)
        except Exception as e:
            try:
                with open(os.path.join(BASKATALOG, "wikipedia_fel.log"), "a",
                          encoding="utf-8") as f:
                    f.write(f"{datetime.datetime.now()}: wikidata {fraga!r} -> {e!r}\n")
            except Exception:
                pass
            return None
    return None


# ============ UNIVERSELL WIKIDATA-UPPSLAGNING ============
# Istallet for hardkodade listor per kategori (lander, foretag, roller):
# extrahera BADE entiteten OCH relationen (vilken egenskap som fragas
# efter) ur fragan, och sok upp bada dynamiskt mot Wikidata. Detta
# tacker i princip VILKEN fraga som helst om en namngiven entitets
# egenskaper (VD, befolkning, grundare, huvudstad, tranare, m.m.) utan
# att vi behover skriva kod for varje ny kategori.
#
# Viktig lardom fran testning: Wikidatas egna tidsstampar ar inte alltid
# fullstandiga for snabbforanderliga roller (t.ex. sporttranare), sa
# aven denna uppslagning kan ge foraldrade svar ibland - det ar en
# begransning i kalldatan, inte nagot vi kan losa bort helt. Darfor
# behandlas dessa svar med SAMMA forsiktighet (grundningskontroll,
# faktamatchning) som allt annat i pipelinen - aldrig blint litade pa.
_FAKTAFRAGA = re.compile(
    r"vd\b|ceo\b|vem (leder|styr|ar chef|ar ordforande|grundade)|"
    r"who (leads|runs|is the ceo|is the chairman|founded)|"
    r"chief executive|ordforande for|chairman of|"
    r"generalsekreterare|secretary.general|generaldirektor|"
    r"director.general|verkstallande direktor|"
    r"grundare|grundades|founder|founded|"
    r"befolkning|invanare|population|"
    r"huvudstad|capital city|"
    r"tranare|coach|manager\b|"
    r"valuta|currency",
    re.IGNORECASE,
)


def _ar_faktafraga(fraga_norm):
    return bool(_FAKTAFRAGA.search(fraga_norm))


def _extrahera_entitet_och_relation(fraga):
    """Anvander Qwen for att plocka ut BADE vilken entitet fragan handlar
    om OCH vilken relation/egenskap som efterfragas, som en KORT
    kanonisk engelsk Wikidata-egenskapsterm (t.ex. 'CEO', 'population',
    'founder', 'capital'). Kortare/mer exakta termer -> snabbare och
    mer traffsakra egenskapssokningar. Format: 'ENTITET | RELATION'."""
    try:
        prompt = bygg_qwen_prompt(
            "Identify two things in this question: 1) the name of the "
            "entity (company/organization/place/team/etc.) being asked "
            "about, 2) the SHORTEST possible canonical Wikidata "
            "property name for what's being asked (1-2 words only, "
            "e.g. 'CEO', 'population', 'founder', 'capital', 'coach', "
            "'currency', 'winner') - NOT a full sentence or phrase. "
            "Respond in EXACTLY this format with no other text: "
            "ENTITY | RELATION",
            fraga
        )
        out = llm_sok(prompt, max_tokens=20, temperature=0.0, stop=_QWEN_STOPP)
        text = out["choices"][0]["text"].strip()
        if "|" not in text:
            return None, None
        entitet, relation = text.split("|", 1)
        entitet = entitet.strip().strip('."\'')
        relation = relation.strip().strip('."\'')
        if not entitet or not relation or len(entitet) > 100 or len(relation) > 40:
            return None, None
        return entitet, relation
    except Exception:
        return None, None


def _wikidata_sok_qid(namn, sprak="sv"):
    """Soker efter Wikidata-entiteter med givet namn, returnerar en lista
    med kandidat-QID:n (flera, eftersom korta namn som 'Tesla' eller
    'Amazon' kan matcha fel post forst - t.ex. floden Amazonas istallet
    for foretaget - sa vi provar flera i tur och ordning)."""
    try:
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities", "search": namn, "language": sprak,
            "format": "json", "limit": 5, "type": "item",
        }
        resp = requests.get(url, params=params,
                            headers={"User-Agent": "AIPrototype/1.0"}, timeout=6)
        if resp.status_code != 200:
            return []
        return [t["id"] for t in resp.json().get("search", [])]
    except Exception:
        return []


def _wikidata_sok_egenskap(beskrivning):
    """Soker efter Wikidata-EGENSKAPER (properties) som matchar en fri
    textbeskrivning (t.ex. 'winner' -> P1346, 'chief executive officer'
    -> P169). Detta ar det som gor uppslagningen universell - ingen
    hardkodad lista over egenskaper per kategori."""
    try:
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities", "search": beskrivning,
            "language": "en", "format": "json", "limit": 3, "type": "property",
        }
        resp = requests.get(url, params=params,
                            headers={"User-Agent": "AIPrototype/1.0"}, timeout=6)
        if resp.status_code != 200:
            return []
        return [t["id"] for t in resp.json().get("search", [])]
    except Exception:
        return []


def _wikidata_hamta_varde(qid, pid):
    """Hamtar vardet for en given (entitet, egenskap)-kombination.
    Sorterar efter tidsstampel (om sadan finns - fangar bade 'point in
    time' och 'start time'-kvalificerare) sa den senaste/aktuella
    varianten foredras for egenskaper som andras over tid. Ignorerar
    orater etiketter (rena Q-ID:n) som inte kunnat oversattas - de ar
    inte anvandbara i ett svar."""
    try:
        sparql = f"""
        SELECT ?valueLabel (COALESCE(?d1, ?d2) AS ?sortdate) WHERE {{
          wd:{qid} p:{pid} ?statement.
          ?statement ps:{pid} ?value.
          OPTIONAL {{ ?statement pq:P585 ?d1. }}
          OPTIONAL {{ ?statement pq:P580 ?d2. }}
          FILTER NOT EXISTS {{ ?statement pq:P582 ?end. }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "sv,en". }}
        }}
        ORDER BY DESC(?sortdate)
        LIMIT 1
        """
        data = _wikidata_query(sparql)
        if not data:
            return None
        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            return None
        varde = bindings[0]["valueLabel"]["value"]
        if re.fullmatch(r"Q\d+", varde):
            return None  # olost etikett, inte anvandbar
        return varde
    except Exception:
        return None


# ============ AMBETSINNEHAVARE (Wikidata P6/P35 - deterministiskt) ============
# Loser rollforvaxlingen som 9B:an gor: "presidente del Consiglio"
# (statsminister = regeringschef) vs "presidente della Repubblica"
# (statschef). Regeringschef -> P6, statschef -> P35 pa landets Wikidata-
# post. _wikidata_hamta_varde ger DAGENS innehavare (sorterar pa startdatum,
# filtrerar avslutade mandat). Multisprakigt via land/rollnycklarna.
_ROLL_REGERINGSCHEF = re.compile(
    r"\b(statsminister\w*|premiarminister\w*|premiar\s?minister\w*|"
    r"prime\s+minister|premier\s+ministre|primo\s+ministro|"
    r"presidente\s+del\s+consiglio|primer\s+ministro|primeiro\s+ministro|"
    r"presidente\s+del\s+gobierno|president\s+of\s+the\s+government|"
    r"regeringschef\w*|regeringssjef\w*|regierungschef\w*|"
    r"premierminister\w*|ministerpresident\w*|bundeskanzler\w*|kanzler\w*|"
    r"kansler\w*|chancellor|taoiseach|\bpremier\b)\b",
    re.IGNORECASE)
_ROLL_STATSCHEF = re.compile(
    r"\b(presidente\s+della\s+repubblica|president\s+de\s+la\s+republique|"
    r"statschef|head\s+of\s+state|prezydent\w*|prezident\w*|"
    # OBS: exakt ordform - INTE 'presidentval'/'presidentkandidat' (compound).
    r"pr[aeé]sident(en|e|es|a|in|inne|innen)?|"
    r"kung\w*|konung\w*|drottning\w*|monark\w*|\bking\b|\bqueen\b|monarch|"
    r"kejsare|emperor)\b",
    re.IGNORECASE)
# Fragor som INTE handlar om vem som SITTER NU - kandidat/val/framtid/
# spekulation. Ambets-uppslaget ska INTE trigga pa dessa (annars svarade
# vi "Donald Trump" pa "vilken demokratisk KANDIDAT ... nasta PRESIDENTVAL").
_AMBETE_EXKLUDERA = re.compile(
    r"\b(kandidat\w*|candidate|candidat\w*|candidato\w*|candidata\w*|"
    r"nominee|nomine\w*|utmanare|motkandidat\w*|challenger|"
    r"election|eleccion\w*|elezion\w*|wahl\w*|primary|primaria\w*|"
    r"mest\s+sannolik\w*|troligast|sannolik\w*|likely|probable\w*|"
    r"vem\s+blir|vem\s+vinner|who\s+will|who\s+wins|qui\s+va|quien\s+sera|"
    r"nasta\b|next\b|kommande|framtida|future|prochain\w*|proxim\w*|"
    r"prossim\w*|representera\w*|represent\w*|repraesenter\w*|"
    r"tror\s+du|tror\s+ni|gissa|guess)\b|presidentval\w*",
    re.IGNORECASE)
# Lands-adjektiv / lokala landsnamn som _LAND_MAP:s fuzzy-match ibland
# missar ("italiano", "Deutschland"). Kompletterar _hitta_land.
_LAND_EXTRA = {
    "italiano": "Italy", "italiana": "Italy", "italienische": "Italy",
    "italienne": "Italy", "italiens": "Italy",
    "deutschland": "Germany", "allemagne": "Germany", "alemania": "Germany",
    "germania": "Germany", "tyska": "Germany",
    "espagne": "Spain", "spagna": "Spain", "spanska": "Spain",
    "espanol": "Spain", "espanola": "Spain", "espana": "Spain",
    "espanya": "Spain", "spanien": "Spain",
    "italia": "Italy", "italie": "Italy", "italien": "Italy",
    "francia": "France", "franska": "France", "frankreich": "France",
    "francaise": "France", "francais": "France", "francuski": "France",
    "sverige": "Sweden", "svenska": "Sweden", "schweden": "Sweden",
    "suede": "Sweden", "suecia": "Sweden", "svezia": "Sweden",
    "norge": "Norway", "norska": "Norway", "norwegen": "Norway",
    "norvege": "Norway", "noruega": "Norway", "norvegia": "Norway",
    "polski": "Poland", "polska": "Poland", "polen": "Poland",
    "pologne": "Poland", "polonia": "Poland",
    "estados unidos": "the United States", "etats-unis": "the United States",
    "royaume-uni": "the United Kingdom", "reino unido": "the United Kingdom",
}


def _hitta_land_extra(norm):
    for k, v in _LAND_EXTRA.items():
        if re.search(r"\b" + re.escape(k) + r"\b", norm):
            return v
    return None


def _namn_overlappar(a, b):
    """Grov namnjamforelse: delar de nagot signifikant ordtoken (>=3 tecken)?
    Anvands for att korsvalidera tva kallor for samma innehavare utan att
    krava exakt strangmatch ('Keir Starmer' vs '[[Keir Starmer]]')."""
    if not a or not b:
        return False
    ta = set(re.findall(r"\w{3,}", _normalisera_for_matchning(a), re.UNICODE))
    tb = set(re.findall(r"\w{3,}", _normalisera_for_matchning(b), re.UNICODE))
    return bool(ta & tb)


def _wiki_infobox_incumbent_namn(roll_en, land_en):
    """Hamtar RATT-namnet ('incumbent') ur engelska Wikipedias ambets-
    infobox for '<roll> of <land>'. Andra oberoende kalla an Wikidata,
    anvands for korsvalidering (fangar t.ex. en vandaliserad P6-post).
    Returnerar raa namnet eller None."""
    try:
        titel = f"{roll_en} of {land_en}"
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query", "prop": "revisions", "rvprop": "content",
            "rvslots": "main", "format": "json", "titles": titel,
            "formatversion": "2", "redirects": "1",
        }
        resp = requests.get(url, params=params, timeout=4,
                            headers={"User-Agent": "AIPrototype/1.0"})
        if resp.status_code != 200:
            return None
        sidor = resp.json().get("query", {}).get("pages", [])
        if not sidor or sidor[0].get("missing"):
            return None
        wt = (sidor[0].get("revisions", [{}])[0]
              .get("slots", {}).get("main", {}).get("content", ""))
        ib = re.search(r"\{\{\s*Infobox\s+(?:official|political)\s+post",
                       wt, re.IGNORECASE)
        start = ib.start() if ib else 0
        m = re.search(r"\|\s*incumbent\s*=\s*\[?\[?([^\]|\n]+)",
                      wt[start:], re.IGNORECASE)
        if not m:
            return None
        namn = m.group(1).strip()
        if 2 < len(namn) <= 80:
            return namn
    except Exception:
        pass
    return None


# GREPP 3: office-entiteter som INTE ar lands-stats/regeringschef och darfor
# inte nas av P6/P35-vagen (vice-roller, EU-organ). Slas upp direkt pa
# office-entitetens P1308 (officeholder). Hardkodade QID = stabila entiteter.
# Alla monster i listan MASTE matcha (roll + kontext) for att trigga.
_LEDARE_TERM = (r"(pr[aeéè]siden\w*|ordf[oö]rand\w*|voorzitter|chef\w*|"
                r"\bhead\b|ledare|leiter|leader|kansler|generalsekret\w*|"
                r"vorsitzend\w*|presiden\w*)")
_ROLL_OFFICE_QID = [
    # USA:s vicepresident
    ("Q11699", [r"\bvice[\s-]?pr[aeéè]siden\w*\b",
                r"\b(usa|united\s+states|amerika\w*|etats-unis|estados\s+"
                r"unidos|vereinigten?\s+staaten|forenta\s+staterna|"
                r"stati\s+uniti|verenigde\s+staten)\b"]),
    # EU-kommissionens ordforande
    ("Q8882", [_LEDARE_TERM, r"europ\w*",
               r"(kommission\w*|commission\w*|comisi\w*|commissione\w*)"]),
    # EU-parlamentets ordforande
    ("Q740126", [_LEDARE_TERM, r"europ\w*",
                 r"(parlament\w*|parliament\w*|parlement\w*)"]),
]


def _office_holder_uppslag(norm):
    for qid, patterns in _ROLL_OFFICE_QID:
        if all(re.search(p, norm, re.IGNORECASE) for p in patterns):
            try:
                namn = _wikidata_hamta_varde(qid, "P1308")
                if namn and 2 < len(namn) <= 80 and not re.fullmatch(
                        r"Q\d+", namn):
                    _debug_logg(f"office-holder {qid} P1308 -> {namn}")
                    return namn
            except Exception:
                pass
    return None


def _amboetsinnehavare_uppslag(fraga, svarssprak="sv"):
    """Deterministiskt Wikidata-uppslag for 'vem ar [roll] i [land] nu'.
    Regeringschef -> P6, statschef -> P35 pa landets post. Returnerar
    innehavarens namn (sprakneutralt) eller None om det inte matchar - da
    tar den vanliga pipelinen over (garanterad fallback)."""
    norm = _normalisera_for_matchning(fraga)
    # Historisk fraga ELLER kandidat-/val-/framtids-/spekulationsfraga
    # (inte "vem SITTER nu") -> hoppa over deterministiska uppslaget.
    if _ar_historisk_fraga(norm) or _AMBETE_EXKLUDERA.search(norm):
        return None
    # GREPP 3: office-entiteter (vice-roller, EU-organ) FORE lands-vagen.
    _off = _office_holder_uppslag(norm)
    if _off:
        return _off
    # Maste vara en NUTIDA innehavar-fraga.
    if not (_ROLL_REALTID.search(norm) or
            (_hitta_roll_sv_nyckel(norm) and
             (_hitta_land_sv_nyckel(norm) or _hitta_land_extra(norm)))):
        return None
    # Regeringschef FORST - annars fastnar "presidente del consiglio" pa
    # den generiska "president"-grinden och ger fel person (statschefen).
    if _ROLL_REGERINGSCHEF.search(norm):
        pid = "P6"
    elif _ROLL_STATSCHEF.search(norm):
        pid = "P35"
    else:
        return None
    land = _hitta_land(norm) or _hitta_land_extra(norm)
    if not land:
        return None
    try:
        for qid in _wikidata_sok_qid(land, "en")[:2]:
            namn = _wikidata_hamta_varde(qid, pid)
            if namn and 2 < len(namn) <= 80 and not re.fullmatch(r"Q\d+", namn):
                # KORSVALIDERING mot en oberoende kalla (engelska Wikipedias
                # infobox). Om Wikidata-namnet klart INTE stammer med
                # Wikipedia-innehavaren (t.ex. en vandaliserad P6-post ->
                # "Andy Burnham" for UK PM) litar vi inte pa det
                # deterministiska svaret utan faller igenom till sok, som
                # laser farska kallor. Matchar de (eller Wikipedia saknas)
                # -> behall det snabba deterministiska svaret.
                # Rollen for Wikipedia-titeln harleds fran PID (P6=regerings-
                # chef, P35=statschef) - INTE fran _hitta_roll_engelska, som
                # mappar "presidente del consiglio" -> "President" och da
                # skulle bygga fel titel ("President of Italy" -> Mattarella)
                # och felaktigt underkanna ett korrekt PM-svar (Meloni).
                roll_en = "Prime Minister" if pid == "P6" else "President"
                wiki_namn = _wiki_infobox_incumbent_namn(roll_en, land)
                if wiki_namn and not _namn_overlappar(namn, wiki_namn):
                    _debug_logg(
                        f"ambetsinnehavare KROCK wd={namn!r} "
                        f"wiki={wiki_namn!r} -> None (faller till sok)")
                    return None
                _debug_logg(f"ambetsinnehavare {pid} {land} ({qid}) -> {namn}")
                return namn
    except Exception:
        pass
    return None


# ============ STRUKTURERAT FAKTAUPPSLAG (exakta Wikidata-egenskaper) ============
# For de vanligaste "vem skapade / vad ar huvudstaden"-fragorna anvander vi
# KANDA Wikidata-egenskaps-ID (P170 skapare, P50 forfattare, P36 huvudstad
# osv) istallet for fritextsok. Det ar exakt och fixar fall som "vem malade
# Sixtinska kapellet" (blev fel/soppa nar det gick via Brave-fritextsok).
_SKAPAR_PROPS = ["P170", "P50", "P86", "P57", "P84", "P58", "P162",
                 "P287", "P61", "P176"]
# P170 creator, P50 author, P86 composer, P57 director, P84 architect,
# P58 screenwriter, P162 producer, P287 designer, P61 discoverer/inventor,
# P176 manufacturer

_SKAPAR_FRAGA = re.compile(
    r"\bwho\s+(painted|wrote|composed|created|made|directed|sculpted|"
    r"designed|built|authored|drew|invented|discovered)\b"
    r"|\bvem\s+(malade|skrev|komponerade|skapade|gjorde|regisserade|"
    r"byggde|ritade|designade|uppfann|upptackte)\b"
    r"|\bwer\s+(hat\s+)?\w*\s*(malte|schrieb|komponierte|schuf|erfand|"
    r"entdeckte)\b"
    r"|\bqui\s+a\s+\w*\s*(peint|ecrit|compose|cree|realise|invente)\b"
    r"|\bquien\s+(pinto|escribio|compuso|creo|invento|construyo)\b"
    r"|\bchi\s+(ha\s+)?\w*\s*(dipinse|scrisse|compose|creo|realizzo|"
    r"invento|dipinto|scritto)\b",
    re.IGNORECASE)

_HUVUDSTAD_FRAGA = re.compile(
    r"\bcapital\s+(city\s+)?of\b|\bhauptstadt\s+von\b|"
    r"\bcapitale\s+d[eui]\b|\bcapital\s+de\b|\bstolica\b|"
    r"\bhuvudstad(en)?\s+(i|for)\b|\b(vad|vilken)\b.{0,25}\bhuvudstad",
    re.IGNORECASE)


def _extrahera_verk(fraga):
    """Plockar ut VERKET (malning/bok/film/lat/byggnad ...) som en
    skapar-fraga handlar om - INTE skaparen. Kritiskt: 'Who painted The
    Starry Night' ska ge 'The Starry Night', inte 'Van Gogh' (annars slar
    vi upp fel entitet). Egen prompt sa modellen inte svarar pa fragan."""
    try:
        prompt = bygg_qwen_prompt(
            "The question asks WHO created a work (painting, book, film, "
            "song, building, sculpture). Output ONLY the name of that WORK "
            "- never its creator, never extra words. If unclear, output "
            "NONE.\n"
            "Q: Who painted The Starry Night?\nWork: The Starry Night\n"
            "Q: Vem skrev Faust?\nWork: Faust\n"
            "Q: Chi dipinse la volta della Cappella Sistina?\n"
            "Work: Sistine Chapel ceiling\n"
            "Q: Wer komponierte die Zauberfloete?\nWork: The Magic Flute\n"
            "Q: Who directed Inception?\nWork: Inception",
            "Q: " + fraga.strip() + "\nWork:")
        out = llm_sok(prompt, max_tokens=20, temperature=0.0,
                      stop=_QWEN_STOPP + ["\n", "Q:", "Work:"])
        v = out["choices"][0]["text"].strip().strip('."\'')
        if not v or v.upper() == "NONE" or len(v) > 80:
            return None
        return v
    except Exception:
        return None


def _props_for_skaparfraga(fn):
    """GREPP 2: verb-specifik property-prioritet. Loser 'Pippi Langstrump'
    -> 'Michael Schaack' (en films REGISSOR returnerades for 'vem SKREV
    boken'). 'skrev/wrote' far BARA forfattare/skapare - aldrig P57 regissor.
    Omatchade verb faller tillbaka pa hela listan (oforandrat beteende)."""
    if re.search(r"\b(wrote|authored|skrev|skrivit|schrieb|verfasste|"
                 r"ecrit|redige|escribio|escrito|scrisse|scritto)\b", fn):
        return ["P50", "P170"]                 # forfattare, sedan skapare
    if re.search(r"\b(painted|malade|malte|gemalt|peint|pinto|"
                 r"dipinse|dipinto|schilderde|geschilderd)\b", fn):
        return ["P170"]                        # skapare av konstverk
    if re.search(r"\b(composed|komponerade|komponierte|composa?|compuso|"
                 r"compose)\b", fn):
        return ["P86", "P170"]                 # kompositor
    if re.search(r"\b(directed|regisserade|realise|realizzo|dirigio)\b", fn):
        return ["P57", "P170"]                 # regissor
    if re.search(r"\b(built|byggde|erbaute|designed|designade|"
                 r"gestaltete)\b", fn):
        return ["P84", "P170", "P287"]         # arkitekt/designer
    if re.search(r"\b(invented|uppfann|erfand|invento|invente|"
                 r"discovered|upptackte|entdeckte|decouvert)\b", fn):
        return ["P61", "P170"]                 # uppfinnare/upptackare
    return _SKAPAR_PROPS                        # created/made/generellt


def _strukturerad_faktauppslag(fraga, sprak="sv"):
    """Exakt Wikidata-uppslag for skapare-av-verk och huvudstad. Returnerar
    ett kort exakt svar (t.ex. 'Vincent van Gogh', 'Canberra') eller None
    (varvid vanliga pipelinen tar over)."""
    fn = _normalisera_for_matchning(fraga)
    ar_skapare = bool(_SKAPAR_FRAGA.search(fn))
    ar_huvudstad = bool(_HUVUDSTAD_FRAGA.search(fn))
    if not (ar_skapare or ar_huvudstad):
        return None
    if ar_skapare:
        entitet = _extrahera_verk(fraga)   # VERKET, inte skaparen
        props = _props_for_skaparfraga(fn)
    else:
        entitet, _rel = _extrahera_entitet_och_relation(fraga)
        props = ["P36"]
    if not entitet:
        return None
    # GREPP 2: samla ALLA kandidat-entiteter forst, skanna sedan
    # property-for-property (INTE entitet-for-entitet). Sa vinner ratt
    # egenskap (t.ex. P50 forfattare) over entitetsordningen - annars
    # returnerar en films regissor (P170 pa filmentiteten som rakar sokas
    # forst) fore bokens forfattare. Loste 'Pippi Langstrump' -> Schaack.
    qids, seen = [], set()
    for sok_sprak in ("en", sprak or "en"):
        for qid in _wikidata_sok_qid(entitet, sok_sprak)[:4]:
            if qid not in seen:
                seen.add(qid); qids.append(qid)
    for pid in props:
        for qid in qids:
            varde = _wikidata_hamta_varde(qid, pid)
            if varde:
                _debug_logg(f"strukturerat: {entitet!r} {pid} -> {varde!r}")
                return varde
    return None


# ---- Utokade strukturerade uppslag: datum (fodd/dod/grundad/slappt) samt
# valuta och officiellt sprak. Samma monster: kant Wikidata-property.
# OBS: release-/publiceringsdatum (P577) hanteras INTE har langre - de
# gick fel pa framtida/oslappta saker (t.ex. "when is GTA 6 released" gav
# 1998 fran fel entitet). Sadana gar till vanlig sok istallet. Har: bara
# palitliga historiska datum (fodd/dod/grundad/byggd).
_DATUM_FRAGA = re.compile(
    r"\bwhen\s+(was|were|did)\b.{0,45}\b(born|die[d]?|found(ed)?|"
    r"establish(ed)?|built)\b"
    r"|\bnar\s+(foddes|fodd|dog|grundades|byggdes)\b"
    r"|\bvilket\s+ar\s+(foddes|fodd|dog|grundades|byggdes)\b"
    r"|\bwann\s+(wurde|starb)\b.{0,30}(geboren|gestorben|gegrundet)?"
    r"|\ben\s+quelle\s+annee\b.{0,30}\b(ne|nee|mort|morte|fonde)\b",
    re.IGNORECASE)

_VALUTA_FRAGA = re.compile(
    r"\bcurrency\s+of\b|\bvaluta(n)?\s+(i|for)\b|\bmonnaie\s+d[eu]\b|"
    r"\bwahrung\s+von\b|\bmoneda\s+de\b", re.IGNORECASE)

_OFF_SPRAK_FRAGA = re.compile(
    r"\bofficial\s+language\b|\bofficiell[at]?\s+sprak|\bvilket\s+sprak\s+"
    r"(talas|pratas|talar\s+man)\b|\blangue\s+officielle\b|"
    r"\bamtssprache\b|\bidioma\s+oficial\b", re.IGNORECASE)


def _valj_datum_prop(fn):
    if re.search(r"\b(born|foddes|fodd|geboren|\bne\b|\bnee\b|nato|nacio)\b", fn):
        return "P569"      # fodelsedatum
    if re.search(r"\b(die[d]?|dog|gestorben|starb|\bmort\b|morte|morto|"
                 r"murio)\b", fn):
        return "P570"      # dodsdatum
    return "P571"          # grundad/byggd/tillkomst (found/establish/built)


def _formatera_datum(varde):
    neg = varde.startswith("-")
    s = varde.lstrip("-")
    m = re.match(r"(\d{1,4})-(\d\d)-(\d\d)", s)
    if m:
        y, mo, d = m.groups()
        suf = " f.Kr." if neg else ""
        if mo == "01" and d == "01":
            return y + suf         # troligen bara aret kant
        return f"{y}-{mo}-{d}" + suf
    m2 = re.match(r"(\d{1,4})", s)
    if m2:
        return m2.group(1) + (" f.Kr." if neg else "")
    return varde


def _strukturerat_extra_uppslag(fraga, sprak="sv"):
    """Datum- (P569/P570/P571/P577), valuta- (P38) och sprak-uppslag (P37)
    mot exakta Wikidata-egenskaper. Returnerar kort exakt svar eller None."""
    fn = _normalisera_for_matchning(fraga)
    ar_datum = bool(_DATUM_FRAGA.search(fn))
    if ar_datum:
        pid = _valj_datum_prop(fn)
    elif _VALUTA_FRAGA.search(fn):
        pid = "P38"
    elif _OFF_SPRAK_FRAGA.search(fn):
        pid = "P37"
    else:
        return None
    entitet, _rel = _extrahera_entitet_och_relation(fraga)
    if not entitet:
        return None
    seen = set()
    for sok_sprak in ("en", sprak or "en"):
        for qid in _wikidata_sok_qid(entitet, sok_sprak)[:4]:
            if qid in seen:
                continue
            seen.add(qid)
            varde = _wikidata_hamta_varde(qid, pid)
            if varde:
                if ar_datum:
                    varde = _formatera_datum(varde)
                _debug_logg(f"strukturerat-extra: {entitet!r} {pid} -> {varde!r}")
                return varde
    return None


# ============ WIKIPEDIA GENERELL UPPSLAGNING (Qwen-driven titelval) ============
# Ersatter behovet av per-kategori hardkodade uppslag (utmarkelser,
# sport, software m.m.). Qwen valjer engelsk Wikipedia-artikeltitel,
# vi hamtar summary via REST API, och skickar den som fakta till Llama.
# Fungerar for allt Wikipedia tacker (~7 miljoner artiklar) med bara
# en hardkodad prompt istallet for tabeller per amne.
_QWEN_WIKIPEDIA_PROMPT = """You are a Wikipedia title finder.

Given a question (may be Swedish or English), output ONE English Wikipedia article title that likely contains the answer. Just the title on one line, no explanation, no quotes.

For most factual questions there IS a Wikipedia article - try hard to find one. Common patterns:
- "Vem ar X i Y" / "Who is X of Y" -> "X of Y" article
- Questions about awards -> the award's own article
- Questions about latest sport championship -> the tournament's own article
- Questions about company CEO -> the company's own article
- Questions about software version -> the software's own article

Output NONE only for math, code requests, personal opinions, or general chitchat.

Examples:
Q: Vem ar statsminister i Sverige?
A: Prime Minister of Sweden

Q: Vem vann Nobelpriset i litteratur senast?
A: Nobel Prize in Literature

Q: Vad ar huvudstaden i Australien?
A: Canberra

Q: Vilken ar senaste versionen av Python?
A: Python (programming language)

Q: Vem ar VD for Volvo Cars?
A: Volvo Cars

Q: Vem vann senaste Super Bowl?
A: Super Bowl

Q: Vem vann Champions League senast?
A: UEFA Champions League

Q: Vem ar dirigent for Berlinfilharmonikerna?
A: Berlin Philharmonic

Q: What is Kubernetes?
A: Kubernetes

Q: Who won the last NBA Finals?
A: NBA Finals

Q: Vad ar 15% av 240?
A: NONE

Q: Skriv en dikt
A: NONE

Q: Hur mar du idag?
A: NONE

Q: {fraga}
A:"""


def wikipedia_generell_uppslag(fraga, sprak="sv"):
    """Generell Wikipedia-uppslagning: Qwen valjer artikeltitel, REST
    API-summary returneras som fakta. Ersatter behovet av hardkodade
    uppslag per amneskategori."""
    prompt = _QWEN_WIKIPEDIA_PROMPT.replace("{fraga}", fraga.strip())
    # Stop-tokens som fangar nasta Q: rad - Qwen har annars tendens
    # att fortsatta generera fler Q/A-par efter sitt svar.
    stop_tokens = _QWEN_STOPP + ["\nQ:", "\n\nQ:", "\n\n"]
    try:
        out = llm_sok(prompt, max_tokens=50, temperature=0.0,
                      stop=stop_tokens)
        titel = out["choices"][0]["text"].strip()
    except Exception:
        return None
    # Ta bara forsta raden om Qwen skulle svarat med flera
    titel = titel.split("\n")[0].strip()
    # Rensa citations och trailing skarpljud
    titel = re.sub(r"^['\"`]+|['\"`.]+$", "", titel).strip()
    if not titel or titel.upper() == "NONE" or len(titel) > 100:
        return None
    _debug_logg(f"wikipedia_generell: qwen valde titel={titel!r}")
    # Steg 2: Hamta summary via Wikipedia REST API. redirect=true
    # foljer omdirigeringar automatiskt (t.ex. "USA" -> "United States").
    try:
        # URL-koda titeln for att stodja special-tecken (t.ex. mellanslag,
        # parenteser). Wikipedia REST tolererar bade _ och %20.
        titel_kodad = titel.replace(" ", "_")
        url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
                + requests.utils.quote(titel_kodad, safe=""))
        resp = requests.get(url, params={"redirect": "true"}, timeout=5,
                            headers={"User-Agent": "AIPrototype/1.0"})
        if resp.status_code != 200:
            return None
        data = resp.json()
        # Skippa disambiguation-sidor - dessa listar bara olika betydelser
        # ("X may refer to: ...") och innehaller inte det faktiska svaret.
        if data.get("type") == "disambiguation":
            return None
        extract = data.get("extract", "").strip()
        if not extract or len(extract) < 30:
            return None
        # Trunkera till sista fullbordade mening under 1500 tecken sa
        # Llama-kontexten inte overbelastas.
        if len(extract) > 1500:
            trunkerad = extract[:1500]
            sista_punkt = trunkerad.rfind(".")
            if sista_punkt > 500:
                extract = trunkerad[:sista_punkt + 1]
            else:
                extract = trunkerad
        _debug_logg(f"wikipedia_generell: extract=({len(extract)} tecken)")
        return extract
    except Exception:
        return None


# Fillerord som ska bort ur en Wikipedia-sokstrang: frageord, artiklar/
# prepositioner (sv/en/de/nl/fr/it/es/pt) OCH generiska kategoriord som
# annars far ett generiskt sok att traffa fel artikel ('cos'e la pro
# vercelli calcio' drog soket mot Juventus/Serie A istallet for klubben).
_SOK_STOPP = {
    "vad", "vilken", "vilket", "vilka", "vem", "ar", "har", "och",
    "what", "which", "who", "is", "are", "the", "a", "an", "of", "and",
    "was", "ist", "wer", "wie", "der", "die", "das", "ein", "eine", "und",
    "wat", "een", "het", "de", "van",
    "quelle", "quel", "qu", "est", "ce", "que", "qui", "le", "la", "les",
    "un", "une", "du", "des", "pour", "et",
    "cos", "che", "cosa", "qual", "quale", "il", "lo", "gli", "di", "del",
    "della", "per", "in", "i", "e",
    "es", "el", "los", "las", "quien", "para", "com", "y",
    "o", "os", "as", "um", "uma", "quem",
    # generiska kategoriord (far ej ensamma avgora relevans)
    "football", "fussball", "calcio", "club", "team", "klubb", "verein",
    "lag", "museum", "museo", "musee", "restaurang", "restaurant",
    "ristorante", "band", "spel", "game", "for", "fotboll",
}


def _wiki_entitetsord(fraga):
    """Distinktiva entitetsord ur en fraga: alla ord utom frageord,
    artiklar/prepositioner och generiska kategoriord (_SOK_STOPP). Anvands
    bade for att BYGGA en fokuserad sokstrang och for titel-relevanskollen
    - sa att fillerord inte drar soket mot generiska artiklar och inte
    heller ensamma godkanner en irrelevant titel."""
    return [o for o in re.findall(r"[a-z0-9]+",
                                  _normalisera_for_matchning(fraga))
            if o not in _SOK_STOPP]


def _wiki_sokstrang(fraga):
    """Fokuserad Wikipedia-sokstrang = entitetsorden. Faller tillbaka pa
    hela fragan om inget entitetsord aterstar."""
    ord = _wiki_entitetsord(fraga)
    return " ".join(ord) if ord else fraga.strip()


# Otvetydiga lokativa prepositioner (ej 'i'/'a'/'en' som ocksa ar artiklar).
# Ordet DIREKT efter ar en platsangivelse ('in hamburg', 'no porto') och far
# INTE ensamt godkanna en titelmatchning - annars matchar 'restaurant lucia
# in hamburg' vilken Hamburg-artikel som helst pa staden.
_LOKATIV_PREP = {"in", "im", "no", "na", "au", "aan", "at"}


def _wiki_distinkta_ord(fraga):
    """Distinktiva ord for TITEL-relevanskollen: entitetsord UTAN rena
    platsangivelser (ord som star direkt efter en lokativ preposition).
    Sa matchar guarden pa sjalva entiteten (lucia) och inte pa staden
    (hamburg) - men klubbar som HETER efter en ort ('pro vercelli', utan
    preposition fore) behalls fortfarande."""
    tokens = re.findall(r"[a-z0-9]+", _normalisera_for_matchning(fraga))
    ut, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in _LOKATIV_PREP and i + 1 < len(tokens):
            i += 2   # hoppa prepositionen OCH platsordet efter den
            continue
        if t not in _SOK_STOPP:
            ut.append(t)
        i += 1
    return ut


def _wiki_sok_titlar(lang, sokstrang, limit=5):
    """Wikipedias full-text sok-API (action=query&list=search): returnerar
    de basta artikeltitlarna for sokstrangen. Till skillnad fran opensearch
    (som bara prefix-matchar TITLAR) hittar detta ratt artikel aven nar
    fragan innehaller extra ord. Vi hamtar FLERA traffar sa att en titel-
    relevanskoll (_titel_matchar_fraga) kan valja ratt och forkasta
    irrelevanta topp-traffar. Ren HTTP, inga modellanrop."""
    try:
        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {"action": "query", "list": "search",
                  "srsearch": sokstrang.strip(), "srlimit": limit,
                  "srnamespace": 0, "format": "json"}
        resp = requests.get(url, params=params, timeout=5,
                            headers={"User-Agent": "AIPrototype/1.0"})
        if resp.status_code != 200:
            return []
        traffar = resp.json().get("query", {}).get("search", [])
        return [t["title"] for t in traffar if t.get("title")]
    except Exception:
        return []


def _titel_matchar_fraga(titel, sokstrang):
    """Skydd mot att sok-API:et returnerar en IRRELEVANT topp-traff (t.ex.
    'vad ar kulturhuset cyklopen i stockholm' -> en orelaterad forfattar-
    artikel, eller 'pro vercelli' -> 'Pro Patria'). Kraver att minst ETT
    sarskiljande innehallsord (>=6 tecken) ur fragan aterfinns i artikel-
    titeln. Annars forkastas traffen sa att vi hellre faller tillbaka pa
    Brave / arligt 'hittade inte' an ger ett sjalvsakert FEL svar."""
    # HELA ORD (tokens) i titeln - inte delstrangar. Ett akta entitetsnamn
    # (cyklopen, teutonia, vercelli) star som eget ord i titeln, medan ett
    # generiskt kategoriord ('restaurang') INTE ska matcha nar det bara ar
    # en delstrang i ett sammansatt titelord ('Automatrestaurang').
    titel_tokens = set(re.findall(r"[a-z0-9]+",
                                  _normalisera_for_matchning(titel)))
    # Distinktiva entitetsord (ej frageord/artikel/kategoriord/platsangivelse
    # efter lokativ preposition), >=6 tecken sa generiska ortnamn (malmo,
    # hamburg) inte ensamma godkanner en orelaterad artikel som rakar namna
    # staden ('Fotboll i Malmo', 'restaurant lucia in Hamburg').
    ord = [o for o in _wiki_distinkta_ord(sokstrang) if len(o) >= 6]
    if not ord:
        return False   # inget distinktivt att verifiera mot -> hellre Brave
    return any(o in titel_tokens for o in ord)


def _wiki_rest_summary(lang, titel):
    """Hamtar REST-summary-extract for en artikeltitel pa given sprak-
    Wikipedia. Foljer omdirigeringar, hoppar disambiguation-sidor och
    trunkerar till sista fullbordade mening under 1500 tecken."""
    try:
        titel_kodad = requests.utils.quote(titel.replace(" ", "_"), safe="")
        url = (f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
               + titel_kodad)
        resp = requests.get(url, params={"redirect": "true"}, timeout=5,
                            headers={"User-Agent": "AIPrototype/1.0"})
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("type") == "disambiguation":
            return None
        extract = data.get("extract", "").strip()
        if not extract or len(extract) < 30:
            return None
        if len(extract) > 1500:
            trunkerad = extract[:1500]
            sista_punkt = trunkerad.rfind(".")
            extract = (trunkerad[:sista_punkt + 1]
                       if sista_punkt > 500 else trunkerad)
        return extract
    except Exception:
        return None


# Spraknamn (fran detektera_svarssprak_namn) -> Wikipedia sprak-kod.
_SPRAKNAMN_TILL_WIKI = {
    "swedish": "sv", "english": "en", "german": "de", "dutch": "nl",
    "french": "fr", "italian": "it", "spanish": "es", "portuguese": "pt",
}


def wikipedia_sok_uppslag(fraga_orig, native_lang, fraga_en=None):
    """B2/A2: recall for nischade entiteter. Manga lokala/nischade
    namngivna saker (klubbar, museer, hus, band) finns bara pa sitt EGET
    sprak i Wikipedia. Sok darfor forst i nativ-sprakets Wikipedia med
    originalfragan, sedan i engelska med den engelska fragan. Ratt titel
    hittas via sok-API:et istallet for att Llama gissar den. Returnerar en
    summary-extract (pa det traffade sprakets sprak) eller None. Ren HTTP,
    inga modellanrop."""
    kandidater = []
    if native_lang and native_lang != "en":
        kandidater.append((native_lang, fraga_orig))
    kandidater.append(("en", fraga_en or fraga_orig))
    for lang, sokstrang in kandidater:
        if not sokstrang or not sokstrang.strip():
            continue
        # Fokuserad sokstrang (fillerord bort) sa soket traffar ratt artikel.
        sokstrang_ren = _wiki_sokstrang(sokstrang)
        titlar = _wiki_sok_titlar(lang, sokstrang_ren)
        dist6 = [o for o in _wiki_distinkta_ord(sokstrang) if len(o) >= 6]
        ent = _wiki_entitetsord(sokstrang)
        huvud = ent[0] if ent else None
        for rank, titel in enumerate(titlar):
            titel_tokens = set(re.findall(r"[a-z0-9]+",
                                          _normalisera_for_matchning(titel)))
            if _titel_matchar_fraga(titel, sokstrang):
                extract = _wiki_rest_summary(lang, titel)
                if extract:
                    _debug_logg(f"wikipedia_sok: {lang}:{titel!r} ({len(extract)} tecken)")
                    return extract
            elif (rank == 0 and dist6 and huvud and len(huvud) >= 5
                  and huvud in titel_tokens):
                # Topptraffens TITEL bar huvudentiteten (t.ex. 'Gekas') men
                # kvalificeraren (>=6, t.ex. 'ullared') sitter i SUMMARY, inte
                # i titeln. Lita pa den BARA om summaryn faktiskt namner
                # kvalificeraren - sa 'gekas i ullared' -> butiken 'Gekas'
                # (summary namner Ullared), men 'lucia in hamburg' -> INTE
                # 'Lucia von Syrakus' (summaryn namner ingen Hamburg).
                extract = _wiki_rest_summary(lang, titel)
                if extract and any(
                        o in _normalisera_for_matchning(extract) for o in dist6):
                    _debug_logg(f"wikipedia_sok: {lang}:{titel!r} (rank0 huvud+summary)")
                    return extract
                _debug_logg(f"wikipedia_sok: rank0 {lang}:{titel!r} saknar kvalificerare")
            else:
                _debug_logg(f"wikipedia_sok: forkastar irrelevant {lang}:{titel!r}")
    return None


# ============ WIKIPEDIA UTMARKELSE-UPPSLAG (mer palitligt an SPARQL) ============
# Wikidata SPARQL med pq:P585 visade sig opalitligt (datum matchar inte
# alltid ceremonidatum - Wikidata's P585 anvander ofta andra typer av
# "point in time"). Wikipedias artiklar for utmarkelser har daremot ett
# 'holder'-falt i sin infobox som redaktorer haller mycket uppdaterat.
# Anvand det i forsta hand for utmarkelser, med gamla Wikidata som
# fallback.
_UTMARKELSE_ARTIKLAR = [
    (re.compile(r"nobelpris.*litteratur|nobel.*prize.*literature|"
                r"nobel litteratur", re.IGNORECASE),
     "Nobel Prize in Literature"),
    (re.compile(r"nobels? ?fred(spris)?|nobel peace prize", re.IGNORECASE),
     "Nobel Peace Prize"),
    (re.compile(r"nobelpris.*fysik|nobel.*prize.*physics|nobel fysik",
                re.IGNORECASE),
     "Nobel Prize in Physics"),
    (re.compile(r"nobelpris.*kemi|nobel.*prize.*chemistry|nobel kemi",
                re.IGNORECASE),
     "Nobel Prize in Chemistry"),
    (re.compile(r"nobelpris.*medicin|nobelpris.*fysiologi|"
                r"nobel.*prize.*(medicine|physiology)", re.IGNORECASE),
     "Nobel Prize in Physiology or Medicine"),
    (re.compile(r"(oscar|academy award).*(basta film|best picture)|"
                r"(basta film|best picture).*(oscar|academy award)",
                re.IGNORECASE),
     "Academy Award for Best Picture"),
]

_HOLDER_LANK_MONSTER = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_HOLDER_AR_MONSTER = re.compile(r"\((\d{4})\)")


def _parsa_wikipedia_holder(holder_text, sprak):
    """Parsa Wikipedia infobox 'holder'-falt till (namn, ar).
    Formats vi hanterar (fran verkliga artiklar):
      '[[László Krasznahorkai]] (2025)'
      '[[María Corina Machado]] [[2025 Nobel Peace Prize|(2025)]]'
      '[[Susumu Kitagawa]], [[Richard Robson (chemist)|Richard Robson]], and [[Omar M. Yaghi|Omar Yaghi]] (2025)'
      \"''[[One Battle After Another]]'' ([[98th Academy Awards]])\"
    """
    ar_match = _HOLDER_AR_MONSTER.search(holder_text)
    ar = ar_match.group(1) if ar_match else None
    lankar = _HOLDER_LANK_MONSTER.findall(holder_text)
    namn_lista = []
    for link_target, display in lankar:
        vald = (display or link_target).strip()
        # Skippa ceremonisidor (t.ex. "98th Academy Awards", "2025 Nobel Peace Prize")
        if re.search(r"academy awards?|academy award ceremony", vald,
                     re.IGNORECASE):
            continue
        if re.search(r"^\d{4}\b.*nobel|nobel.*\d{4}", vald,
                     re.IGNORECASE):
            continue
        if re.fullmatch(r"\(?\d{4}\)?", vald):
            continue
        # Rensa disambiguerings-parenteser (t.ex. "John Clarke (physicist)")
        vald = re.sub(r"\s*\([^)]+\)\s*$", "", vald).strip()
        if vald and vald not in namn_lista:
            namn_lista.append(vald)
    if not namn_lista:
        return None, ar
    binder = _str("namn_binder", sprak)
    if len(namn_lista) == 1:
        namn_str = namn_lista[0]
    elif len(namn_lista) == 2:
        namn_str = f"{namn_lista[0]}{binder}{namn_lista[1]}"
    else:
        namn_str = ", ".join(namn_lista[:-1]) + f",{binder}{namn_lista[-1]}"
    return namn_str, ar


def wikipedia_utmarkelse_uppslag(fraga, sprak="sv"):
    """Hamtar senaste vinnaren av en utmarkelse direkt fran Wikipedias
    infobox 'holder'-falt. Ersatter tidigare Wikidata SPARQL-approach
    som visade sig opalitlig for arsangivelser. Wikipedias holder-falt
    uppdateras snabbt av redaktorer och matchar prisceremonins ar korrekt."""
    fraga_norm = _normalisera_for_matchning(fraga)
    for monster, artikel in _UTMARKELSE_ARTIKLAR:
        if not monster.search(fraga_norm):
            continue
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query", "prop": "revisions", "rvprop": "content",
                "rvslots": "main", "format": "json", "titles": artikel,
                "formatversion": "2", "redirects": "1",
            }
            resp = None
            for forsok in range(3):
                try:
                    resp = requests.get(url, params=params, timeout=5,
                                        headers={"User-Agent": "AIPrototype/1.0"})
                    if resp.status_code == 200:
                        break
                except requests.exceptions.RequestException:
                    resp = None
                time.sleep(0.3 * (2 ** forsok))
            if resp is None or resp.status_code != 200:
                return None
            data = resp.json()
            sidor = data.get("query", {}).get("pages", [])
            if not sidor or sidor[0].get("missing"):
                return None
            wikitext = (sidor[0].get("revisions", [{}])[0]
                        .get("slots", {}).get("main", {}).get("content", ""))
            # Matcha till nasta faltrad (\n|) eller infobox-slut (\n}}) -
            # anvander lookahead sa vi inte snor undan pa | inuti langar
            # (t.ex. '[[Richard Robson (chemist)|Richard Robson]]').
            holder_match = re.search(
                r"\|\s*holder\s*=\s*(.+?)(?=\n\s*[|}])",
                wikitext, re.IGNORECASE)
            if not holder_match:
                return None
            holder_text = holder_match.group(1).strip()
            namn, ar = _parsa_wikipedia_holder(holder_text, sprak)
            if not namn:
                return None
            if ar:
                return _str("utmarkelse_med_ar", sprak, ar=ar, namn=namn)
            return _str("utmarkelse_utan_ar", sprak, namn=namn)
        except Exception as e:
            try:
                with open(os.path.join(BASKATALOG, "wikipedia_fel.log"), "a",
                          encoding="utf-8") as f:
                    f.write(f"{datetime.datetime.now()}: utmarkelse "
                             f"{fraga!r} -> {e!r}\n")
            except Exception:
                pass
            return None
    return None


# ============ WIKIPEDIA SPORT-UPPSLAG (fanger senaste champions) ============
# Wikipedia-artiklar for stora sportturneringar har infobox-falt for
# senaste vinnaren - men faltet heter olika saker per sport. Konfigurera
# vilka falt som ska provas per artikel, i prioriteringsordning.
# Super Bowl har en unik struktur (season2025 med lag+resultat i en
# UBL-lista) och hanteras separat.
_SPORT_ARTIKLAR = [
    (re.compile(r"champions? league|uefa cl|championsligan", re.IGNORECASE),
     "UEFA Champions League",
     ["current champions", "current_champions"]),
    (re.compile(r"world cup|fotbolls.?vm|fifa.*vm|fifa world cup",
                re.IGNORECASE),
     "FIFA World Cup",
     ["current champions", "current_champions"]),
    (re.compile(r"nba finals?|nba.mastar(skap)?", re.IGNORECASE),
     "NBA Finals",
     ["defending_champs", "defending champs"]),
    (re.compile(r"stanley cup|nhl (finals?|mastar)", re.IGNORECASE),
     "Stanley Cup",
     ["mostrecent"]),
    (re.compile(r"premier league|engelska ligan|england(s)? liga",
                re.IGNORECASE),
     "Premier League",
     ["champions"]),
    (re.compile(r"allsvenskan|svenska ligan", re.IGNORECASE),
     "Allsvenskan",
     ["champions"]),
]

_SPORT_SUPERBOWL = re.compile(r"super.?bowl", re.IGNORECASE)


def _parsa_sport_champion(text, sprak):
    """Extrahera lagnamn fran ett Wikipedia sport-falt. Formaten:
      '{{nowrap|{{fbaicon|FRA}} [[Paris Saint-Germain FC|Paris Saint-Germain]]<br />(2nd title)}}'
      '[[Arsenal F.C.|Arsenal]] (4th title)'
      '[[Carolina Hurricanes]]<br />(2nd title)'
      '[[2025-26 Carolina Hurricanes season|Carolina Hurricanes]] (2)'
    Ta forsta wiki-lanken som ser ut som ett lagnamn."""
    lankar = re.findall(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", text)
    for target, display in lankar:
        vald = (display or target).strip()
        # Skippa arstal, seasons, referensartiklar
        if re.search(r"^\d{4}", vald) or "season" in vald.lower():
            continue
        # Rensa disambiguering (t.ex. "Arsenal F.C." -> "Arsenal")
        namn = re.sub(r"\s+F\.?C\.?\s*$", "", vald)
        namn = re.sub(r"\s+CF\s*$", "", namn)
        namn = namn.strip()
        if namn:
            return namn
    return None


def _parsa_superbowl_infobox(wikitext, sprak):
    """Super Bowl har unik infobox-struktur - de senaste seasons har eget
    numrerat falt (t.ex. season2025) med UBL-list innehallande:
      spelnamn (bold), plats, datum, vinnare (bold) score<br>forlorare score.
    BAADE spelnamnet ('Super Bowl LX') och vinnaren har bold-markering,
    men vinnaren ar den sista bold-lanken (spelnamnet forst).
    Hitta hogsta season-nummer och extrahera vinnaren."""
    seasons = re.findall(
        r"\|\s*season(\d{4})\s*=\s*(.+?)(?=\n\s*[|}])",
        wikitext, re.IGNORECASE)
    if not seasons:
        return None, None
    seasons.sort(key=lambda x: int(x[0]), reverse=True)
    for ar_str, innehall in seasons:
        bold_lag = re.findall(r"'''\[\[([^\]|]+)(?:\|([^\]]+))?\]\]'''",
                              innehall)
        if not bold_lag:
            continue
        # Ta sista bold-lanken - spelnamnet ("Super Bowl LX") kommer
        # forst, sedan vinnarlaget. Skippa turneringsnamn.
        for target, display in reversed(bold_lag):
            vald = (display or target).strip()
            if re.match(r"^super\s+bowl\s+[ivxlcdm]+$", vald,
                        re.IGNORECASE):
                continue
            # Rensa "2025 Seattle Seahawks season" -> "Seattle Seahawks"
            vald = re.sub(r"^\d{4}\s+", "", vald)
            vald = re.sub(r"\s+season\s*$", "", vald)
            if vald.strip():
                return vald.strip(), ar_str
    return None, None


def wikipedia_sport_uppslag(fraga, sprak="sv"):
    """Hamtar senaste champion for stora sportturneringar direkt fran
    Wikipedias infobox. Ersatter DDGS-fallback som var sportens svaga
    lank (0-10% korrekt enligt tidigare tester)."""
    fraga_norm = _normalisera_for_matchning(fraga)
    # Super Bowl - separat hantering pga unik infobox-struktur
    if _SPORT_SUPERBOWL.search(fraga_norm):
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {"action": "query", "prop": "revisions",
                      "rvprop": "content", "rvslots": "main",
                      "format": "json", "titles": "Super Bowl",
                      "formatversion": "2", "redirects": "1"}
            resp = requests.get(url, params=params, timeout=5,
                                headers={"User-Agent": "AIPrototype/1.0"})
            if resp.status_code == 200:
                wt = (resp.json().get("query", {}).get("pages", [{}])[0]
                      .get("revisions", [{}])[0].get("slots", {})
                      .get("main", {}).get("content", ""))
                vinnare, ar = _parsa_superbowl_infobox(wt, sprak)
                if vinnare:
                    if ar:
                        return _str("sport_superbowl_med_ar", sprak,
                                     vinnare=vinnare, ar=ar)
                    return _str("sport_superbowl_utan_ar", sprak,
                                 vinnare=vinnare)
        except Exception:
            pass
    # Ovriga sport-artiklar
    for monster, artikel, falt_lista in _SPORT_ARTIKLAR:
        if not monster.search(fraga_norm):
            continue
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {"action": "query", "prop": "revisions",
                      "rvprop": "content", "rvslots": "main",
                      "format": "json", "titles": artikel,
                      "formatversion": "2", "redirects": "1"}
            resp = requests.get(url, params=params, timeout=5,
                                headers={"User-Agent": "AIPrototype/1.0"})
            if resp.status_code != 200:
                return None
            wt = (resp.json().get("query", {}).get("pages", [{}])[0]
                  .get("revisions", [{}])[0].get("slots", {})
                  .get("main", {}).get("content", ""))
            for falt in falt_lista:
                m = re.search(
                    rf"\|\s*{re.escape(falt)}\s*=\s*(.+?)(?=\n\s*[|}}])",
                    wt, re.IGNORECASE)
                if m:
                    champion = _parsa_sport_champion(m.group(1), sprak)
                    if champion:
                        return _str("sport_generisk", sprak,
                                     artikel=artikel, champion=champion)
        except Exception:
            pass
    return None


# ============ WIKIDATA SOFTWARE VERSION-UPPSLAG (P348) ============
# Populara mjukvaror har Q-ID i Wikidata dar property P348 haller senaste
# versionen. Preferred rank pekar pa "current stable release". Fallback
# till senaste datum om ingen preferred markering finns.
_SOFTWARE_Q_ID = {
    "python": "Q28865",
    "node.js": "Q756100", "nodejs": "Q756100", "node": "Q756100",
    "react": "Q19399674",
    "typescript": "Q978185",
    "docker": "Q15206305",
    "ubuntu": "Q381",
    "git": "Q186055",
    "kubernetes": "Q19951979",
    "rust": "Q575650",
    "go language": "Q37227", "golang": "Q37227",
}

_SOFTWARE_FRAGA = re.compile(
    r"(senaste|latest|current|nyaste|nuvarande|slappt|slappts|"
    r"stable|version).{0,25}(python|node\.?js|node|react|typescript|"
    r"docker|ubuntu|kubernetes|git|rust|go)|"
    r"(python|node\.?js|node|react|typescript|docker|ubuntu|"
    r"kubernetes|git|rust|go).{0,25}(senaste|latest|current|"
    r"nyaste|nuvarande|slappt|slappts|version)",
    re.IGNORECASE,
)


def wikidata_software_version_uppslag(fraga, sprak="sv"):
    """Fragar Wikidata's P348 (software version identifier) for att fa
    aktuell stabil version. Filter: skippa alpha/beta/preview. Foredra
    preferred rank; annars ta senaste datum."""
    if not _SOFTWARE_FRAGA.search(fraga):
        return None
    # Vilken mjukvara namns i fragan?
    fraga_low = fraga.lower()
    valt_qid = None
    valt_namn = None
    for namn, qid in _SOFTWARE_Q_ID.items():
        if namn in fraga_low:
            valt_qid = qid
            valt_namn = namn.title()
            break
    if not valt_qid:
        return None
    # Steg 1: probera efter preferred rank
    sparql_pref = f"""
    SELECT ?version ?date WHERE {{
      wd:{valt_qid} p:P348 ?stmt.
      ?stmt ps:P348 ?version.
      ?stmt wikibase:rank wikibase:PreferredRank.
      OPTIONAL {{ ?stmt pq:P577 ?date. }}
    }}
    LIMIT 1
    """
    # Steg 2: fallback - senaste datum, skippa preview/beta/alpha
    sparql_latest = f"""
    SELECT ?version ?date WHERE {{
      wd:{valt_qid} p:P348 ?stmt.
      ?stmt ps:P348 ?version.
      OPTIONAL {{ ?stmt pq:P577 ?date. }}
      FILTER NOT EXISTS {{ ?stmt pq:P548 wd:Q2158849 }}
      FILTER NOT EXISTS {{ ?stmt pq:P548 wd:Q3295609 }}
      FILTER NOT EXISTS {{ ?stmt pq:P548 wd:Q107130506 }}
    }}
    ORDER BY DESC(?date)
    LIMIT 1
    """
    for sparql in (sparql_pref, sparql_latest):
        try:
            data = _wikidata_query(sparql)
            if not data:
                continue
            b = data.get("results", {}).get("bindings", [])
            if not b:
                continue
            version = b[0].get("version", {}).get("value", "")
            datum = b[0].get("date", {}).get("value", "")[:10]
            if not version:
                continue
            if datum:
                return _str("software_med_datum", sprak,
                             namn=valt_namn, version=version, datum=datum)
            return _str("software_utan_datum", sprak,
                         namn=valt_namn, version=version)
        except Exception:
            continue
    return None


def wikidata_universell_uppslagning(fraga, sprak="sv"):
    """Universell uppslagning - fungerar for VILKEN entitet och VILKEN
    egenskap som helst (VD, befolkning, grundare, huvudstad, tranare,
    m.m.), inte bara hardkodade kategorier. Returnerar None om nagot
    inte stammer, da tar sok+AI-pipelinen over som vanligt (sakert).

    Optimering: provar bara ett begransat antal kandidater (3 entiteter
    x 2 egenskaper = max 6 kombinationer, ner fran tidigare 15), och
    kor dem PARALLELLT istallet for i sekvens - det som tidigare tog
    upp till 22 sekunder (6-15 natverksanrop i rad) tar nu ungefar
    lika lange som EN forfragan, eftersom de skickas samtidigt."""
    fraga_norm = _normalisera_for_matchning(fraga)
    if not _ar_faktafraga(fraga_norm):
        return None
    entitet, relation = _extrahera_entitet_och_relation(fraga)
    if not entitet or not relation:
        return None
    qid_kandidater = _wikidata_sok_qid(entitet, sprak=sprak)[:3]
    pid_kandidater = _wikidata_sok_egenskap(relation)[:2]
    if not qid_kandidater or not pid_kandidater:
        return None

    kombinationer = [(qid, pid) for qid in qid_kandidater for pid in pid_kandidater]
    with ThreadPoolExecutor(max_workers=len(kombinationer)) as pool:
        framtider = {pool.submit(_wikidata_hamta_varde, qid, pid): i
                     for i, (qid, pid) in enumerate(kombinationer)}
        resultat_per_index = {}
        for framtid in as_completed(framtider):
            resultat_per_index[framtider[framtid]] = framtid.result()

    # Behall prioritetsordningen (basta entitets-/egenskapsgissningen
    # forst) aven om svaren kom tillbaka i annan ordning parallellt.
    for i in range(len(kombinationer)):
        varde = resultat_per_index.get(i)
        if varde and len(varde) <= 150:
            return _str("wikidata_universell", sprak,
                         relation=relation, entitet=entitet, varde=varde)
    return None


# ============ WEBBSOKNING ============
def _oversatt_till_engelska(fraga):
    """Oversatter en svensk fraga till engelska for battre sokresultat -
    engelska sokresultat ar ofta tydligare/mer atkomliga an svenska.
    Returnerar originalfragan om nagot gar fel (sakert fallback)."""
    try:
        # KRITISKT: en 3B-modell tenderar att SVARA pa en fraga istallet
        # for att oversatta den (t.ex. "Wer ist Bundeskanzler?" -> "Olaf
        # Scholz" ur foraldrat minne), vilket forgiftar sokfragan. Few-
        # shot-exempel dar FRAGOR oversatts (aldrig besvaras) tvingar
        # modellen till ren oversattning. Namn/artal i indata ar ALLTID
        # fel har (modellen ska aldrig fylla i sadant sjalv).
        prompt = bygg_qwen_prompt(
            "You are a translation engine. Translate the user's text into "
            "English. This is a TRANSLATION task, NOT a question-answering "
            "task: even when the text is a question, translate the QUESTION "
            "itself word-for-word - NEVER answer it, NEVER add a name, date "
            "or fact that is not in the source. Keep every noun literal "
            "('glodlampa' -> 'light bulb', NOT 'globe'). Output ONLY the "
            "English sentence, nothing else.\n\n"
            "Text: Wer ist aktuell Bundeskanzler von Deutschland?\n"
            "English: Who is currently the Chancellor of Germany?\n\n"
            "Text: Wie is nu de premier van Nederland?\n"
            "English: Who is now the prime minister of the Netherlands?\n\n"
            "Text: Quelle est la derniere version de Python?\n"
            "English: What is the latest version of Python?\n\n"
            "Text: Quien gano el ultimo Super Bowl?\n"
            "English: Who won the latest Super Bowl?\n\n"
            "Text: Vem ar statsminister i Sverige just nu?\n"
            "English: Who is the prime minister of Sweden right now?",
            "Text: " + fraga.strip() + "\nEnglish:"
        )
        out = llm_sok(prompt, max_tokens=60, temperature=0.0,
                      stop=_QWEN_STOPP + ["\nText:", "\nEnglish:", "\n\n"])
        oversattning = out["choices"][0]["text"].strip().strip('"')
        # Ta bort en ev. kvarhangande etikett ("English:") och nyrad.
        oversattning = oversattning.split("\n")[0].strip()
        if oversattning.lower().startswith("english:"):
            oversattning = oversattning[len("english:"):].strip()
        if oversattning and len(oversattning) > 2:
            return oversattning
    except Exception:
        pass
    return fraga


def _oversatt_fran_engelska(text, malsprak_namn):
    """Speglar en fast (engelsk) standardtext till anvandarens sprak sa
    att aven kanned fallback-meddelanden foljer samma 'svara pa anvandarens
    sprak'-princip. Returnerar originalet oforandrat om malspraket ar
    engelska/okant eller vid fel."""
    if (not text or not malsprak_namn
            or malsprak_namn.strip().lower() == "english"):
        return text
    try:
        prompt = bygg_qwen_prompt(
            f"Translate the following message into {malsprak_namn}. Keep "
            f"the meaning and tone; reply with ONLY the translation.",
            text)
        out = llm(prompt, max_tokens=220, temperature=0.0, stop=_LLAMA_STOPP)
        ut = out["choices"][0]["text"].strip()
        return ut if len(ut) > 2 else text
    except Exception:
        return text


_NYHETSFRAGA = re.compile(
    r"senaste|nyheter|news|latest|vad hande|what happened|"
    r"uppdatering|update|hant idag|hande idag|breaking|"
    r"today|storsta nyhet|recent development|just nu i varlden",
    re.IGNORECASE,
)


# Regelbaserade/juridiska begrepp dar hallucination ar farligt: modellen
# kanner ofta orden men inte den aktuella siffran/regeln, och tenderar
# svara sjalvsakert med gammal/felaktig info. Utan Wikipedia/Wikidata-
# trafft ska vi hellre punkta anvandaren mot officiell kalla an att
# lata Llama chansa. Om nagon av dessa matchar OCH ingen verifierad
# fakta hittades - direktreturn med "kolla officiell kalla"-meddelande.
_KRAVER_KALLA = re.compile(
    r"\bvisum(krav)?\b|\besta\b|\bstyrranta(n)?\b|\bmoms(sats)?\b|"
    r"\bskattesats(en)?\b|\barvsskatt\b|\bgavoskatt\b|\bbolagsskatt\b|"
    r"\bfastighetsskatt\b|\brotavdrag\b|\brutavdrag\b|"
    r"\bfederal reserve rate\b|\binterest rate\b|\btax rate\b|"
    r"\bvisa requirement(s)?\b|\bentry requirement(s)?\b|"
    r"\bimmigration (policy|rule|requirement)\b|"
    # Utokad: lone- och jobbmarknadsfragor dar Llama tenderar hallucinera
    # sjalvsakert med foraldrade siffror. Regelbaserat med officiella
    # kallor (SCB, LO, Arbetsformedlingen) - hellre punkta anvandaren dit.
    r"\bgenomsnittslon(en)?\b|\bmedianlon(en)?\b|\bmedian salary\b|"
    r"\baverage salary\b|\bmedian wage\b|\bmedelinkomst(en)?\b|"
    r"\bhogst efterfragan\b|\bhighest paying\b|\bmost in-demand\b|"
    r"\bhiring the most\b|"
    # Aktiekurser/bolagsvarde: ingen finans-API, och webbsnuttar (t.ex.
    # kvartalsrapporter) ger sallan ett rent aktuellt borsvarde. Hellre
    # arligt "kolla en finanskalla" an ett foraldrat/gissat varde.
    r"\bmarket cap(italization)?\b|\bstock price\b|\bshare price\b|"
    r"\bnet worth\b|\bworth now\b|\bhow much is .{1,40} worth\b|"
    r"\baktiekurs(en)?\b|\bborsvarde(t)?\b|\bmarknadsvarde(t)?\b",
    re.IGNORECASE,
)


def _kraver_kalla(fraga_norm):
    return bool(_KRAVER_KALLA.search(fraga_norm))


def _ar_nyhetsfraga(fraga_norm):
    return bool(_NYHETSFRAGA.search(fraga_norm))


_RELATIV_ALDER = re.compile(
    r"(\d+)\s*(minut|hour|timm|dag|day|week|veck|month|manad)",
    re.IGNORECASE,
)
_ALDER_VIKT = {"minut": 1, "hour": 1, "timm": 1, "dag": 24, "day": 24,
               "week": 168, "veck": 168, "month": 720, "manad": 720}


def _kalla_aktualitet_timmar(text):
    """Grov uppskattning av kallans alder i timmar, baserat pa fraser
    som 'X hours ago', '2 dagar sedan', etc. i sokresultatets text.
    Anvands for att prioritera farskare kallor forst. Okand alder ->
    mellanhog rank (varken bast eller sist)."""
    match = _RELATIV_ALDER.search(text.lower())
    if not match:
        return 999999
    antal = int(match.group(1))
    enhet = match.group(2)
    for prefix, vikt in _ALDER_VIKT.items():
        if enhet.startswith(prefix[:4]):
            return antal * vikt
    return 999999


# A3: kallrankning - foredra Wikipedia/uppslagsverk/officiella sidor framfor
# roriga resultattavlor/aggregatorer (t.ex. sport-statistiksajter som gav fel
# svar for "Limhamn Griffins"). Lagre tal = hogre prioritet i sok-kontexten.
_KALLA_HOG = ("wikipedia.org", "wikidata.org", "britannica.com", ".gov",
              ".edu", "ne.se", "nationalencyklopedin")
_KALLA_LAG = ("sofascore.", "flashscore.", "transfermarkt.", "fotmob.",
              "besoccer.", "livescore.", "livesport.", "eloratings.",
              "worldfootball.", "fbref.", "soccerway.", "eurosport.",
              "national-football-teams.", "thesportsdb.", "scoreboard.",
              "sportstats.")


def _kalla_kvalitet_rank(url):
    """0 = hog (Wikipedia/officiellt), 2 = lag (resultattavla/aggregator),
    1 = ovrigt. Anvands som primar sorteringsnyckel for sok-traffar."""
    u = (url or "").lower()
    if any(d in u for d in _KALLA_HOG):
        return 0
    if any(d in u for d in _KALLA_LAG):
        return 2
    return 1


def _brave_sok(query, count=4, nyheter=False):
    """Anropar Brave Search API (web- eller news-endpoint). Returnerar en
    lista av dict {titel, body, url, datum}, eller None vid saknad nyckel,
    fel eller inga traffar. Ersatter DDGS-anropen."""
    if not _BRAVE_API_KEY:
        return None
    _cache_nyckel = (query.strip().lower(), count, nyheter)
    if _cache_nyckel in _BRAVE_CACHE:
        return _BRAVE_CACHE[_cache_nyckel]
    url = _BRAVE_NEWS_URL if nyheter else _BRAVE_WEB_URL
    params = {"q": query, "count": count}
    headers = {"Accept": "application/json",
               "Accept-Encoding": "gzip",
               "X-Subscription-Token": _BRAVE_API_KEY}
    for forsok in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=6)
            if resp.status_code == 429:  # rate limit - kort backoff, forsok igen
                time.sleep(1.0 * (forsok + 1))
                continue
            if resp.status_code != 200:
                _debug_logg(f"Brave HTTP {resp.status_code}: {resp.text[:200]}")
                return None
            data = resp.json()
            raa = (data.get("results", []) if nyheter
                   else data.get("web", {}).get("results", []))
            ut = []
            for r in raa[:count]:
                ut.append({
                    "titel": _HTML_TAGG.sub("", r.get("title", "")),
                    "body": _HTML_TAGG.sub("", r.get("description", "")),
                    "url": r.get("url", ""),
                    # page_age ar ISO-datum; age ar t.ex. "2 hours ago".
                    "datum": (r.get("page_age") or r.get("age") or "")[:10],
                })
            resultat = ut or None
            if resultat is not None:          # cacha bara lyckade traffar
                _BRAVE_CACHE[_cache_nyckel] = resultat
            return resultat
        except requests.exceptions.RequestException as e:
            _debug_logg(f"Brave natverksfel: {e}")
            time.sleep(0.3 * (2 ** forsok))
        except Exception as e:
            _debug_logg(f"Brave ovantat fel: {e}")
            return None
    return None


def sok_pa_natet(fraga, max_resultat=4, oversatt=True):
    if not _SOKNING_TILLGANGLIG:
        return None
    ar = datetime.date.today().year
    # oversatt=False: sok pa RATT ord (P1) - anvands nar vi medvetet vill
    # soka pa originalspraket sa entiteter/akronymer/egennamn (t.ex.
    # "tga", lokala namn) inte forsvinner i oversattningen.
    sok_fraga_att_anvanda = fraga
    if oversatt and detektera_sprak(fraga) == "sv":
        sok_fraga_att_anvanda = _oversatt_till_engelska(fraga)
        _debug_logg(f"oversatt sokfraga: {sok_fraga_att_anvanda!r}")

    # For nyhetsartade fragor ("senaste nyheterna om X"): Brave har ett
    # dedikerat news-endpoint med datum per artikel, battre traffar an
    # generisk websokning for just denna fragetyp.
    if _ar_nyhetsfraga(_normalisera_for_matchning(fraga)):
        nyheter = _brave_sok(sok_fraga_att_anvanda, count=max_resultat, nyheter=True)
        if nyheter:
            nyheter_sorterade = sorted(
                nyheter, key=lambda n: n.get("datum", ""), reverse=True)
            rader = [f"- [{n['datum']}] {n['titel']}: {n['body']} "
                     f"(Kalla: {n['url']})" for n in nyheter_sorterade]
            return "\n".join(rader)
        # annars: falla tillbaka pa vanlig websokning nedan

    sokfraga = f"{sok_fraga_att_anvanda} {ar}"
    traffar = _brave_sok(sokfraga, count=max_resultat, nyheter=False)
    if not traffar:
        # ROBUSTHET: forsta soket (med artal) gav inget - forsok igen UTAN
        # artal (tidlosa fragor far ofta fler/battre traffar sa) innan vi
        # ger upp. Sa slipper vi "hittade inget" pa fragor som egentligen
        # har svar.
        traffar = _brave_sok(sok_fraga_att_anvanda, count=max_resultat,
                             nyheter=False)
    if not traffar:
        return None
    # A3: sortera PRIMART pa kallkvalitet (Wikipedia/officiellt fore roriga
    # resultattavlor) och SEKUNDART pa farskhet ("X dagar/timmar sedan" i
    # snippet-texten). Sa hamnar en trovardig kalla forst i kontexten aven
    # nar en aggregator ligger hogre i Braves egen rankning.
    traffar_sorterade = sorted(
        traffar, key=lambda t: (_kalla_kvalitet_rank(t.get("url", "")),
                                _kalla_aktualitet_timmar(t.get("body", ""))))
    rader = [f"- {t['titel']}: {t['body']} (Kalla: {t['url']})"
             for t in traffar_sorterade]
    return "\n".join(rader)


def _sok_brave_parallellt(*fragor):
    """Kor flera Brave-sok SAMTIDIGT. Natverks-I/O slapper GIL:en, sa
    anropen overlappar och tva sok tar ~en rundtur istallet for tva. Slar
    ihop kontexten och bevarar inbordes ordning (forsta fragan forst).
    OBS: anvands bara med sok_pa_natet(oversatt=False), som INTE anropar
    GPU-modellen - trad-parallellism mot den enda llama-instansen vore
    annars osaker."""
    fragor = [f for f in fragor if f and f.strip()]
    if not fragor:
        return None
    with ThreadPoolExecutor(max_workers=len(fragor)) as ex:
        resultat = list(ex.map(
            lambda f: sok_pa_natet(f, oversatt=False), fragor))
    kontext = None
    for r in resultat:
        kontext = _slaihop_kontext(kontext, r)
    return kontext or None


def _forsta_uppslag_parallellt(fraga, sprak):
    """Kor de fyra oversta RENA HTTP-uppslagen samtidigt (ingen av dem
    anropar GPU-modellen) och returnerar den hogst prioriterade icke-tomma
    traffen. Bevarar exakt samma prioordning som den gamla sekventiella
    kedjan: direkt > utmarkelse > sport > software."""
    uppslag = (
        wikipedia_direktuppslag,
        wikipedia_utmarkelse_uppslag,
        wikipedia_sport_uppslag,
        wikidata_software_version_uppslag,
    )
    resultat = [None] * len(uppslag)
    with ThreadPoolExecutor(max_workers=len(uppslag)) as ex:
        framtid = {ex.submit(fn, fraga, sprak=sprak): i
                   for i, fn in enumerate(uppslag)}
        for fut in as_completed(framtid):
            i = framtid[fut]
            try:
                resultat[i] = fut.result()
            except Exception as e:
                _debug_logg(f"parallellt uppslag {i} fel: {e}")
    for r in resultat:            # prioordning bevarad via listindex
        if r:
            return r
    return None


# ============ PROMPT-BYGGARE ============
def bygg_system_prompt(sprak):
    return _str("system_prompt", sprak, datum=_DATUM_IDAG)


# Otvetydigt svenska ord (finns INTE som vanliga tyska/engelska ord och
# ar darfor sakra att lita pa). "du" ar med FLIT uteslutet: det ar aven
# tyska/danska och var precis det som forr fick tyska fragor att
# fellasas som svenska.
_UNIK_SVENSKA_ORD = {
    "vem", "vad", "vilka", "vilken", "vilket", "hur", "nar", "varfor",
    "jag", "och", "inte", "aven", "ocksa", "bara", "redan", "annu",
    "hej", "hejsan", "tack", "nasta", "senaste", "beratta", "ar", "har",
    "vart", "vara", "mitt", "mina", "ditt", "dina", "honom", "henne",
    "kanske", "mycket", "manga", "nagon", "nagot", "gor", "kung",
}


_SPRAK_LISTA = ("Swedish", "English", "German", "Dutch", "French",
                "Italian", "Spanish", "Portuguese", "Polish", "Finnish",
                "Danish", "Norwegian", "Czech", "Greek")

# Few-shot-instruktion for LLM-baserad sprakidentifiering. Llama 3.2 3B
# (redan laddad) ar langt mer palitlig an ordlistor - aven for korta,
# felstavade fragor och foljdfragor. Exemplen tacker de klassiska
# fallgroparna: engelsk fraga som NAMNER ett land/spel ("sweden",
# "planet of lana"), svensk fraga som namner ett engelskt spelnamn, och
# "wie" (tyska: hur / nederlandska: vem). Regel: bedom spraket pa ORDEN,
# inte pa vad texten namner.
_LID_SYS = (
    "You identify the LANGUAGE that a text is written in. Reply with ONE "
    "word only, chosen from exactly this list: Swedish, English, German, "
    "Dutch, French, Italian, Spanish, Portuguese, Polish, Finnish, Danish, "
    "Norwegian, Czech, Greek. Judge by the language of "
    "the WORDS themselves - NOT by any country, city, person, company or "
    "product that the text happens to mention. If the language is not one "
    "of those, or you are unsure, reply English.\n\n"
    "Text: what is planet of lana\nLanguage: English\n\n"
    "Text: its game developed by wishfully\nLanguage: English\n\n"
    "Text: when was sweden knocked out of the world cup\nLanguage: English\n\n"
    "Text: who is the ceo of google right now\nLanguage: English\n\n"
    "Text: vem ar statsminister i sverige just nu\nLanguage: Swedish\n\n"
    "Text: vilket lag vann senaste vm i fotboll\nLanguage: Swedish\n\n"
    "Text: vem malade mona lisa\nLanguage: Swedish\n\n"
    "Text: vem uppfann telefonen\nLanguage: Swedish\n\n"
    "Text: vad kostar apple aktien\nLanguage: Swedish\n\n"
    "Text: hur ar vadret i london\nLanguage: Swedish\n\n"
    "Text: vad handlar spelet the last of us om\nLanguage: Swedish\n\n"
    "Text: Wer ist aktuell Bundeskanzler von Deutschland?\nLanguage: German\n\n"
    "Text: Wie is nu de premier van Nederland?\nLanguage: Dutch\n\n"
    "Text: Quelle est la capitale de la France?\nLanguage: French\n\n"
    "Text: Qual e l'ultima versione di Python?\nLanguage: Italian\n\n"
    "Text: Cuando sale GTA 6?\nLanguage: Spanish\n\n"
    "Text: Quem ganhou o Nobel da literatura?\nLanguage: Portuguese\n\n"
    "Text: Kto jest obecnie prezydentem Polski?\nLanguage: Polish\n\n"
    "Text: Mika on Suomen paakaupunki?\nLanguage: Finnish\n\n"
    "Text: Hvem er statsminister i Danmark nu?\nLanguage: Danish\n\n"
    "Text: Hvem er statsminister i Norge?\nLanguage: Norwegian\n\n"
    "Text: Kdo je prezidentem Ceske republiky?\nLanguage: Czech\n\n"
    "Text: Poia einai i protevousa tis Elladas?\nLanguage: Greek")


# ---- Sprakdetektor via lingua: deterministisk, tacker alla vanliga
# EU-sprak (inkl. polska, norska, danska, finska, tjeckiska ...) och ar
# palitlig aven pa KORTA texter och pa att skilja nordiska sprak at.
# Byggs en gang (lazy). Fixar drift dar tidigare 8-sprakslistan tvingade
# t.ex. polska/norska till engelska.
_LINGUA_DETEKTOR = None
_LINGUA_TILL_NAMN = {
    "ENGLISH": "English", "SWEDISH": "Swedish", "GERMAN": "German",
    "DUTCH": "Dutch", "FRENCH": "French", "ITALIAN": "Italian",
    "SPANISH": "Spanish", "PORTUGUESE": "Portuguese", "POLISH": "Polish",
    "DANISH": "Danish", "FINNISH": "Finnish", "CZECH": "Czech",
    "GREEK": "Greek", "HUNGARIAN": "Hungarian", "ROMANIAN": "Romanian",
    "BOKMAL": "Norwegian", "NYNORSK": "Norwegian",
}


def _lingua_detektor():
    global _LINGUA_DETEKTOR
    if _LINGUA_DETEKTOR is not None:
        return _LINGUA_DETEKTOR
    try:
        from lingua import Language, LanguageDetectorBuilder
        sprak = [Language.ENGLISH, Language.SWEDISH, Language.GERMAN,
                 Language.DUTCH, Language.FRENCH, Language.ITALIAN,
                 Language.SPANISH, Language.PORTUGUESE, Language.POLISH,
                 Language.DANISH, Language.FINNISH, Language.CZECH,
                 Language.GREEK, Language.HUNGARIAN, Language.ROMANIAN,
                 Language.BOKMAL]
        _LINGUA_DETEKTOR = (LanguageDetectorBuilder
                            .from_languages(*sprak).build())
    except Exception:
        _LINGUA_DETEKTOR = False   # otillganglig -> falla tillbaka pa LLM
    return _LINGUA_DETEKTOR


def detektera_svarssprak_namn(fraga):
    """Identifiera anvandarens sprak (for att spegla svaret tillbaka pa
    SAMMA sprak) med Llama 3.2 3B istallet for ordlistor. Modellen ar
    redan laddad, sa det kostar bara ~0.1 s. Returnerar ett engelskt
    spraknamn ur _SPRAK_LISTA; faller tillbaka till engelska vid tvivel,
    fel, eller om modellen annu inte ar laddad. (Tidigare ordliste-
    heuristik fellaste standigt korta/foljdfragor - t.ex. engelska
    'its game developed by wishfully' blev okant -> fel svarssprak.)"""
    return _dsn_inner(fraga)


# GREPP 3: lingua forvaxlar nara sprakpar pa KORTA fragor - portugisiska ->
# spanska, svenska -> danska (200-testet: pt-fraga blev es-svar, sv-fraga blev
# da-svar). Hogprecisions-markorord korrigerar lingua-gissningen.
_PT_MARKORER = re.compile(r"\bo que\b|não|você|voce|obrigad|então|entao|"
                          r"ção|ções|ões|\bàs\b|\bmarés\b|\bé\b|\bvocê\b", re.I)
_ES_MARKORER = re.compile(r"¿|¡|\bqué\b|\bcómo\b|\bcuál\b|\bestá\b|\busted\b|"
                          r"\bñ|\bhola\b|\bespaña\b", re.I)
_SV_MARKORER = re.compile(r"\bvad\b|\bär\b|\bjag\b|\boch\b|\bfrån\b|\bhär\b|"
                          r"\bdär\b|\binte\b|\bvilken\b|\bvilket\b|\bhur\b|ä", re.I)
_DA_MARKORER = re.compile(r"\bhvad\b|\bhvilken\b|\bhvor\b|\bjeg\b|\bikke\b|"
                          r"\bog\b|\bder er\b", re.I)


def _diskriminera_narsprak(namn, text):
    t = " " + (text or "").lower() + " "
    if namn == "Spanish" and _PT_MARKORER.search(t):
        return "Portuguese"
    if namn == "Portuguese" and _ES_MARKORER.search(t):
        return "Spanish"
    if namn == "Danish" and _SV_MARKORER.search(t) and not _DA_MARKORER.search(t):
        return "Swedish"
    if namn == "Norwegian" and _SV_MARKORER.search(t) and not _DA_MARKORER.search(t):
        return "Swedish"
    if namn == "Swedish" and _DA_MARKORER.search(t):
        return "Danish"
    return namn


def _dsn_inner(fraga):
    text = (fraga or "").strip()
    if len(text) < 2:
        return "English"
    # Enda deterministiska genvagen (INTE en ordlista): bokstaven 'a med
    # ring' anvands bara av svenskan bland vara sprak - ett sakert,
    # aldrig felande svensk-skydd for anvandaren, sparar dessutom ett
    # modellanrop for de flesta svenska fragor.
    if "\u00e5" in text.lower():
        return "Swedish"
    # PRIMART: lingua (deterministisk, ~16 EU-sprak). Skiljer aven
    # polska/norska/danska/finska korrekt - tidigare fastnade de pa
    # engelska. Faller tillbaka pa LLM-klassificeraren om lingua saknas.
    det = _lingua_detektor()
    if det:
        try:
            spr = det.detect_language_of(text)
            if spr is not None:
                namn = _LINGUA_TILL_NAMN.get(spr.name)
                if namn:
                    return _diskriminera_narsprak(namn, text)
        except Exception:
            pass
    try:
        prompt = bygg_qwen_prompt(_LID_SYS, "Text: " + text + "\nLanguage:")
        out = llm(prompt, max_tokens=5, temperature=0.0,
                  stop=_LLAMA_STOPP + ["\n", "Text:", "Language:"])
        rad = out["choices"][0]["text"].strip()
        traff = re.search(r"[A-Za-z]+", rad)
        if traff:
            namn = traff.group(0).capitalize()
            if namn in _SPRAK_LISTA:
                return namn
    except Exception:
        pass
    return "English"


def _sprak_direktiv(svarssprak_namn):
    """Stark, konkret instruktion (placeras SIST i systemprompten =
    starkaste positionen) som tvingar svaret till anvandarens eget
    sprak. Ersatter den tidigare vaga 'svara pa samma sprak'-varianten
    som en 3B-modell inte foljde tillrackligt palitligt."""
    if not svarssprak_namn:
        # Okant sprak (heuristiken kande inte igen det) - tvinga inte
        # fram ett felaktigt spraknamn, be modellen spegla anvandarens
        # egna sprak istallet.
        return (
            "\n\nCRITICAL LANGUAGE RULE: Write your ENTIRE reply in the "
            "EXACT same language as the user's latest message - regardless "
            "of the language of any facts, search results, system notes, "
            "or earlier turns above."
        )
    return (
        f"\n\nCRITICAL LANGUAGE RULE: The user's latest message is written "
        f"in {svarssprak_namn}. Write your ENTIRE reply in {svarssprak_namn} "
        f"and in NO other language - regardless of the language of any "
        f"facts, search results, system notes, or earlier turns above."
    )


_STOPP = _LLAMA_STOPP + [
    "\nFraga:", "\nFråga:", "\nQuestion:",
    "\n\nFraga:", "\n\nFråga:", "\n\nQuestion:",
    "\nSvar:", "\nAnswer:", "\nA:", "\nQ:",
    "\nUser:", "\nAssistant:", "\nHuman:", "\nAI:",
    "(Swedish", "(swedish", "(English", "(english",
    "(Note", "(note", "(Translation", "(translation",
]


# A2: tak for slutsvarets langd. Hojt fran 700 -> 1400 sa langa svar
# (syntes, planer, pedagogiska forklaringar, rad) inte huggs av mitt i en
# mening. Extraktion/verifiering har egna, lagre tak och rors inte.
_MAX_SVAR_TOKENS = 1400


def _polera_avslut(text):
    # A2: om ett LANGT svar rakade bli avhugget (nadde token-taket) -
    # stada avslutet. Ror ALDRIG korta svar. (1) strippa en dinglande
    # listmarkor pa sista raden (ren trunkerings-skrap, t.ex. "\n* " eller
    # "\n   2."); (2) om avslutet anda inte ar en hel mening -> trimma
    # tillbaka till sista hela meningen, men bara om >=70% behalls.
    try:
        t = (text or "").rstrip()
        if len(t) < 1200:
            return text
        t = re.sub(r"\n[ \t]*(?:[*\-•]|\d+[.)])[ \t]*$", "", t).rstrip()
        _karna = t.rstrip("*`_ ")  # bortse fran avslutande markdown-emfas
        if _karna.endswith((".", "!", "?", "…", "\"", "'", "”", "’",
                            ")", "]", "}")):
            return t
        traffar = list(re.finditer(r"[.!?…](?:[\"'”’)\]]+)?(?:\s|$)", t))
        if traffar:
            kvar = t[:traffar[-1].end()].rstrip()
            if len(kvar) >= 0.7 * len(t):
                return kvar
        return t
    except Exception:
        return text


def _generera(prompt, temperature, repeat_penalty, token_cb=None):
    # A2: adaptivt tak. Vill ge langa svar plats (upp till _MAX_SVAR_TOKENS)
    # UTAN att spranga kontextfonstret nar prompten redan ar stor (t.ex.
    # inhamtad fakta) - da krymps completion sa prompt+svar ryms i n_ctx.
    _mx = _MAX_SVAR_TOKENS
    try:
        try:
            _n_ctx = int(llm.n_ctx())
        except Exception:
            _n_ctx = 4096
        try:
            _p_tok = len(llm.tokenize(prompt.encode("utf-8", "ignore")))
        except Exception:
            _p_tok = int(len(prompt) / 3.2)  # konservativ teckenuppskattning
        _rum = _n_ctx - _p_tok - 48
        if _rum < _mx:
            _mx = max(200, _rum)
    except Exception:
        _mx = 700  # sista utvag: gamla sakra taket
    if token_cb is None:
        out = llm(
            prompt,
            max_tokens=_mx,
            stop=_STOPP,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            top_p=0.9,
        )
        return _polera_avslut(_strip_think(out["choices"][0]["text"]).strip())
    # STROMMANDE: mata ut text live till UI:t medan modellen genererar, sa
    # svaret "skrivs ut" ord for ord istallet for att dyka upp allt pa en
    # gang. token_cb far den ACKUMULERADE texten hittills. Guards kors anda
    # pa den FARDIGA texten efterat - om ett svar underkanns byter UI:t ut
    # det streamade mot det slutliga (sker sallan).
    delar = []
    try:
        for chunk in llm(
                prompt, max_tokens=_mx, stop=_STOPP,
                temperature=temperature,
                repeat_penalty=repeat_penalty, top_p=0.9, stream=True):
            bit = chunk["choices"][0].get("text", "")
            if bit:
                delar.append(bit)
                try:
                    token_cb(_strip_think("".join(delar).lstrip()))
                except Exception:
                    pass
    except Exception:
        if not delar:  # streaming small - falla tillbaka pa vanligt anrop
            out = llm(prompt, max_tokens=_mx, stop=_STOPP,
                      temperature=temperature, repeat_penalty=repeat_penalty,
                      top_p=0.9)
            return _polera_avslut(_strip_think(out["choices"][0]["text"]).strip())
    return _polera_avslut(_strip_think("".join(delar)).strip())


# ============ FAKTA-EXTRAKTION ============
def _extrahera_fakta_engang(fraga, sok_kontext, temperature=0.0):
    # VIKTIGT: prompten ar pa ENGELSKA och faktan extraheras pa ENGELSKA.
    # Hela retrieval-pipelinen kors internt pa engelska (sok, kallor,
    # wiki_fakta ar redan engelska); en svensk extraktionsprompt fick
    # modellen att blanda sprak ("En young girl ...") och slutsvaret att
    # flippa till fel sprak. Slutsvaret speglas anda till anvandarens
    # sprak via _sprak_direktiv, sa en engelsk fakta ar ratt val.
    prompt = bygg_llama_prompt(
        "You are precise and methodical.",
        "Here are web search results (with source):\n\n" + sok_kontext +
        f"\n\nQuestion: {fraga}\n\n"
        "STEP 1 - RELEVANCE CHECK: first check whether the sources are "
        "actually about the TOPIC, COUNTRY and CONTEXT the question is "
        "about (e.g. if the question is about the US Senate but the "
        "sources are about a Swedish election, they are NOT relevant just "
        "because both involve a 'vote'). If the sources are about the "
        "WRONG country/topic/context: write ONLY 'ANSWER: Could not find "
        "relevant information about this.' and stop there.\n\n"
        "STEP 2 - IF THE SOURCES ARE RELEVANT: when a source has a title "
        "like \"President X attends...\" or \"President X gives a "
        "speech...\", the word \"President\" before the name means that "
        "person IS the current president (or the corresponding role/title "
        "the question is about) - it is not a guess, that is how news "
        "headlines work. The same logic applies to other roles (e.g. "
        "\"Prime Minister X\", \"CEO X\").\n\n"
        "Look in the relevant sources for the pattern \"[Role] [Name]\" "
        "matching the question. Ignore sources about OTHER roles (e.g. "
        "vice president when the question is about the president). Also "
        "ignore headlines about future elections written in the future "
        "tense (\"will be\").\n\n"
        "NOTE: the question does NOT have to be about a role/title. For "
        "any other factual question (e.g. draft pick, winner, inventor, "
        "number, percentage, date, record, biggest/highest, or what "
        "something is) - extract the DIRECT answer (a name, number, date "
        "or short fact) that the sources actually support. Only if the "
        "sources really do not answer the question, write 'Could not find "
        "relevant information about this.'\n\n"
        "Write one line starting with 'ANSWER:' followed by ONE short "
        "sentence, in ENGLISH, that answers the question."
    )
    out = llm(prompt, max_tokens=200, temperature=temperature, stop=_LLAMA_STOPP)
    text = out["choices"][0]["text"].strip()
    for markor in ("ANSWER:", "SVAR:"):
        if markor in text:
            return text.split(markor)[-1].strip()
    return text


_SISTA_NAMN = re.compile(r"([A-ZÅÄÖ][a-zåäö]+(?:\s+[A-ZÅÄÖ][a-zåäö]+)*)\.?\s*$")


def _extrahera_namn_ur_text(text):
    """Plockar ut det sista namnet (proper noun) i en text - anvands for
    att jamfora om flera extraktions-forsok kom fram till samma person/
    sak, aven om formuleringen runtomkring skiljer sig nagot."""
    match = _SISTA_NAMN.search(text.strip())
    return match.group(1).strip().lower() if match else None


def extrahera_fakta(fraga, sok_kontext):
    """Kor extraktionen 3 ganger (sjalvkonsistens/ensemble) och kraver
    att minst 2 av 3 forsok kommer fram till samma namn/svar innan vi
    litar pa det - fangar upp fall dar en enskild korning rakar tolka
    sokresultaten fel, aven om den anvander laga temperaturer. Kostar
    extra tid men bara i sok-grenen, paverkar aldrig icke-sokfragor."""
    # A6-latens: kor forst TVA forsok. Ar de redan eniga (samma proper
    # noun ELLER samma normaliserade text, och inte oklara) litar vi pa
    # det direkt och hoppar over den tredje ~200-token-genereringen. Bara
    # vid oenighet kors ett tredje forsok som tie-breaker - da galler exakt
    # samma >=2-rostning som forr (namn -> exakt -> fuzzy -> ensam-grundad).
    _a = _extrahera_fakta_engang(fraga, sok_kontext, temperature=0.0)
    # SNABBVAG (latens): om det forsta deterministiska forsoket ar tydligt
    # OCH grundat i kallorna litar vi pa det direkt - hoppar over bade det
    # andra och tredje extraktionsanropet. Sparar ~1-2 tunga genereringar
    # per sokfraga. Bara oklara/ogrundade forsta-svar gar vidare till hela
    # ensemblen (sjalvkonsistens) precis som forr, sa robustheten behalls
    # dar den behovs.
    if not _fakta_ar_oklar(_a) and _svar_ar_grundat(_a, sok_kontext):
        _debug_logg(f"extrahera_fakta snabbvag (grundat 1/1)={_a!r}")
        return _a
    _b = _extrahera_fakta_engang(fraga, sok_kontext, temperature=0.3)
    _na, _nb = _extrahera_namn_ur_text(_a), _extrahera_namn_ur_text(_b)
    if (_na is not None and _na == _nb) or (
            _normalisera_for_matchning(_a) == _normalisera_for_matchning(_b)
            and not _fakta_ar_oklar(_a)):
        _debug_logg(f"extrahera_fakta tidig enighet (2/2)={_a!r}")
        return _a
    forsok = [_a, _b,
              _extrahera_fakta_engang(fraga, sok_kontext, temperature=0.3)]
    _debug_logg(f"extrahera_fakta ensemble-forsok={forsok!r}")
    namn_per_forsok = [_extrahera_namn_ur_text(f) for f in forsok]
    for i, namn in enumerate(namn_per_forsok):
        if namn is None:
            continue
        antal_matchande = sum(1 for n in namn_per_forsok if n == namn)
        if antal_matchande >= 2:
            return forsok[i]
    # NYTT: for svar UTAN proper noun (siffror, procent, datum, korta
    # fakta som "3.5%") - om minst 2 av 3 forsok ar (nastan) identiska
    # och inte oklara, lita pa det. Namn-rostningen ovan slang annars
    # sadana svar trots att modellen var helt enig.
    norm = [_normalisera_for_matchning(f) for f in forsok]
    for i in range(len(forsok)):
        if _fakta_ar_oklar(forsok[i]):
            continue
        if sum(1 for x in norm if x == norm[i]) >= 2:
            return forsok[i]
    # NYTT: fuzzy-samstammighet. Beskrivande svar formuleras ofta olika
    # mellan forsoken ("Planet of Lana is a puzzle adventure game" vs
    # "...is a colourful side-scrolling puzzle adventure game created by
    # Wishfully") - da matchar varken namn- eller exakt-rostningen, trots
    # att forsoken tydligt SAGER SAMMA SAK. Om >=2 icke-oklara forsok
    # delar tillrackligt manga innehallsord (Jaccard >= 0.5) litar vi pa
    # dem och valjer det LANGSTA (mest kompletta) svaret. Genuint olika
    # svar har lag overlappning och faller fortfarande igenom till oklart.
    konfident = [f for f in forsok if not _fakta_ar_oklar(f)]
    ordmangder = [set(_normalisera_for_matchning(f).split()) for f in konfident]
    for i in range(len(konfident)):
        if len(ordmangder[i]) < 3:
            continue
        klunga = [konfident[i]]
        for j in range(len(konfident)):
            if j == i or not ordmangder[j]:
                continue
            union = ordmangder[i] | ordmangder[j]
            jaccard = len(ordmangder[i] & ordmangder[j]) / len(union)
            if jaccard >= 0.35:
                klunga.append(konfident[j])
        if len(klunga) >= 2:
            return max(klunga, key=len)
    # P3/P4: exakt ETT sjalvsakert forsok (de ovriga hedgade "not
    # mentioned" - de MOTSADE inte, de hittade bara inte svaret). Lita pa
    # det ENDA om det ar GRUNDAT i kallorna (dess egennamn finns dar) -
    # grundningskollen hindrar ett ensamt hallucinerat forsok fran att
    # slinka igenom. Loser t.ex. "San Carlos High School" dar 2 av 3
    # forsok bara sa "not mentioned".
    if len(konfident) == 1 and _svar_ar_grundat(konfident[0], sok_kontext):
        return konfident[0]
    # Ingen majoritet - bygg ett tydligt "oklart"-svar sa den befintliga
    # oklar-hanteringen tar over (battre an att gissa ett av de tre).
    return ("ANSWER: The sources gave different answers across attempts; "
            "unclear which one is correct.")


_OKLAR_FRASER = [
    "inte namnt", "ikke namnt", "hittar inte", "hittas inte",
    "kunde inte hitta", "inte relevant",
    "inget svar", "inte tydligt", "oklart", "kan inte avgora",
    "inte specificerat", "ingen tydlig", "framgar inte", "inte klart",
    "not mentioned", "not found", "unclear", "cannot determine",
    "could not find", "couldn't find", "not relevant", "no answer",
    "different answers", "no relevant", "cannot find", "can't find",
]


def _fakta_ar_oklar(fakta):
    fakta_norm = _normalisera_for_matchning(fakta)
    return any(fras in fakta_norm for fras in _OKLAR_FRASER)


# P5: list-fragor ("vilka X finns i Y", "what X are there") behover FLERA
# poster, inte ett enda faktum. Konservativ regex - hellre missa an
# felklassa en enkel faktafraga.
_LISTFRAGA = re.compile(
    r"(are there\b|are ther\b|are in\b|finns det\b|finns i\b|"
    r"list of\b|list all\b|name all\b|name the\b|namnge\b|lista upp\b|"
    r"\bvilka\b|\bwhich\b.{0,50}\bare\b|"
    r"\bwhat\b.{0,60}\b(high ?schools?|schools?|restaurants?|hospitals?|"
    r"teams?|clubs?|companies|universit(y|ies)|colleges?|bars?|hotels?|"
    r"museums?|shops?|stores?|parks?)\b|"
    r"\bwhat\b.{0,50}\bare (there|ther|in)\b)",
    re.IGNORECASE)


def _ar_listfraga(fraga_norm):
    return bool(_LISTFRAGA.search(fraga_norm))


def extrahera_lista(fraga, sok_kontext):
    """Extraherar en LISTA av poster (namn) ur sokresultaten for list-
    fragor. Engelsk prompt (intern pipeline). Returnerar en kort
    kommaseparerad mening, eller en oklar-markering om inget hittas."""
    prompt = bygg_llama_prompt(
        "You are precise and methodical.",
        "Here are web search results (with source):\n\n" + sok_kontext +
        f"\n\nQuestion: {fraga}\n\n"
        "List the specific items (names) from the sources that answer the "
        "question. Use ONLY items actually present in the sources - do NOT "
        "invent or add any from your own memory. Include ONLY items that "
        "match the SUBJECT and the COUNTRY/CONTEXT of the question; if the "
        "sources also contain items from a DIFFERENT country, topic or "
        "event (e.g. British parties when the question is about Sweden), "
        "EXCLUDE those. If the sources do not "
        "list any, reply 'Could not find relevant information about "
        "this.'\n\nWrite one line starting with 'ANSWER:' followed by the "
        "items as a short comma-separated sentence, in ENGLISH."
    )
    out = llm(prompt, max_tokens=220, temperature=0.0, stop=_LLAMA_STOPP)
    text = out["choices"][0]["text"].strip()
    for markor in ("ANSWER:", "SVAR:"):
        if markor in text:
            text = text.split(markor)[-1].strip()
            break
    if _fakta_ar_oklar(text):
        return text
    # GRUNDNING: behall bara poster vars ord faktiskt finns i kallorna -
    # modellen tenderar annars att FYLLA PA listan ur eget minne (t.ex.
    # "Carlmont, Tide Academy" som inte fanns i sokresultaten). Om inget
    # ar grundat -> arligt "hittade inte".
    kontext_low = (sok_kontext or "").lower()
    poster = [p.strip(" .") for p in re.split(r"[;,\n]|\band\b|\boch\b", text)
              if p.strip(" .")]
    grundade = []
    for p in poster:
        nyckelord = re.findall(r"[a-zA-Z0-9åäö]{4,}", p.lower())
        if nyckelord and any(o in kontext_low for o in nyckelord):
            grundade.append(p)
    if not grundade:
        return "Could not find relevant information about this."
    return ", ".join(grundade)


def _extrahera_ja_nej_engang(fraga, sok_kontext, temperature=0.0):
    """Specialiserad analys for ja/nej-fragor ('Har X hant Y an?').
    Tvingar modellen att uttryckligen resonera JA/NEJ/OKLART utifran
    bevisen i sokresultaten, istallet for att fritt formulera ett svar
    - vilket visat sig kunna leda till fel slutsats aven nar bevisen
    faktiskt racker for att svara ratt (t.ex. missar att en lista med
    bara obemannade flygningar betyder NEJ pa en fraga om bemannad
    flygning)."""
    # VIKTIGT: prompten och beslutet ar pa ENGELSKA - precis som resten av
    # retrieval-/extraktionspipelinen (sok, kallor, extrahera_fakta kors
    # internt pa engelska). Slutsvaret speglas sedan till anvandarens sprak
    # via _sprak_direktiv/_oversatt_fran_engelska precis som ovrig fakta. En
    # SVENSK ja/nej-analys gav ett svenskt fakta-block ("Svaret ar NEJ...")
    # som den lilla 3B-modellen ekade rakt av - sa en engelsk fraga fick ett
    # svenskt svar. Engelsk analys loser det (samma vag som all annan fakta).
    prompt = bygg_llama_prompt(
        "You are precise and methodical. You NEVER draw a conclusion the "
        "search results do not explicitly support.",
        "Here are web search results:\n\n" + sok_kontext +
        f"\n\nYes/no question: {fraga}\n\n"
        "Analyze CAREFULLY what the sources actually say (e.g. if they only "
        "mention ATTEMPTS/TESTS without a crew, that means NO to a question "
        "about something WITH a crew - do not be fooled by similar but not "
        "identical claims). Then write ONE line that starts with exactly "
        "'ANSWER: YES', 'ANSWER: NO' or 'ANSWER: UNCLEAR', followed by a "
        "short justification on the same line, in English."
    )
    out = llm(prompt, max_tokens=150, temperature=temperature, stop=_LLAMA_STOPP)
    text = out["choices"][0]["text"].strip()
    _debug_logg(f"extrahera_ja_nej RAW={text!r}")
    # Tala mot variation i formatet (markdown-asterisker, saknad etikett,
    # och aven aldre svenska SVAR:/JA/NEJ ifall modellen faller tillbaka pa
    # svenska) - forst med etikett, annars om raden borjar direkt med beslutet.
    _ordval = r"(YES|NO|UNCLEAR|JA|NEJ|OKLART)"
    match = re.search(r"(?:ANSWER|SVAR)[:\s]*\**\s*" + _ordval + r"\b[.:,]?\s*(.*)",
                      text, re.IGNORECASE)
    if not match:
        match = re.match(r"\**\s*" + _ordval + r"\b[.:,]?\s*(.*)",
                         text.strip(), re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).upper()
    motivering = match.group(2).strip()
    beslut = {"JA": "YES", "NEJ": "NO", "OKLART": "UNCLEAR"}.get(raw, raw)
    if beslut == "UNCLEAR":
        return None
    return beslut, motivering


def extrahera_ja_nej(fraga, sok_kontext):
    """Kor ja/nej-analysen 3 ganger (sjalvkonsistens/ensemble) och kraver
    att minst 2 av 3 forsok kommer fram till samma beslut (JA/NEJ/
    OKLART) innan vi litar pa det - fangar upp fall dar en enskild
    korning drar fel slutsats av misstag, aven med laga temperaturer."""
    forsok = [_extrahera_ja_nej_engang(fraga, sok_kontext, temperature=t)
              for t in (0.0, 0.3, 0.3)]
    _debug_logg(f"extrahera_ja_nej ensemble-forsok={forsok!r}")
    beslut_lista = [f[0] if f else "UNCLEAR" for f in forsok]
    for beslut in ("YES", "NO"):
        if beslut_lista.count(beslut) >= 2:
            # Anvand motiveringen fran forsta forsoket som gav detta beslut
            for f in forsok:
                if f and f[0] == beslut:
                    return f
    return None


_VANLIGA_STAVFEL = [
    (re.compile(r"\bstadsminister(n|arna)?\b", re.IGNORECASE), "statsminister"),
]


def _ratta_vanliga_stavfel(fraga):
    """Rattar nagra latta men vanliga stavfel (t.ex. 'stadsminister' -
    stad/city - istallet for 'statsminister' - stat/regeringschef) sa
    varken klassificering, sokning eller extraktion blir vilseledd."""
    for monster, ersattning in _VANLIGA_STAVFEL:
        fraga = monster.sub(ersattning, fraga)
    return fraga


def _fakta_matchar_svar(fakta, svar):
    """Kollar att modellens svar faktiskt namner samma sak som den
    givna faktan sager - fangar fallet dar modellen INTE hedgar
    (sager 'vet ej') utan istallet lugnt pastar nagot som direkt
    MOTSAGER faktan (t.ex. fakta sager 'Trump', svaret sager 'Biden')."""
    if not fakta:
        return True
    match = re.search(r"\b(?:is|ar)\s+([^.]+)\.?\s*$",
                      _normalisera_for_matchning(fakta), re.IGNORECASE)
    if not match:
        return True
    namn_ord = [o for o in match.group(1).strip().split() if len(o) > 2]
    if not namn_ord:
        return True
    sista_ordet = namn_ord[-1]
    return sista_ordet in _normalisera_for_matchning(svar)


_NAMN_KANDIDAT = re.compile(
    r"\b[A-ZÅÄÖ][a-zåäö]+(?:\s+[A-ZÅÄÖ][a-zåäö]+)+\b")


# Fix 2: skydda arsangivelser fran Llama-hallucination. Wikidata levererar
# ofta strukturerad fakta med specifikt ar (t.ex. "Vinnaren senast
# (ar 2025) ar László Krasznahorkai."), men Llama tenderar att grada in
# ett annat ar nar den omformulerar - t.ex. sager "2026" for att det ar
# dagens ar. Kollar att inget ar i svaret som saknas i faktan gar igenom.
_AR_MONSTER = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _fakta_ar_bevarat(fakta, svar):
    """Om faktan namner ett specifikt ar (som Wikidata skickar med for
    utmarkelser), sa MASTE svaret antingen anvanda samma ar eller inte
    namna nagot ar alls. Om svaret namner ett annat ar - trasigt."""
    if not fakta or not svar:
        return True
    fakta_ar = set(_AR_MONSTER.findall(fakta))
    if not fakta_ar:
        return True  # Faktan har inget arsangivelse - inget att skydda
    svar_ar = set(_AR_MONSTER.findall(svar))
    if not svar_ar:
        return True  # Svaret namner inget ar - ok
    # Varje ar i svaret maste ocksa finnas i faktan
    return svar_ar.issubset(fakta_ar)


# Fix 1b: en hedge-krav-check. Nar fragan kraver aktuell info men INGEN
# fakta hittades (Wikipedia/Wikidata missade, sok gav inget) - systemet
# ber redan Llama vara arlig. Men Llama ignorerar ofta detta pa svenska
# och svarar sjalvsakert med hallucinerad info istallet. Kravet: svaret
# maste innehalla nagon hedge-fras. Annars ny generering, sen fallback.
def _svar_har_hedge(svar):
    if not svar:
        return False
    svar_lower = svar.lower()
    return any(f in svar_lower for f in _HEDGE_FRASER)


def _svar_ar_grundat(svar, sok_kontext):
    """Kollar att namn/proper-nouns modellen namner i sitt svar faktiskt
    forekommer nagonstans i sokresultaten den fick - fangar generella
    hallucinationer dar modellen namner ngt (t.ex. en pristagare) som
    inte alls finns i kallorna, aven for 'vem/vad'-fragor (inte bara
    ja/nej). Om inget namn hittas i svaret, eller ingen sok_kontext
    finns att jamfora mot, antar vi att det ar okej (kan inte
    verifiera -> stoppar inte upp ett troligen ofarligt svar)."""
    if not sok_kontext:
        return True
    kandidater = _NAMN_KANDIDAT.findall(svar)
    if not kandidater:
        return True
    kontext_lower = sok_kontext.lower()
    for namn in kandidater:
        if namn.lower() not in kontext_lower:
            return False
    return True


def _debug_logg(text):
    try:
        with open(os.path.join(BASKATALOG, "debug.log"), "a",
                  encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now()}: {text}\n")
    except Exception:
        pass


# ============ LOKALT-FORST MED SOK-ESKALERING ============
# Basmodellen svarar forst ur eget minne (snabbt, offline). Om den INTE
# kan svara palitligt eskalerar vi till webbsok/Wikipedia - sa appen kan
# svara pa saker Llama inte kan (t.ex. ett specifikt indiespel), utan att
# gora enkla kanda fragor langsamma.
_SENTINEL_SOK = "SEARCH_NEEDED"

# Laggs till i systemprompten vid det LOKALA forsoket: uppmuntrar arlighet
# (punta hellre an hallucinera). Modellen behover inte alltid traffa
# token exakt - _svar_ar_ickesvar fangar aven vanliga "vet inte"-svar.
_LOKAL_SENTINEL_INSTR = (
    "\n\nIMPORTANT - only answer from facts you are genuinely confident "
    "about. If the question is about a specific named thing (a game, app, "
    "product, movie, book, song, company, person, place or event) that you "
    "do NOT reliably know, do NOT guess or make something up. In that case "
    "reply with EXACTLY this single token and nothing else: " + _SENTINEL_SOK)


def _ar_sentinel_text(text):
    """True om den ACKUMULERADE streamade texten ar sentinel-token
    (SEARCH_NEEDED) - eller annu en prefix som fortfarande kan vaxa till
    den. Sadan text ar en INTERN sok-signal, inte ett svar, och far aldrig
    visas live i UI:t (annars blinkar 'SEARCH_NEEDED' till pa skarmen innan
    vi eskalerar till webbsok)."""
    t = (text or "").strip().upper()
    s = _SENTINEL_SOK.upper()
    return s.startswith(t) or s in t


# Few-shot: lat Llama sjalv avgora om ett svar faktiskt besvarar fragan
# eller ar ett "vet inte". Sprakoberoende (funkar for sv/en/de/fr/it/...).
_ICKESVAR_PROMPT = """You are given a QUESTION and an ANSWER. Decide whether the ANSWER actually provides the information the question asks for, or whether it FAILS to (it says it does not know, is unsure, cannot find it, has no data/information, only says it lacks real-time/current info, or asks the user for clarification instead of answering). Reply with ONLY one word: ANSWERED or UNKNOWN.

QUESTION: What is the capital of France?
ANSWER: Paris.
VERDICT: ANSWERED

QUESTION: what is planet of lana
ANSWER: I couldn't find any information on "Planet of Lana". It's possible that it's a lesser-known or upcoming project.
VERDICT: UNKNOWN

QUESTION: who is the studio behind Cocoon game
ANSWER: I'm unsure.
VERDICT: UNKNOWN

QUESTION: who wrote Hamlet?
ANSWER: William Shakespeare.
VERDICT: ANSWERED

QUESTION: wer hat planet of lana entwickelt
ANSWER: Ich bin mir nicht sicher, wer das Spiel entwickelt hat.
VERDICT: UNKNOWN

QUESTION: {q}
ANSWER: {a}
VERDICT:"""


def _svar_ar_ickesvar(fraga, svar):
    """True om ANSWER inte faktiskt besvarar fragan (Llama vet inte / ar
    osaker). Kort Llama-klassificering, sprakoberoende. Vid fel: False
    (behall det lokala svaret - eskalera inte i onodan)."""
    try:
        p = _ICKESVAR_PROMPT.replace("{q}", fraga.strip())
        p = p.replace("{a}", svar.strip()[:600])
        out = llm(p, max_tokens=3, temperature=0.0,
                  stop=_LLAMA_STOPP + ["\nQUESTION:", "\n\n"])
        v = out["choices"][0]["text"].strip().upper()
        return v.startswith("UNKNOWN")
    except Exception:
        return False


# Few-shot: avgor om en fraga galler en SPECIFIK/nisch namngiven entitet
# (lokal verksamhet, produkt, spel, foretag, icke-varldskand person) dar
# Llamas minne ar opalitligt. Sadana ska ALLTID sokas - annars svarar
# Llama ofta sjalvsakert FEL (t.ex. "bobergs pa hamngarden = ett par skor")
# och lokalt-forst-kontrollen fangar det inte (svaret LATER sakert).
_VERIFIERA_PROMPT = """Decide whether answering the QUESTION accurately requires looking up a SPECIFIC named thing that is not common knowledge - for example a particular local business, restaurant, shop, hotel, place, a specific product, app, game, movie, book, song, a company/organization/team, or a person who is not world-famous. Reply with ONE word: LOOKUP or KNOWN.

Reply KNOWN for general concepts, definitions, science, history, math, world-famous people/works, capital cities, and things most educated people know.
Reply LOOKUP for specific or niche named entities you may not reliably know from memory.

Q: what is a mutex
A: KNOWN
Q: who wrote Hamlet
A: KNOWN
Q: what is the capital of France
A: KNOWN
Q: what is photosynthesis
A: KNOWN
Q: what is democracy
A: KNOWN
Q: vad ar bobergs pa hamngarden
A: LOOKUP
Q: what is planet of lana
A: LOOKUP
Q: vad ar limhamn griffins
A: LOOKUP
Q: what is the game cocoon
A: LOOKUP
Q: vad ar tga for utbildning i malmo
A: LOOKUP
Q: {fraga}
A:"""


def _bor_soka_entitet(fraga):
    """True om fragan galler en specifik/nisch namngiven entitet dar
    Llamas minne ar opalitligt -> sok direkt istallet for att riskera en
    sjalvsaker hallucination. Vid fel: False (behall lokalt-forst)."""
    try:
        p = _VERIFIERA_PROMPT.replace("{fraga}", fraga.strip())
        out = llm(p, max_tokens=3, temperature=0.0,
                  stop=_LLAMA_STOPP + ["\nQ:", "\n\n"])
        return out["choices"][0]["text"].strip().upper().startswith("LOOKUP")
    except Exception:
        return False


_JANEJ_INTENT_PROMPT = """Decide whether the QUESTION is a yes/no question - one that expects the answer "yes" or "no" - or an OPEN question that asks for information, an explanation or a definition. Reply with ONE word: POLAR or OPEN.

Q: was ist die band betontod
A: OPEN
Q: qu'est-ce que le fc chambly
A: OPEN
Q: vad ar bobergs pa hamngarden
A: OPEN
Q: what is planet of lana
A: OPEN
Q: quelle est la capitale de la france
A: OPEN
Q: wat is de albert cuypmarkt
A: OPEN
Q: ist berlin die hauptstadt von deutschland
A: POLAR
Q: har sverige nagonsin vunnit fotbolls vm
A: POLAR
Q: is limhamn griffins a hockey team
A: POLAR
Q: does python 3.13 exist yet
A: POLAR
Q: {fraga}
A:"""


def _ar_akta_ja_nej(fraga):
    """B1/A1: bekraftar att en fraga som regexen _ar_ja_nej_fraga flaggat
    VERKLIGEN ar en ja/nej-fraga. Regexen ger falska traffar pa utländska
    'was ist X' / 'vad ar X' (definitionsfragor) eftersom tyska 'was'
    krockar med engelska 'was' i monstret. En liten sprakoberoende Llama-
    klassificering (som langid/entitet) vetar de falska traffarna sa att
    definitionsfragor inte langre prefixas 'Die Antwort ist JA...'. Vid
    fel: True (behall regexens beslut - veta inte bort en akta analys)."""
    try:
        p = _JANEJ_INTENT_PROMPT.replace("{fraga}", fraga.strip())
        out = llm(p, max_tokens=3, temperature=0.0,
                  stop=_LLAMA_STOPP + ["\nQ:", "\n\n"])
        return out["choices"][0]["text"].strip().upper().startswith("POLAR")
    except Exception:
        return True


def _forsok_lokalt_svar(fraga_orig, svarssprak, sprak, historik, token_cb=None):
    """Lokalt forsta forsok UTAN webbsok - snabbt svar ur modellens eget
    minne. Returnerar ett sjalvsakert svar (i anvandarens sprak), ELLER
    None om modellen inte kunde svara palitligt (token, klassad som icke-
    svar, eller trasigt) - vilket signalerar till fraga_ai att eskalera
    till webbsok."""
    system_msg = bygg_system_prompt(sprak) + _LOKAL_SENTINEL_INSTR
    system_msg += _sprak_direktiv(svarssprak)
    prompt = bygg_llama_prompt_med_historik(system_msg, historik, fraga_orig)
    # STROMNINGS-SKYDD: det HAR ar ett spekulativt forsta forsok som kan
    # forkastas (sentinel/icke-svar/trasigt) och da eskalera till webbsok.
    # Filtrera bort sentinel-token ur den live-streamade texten sa den
    # aldrig blinkar till pa skarmen; det riktiga svaret streamas rent efter
    # ev. eskalering (UI:t nollstalls i status-callbacken 'sok').
    _lokal_cb = token_cb
    if token_cb is not None:
        def _lokal_cb(text_hittills):
            if _ar_sentinel_text(text_hittills):
                return
            token_cb(text_hittills)
    # SAMPLING: lagre repeat_penalty (1.10 ist. for 1.15) ger naturligare,
    # mindre grumlig text - hog penalty tvingade fram udda ordval (syns tydligt
    # i svensk prosa/dikt). 1.10 behaller loop-skyddet men later spraket floda.
    svar = _generera(prompt, temperature=0.4, repeat_penalty=1.10,
                     token_cb=_lokal_cb)
    if ar_svaret_trasigt(svar, hade_fakta=False, sprak=sprak):
        return None
    if _SENTINEL_SOK.lower() in svar.lower():
        _debug_logg("lokalt: modellen bad om sok (token)")
        return None
    if _svar_ar_ickesvar(fraga_orig, svar):
        _debug_logg("lokalt: klassat som icke-svar")
        return None
    return _stada_hallucinerad_landsparentes(svar, sprak)


def _slaihop_kontext(a, b):
    """Slar ihop tva sok-kontexter (radbaserade '- titel: text (Kalla)')
    och tar bort dubbletter. Anvands for att kombinera sok pa
    originalsprak + engelska (P1)."""
    if not a:
        return b or ""
    if not b:
        return a or ""
    rader, sedda = [], set()
    for kalla in (a, b):
        for rad in kalla.split("\n"):
            nyckel = rad.strip().lower()[:80]
            if nyckel and nyckel not in sedda:
                sedda.add(nyckel)
                rader.append(rad)
    return "\n".join(rader)


_FOLJD_PRONOMEN = re.compile(
    r"\b(it|its|they|them|their|his|her|hers|"
    r"den|det|denna|detta|dess|hans|hennes|deras|dem|"
    r"sie|ihr|ihre|dessen|"
    r"elle|lui|leur|"
    r"suo|sua|"
    r"ella|"
    r"ele|ela|seu)\b", re.IGNORECASE)


def _ar_foljdfraga(fraga, historik):
    """A5: heuristik for en (mojligen) tillbakasyftande foljdfraga - en kort
    fraga med ett bakatsyftande pronomen OCH en historik att losa det mot.
    Konservativ: hellre missa an skriva om en tydlig fristaende fraga."""
    if not historik:
        return False
    norm = _normalisera_for_matchning(fraga)
    n_ord = len(norm.split())
    if n_ord > 12:
        return False
    # (a) Tydligt tillbakasyftande pronomen ('den', 'det', 'they', ...)
    #     -> nastan sakert en foljdfraga.
    if _FOLJD_PRONOMEN.search(norm):
        return True
    # (b) Fraga UTAN eget ankare (inget artal, inget egennamn med versal
    #     bortsett fran ev. meningsforsta ordet) ar nastan alltid en
    #     AMNESFORTSATTNING som forst blir begriplig mot historiken:
    #     'vem tror du vinner', 'kan du gissa vilka partier som bygger
    #     regeringen tillsammans'. Saknar entitet att soka pa. Omskrivaren
    #     returnerar fragan OFORANDRAD om den redan ar fristaende.
    _rest = fraga.strip().split(None, 1)
    _efter_forsta = _rest[1] if len(_rest) > 1 else ""
    har_ankare = (bool(re.search(r"\b(19|20)\d{2}\b", fraga)) or
                  bool(re.search(r"[A-ZÅÄÖ][a-zåäö]{2,}", _efter_forsta)))
    if not har_ankare and n_ord <= 10:
        return True
    # (c) Mycket kort fraga aven med litet ankare -> lat omskrivaren avgora.
    return n_ord <= 6


def _skriv_om_foljdfraga(fraga, historik):
    """A5: skriver om en tillbakasyftande foljdfraga ('its game by wishfully')
    till en FRISTAENDE fraga med hjalp av konversationshistoriken, sa att
    sok/retrieval far en fraga med tydlig entitet. Behaller spraket.
    Returnerar originalet om nagot gar fel eller om modellen inte gor den
    tydligare."""
    try:
        rader = []
        for varv in historik[-3:]:
            fr = (varv.get("fraga") or "").strip()
            sv = (varv.get("svar") or "").strip()
            if fr:
                rader.append("User: " + fr)
            if sv:
                rader.append("Assistant: " + sv)
        kontext = "\n".join(rader)
        if not kontext:
            return fraga
        prompt = bygg_qwen_prompt(
            "You rewrite a possibly-elliptical follow-up question into ONE "
            "self-contained question, using the conversation so far to "
            "resolve references like 'it', 'they', 'this'. Keep the SAME "
            "language as the follow-up. If it is already self-contained, "
            "return it unchanged. Output ONLY the rewritten question.",
            f"Conversation so far:\n{kontext}\n\n"
            f"Follow-up: {fraga.strip()}\n\nRewritten:")
        out = llm(prompt, max_tokens=48, temperature=0.0,
                  stop=_LLAMA_STOPP + ["\nFollow-up:", "\nRewritten:", "\n\n"])
        ny = out["choices"][0]["text"].strip().strip('"').split("\n")[0].strip()
        if ny and 2 < len(ny) <= 200:
            return ny
    except Exception:
        pass
    return fraga


# ============ VADER (open-meteo, ingen API-nyckel) ============
# WMO-vaderkoder -> engelsk text. Svaret speglas sedan till anvandarens
# sprak via _sprak_direktiv precis som ovrig fakta.
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle",
    55: "dense drizzle", 56: "freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "rain", 65: "heavy rain", 66: "freezing rain",
    67: "heavy freezing rain", 71: "slight snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "slight rain showers", 81: "rain showers",
    82: "violent rain showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}

# Svenska WMO-texter: nar anvandaren fragar pa svenska bygger vi fakta-texten
# direkt pa svenska istallet for att lata 3B-modellen oversatta vaderjargong
# (den gav "overcast"->"tackt", "precipitation chance"->"nederbordssannfard").
_WMO_SV = {
    0: "klart", 1: "mestadels klart", 2: "delvis molnigt", 3: "mulet",
    45: "dimma", 48: "underkyld dimma", 51: "lätt duggregn", 53: "duggregn",
    55: "tätt duggregn", 56: "underkylt duggregn", 57: "tätt underkylt duggregn",
    61: "lätt regn", 63: "regn", 65: "kraftigt regn", 66: "underkylt regn",
    67: "kraftigt underkylt regn", 71: "lätt snöfall", 73: "snöfall",
    75: "kraftigt snöfall", 77: "snökorn", 80: "lätta regnskurar",
    81: "regnskurar", 82: "kraftiga regnskurar", 85: "snöbyar",
    86: "kraftiga snöbyar", 95: "åska", 96: "åska med hagel",
    99: "åska med kraftigt hagel",
}

_VADER_FRAGA = re.compile(
    r"\b(vader|vadret|vaderprognos|weather|forecast|temperatur(en)?|"
    r"regnar|snoar|wetter|weer|meteo|meteo|previsioni|clima|tiempo|"
    r"temps|tempo|hur (varmt|kallt)|graders?)\b", re.IGNORECASE)


def _ar_vaderfraga(fraga_norm):
    return bool(_VADER_FRAGA.search(fraga_norm))


_PLATS_PROMPT = """Extract ONLY the location (city/town/place) that this weather question is about. Reply with the place name alone, or NONE if the question names no place.

Q: vad ar vadret i stockholm imorgon
A: Stockholm
Q: hur blir vadret i goteborg pa lordag
A: Goteborg
Q: what is the weather in new york today
A: New York
Q: wie ist das wetter in berlin morgen
A: Berlin
Q: quel temps fait-il a paris
A: Paris
Q: che tempo fa a milano
A: Milano
Q: vad ar vadret
A: NONE
Q: {fraga}
A:"""


def _extrahera_plats(fraga):
    """Plockar ut ortnamnet ur en vaderfraga med Llama (redan laddad)."""
    try:
        p = _PLATS_PROMPT.replace("{fraga}", fraga.strip())
        out = llm(p, max_tokens=12, temperature=0.0,
                  stop=_LLAMA_STOPP + ["\nQ:", "\n"])
        s = out["choices"][0]["text"].strip().strip('".')
        if s and s.upper() != "NONE" and 1 < len(s) <= 60:
            return s
    except Exception:
        pass
    return None


def _geocode(plats):
    try:
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": plats, "count": 1, "language": "en"},
                         timeout=6, headers={"User-Agent": "AIPrototype/1.0"})
        res = (r.json().get("results") or []) if r.status_code == 200 else []
        if not res:
            return None
        d = res[0]
        return (d["latitude"], d["longitude"],
                d.get("name", plats), d.get("country", ""))
    except Exception:
        return None


def vader_uppslag(fraga_orig, spraknamn="English"):
    """Riktig vaderprognos via open-meteo (nyckellost). Returnerar en
    fakta-text (nu + 3 dygn) - pa SVENSKA om anvandaren fragar pa svenska
    (annars engelska, som genereringen speglar till ovriga sprak). Svensk
    text byggs direkt for att slippa 3B-modellens vaderjargong-oversattning.
    None om plats/prognos saknas."""
    plats = _extrahera_plats(fraga_orig)
    if not plats:
        return None
    geo = _geocode(plats)
    if not geo:
        return None
    lat, lon, namn, land = geo
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "daily": ("weather_code,temperature_2m_max,temperature_2m_min,"
                      "precipitation_probability_max"),
            "timezone": "auto", "forecast_days": 3},
            timeout=6, headers={"User-Agent": "AIPrototype/1.0"})
        if r.status_code != 200:
            return None
        j = r.json()
    except Exception:
        return None

    ar_sv = (spraknamn or "").strip().lower() == "swedish"
    wmo = _WMO_SV if ar_sv else _WMO
    okand = "okända förhållanden" if ar_sv else "unknown conditions"

    def _desc(c):
        try:
            return wmo.get(int(c), okand)
        except Exception:
            return okand

    cur = j.get("current", {}) or {}
    daily = j.get("daily", {}) or {}
    dt = daily.get("time", [])
    mx = daily.get("temperature_2m_max", [])
    mn = daily.get("temperature_2m_min", [])
    wc = daily.get("weather_code", [])
    pp = daily.get("precipitation_probability_max", [])
    if ar_sv:
        rader = [f"Väderdata för {namn}, {land} (källa: open-meteo; "
                 f"temperatur i Celsius):"]
        if cur.get("temperature_2m") is not None:
            rader.append(f"- Just nu: {cur['temperature_2m']}°C, "
                         f"{_desc(cur.get('weather_code'))}, vind "
                         f"{cur.get('wind_speed_10m')} km/h.")
        labels = ["Idag", "Imorgon", "I övermorgon"]
        for i in range(min(len(dt), 3)):
            rader.append(
                f"- {labels[i]} ({dt[i]}): högst {mx[i]}°C, lägst {mn[i]}°C, "
                f"{_desc(wc[i])}, nederbördsrisk {pp[i]}%.")
    else:
        rader = [f"Weather data for {namn}, {land} (source: open-meteo; "
                 f"temperatures in Celsius):"]
        if cur.get("temperature_2m") is not None:
            rader.append(f"- Right now: {cur['temperature_2m']}C, "
                         f"{_desc(cur.get('weather_code'))}, wind "
                         f"{cur.get('wind_speed_10m')} km/h.")
        labels = ["Today", "Tomorrow", "The day after tomorrow"]
        for i in range(min(len(dt), 3)):
            rader.append(
                f"- {labels[i]} ({dt[i]}): high {mx[i]}C, low {mn[i]}C, "
                f"{_desc(wc[i])}, precipitation chance {pp[i]}%.")
    if len(rader) < 2:
        return None
    return "\n".join(rader)


# ============ AKTIE-/KRYPTOKURS (Yahoo Finance, ingen API-nyckel) ============
_AKTIE_FRAGA = re.compile(
    r"\b(aktie(kurs(en)?|n|rna|r|pris(et)?)?|borskurs(en)?|"
    r"stock price|share price|stock quote|"
    r"bitcoin|ethereum|krypto(valuta)?|crypto(currency)?)\b", re.IGNORECASE)

_AKTIE_ORD = {
    "aktie", "aktien", "aktier", "aktierna", "aktiekurs", "aktiekursen",
    "aktiepris", "aktiepriset", "kurs", "kursen", "borskurs", "borskursen",
    "stock", "stocks", "share", "shares", "price", "quote", "kostar",
    "kosta", "vard", "varde", "vardet", "worth", "market", "cap", "ticker",
    "just", "nu", "idag", "today", "now", "current",
}


def _ar_aktiefraga(fraga_norm):
    return bool(_AKTIE_FRAGA.search(fraga_norm))


# Kanda bolag/krypto -> Yahoo-symbol. Loser de vanligaste DIREKT sa vi
# slipper Yahoos fritextsok - det gav for "apple s" (genitiv) en obskyr fond
# ("Appletree Debt Fund") istallet for Apple Inc, och for "how much bitcoin
# right" ingen traff alls. Aliaslistan ar robust och sprakoberoende.
_KANDA_TICKERS = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
    "alphabet": "GOOGL", "amazon": "AMZN", "meta": "META",
    "facebook": "META", "netflix": "NFLX", "nvidia": "NVDA",
    "tesla": "TSLA", "intel": "INTC", "amd": "AMD", "ibm": "IBM",
    "oracle": "ORCL", "adobe": "ADBE", "salesforce": "CRM",
    "spotify": "SPOT", "paypal": "PYPL", "disney": "DIS", "boeing": "BA",
    "coca cola": "KO", "cocacola": "KO", "pepsi": "PEP",
    "mcdonalds": "MCD", "walmart": "WMT", "nike": "NKE",
    "starbucks": "SBUX", "mastercard": "MA", "berkshire": "BRK-B",
    "samsung": "005930.KS", "toyota": "TM",
    # Nordiska bolag - hemmalistning (Stockholm, SEK)
    "volvo": "VOLV-B.ST", "ericsson": "ERIC-B.ST", "investor": "INVE-B.ST",
    "handelsbanken": "SHB-A.ST", "swedbank": "SWED-A.ST",
    "atlas copco": "ATCO-A.ST", "saab": "SAAB-B.ST", "evolution": "EVO.ST",
    "sinch": "SINCH.ST", "kinnevik": "KINV-B.ST", "electrolux": "ELUX-B.ST",
    "nordea": "NDA-SE.ST", "telia": "TELIA.ST", "sandvik": "SAND.ST",
    # Krypto
    "bitcoin": "BTC-USD", "btc": "BTC-USD", "ethereum": "ETH-USD",
    "ether": "ETH-USD", "eth": "ETH-USD", "dogecoin": "DOGE-USD",
    "doge": "DOGE-USD", "solana": "SOL-USD", "cardano": "ADA-USD",
    "ripple": "XRP-USD", "xrp": "XRP-USD", "litecoin": "LTC-USD",
    "polkadot": "DOT-USD",
}
# Langsta alias forst -> "coca cola" matchar fore ev. kortare delalias.
_TICKER_ALIAS = sorted(_KANDA_TICKERS, key=len, reverse=True)

# Extra fyllnadsord som inte ska bli en del av bolagsnamnet i fritextsok.
_AKTIE_EXTRA_STOPP = {
    "how", "much", "many", "right", "el", "la", "cuanto", "quanto",
    "combien", "wie", "viel", "vale", "vaut", "quel", "quanta", "der",
}


def _kand_ticker(fraga_norm):
    """Matchar ett kant bolag/kryptoalias (helt ord, ev. genitiv-s) och
    returnerar Yahoo-symbolen - annars None."""
    for alias in _TICKER_ALIAS:
        if re.search(r"\b" + re.escape(alias) + r"s?\b", fraga_norm):
            return _KANDA_TICKERS[alias]
    return None


def _valj_kvot(quotes):
    """Valjer basta Yahoo-traffen: riktiga bolag/krypto fore fonder/index
    (sa 'apple' inte langre landar pa en obskyr 'Appletree'-fond)."""
    ordning = {"EQUITY": 0, "CRYPTOCURRENCY": 0, "ETF": 1, "CURRENCY": 1,
               "INDEX": 2, "MUTUALFUND": 3, "FUTURE": 3}
    kand = [q for q in quotes if q.get("symbol")]
    if not kand:
        return None
    kand.sort(key=lambda q: ordning.get(q.get("quoteType", ""), 4))
    return kand[0]


def _aktie_query(fraga):
    tokens = re.findall(r"[a-z0-9]+", _normalisera_for_matchning(fraga))
    kvar = [o for o in tokens
            if o not in _SOK_STOPP and o not in _AKTIE_ORD
            and o not in _AKTIE_EXTRA_STOPP and len(o) > 1]
    return " ".join(kvar).strip() or fraga.strip()


def _hamta_aktie(fraga_orig):
    """Riktig kurs via Yahoo Finance (nyckellos): sok-endpoint loser symbol
    (foretagsnamn/krypto -> ticker), chart-endpoint ger senaste pris.
    Returnerar en arlig fakta-text med pris, kalla och tidsstampel, eller
    None (da faller appen tillbaka pa 'kolla en finanskalla')."""
    hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AIPrototype/1.0"}
    norm = _normalisera_for_matchning(fraga_orig)
    sym = _kand_ticker(norm)      # kant bolag/krypto -> symbol direkt
    namn = None
    try:
        if sym is None:
            q = _aktie_query(fraga_orig)
            if not q:
                return None

            def _yahoo_sok(term):
                s = requests.get(
                    "https://query1.finance.yahoo.com/v1/finance/search",
                    params={"q": term, "quotesCount": 6, "newsCount": 0},
                    timeout=6, headers=hdr)
                return s.json().get("quotes", []) if s.status_code == 200 else []

            quotes = _yahoo_sok(q)
            if not quotes and " " not in q and len(q) > 3 and q.endswith("s"):
                # Svensk genitiv/plural ('apples' -> 'apple') gav ingen traff.
                quotes = _yahoo_sok(q[:-1])
            val = _valj_kvot(quotes)   # foredra bolag/krypto fore fonder/index
            if not val:
                return None
            sym = val.get("symbol")
            namn = val.get("shortname") or val.get("longname") or sym
        if not sym:
            return None
        c = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
            timeout=6, headers=hdr)
        if c.status_code != 200:
            return None
        m = c.json()["chart"]["result"][0]["meta"]
        pris = m.get("regularMarketPrice")
        if pris is None:
            return None
        if not namn:
            namn = m.get("shortName") or m.get("longName") or sym
        namn = " ".join(str(namn).split())   # stada dubbla mellanslag/skrap
        cur = m.get("currency", "")
        exch = m.get("exchangeName", "")
        prev = m.get("chartPreviousClose") or m.get("previousClose")
        andring = ""
        if prev:
            diff = pris - prev
            pct = (diff / prev * 100) if prev else 0
            tecken = "+" if diff >= 0 else ""
            andring = (f", change {tecken}{round(diff, 2)} {cur} "
                       f"({tecken}{round(pct, 2)}%) vs previous close")
        tid = m.get("regularMarketTime")
        nar = ""
        if tid:
            try:
                nar = (" as of " + datetime.datetime.utcfromtimestamp(
                    int(tid)).strftime("%Y-%m-%d %H:%M UTC"))
            except Exception:
                nar = ""
        return (f"{namn} ({sym}, {exch}) latest price: {pris} {cur}"
                f"{andring}{nar}. Source: Yahoo Finance; market quotes can be "
                f"delayed ~15 minutes.")
    except Exception:
        return None


# ============ DIREKT OVERSATTNING ============
# Explicita 'oversatt X till Y'-fragor. Den generella modellen klarade dem
# daligt ('good morning' -> 'Hej!', 'tack' pa tyska -> 'T-H-A-N-K'). En
# fokuserad few-shot ger ratt oversattning. VERB-baserad detektering (ej
# 'vad heter'/'what is', som ocksa ar vanliga faktafragor) undviker falska
# traffar.
_OVERSATT_FRAGA = re.compile(
    r"\b(translate|translation|how (do|would|can) (you|i|we) say|how to say|"
    r"oversatt|hur sager man|hur skriver man|"
    r"ubersetz|wie sagt man|wie heisst man|"
    r"traduis|traduire|traduction|comment dit|comment on dit|"
    r"traduci|traduzione|come si dice|"
    r"traducir|traduce|traduccion|como se dice)\b", re.IGNORECASE)


def _ar_oversattningsfraga(fraga_norm):
    return bool(_OVERSATT_FRAGA.search(fraga_norm))


_OVERSATT_PROMPT_SYS = (
    "You are a precise translator. The user's message is a translation "
    "request - 'translate X to LANGUAGE', 'how do you say X in LANGUAGE', or "
    "the same in another language (Swedish 'hur sager man X pa SPRAK', German "
    "'wie sagt man X auf SPRACHE', French 'comment dit-on X en LANGUE', etc.). "
    "Work out the phrase and the TARGET language, then output ONLY the "
    "translation in that target language - just the translated text itself. "
    "No quotes, no original, no explanation, no language label.")


def _direkt_oversattning(fraga):
    """Dedikerad vag for explicita oversattningsfragor. Returnerar
    oversattningen pa MALSPRAKET (ingen spegling) eller None."""
    try:
        prompt = bygg_llama_prompt(
            _OVERSATT_PROMPT_SYS,
            "Request: Translate 'good morning' to Swedish.\nTranslation: God morgon\n\n"
            "Request: Hur sager man 'tack' pa tyska?\nTranslation: Danke\n\n"
            "Request: How do you say 'thank you' in French?\nTranslation: Merci\n\n"
            "Request: What is 'water' in Spanish?\nTranslation: Agua\n\n"
            "Request: Ubersetze 'Katze' ins Englische.\nTranslation: Cat\n\n"
            "Request: Come si dice 'grazie' in inglese?\nTranslation: Thank you\n\n"
            f"Request: {fraga.strip()}\nTranslation:")
        out = llm(prompt, max_tokens=60, temperature=0.0,
                  stop=_LLAMA_STOPP + ["\nRequest:", "\nTranslation:", "\n\n"])
        svar = out["choices"][0]["text"].strip().strip('"').strip()
        svar = svar.split("\n")[0].strip()
        if svar and 0 < len(svar) <= 200:
            return svar
    except Exception:
        pass
    return None


# ============ EXAKT RAKNARE (deterministisk aritmetik) ============
# Llama 3.2 3B raknar fel pa rak aritmetik (gav t.ex. 137*24 = 3296, ratt ar
# 3288). Fanga rena raknefragor och berakna EXAKT i Python. Sakert: ingen ra
# eval() pa fri input - bara AST med VITLISTADE noder (tal + aritmetik).
import ast as _ast

_RAKNE_ORD = [
    (r"\bplus\b", "+"),
    (r"\b(minus|subtract(ed by)?|subtrahera(t)? med)\b", "-"),
    (r"\b(times|multiplied by|multiplicerat med|ganger|gange)\b", "*"),
    (r"\b(divided by|dividerat med|delat med|delat pa|delat)\b", "/"),
    (r"\b(to the power of|upphojt till|upphojd till|raised to)\b", "**"),
]

_RAKNE_SIGNAL = re.compile(
    # +, *, / far sitta tatt (100/4, 7*8). Men MINUS kraver mellanslag runt
    # sig - annars fangas prosa-intervall som "3-4 pass"/"5-6 timmar" och
    # rknas felaktigt som subtraktion ("3-4 = -1"). "100 - 25" fungerar.
    r"(\d[\d\s.,]*[+*/]\s*\d)|(\d[\d\s.,]*\s-\s\d)|"
    r"\b(plus|minus|times|multiplied by|divided by|ganger|gange|"
    r"delat|dividerat|multiplicerat|upphojt|squared|i kvadrat|"
    r"square root|kvadratroten|roten ur|sqrt|"
    r"percent of|procent av)\b|(%\s*(of|av))|"
    r"\b(rakna ut|berakna|calculate|compute|what is|whats|vad ar|vad blir|"
    r"how much is)\b.*\d", re.IGNORECASE)


def _ar_raknefraga(fraga_norm):
    return bool(_RAKNE_SIGNAL.search(fraga_norm))


_TILLATNA_AST = (_ast.Expression, _ast.BinOp, _ast.UnaryOp, _ast.Constant,
                 _ast.Add, _ast.Sub, _ast.Mult, _ast.Div, _ast.FloorDiv,
                 _ast.Mod, _ast.Pow, _ast.USub, _ast.UAdd)


def _safe_eval(uttryck):
    """Utvarderar ett aritmetiskt uttryck SAKERT - endast vitlistade
    AST-noder (tal + de fyra raknesatten, potens, parenteser). Aldrig
    namn/anrop/attribut. None vid syntaxfel eller otillaten nod."""
    try:
        trad = _ast.parse(uttryck, mode="eval")
    except Exception:
        return None
    for nod in _ast.walk(trad):
        if not isinstance(nod, _TILLATNA_AST):
            return None
        if isinstance(nod, _ast.Constant) and not isinstance(nod.value, (int, float)):
            return None
    try:
        return eval(compile(trad, "<calc>", "eval"), {"__builtins__": {}}, {})
    except Exception:
        return None


def _formatera_tal(res):
    if isinstance(res, bool):
        return None
    if isinstance(res, int):
        return str(res)
    if isinstance(res, float):
        if res.is_integer():
            return str(int(res))
        return ("%.6f" % res).rstrip("0").rstrip(".")
    return str(res)


def _berakna(fraga, spraknamn="English"):
    """Extraherar och beraknar ett rent aritmetiskt uttryck ur fragan.
    Returnerar (visningstext, resultat) eller None om det inte ar en ren
    rakneuppgift (da faller appen tillbaka pa vanlig hantering)."""
    ar_sv = (spraknamn or "").strip().lower() == "swedish"
    av = "av" if ar_sv else "of"
    t = " " + fraga.lower().strip() + " "
    # Procent: "X% of Y" / "X procent av Y" -> (X/100)*Y
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:%|percent|procent)\s*(?:of|av)\s*"
                  r"(\d+(?:[.,]\d+)?)", t)
    if m:
        a = float(m.group(1).replace(",", ".")); b = float(m.group(2).replace(",", "."))
        return (f"{m.group(1)}% {av} {m.group(2)}", a / 100 * b)
    # Kvadratrot
    m = re.search(r"(?:square root of|kvadratroten ur|roten ur|sqrt)\s*"
                  r"\(?\s*(\d+(?:[.,]\d+)?)", t)
    if m:
        x = float(m.group(1).replace(",", "."))
        if x >= 0:
            return (f"√{m.group(1)}", x ** 0.5)
    # Ordoperatorer -> symboler
    for monster, sym in _RAKNE_ORD:
        t = re.sub(monster, " " + sym + " ", t)
    t = re.sub(r"\bsquared\b|\bi kvadrat\b", " ** 2 ", t)
    # Svensk decimalkomma mellan siffror -> punkt (3,5 -> 3.5)
    t = re.sub(r"(\d),(\d)", r"\1.\2", t)
    # Plocka ut uttrycket (ev. med inledande parentes)
    m = re.search(r"[-+(]*\s*\d[\d\s.+\-*/%()]*", t)
    if not m:
        return None
    uttryck = m.group(0).strip()
    # Balansera parenteser (ta bort omatchade)
    while uttryck.count(")") > uttryck.count("("):
        i = uttryck.rfind(")")
        uttryck = uttryck[:i] + uttryck[i + 1:]
    while uttryck.count("(") > uttryck.count(")"):
        i = uttryck.find("(")
        uttryck = uttryck[:i] + uttryck[i + 1:]
    uttryck = uttryck.strip().rstrip("+-*/ .").strip()
    if not re.search(r"\d", uttryck):
        return None
    # Kraver minst en operator (annars ar det bara ett tal)
    if not re.search(r"[-+*/%]", uttryck):
        return None
    # Skydda mot gigantiska exponenter (undvik hang/minne)
    for m2 in re.finditer(r"\*\*\s*(\d+)", uttryck):
        if int(m2.group(1)) > 1000:
            return None
    res = _safe_eval(uttryck)
    if res is None or isinstance(res, bool):
        return None
    visning = uttryck.replace("**", " ^ ").replace("*", " × ").replace("/", " ÷ ")
    visning = re.sub(r"\s+", " ", visning).strip()
    return (visning, res)


# ============ PERSONLIG / KANSLOMASSIG VAG ============
# Kansliga, relationella eller personliga meddelanden (jag mar daligt, ar
# du min van, alskar du mig, jag ar ensam ...) ska INTE ga till webbsok -
# da mangas de till "kallorna bekraftar ..."-svar eller kliniska dumpar.
# Istallet: svara varmt och direkt fran sakerhets-riktlinjerna i
# systemprompten, utan sok och utan pafosad fakta.
_PERSONLIG_KLASS = (
    "Decide whether the MESSAGE is personal or emotional - the user "
    "sharing feelings, distress, loneliness or self-worth, or asking about "
    "your feelings or your relationship to them (love, friendship, caring) "
    "- rather than a request for information.\n"
    "Answer PERSONAL for feelings, venting, distress, loneliness, "
    "self-doubt, or questions about your feelings/relationship to the "
    "user.\n"
    "Answer INFO for factual, practical or knowledge questions.\n"
    "Reply with ONE word: PERSONAL or INFO.\n\n"
    "MESSAGE: I feel completely worthless and alone.\nPERSONAL\n\n"
    "MESSAGE: Do you love me? You're my best friend.\nPERSONAL\n\n"
    "MESSAGE: What is the capital of France?\nINFO\n\n"
    "MESSAGE: Who won the World Cup?\nINFO\n\n"
    "MESSAGE: I'm so stressed I can't sleep and nothing feels worth it.\n"
    "PERSONAL\n\n"
    "MESSAGE: how do i center a div in css\nINFO\n\n"
    "MESSAGE: {q}\n"
)

_PERSONLIG_SYS_EXTRA = (
    " VIKTIGT: Anvandarens meddelande ar personligt eller kanslomassigt. "
    "Svara kort, varmt och arligt. Sok INTE och citera INGA kallor. Pasta "
    "inte att du har kanslor eller ar deras van - du ar ett verktyg, och "
    "det far du saga vanligt. Verkar de ha det svart: foresla varsamt att "
    "prata med nagon de litar pa eller en fackperson. Hall det manskligt "
    "och osokt."
)


_PERSONLIG_REGEX = re.compile(
    r"\bi feel\b|\bi'm feeling\b|\bi am feeling\b|jag kanner mig|jag mar\b|"
    r"worthless|hopeless|vardelos|meningslos|"
    r"i'm so |i am so |feel so (sad|alone|lonely|depressed|anxious|empty)|"
    r"kanner mig (ensam|vardelos|nere|deppig)|"
    r"\bensam\b|\blonely\b|\balone\b|feel invisible|kanner mig osynlig|"
    r"do you love me|alskar du mig|are you my friend|ar du min van|"
    r"do you have feelings|har du kanslor|are you conscious|are you sentient|"
    r"best friend|basta van|only one who (understands|gets) me|"
    r"nobody (cares|loves|understands|likes)|"
    r"ingen (bryr sig|forstar mig|alskar mig|gillar mig)|"
    r"i want to (die|disappear|give up)|jag vill (do|forsvinna|ge upp)|"
    r"no reason to (live|go on)|ingen mening med|nothing matters|"
    r"inget spelar roll|i hate myself|jag hatar mig",
    re.IGNORECASE)


def _ar_personlig_fraga(fraga):
    """KONSERVATIV regex: bara TYDLIGA personliga/kanslomassiga signaler
    triggar den varma lokala vagen. (En LLM-klassificerare triggade pa
    ALLT och stal faktafragor fran sok-vagen.) Hellre missa nagot enstaka
    an ratta bort en faktafraga."""
    return bool(_PERSONLIG_REGEX.search(_normalisera_for_matchning(fraga)))


# ============ VEM VANN (fokuserad vinnar-extraktion) ============
# "Vem vann senaste Serie A / Ballon d'Or / F1 / Copa del Rey ..." dumpade
# forr tavlingens uppslagsverksbeskrivning istallet for att namna vinnaren.
# Har: sok pa senaste vinnaren och extrahera BARA namnet.
_MASTAR_FRAGA = re.compile(
    r"\bwho\s+won\b|\bwho\s+is\s+the\s+(current|reigning|defending|latest)\s+"
    r"(champion|winner|title)\b"
    r"|\bvem\s+vann\b|\bvem\s+ar\s+(regerande|nuvarande)\s+(mastare|vinnare)\b"
    r"|\bchi\s+ha\s+vinto\b|\bchi\s+e\s+.{0,20}campione\b"
    r"|\bqui\s+a\s+(gagne|remporte)\b|\bqui\s+est\s+.{0,20}champion\b"
    r"|\bquien\s+gano\b|\bque\s+equipo\s+gano\b|\bquien\s+es\s+.{0,20}campeon\b"
    r"|\bwer\s+hat\s+.{0,50}gewonnen\b|\bwer\s+ist\s+.{0,40}"
    r"(meister|weltmeister|champion|sieger)\b|\bwer\s+gewann\b",
    re.IGNORECASE)


def _mastare_uppslag(fraga_orig):
    """Fokuserad 'vem vann'-vag: sok pa den senaste vinnaren och extrahera
    BARA namnet (inte en beskrivning av tavlingen). Returnerar vinnarnamn
    eller None (varvid vanliga pipelinen tar over)."""
    if not _SOKNING_TILLGANGLIG:
        return None
    if not _MASTAR_FRAGA.search(_normalisera_for_matchning(fraga_orig)):
        return None
    ar = datetime.date.today().year
    fraga_en = fraga_orig
    try:
        if detektera_sprak(fraga_orig) != "en":
            fraga_en = _oversatt_till_engelska(fraga_orig)
    except Exception:
        pass
    # EN enda sokning (flera varianter hamrade Brave -> rate-limit/mycket
    # langsamt). Fokuserad fraga; faller tillbaka pa raa fragan om tom.
    kontext = None
    for q in (f"{fraga_en} latest winner", fraga_en):
        try:
            kontext = sok_pa_natet(q, max_resultat=6, oversatt=False)
        except Exception:
            kontext = None
        if kontext:
            break
    if not kontext:
        return None
    kontext = kontext[:2500]
    try:
        prompt = bygg_qwen_prompt(
            "From the web search results below, identify WHO WON the "
            "competition or award the question is about (the most recent "
            "completed edition). Reply with ONLY the winner's name (person "
            "or team) and the year in parentheses - nothing else, no "
            "description. A winner's name is almost always present - extract "
            "the most recent one. Reply UNKNOWN only if there is genuinely "
            "no person or team name anywhere in the results.",
            f"Question: {fraga_en}\n\nSearch results:\n{kontext}\n\nWinner:")
        out = llm_sok(prompt, max_tokens=24, temperature=0.0,
                      stop=_QWEN_STOPP + ["\n"])
        namn = out["choices"][0]["text"].strip().strip('."\' ')
    except Exception:
        return None
    if (not namn or namn.upper().startswith("UNKNOWN")
            or len(namn) > 60 or len(namn) < 2):
        return None
    _debug_logg(f"mastare-uppslag -> {namn!r}")
    return namn


# ============ KOD-LAGE: ROUTER (9B med kod-systemprompt) ============
# Kodfragor besvaras av samma 9B-modell som allt annat, men med en
# kod-inriktad systemprompt (_KOD_SYSTEM). Ingen modellvaxling langre.
# _SENAST_KOD ar en enkel hysteres-flagga: en kort foljdfraga ('gor den
# snabbare') efter en kodfraga halls kvar i kod-lage.

_SENAST_KOD = False

_KOD_VERB = (r"skriv\w*|g[oö]r\w*|skapa\w*|generer\w*|fixa\w*|r[aä]tta\w*|"
             r"optimer\w*|implementer\w*|koda\w*|programmer\w*|debugg\w*|"
             r"fels[oö]k\w*|refaktor\w*|kompiler\w*|f[oö]rklar\w*|betyd\w*|"
             r"write|creat\w*|generat\w*|\bfix\w*|debug\w*|implement\w*|"
             r"refactor\w*|optimi\w*|explain\w*")
_KOD_NOUN = (r"funktion\w*|klass\w*|metod\w*|skript\w*|program\w*|kod\w*|"
             r"algoritm\w*|loop\w*|regex\w*|reg(exp|uljärt)\w*|array\w*|"
             r"variabl?\w*|k[aä]llkod\w*|query|queries|endpoint\w*|\bapi\b|"
             r"quicksort|bugg\w*|function\w*|class\w*|method\w*|"
             r"script\w*|\bcode\w*|algorithm\w*")
_KOD_SPRAK = (r"python|javascript|\bjs\b|typescript|\bjava\b|c\+\+|\bc#\b|"
              r"csharp|\brust\b|golang|\bphp\b|\bruby\b|kotlin|\bswift\b|"
              r"\bsql\b|\bhtml\b|\bcss\b|\bbash\b|powershell|\breact\b|"
              r"node\.?js")
_KOD_STARK = re.compile(
    r"```"
    r"|\b(debugg\w*|fels[oö]k\w*|refaktor\w*|kompiler\w*|stack ?trace|"
    r"traceback|syntax ?error|kompileringsfel|segfault|null ?pointer|"
    r"pip install|npm (install|run)|git (commit|push|clone)|dockerfile|"
    r"programmer\w*|reguljärt uttryck|regex\w*)\b"
    r"|\b(?:" + _KOD_VERB + r")\b[^.?!\n]{0,45}\b(?:" + _KOD_NOUN + r")\b"
    r"|\b(?:" + _KOD_NOUN + r")\b[^.?!\n]{0,45}\b(?:" + _KOD_VERB + r")\b"
    r"|\b(?:" + _KOD_VERB + r")\b[^.?!\n]{0,45}\b(?:" + _KOD_SPRAK + r")\b"
    r"|\b(?:" + _KOD_SPRAK + r")\b[^.?!\n]{0,35}\b(?:kod\w*|program\w*|"
    r"funktion\w*|loop\w*|" + _KOD_NOUN + r"|" + _KOD_VERB + r")\b",
    re.IGNORECASE)


def _ar_kodfraga(fraga):
    """Router: True om fragan ar en programmerings-/kodfraga. Konservativ
    mot falska positiva sa vanliga fragor inte utloser en dyr modellvaxling.
    (Validerad mot testset: 0 missar, 0 falska positiva.)"""
    return bool(_KOD_STARK.search(fraga or ""))


def _ar_kod_foljdfraga(fraga):
    """Hysteres i kod-lage: en kort/tillbakasyftande foljdfraga ('gor den
    snabbare', 'forklara rad 3') halls kvar i kod-lage aven utan explicita
    kod-ord, sa en kodsession inte staller om till 9B mitt i."""
    norm = _normalisera_for_matchning(fraga)
    n = len(norm.split())
    if n == 0 or n > 10:
        return False
    return bool(_FOLJD_PRONOMEN.search(norm)) or n <= 4


_KOD_SYSTEM = (
    "You are an expert programming assistant. "
    "Write correct, idiomatic, working code. Give a short explanation and a "
    "complete solution inside a fenced code block. IMPORTANT: write the "
    "explanation text in the SAME language as the user's question (English "
    "question -> English explanation, Swedish question -> Swedish "
    "explanation, and so on). Always keep code, keywords and identifiers "
    "unchanged - never translate code. Do not repeat any sentence or "
    "paragraph. Never invent APIs or libraries that do not exist. Be concise."
)


def _dedup_stycken(text):
    """Tar bort ett stycke som upprepas OMEDELBART identiskt (Coder-modellen
    dubblerade ibland forklaringen). Ror inte kodblock/olika stycken."""
    if not text:
        return text
    delar = re.split(r"\n\s*\n", text)
    ut = []
    for d in delar:
        if ut and d.strip() and d.strip() == ut[-1].strip():
            continue
        ut.append(d)
    return "\n\n".join(ut)


def _svara_kod(fraga, historik, svarssprak=None):
    """Genererar ett kodsvar med 9B-modellen och kod-systemprompten
    (_KOD_SYSTEM). Ingen webbsok - forklaringen halls pa anvandarens sprak
    via systemprompten. Historik ger foljdfragor ('gor den snabbare')
    kontext. Storre max_tokens for langre kod."""
    try:
        # Sprakspegling styrs nu helt av _KOD_SYSTEM ("samma sprak som
        # fragan") - detektera-och-injicera togs bort da sprakdetektorn
        # ibland fellaste svenska kodfragor som engelska. svarssprak-
        # parametern behalls for bakatkompatibilitet men anvands ej.
        prompt = bygg_llama_prompt_med_historik(_KOD_SYSTEM, historik, fraga)
        _out = llm(prompt, max_tokens=1536, stop=_LLAMA_STOPP,
                   temperature=0.2, repeat_penalty=1.05, top_p=0.9)
        svar = _strip_think(_out["choices"][0]["text"]).strip()
        if not svar:
            _out = llm(prompt, max_tokens=1536, stop=_LLAMA_STOPP,
                       temperature=0.4, repeat_penalty=1.1, top_p=0.9)
            svar = _strip_think(_out["choices"][0]["text"]).strip()
        return _dedup_stycken(svar) or "[Kunde inte generera ett kodsvar - forsok igen.]"
    except Exception as _e:
        return f"[Fel i kod-laget: {_e}]"


# Detalj-/resultatfragor som Wikipedias INGRESS sallan svarar pa (svaret
# ligger i en tabell langre ned) -> eskalera till djup-vagen (hamta hela
# sidan). Loser t.ex. "vem toppade medaljligan vid OS 2026".
_RESULTAT_DETALJ = re.compile(
    r"medal\s*table|medalj(lig\w*|tabell\w*)|"
    r"who\s+(topped|won|led|finished)|vem\s+(toppade|vann|ledde|slutade)|"
    r"final\s+(score|standings|result)|slutresultat|slutst[aä]llning\w*|"
    r"\bstandings\b|\branking\b|topplacer\w*|"
    r"how\s+many\s+(medals|golds|goals|points|wins)|"
    r"hur\s+m[aå]nga\s+(medaljer|guld|m[aå]l|po[aä]ng|vinster)",
    re.IGNORECASE)


# SPORT-/UTMARKELSE-RESULTAT: "vem vann senaste X". Wiki-ingressen ger ofta
# bara en DEFINITION av tavlingen ("Ballon d'Or ar ett arligt pris ...") i
# stallet for vinnaren, sa vi styr om till en steerad djup-sokning pa
# VINNAREN. Kraver bade ett tavlings-/utmarkelse-substantiv OCH en
# resultat-framing, sa vanliga fragor ("forklara offside i Premier League")
# inte trilar in.
_SPORT_EVENT = re.compile(
    r"ballon\s*d.?or|bal[oó]n\s*de\s*oro|golden\s*ball|"
    r"serie\s*a|premier\s*league|la\s*liga|bundesliga|ligue\s*1|allsvenskan|"
    r"eredivisie|scudetto|"
    r"champions\s*league|europa\s*league|world\s*cup|v[aä]rldsm[aä]sterskap\w*|"
    r"\bvm\b|\bem\b|mundial|coupe\s*du\s*monde|weltmeister\w*|"
    r"tour\s*de\s*france|giro\b|vuelta|wimbledon|roland\s*garros|us\s*open|"
    r"australian\s*open|grand\s*slam|formula\s*1|formel\s*1|\bf1\b|"
    r"grand\s*prix|super\s*bowl|\bnba\b|\bnhl\b|stanley\s*cup|ryder\s*cup|"
    r"olympi\w*|\bos\b|nobel\w*|oscar\w*|academy\s*award\w*|eurovision|"
    r"campionato|campeonato|gran\s*premio|mästerskap\w*|championship",
    re.IGNORECASE)
_SPORT_FRAMING = re.compile(
    r"\b(who|vem|chi|qui|qui[eé]n|quem|wer|wie)\b.{0,45}\b(won|win|winner|"
    r"vann|vinnare|vinner|vinto|vinse|remport\w*|gagn\w*|gan[oó]|ganou|"
    r"gewann|gewonnen|champion|m[aä]stare|campion\w*|campe[oó]n|campe[aã]o|"
    r"vincitore|vainqueur|topped)\b"
    r"|last\s+(winner|champion)|latest\s+(winner|champion)|"
    r"most\s+recent\s+(winner|champion)|senaste\s+(vinnare|m[aä]star\w*)|"
    r"\b(last|latest|most recent|senaste|ultim\w*|dernier\w*|letzt\w*|"
    r"[uú]ltim\w*)\b.{0,25}\b(won|win|vann|title|titel|championship|"
    r"m[aä]sterskap\w*)",
    re.IGNORECASE)


def _ar_sport_resultat(fraga_norm):
    return bool(_SPORT_EVENT.search(fraga_norm)
                and _SPORT_FRAMING.search(fraga_norm))


def _djup_pa_farsk_sok(fraga_en, steer="", sasong=False):
    """Djup-vag pa en FARSK Brave-sokning: sok -> hamta-och-las hela
    sidorna -> ranka passager -> extrahera fakta. For detalj-/resultat-
    fragor dar wiki-ingressen inte racker. Grundad fakta eller None.
    'steer' laggs till i SOKFRAGAN (t.ex. 'winner champion final result')
    for att styra Brave mot sjalva resultatet i stallet for en definition.
    A4: 'sasong=True' for lig-/serie-sport dar en sasong loper over tva ar
    (2025-26) - da soker vi pa sasongsstrangar sa vi traffar den SENAST
    AVSLUTADE sasongens tabell i stallet for en gammal sida (Serie A lag
    annars en sasong efter)."""
    if retrieval_deep is None:
        return None
    try:
        _ar = datetime.date.today().year
        _q = (fraga_en + " " + steer).strip() if steer else fraga_en
        traffar = None
        if sasong:
            # senast avslutade sasongen forst (ar-1 / ar), sedan innevarande.
            _sasonger = [f"{_ar - 1}-{str(_ar)[2:]}", f"{_ar - 1}/{str(_ar)[2:]}",
                         f"{_ar}-{str(_ar + 1)[2:]}", f"{_ar}/{str(_ar + 1)[2:]}"]
            for _ss in _sasonger:
                traffar = _brave_sok(f"{_q} {_ss} season final standings winner",
                                     count=5)
                if traffar:
                    _debug_logg(f"A4 sasongssok traff pa {_ss!r}")
                    break
        if not traffar:
            traffar = (_brave_sok(f"{_q} {_ar}", count=5)
                       or _brave_sok(f"{_q} {_ar - 1}", count=5)
                       or _brave_sok(_q, count=5))
        if not traffar:
            return None
        dj = retrieval_deep.djup_kontext(fraga_en, traffar, n_sidor=4, topp=6)
        if not dj:
            return None
        # A4: for sasongssport - be extraktionen explicit om den SENAST
        # AVSLUTADE sasongens vinnare/tabell, inte en pagaende eller aldre.
        _fraga_x = (fraga_en + " (give the winner/result of the most recently "
                    "COMPLETED season, not an older season or an ongoing one)"
                    if sasong else fraga_en)
        f = extrahera_fakta(_fraga_x, dj)
        if f and not _fakta_ar_oklar(f):
            return f
    except Exception as e:
        _debug_logg(f"_djup_pa_farsk_sok fel: {e!r}")
    return None


def _verifiera_svar(svar, kontext, svarssprak, fraga=None):
    """SJALV-VERIFIERING (djup-vag, steg 2): laser UTKASTET mot de hamtade
    KALLORNA och korrigerar faktafel - sarskilt personers namn, deras
    roller/titlar, vem som leder/innehar en position, partitillhorighet och
    datum. Kors BARA nar vi har riktig kontext att stamma av mot (sok/djup),
    sa den snabba lokala vagen ar oforandrad. Ett enda extra modellanrop
    (~nagra sekunder) - medveten avvagning: langsammare men mer korrekt pa
    svara/faktatunga fragor. Vid minsta osakerhet returneras originalsvaret
    oforandrat (aldrig samre an utan verifiering)."""
    try:
        if not svar or not kontext:
            return svar
        kontext_kort = kontext if len(kontext) <= 4000 else kontext[:4000]
        sys = (
            "You are finalizing an assistant's answer. You are given the "
            "QUESTION, SOURCE excerpts, and a DRAFT answer. Do ALL of the "
            "following, then output only the final answer:\n"
            "1. ANSWER THE QUESTION. If the draft is a generic definition or "
            "description of the topic (e.g. it explains what an office or a "
            "competition IS) instead of answering the specific question, "
            "REWRITE it to answer the question directly using the sources - "
            "name the specific person, winner, value or fact asked for. If "
            "the sources do NOT contain the specific answer, reply with ONE "
            "short honest sentence saying you don't have that specific "
            "information; do NOT pad it with a definition.\n"
            "1b. ANSWER FIRST. The VERY FIRST sentence must state the direct "
            "answer (the name / value / winner asked for). Any definition, "
            "history or background may come only AFTER that, and only if it "
            "adds value. NEVER open with a general description of the topic "
            "and bury the answer at the end - if the draft does that, move "
            "the answer to the front and cut most of the preamble.\n"
            "2. FACT-CHECK. Correct any name, title, who-holds-which-"
            "position, party affiliation, winner or date that contradicts "
            "the sources. Do NOT add claims the sources don't support.\n"
            "3. LANGUAGE. Write the final answer in the SAME language as the "
            "QUESTION.\n"
            "Keep the draft's tone and formatting where it is already "
            "correct. Output ONLY the final answer - no preamble, no notes.\n\n"
            "QUESTION:\n" + (fraga or "") + "\n\nSOURCES:\n" + kontext_kort +
            "\n\nDRAFT:\n" + svar
        )
        sys += _sprak_direktiv(svarssprak)
        prompt = bygg_llama_prompt(sys, "Return the final answer.")
        korr = _generera(prompt, temperature=0.1, repeat_penalty=1.1)
        # Behall korrigeringen om den ar icke-tom och inte stympad. OBS:
        # tillat AVSEVART kortare svar - en dump som ersatts av ett kort
        # arligt "vet ej" eller ett extraherat kort faktum ar onskat.
        if korr and len(korr.strip()) >= 6:
            _debug_logg("svarskontrakt: utkast granskat mot fraga+kallor")
            return korr
    except Exception as _ve:
        _debug_logg(f"verifiering fel: {_ve!r}")
    return svar


def _sprakgrind(svar, malsprak_namn):
    """UTGANGSGRIND (sprak): sakerstaller att svaret ar pa ANVANDARENS
    sprak. Fangar de fall dar pipelinen speglar tillbaka fel sprak (t.ex.
    fransk fraga -> spanskt svar, finsk fraga -> engelskt svar). Kors pa
    ALLA svar via wrappern nedan. Billig lingua-detektering forst; oversatter
    bara vid tydlig missmatch. Hoppar korta svar och kodblock."""
    try:
        if not svar or not malsprak_namn:
            return svar
        s = svar.strip()
        if len(s) < 40 or "```" in svar:
            return svar
        mal = malsprak_namn.strip()
        if mal.lower() == "english":
            return svar  # engelska svar speglas sallan fel; spara kostnad
        # GREPP 3: INGEN "a-ring"-genvag langre - danska/norska anvander ocksa
        # a-ring, sa den slapper igenom sv->da/no-drift.
        # Detektera svarets FAKTISKA sprak med lingua (palitligt for medellanga+
        # texter). Redan malspraket -> klart. Skiljer det sig (danska nar mal ar
        # svenska, spanska nar mal ar portugisiska) -> oversatt och NAMN
        # kallspraket explicit. Att namna kallan ('this is Danish, translate to
        # Swedish') ar mycket starkare an ett vagt grannsprak-varning, som
        # modellen ignorerar for nara sprak.
        kalla = None
        _det = _lingua_detektor()
        if _det and len(s) >= 60:
            try:
                _spr = _det.detect_language_of(s)
                if _spr:
                    kalla = _LINGUA_TILL_NAMN.get(_spr.name)
            except Exception:
                kalla = None
        if kalla and kalla.lower() == mal.lower():
            return svar  # redan pa ratt sprak - hoppa (kostnad + ingen risk)
        if kalla and kalla.lower() != mal.lower():
            _instr = (f"The text below is written in {kalla}. Translate the "
                      f"ENTIRE text into natural, correct {mal}. Translate "
                      f"every single word - leave NO {kalla} words. Keep the "
                      f"meaning and the formatting. Reply with ONLY the {mal} "
                      f"text - no notes, no explanation.")
        else:
            _instr = (f"The text below must be written in {mal}. If it is "
                      f"ALREADY fully and correctly in {mal}, repeat it "
                      f"EXACTLY. Otherwise translate the ENTIRE text into "
                      f"natural {mal}, keeping meaning and formatting. Reply "
                      f"with ONLY the {mal} text - no notes.")
        prompt = bygg_qwen_prompt(_instr, s)
        n_tok = min(1300, max(200, int(len(s) * 0.9)))
        out = llm(prompt, max_tokens=n_tok, temperature=0.0, stop=_LLAMA_STOPP)
        ut = out["choices"][0]["text"].strip()
        if ut and len(ut) >= 0.5 * len(s):
            if ut != s:
                _debug_logg(f"sprakgrind: sakerstallde {mal}")
            return ut
    except Exception as _se:
        _debug_logg(f"sprakgrind fel: {_se!r}")
    return svar


_SVARA_FORST_FRAGEORD = re.compile(
    r"^\s*\W*(vem|vad|vilken|vilket|who|what|which|wer|was|welche[rs]?|"
    r"wie|qui|quel|quelle|quels|quelles|chi|quien|quién|cual|cuál|kto|"
    r"kdo|ile|hvem|hvad|hvilken|wat)\b", re.IGNORECASE)


def _svara_forst_grind(svar, fraga):
    """GREPP 2 (svara-forst): for en KORT-SVARS-fraga ('vem/vad ar ...') dar
    svaret blev LANGT och inleds med en DEFINITION i stallet for svaret -
    lyft fram svaret till forsta meningen. Loser 'paus'-dumpen: ratt namn
    (Leo XIV) begravt sist efter ett stycke om ambetet. Mycket konservativ:
    kors bara nar (a) fragan ar en vem/vad-fraga OCH (b) svaret ar langt.
    Korta svar, essaer och foljdfragor rors aldrig."""
    try:
        if not svar or not fraga:
            return svar
        s = svar.strip()
        # bara langa svar (korta faktasvar ar redan svara-forst) och ingen kod
        if len(s) < 240 or "```" in s:
            return svar
        if not _SVARA_FORST_FRAGEORD.match(fraga.strip()):
            return svar
        prompt = bygg_llama_prompt(
            "Reorder the ANSWER so its FIRST sentence directly answers the "
            "QUESTION (state the name / value / person asked for right away). "
            "Keep every fact, keep the SAME language, keep useful context but "
            "move it AFTER the direct answer. If the answer ALREADY opens with "
            "the direct answer, return it completely unchanged. Output ONLY "
            "the answer text - no preamble.\n\n"
            "QUESTION: " + fraga.strip() + "\n\nANSWER: " + s,
            "Return the answer.")
        ut = _generera(prompt, temperature=0.0, repeat_penalty=1.1)
        if ut and len(ut.strip()) >= 0.4 * len(s):
            ut = ut.strip()
            if ut != s:
                _debug_logg("svara-forst: lyfte fram svaret ur dump")
            return ut
    except Exception as _af:
        _debug_logg(f"svara-forst fel: {_af!r}")
    return svar


def fraga_ai(fraga, status_callback=None, historik=None, token_callback=None):
    """UTGANGSGRIND-wrapper: kor sjalva pipelinen (_fraga_ai_inner) och lagger
    en svara-forst-grind + sprakgrind pa slutsvaret. All ovrig logik ligger i
    _fraga_ai_inner."""
    svar = _fraga_ai_inner(fraga, status_callback=status_callback,
                           historik=historik, token_callback=token_callback)
    try:
        # Svara-forst kors BARA pa enskilda faktafragor (inte foljdfragor i
        # ett samtal - dar ar 'vem/vad' sallan en uppslagsfraga).
        if not historik:
            svar = _svara_forst_grind(svar, _ratta_vanliga_stavfel(fraga))
    except Exception:
        pass
    try:
        mal = detektera_svarssprak_namn(_ratta_vanliga_stavfel(fraga))
        svar = _sprakgrind(svar, mal)
    except Exception:
        pass
    return svar


# GREPP 3: kvantitativ foljdfraga i ett samtal - "hur mycket blir det pa ett
# ar", "och per person?" - bygger pa siffror som redan namnts. Ska besvaras
# LOKALT med resonemang, ALDRIG via retrieval (som annars avstar nar soket ar
# tomt: "kunde inte hitta nagot palitligt svar"). Loste #155 i 200-testet.
_KVANT_FOLJD = re.compile(
    r"\b(hur\s+(mycket|manga)|how\s+(much|many)|wie\s+viel\w*|combien|"
    r"cuant[oa]s?|quant[oie]|hoeveel|total\w*|tillsammans|altogether|"
    r"\beach\b|per\s+(person|manad|month|ar|year|styck|dag|day|nott|semaine|"
    r"woche|settimana)|nach\s+einem\s+jahr|por\s+persona|a\s+testa)\b",
    re.IGNORECASE)
_REALTID_ORD = re.compile(
    r"\b(kostar\s+(nu|idag)|\bprice\b|pris\s+(nu|idag)|today|idag|"
    r"aktuell\w*|latest|senaste|\bvader\b|weather|temperatur\w*|\bkurs\b|"
    r"just\s+nu|right\s+now|currently|\bnu\b|heute|aujourd|hoy|oggi)\b",
    re.IGNORECASE)


def _ar_kvantitativ_foljdfraga(fraga, historik):
    if not historik:
        return False
    norm = _normalisera_for_matchning(fraga)
    if not _KVANT_FOLJD.search(norm) or _REALTID_ORD.search(norm):
        return False
    # Maste bygga pa tidigare siffror (annars ar det ingen berakning).
    recent = " ".join((h.get("svar", "") + " " + h.get("fraga", ""))
                      for h in historik[-2:])
    return bool(re.search(r"\d", recent))


# GREPP 4: SYNTES-TUR. Pa den avslutande "sammanfatta / satt ihop / gor en
# pitch / en plan"-turen finns inget nytt att SOKA - materialet ligger i
# konversationen. Kors retrieval anda far den antingen tomt (-> avstar: "finns
# inte i kallmaterialet") eller ett IRRELEVANT dokument (-> hallucinerar, t.ex.
# en SaaS-trad som slutar i "vertical gardening"). Losning: kanna igen syntes-
# avsikt i ett pagaende samtal och svara ENBART fran historiken, aldrig via sok.
_SYNTES_INTENT = re.compile(
    r"\b(sammanfatta\w*|sammanfattning\w*|satt\s+ihop|summera\w*|summari[sz]e|"
    r"sum\s+up|wrap\s+(it\s+)?up|put\s+(it\s+|this\s+)?together|assemble|"
    r"in\s+short|elevator\s*pitch|\bpitch\b|to\s+summari[sz]e|"
    r"en\s+(kort\s+|enkel\s+)?plan\b|veckoplan\w*|veckoschema\w*|"
    r"\d+\s*-?\s*veckors?plan|weekly\s+(plan|schedule)|action\s+plan|"
    r"zusammenfass\w*|resume\w*|résum\w*|resumen|resumir|riassum\w*|"
    r"utifran\s+(allt|det)\s+vi|based\s+on\s+(all|everything|our)|"
    r"allt\s+vi\s+(pratat|diskuterat)|allt\s+ovan)\b",
    re.IGNORECASE)


# Smal realtids-vakt (INTE hela _REALTID_ORD - dess "nu" fangar fel, t.ex.
# "sammanfatta NU hela affarsiden"). Bara nar fragan uttryckligen vill ha
# FARSK extern info hoppar vi syntes-vagen och behaller sok.
_SYNTES_REALTID = re.compile(
    r"\b(senaste\s+nytt|latest\s+news|aktuell\w*\s+(kurs|pris|läge|status|"
    r"nyheter)|dagens\s+(kurs|pris|nyheter)|current\s+(price|news|status))\b",
    re.IGNORECASE)


def _ar_syntesfraga(fraga, historik):
    if not historik:
        return False
    norm = _normalisera_for_matchning(fraga)
    if _SYNTES_REALTID.search(norm):
        return False
    return bool(_SYNTES_INTENT.search(norm))


def _svara_syntes(fraga, historik, svarssprak):
    try:
        _sys = ("You are wrapping up THIS conversation. The user asks you to "
                "summarize, assemble or condense what has ALREADY been "
                "discussed above into the exact form they request (a summary, "
                "an elevator pitch, a plan, a schedule). Use ONLY information "
                "from the conversation above - the business, product, numbers "
                "or situation the user described earlier. Do NOT search for or "
                "invent any new company, product, source, fact or example, and "
                "NEVER say the information is missing - it is in the "
                "conversation. Produce exactly what they asked for, directly.")
        _sys += _sprak_direktiv(svarssprak)
        # KOMPRIMERA historiken: behall VARJE turs hela FRAGA (kort och
        # ramar in amnet - t.ex. C5:s tur 1 beskriver hela produkten) men
        # korta ner de langa SVAREN. Sa ryms HELA samtalets bage inom
        # kontextfonstret - annars beskar historik-trunkeringen bort tur 1
        # (setupen) och modellen avstod ("details were not in our history").
        _komp = []
        for _h in (historik or []):
            _sv = (_h.get("svar") or "")
            if len(_sv) > 420:
                _sv = _sv[:420].rsplit(" ", 1)[0] + " […]"
            _komp.append({"fraga": _h.get("fraga", ""), "svar": _sv})
        _prompt = bygg_llama_prompt_med_historik(_sys, _komp, fraga)
        _ut = _generera(_prompt, temperature=0.4, repeat_penalty=1.1)
        if _ut and len(_ut.strip()) >= 20 and not ar_svaret_trasigt(_ut):
            _debug_logg("syntes-tur -> lokalt svar fran historik (ingen sok)")
            return _ut.strip()
    except Exception as _sye:
        _debug_logg(f"syntes-tur fel: {_sye!r}")
    return None


# ============================================================
# GREPP 5: AVSIKTSROUTING (begrepp + rad/asikt)
# Klassificerar turen och valjer vag. Loser tva monster fran de 8
# konsumentsamtalen: (1) BEGREPPS-VAGRAN - "forklara hur styrrantan
# fungerar" mottes av system-promptens "jag svarar bara pa regel-/
# sifferbaserade fragor"; (2) RAD-OVERVARSAMHET - "borde jag starta
# X?" drunknade i forbehall. Bada besvaras nu lokalt ur modellens
# kunnande FORE retrieval. Gatas av 'inte snabbcheck_aktuell_info'
# sa aktuella fragor (dagskurs, senaste nytt) gar vidare till sok.
# ============================================================

# Begreppsfraga: starka forklarings-verb pa flera sprak. INTE bara
# "vad ar" - det skulle sluka faktafragor som bor ga till sok.
# OBS: matchas mot _normalisera_for_matchning() (foldad ASCII, gemener)
# sa monstren skrivs UTAN diakritik: "forklara" traffar aven "förklara".
_BEGREPP_INTENT = re.compile(
    r"(forklara|forklar\b|forklarar|hur fungerar|hur funkar|hur gar det till|"
    r"vad menas med|vad innebar|vad betyder|hur kommer det sig|"
    r"explain|how does|how do |how is it|what does .*mean|why (do|does|is|are)|"
    r"erklar|erklare|wie funktioniert|was bedeutet|"
    r"explique|comment fonctionne|"
    r"spiega|come funziona|"
    r"explica|como funciona)")

# Rad/asikt/brainstorm: den ber om ett omdome/forslag/plan, inte en fakta.
# Aven "om du fick / vad skulle du valja / hur skulle en ... se ut"-framing.
_RAD_INTENT = re.compile(
    r"(borde jag|ska jag |bor jag |vad tycker du|vad anser du|vad tror du om|"
    r"ge mig (tips|rad|forslag|ideer|ideer)|tips pa|forslag pa|"
    r"hjalp\w* mig|brainstorma|pitcha|"
    r"vilka (ar )?fordelar|for- och nackdelar|vad ar basta|basta sattet|"
    r"om du fick|om du var |om du skulle|om du vore|"
    r"vad skulle du|vilka skulle du|hur skulle du|hur skulle en|hur skulle man|"
    r"ga igenom|vad kravs for|hur borde|hur ska jag|hur bor jag|"
    r"should i |do you think|what do you think|give me (tips|advice|ideas|suggestions)|"
    r"help me\b|brainstorm|pitch me|recommend|what.s the best|best way to|"
    r"how should i|if you (were|had to)|what would you|which would you|how would you)")

# Fermi-/uppskattningsfragor ('gissa och forklara hur manga ...'). De bar
# ofta "forklara"/"explain" men ska INTE till begrepps-/radvagen - de har
# sin egen resonerande gissningshantering langre ner i pipelinen.
_EST_GISSNING = re.compile(
    r"\bgissa\b|\bguess\b|uppskatta|\bestimat|approxim|\bdevine\b|"
    r"fais une estimation|\bstima\b|schatz|"
    r"roughly how many|how many .{0,40}(fit|do you think)|"
    r"hur manga .{0,40}(tror du|far plats|ryms)")


def _ar_begreppsfraga(fraga):
    try:
        if snabbcheck_aktuell_info(fraga):
            return False
        _n = _normalisera_for_matchning(fraga or "")
        # Spekulativa Fermi-/gissningsfragor ("gissa och forklara",
        # "uppskatta") har sin egen vag (tvingar sok for att grunda
        # gissningen) - lat dem inte fangas har bara for att de bar
        # ordet "forklara"/"explain".
        if _ar_spekulativ_fraga(_n) or _EST_GISSNING.search(_n):
            return False
        return bool(_BEGREPP_INTENT.search(_n))
    except Exception:
        return False


def _ar_radsfraga(fraga):
    try:
        if snabbcheck_aktuell_info(fraga):
            return False
        _n = _normalisera_for_matchning(fraga or "")
        if _ar_spekulativ_fraga(_n) or _EST_GISSNING.search(_n):
            return False
        return bool(_RAD_INTENT.search(_n))
    except Exception:
        return False


def _svara_begrepp(fraga, historik, svarssprak):
    # Egen system-prompt UTAN regel/siffer-vagran - modellen ska
    # forklara begreppet pedagogiskt ur eget kunnande.
    try:
        _sys = ("You are a knowledgeable, clear-spoken assistant. The user is "
                "asking you to EXPLAIN a concept or how something works. Answer "
                "directly and pedagogically from your own general knowledge. "
                "Give a concrete, well-structured explanation, with a simple "
                "everyday example where it helps understanding. Do NOT refuse, "
                "do NOT say you only answer rule- or number-based questions, and "
                "do NOT claim you lack current data - a concept does not require "
                "live data. Keep it focused and genuinely useful.")
        _sys += _sprak_direktiv(svarssprak)
        _prompt = bygg_llama_prompt_med_historik(_sys, historik, fraga)
        _ut = _generera(_prompt, temperature=0.4, repeat_penalty=1.1)
        if _ut and len(_ut.strip()) >= 20 and not ar_svaret_trasigt(_ut):
            _debug_logg("avsiktsrouting: begrepp -> lokalt svar")
            return _ut.strip()
    except Exception as _be:
        _debug_logg(f"begreppsvag fel: {_be!r}")
    return None


def _svara_rad(fraga, historik, svarssprak):
    # Resonerat, beslutsamt rad + hogst ETT latt forbehall. Ingen
    # mur av varningar och ingen "jag svarar bara pa fakta"-vagran.
    try:
        _sys = ("You are a thoughtful, decisive advisor. The user wants your "
                "reasoning, advice, opinion or ideas - not a wall of caveats. "
                "Give a clear, concrete, actionable answer: take a position, "
                "lay out the key trade-offs briefly, and end with a practical "
                "recommendation or a few concrete ideas. You MAY add at most ONE "
                "short caveat, and only if it is genuinely important. Do NOT "
                "refuse, do NOT hedge excessively, and do NOT say you only answer "
                "factual questions. Be specific and useful.")
        _sys += _sprak_direktiv(svarssprak)
        _prompt = bygg_llama_prompt_med_historik(_sys, historik, fraga)
        _ut = _generera(_prompt, temperature=0.6, repeat_penalty=1.1)
        if _ut and len(_ut.strip()) >= 20 and not ar_svaret_trasigt(_ut):
            _debug_logg("avsiktsrouting: rad/asikt -> lokalt svar")
            return _ut.strip()
    except Exception as _re:
        _debug_logg(f"radvag fel: {_re!r}")
    return None


# ============================================================
# A1: RELEVANSGRIND PA HAMTAD SOK-KONTEXT
# Retrieval kan blanda in ett stycke fran en helt orelaterad kalla
# (t.ex. ett cancer-vaccin-stycke mitt i ett energisvar). Vi delar
# kontexten i kall-block, poangsatter varje block pa innehallsord-
# overlapp mot fragan och slapper igenom bara de relevanta.
# Sakerhetsnat: filtrerar ALDRIG bort allt, och ror inte kontexten alls
# om den inte gar att bedoma palitligt.
# ============================================================
_STOPP_ORD_REL = set((
    "the a an of to in on at for and or but is are was were be been being "
    "this that these those with from by as it its into about over under "
    "after before between not no yes do does did have has had will would "
    "en ett och eller men som att till med pa av om det den de har ar var "
    "vad vem vilka vilken hur nar varfor jag du vi han hon kan ska vill "
    "maste finns vara blir mer mest ocksa bara samt kring mot enligt"
).split())


def _innehallsord(text):
    return {w for w in re.findall(
        r"[a-z0-9]{3,}", _normalisera_for_matchning(text or ""))
        if w not in _STOPP_ORD_REL}


def _rel_traff(q_ord, b_ord):
    # Antal fragord som "finns" i blocket. Prefix-tolerant sa "hydro" i
    # fragan traffar "hydropower" i kallan (och tvartom) - annars sallar
    # grinden bort relevanta block pa ren ordbojning.
    n = 0
    for q in q_ord:
        for b in b_ord:
            if q == b or (len(q) >= 4 and len(b) >= 4
                          and (b.startswith(q) or q.startswith(b))):
                n += 1
                break
    return n


def _relevansfiltrera_kontext(sok_kontext, fraga):
    # KONSERVATIV: slapper bara block som ar HELT orelaterade (noll
    # fragord-traffar) och bara nar minst ett block ar tydligt relevant
    # (>=2 traffar). Sa vi tar bort blodning utan att beskara ett
    # relevant-men-annorlunda-formulerat block.
    try:
        if not sok_kontext or not isinstance(sok_kontext, str):
            return sok_kontext
        q_ord = _innehallsord(fraga)
        if len(q_ord) < 2:
            return sok_kontext  # for tunn fraga -> ingen palitlig grind
        block = [b.strip() for b in re.split(r"\n\s*\n", sok_kontext)
                 if b.strip()]
        if len(block) < 2:
            return sok_kontext  # inget att salla
        poang = [(_rel_traff(q_ord, _innehallsord(b)), b) for b in block]
        maxtraff = max(p for p, _ in poang)
        if maxtraff < 2:
            return sok_kontext  # for otydligt -> ror inte kontexten
        behallna = [b for p, b in poang if p >= 1]  # slapp bara HELT orelaterade
        if not behallna or len(behallna) == len(block):
            return sok_kontext  # inget filtrerades bort
        _debug_logg(f"relevansgrind: {len(block)} block -> "
                    f"{len(behallna)} (max={maxtraff})")
        return "\n\n".join(behallna)
    except Exception as _fe:
        _debug_logg(f"relevansgrind fel: {_fe!r}")
        return sok_kontext


# ============================================================
# A5: DETERMINISTISK KALENDER-/DATUMHJALPARE
# Tva monster som 9B:an klantar till: (1) veckodagsrakning ("onsdag +10
# dagar" -> lordag; #102) och (2) period-konstanter ("hur manga dagar pa
# ett ar" -> 365, dar modellen svarade "en manad har 30 dagar"; #33).
# Vi RAKNAR faktumet deterministiskt (sprakneutralt, pa engelska) och later
# sedan modellen formulera det EXAKTA faktumet pa anvandarens sprak - sa
# svaret blir rimligt bade i sak OCH sprak (viktigt for flersprakstestet).
# ============================================================
_VECKODAG_IDX = {
    "mandag": 0, "tisdag": 1, "onsdag": 2, "torsdag": 3, "fredag": 4,
    "lordag": 5, "sondag": 6,                                    # sv/da/no
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,                     # en
    "montag": 0, "dienstag": 1, "mittwoch": 2, "donnerstag": 3,
    "freitag": 4, "samstag": 5, "sonnabend": 5, "sonntag": 6,    # de
    "lundi": 0, "mardi": 1, "mercredi": 2, "jeudi": 3, "vendredi": 4,
    "samedi": 5, "dimanche": 6,                                  # fr
    "lunes": 0, "martes": 1, "miercoles": 2, "jueves": 3,
    "viernes": 4, "sabado": 5, "domingo": 6,                     # es/pt
    "lunedi": 0, "martedi": 1, "mercoledi": 2, "giovedi": 3,
    "venerdi": 4, "sabato": 5, "domenica": 6,                    # it
    "tirsdag": 1,                                                # da/no tisdag
    "maandag": 0, "dinsdag": 1, "woensdag": 2, "donderdag": 3,
    "vrijdag": 4, "zaterdag": 5, "zondag": 6,                    # nl
    "maanantai": 0, "tiistai": 1, "keskiviikko": 2, "torstai": 3,
    "perjantai": 4, "lauantai": 5, "sunnuntai": 6,              # fi
}
_VECKODAG_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "Saturday", "Sunday"]
# Riktningsord (fram i tiden / bakat i tiden), foldad ASCII.
_DATUM_PLUS = ("after", "later", "hence", "forward", "efter", "senare",
               "framat", "dopo", "despues", "apres", "spater", "nach")
_DATUM_MINUS = ("before", "earlier", "ago", "prior", "fore", "innan",
                "tidigare", "prima", "antes", "avant", "vor")
# Antal dagar knutet till ett dag-ord (undviker att plocka ett arstal).
_DAG_ANTAL = re.compile(
    r"(\d{1,4})\s*(?:days?|dagar|dag|tage|tagen|jours?|dias|giorni|"
    r"dni|paiv\w*|dage|dager)", re.IGNORECASE)


def _datum_fakta(fraga, _n):
    # Returnerar en ENGELSK faktamening, eller None om det inte ar en
    # kalender-/datumfraga vi hanterar deterministiskt.
    # ---- (1) VECKODAGSARITMETIK ----
    _funna = [(m.start(), _VECKODAG_IDX[w])
              for w in _VECKODAG_IDX
              for m in re.finditer(r"\b" + re.escape(w) + r"\b", _n)]
    _m_ant = _DAG_ANTAL.search(_n)
    if _m_ant:
        _antal = int(_m_ant.group(1))
        _tecken = -1 if any(o in _n for o in _DATUM_MINUS) else 1
        if not any(o in _n for o in _DATUM_PLUS) and _tecken == 1:
            # inget uttryckligt riktningsord + inget minus -> tolka bara
            # som framat om det finns "in/om" (t.ex. "in 10 days", "om 10 dagar")
            if not re.search(r"\b(in|om|within|inom)\b", _n):
                _tecken = 1  # standard: framat
        if _funna:
            _bas = min(_funna, key=lambda x: x[0])[1]  # forsta veckodagen
            _bas_namn = _VECKODAG_EN[_bas]
            _res = (_bas + _tecken * _antal) % 7
            _rikt = "after" if _tecken > 0 else "before"
            return (f"{_antal} day(s) {_rikt} {_bas_namn} is "
                    f"{_VECKODAG_EN[_res]}.")
        # ingen namngiven veckodag men "in N days"/"om N dagar" -> fran idag
        if re.search(r"\b(in|om|within|inom)\b", _n) and _antal <= 3660:
            _idag = datetime.date.today()
            _res = (_idag + datetime.timedelta(days=_antal))
            return (f"In {_antal} day(s) from today ("
                    f"{_idag.isoformat()}) it will be "
                    f"{_VECKODAG_EN[_res.weekday()]}, {_res.isoformat()}.")

    # ---- (2) PERIOD-KONSTANTER ("hur manga X i en Y") ----
    # Uteslut riktiga nedrakningar ("dagar kvar till jul") - dem hanterar
    # vi inte deterministiskt.
    if re.search(r"\b(until|till|fram till|kvar|innan|before|jusqu|hasta|"
                 r"bis|fino a|tot)\b", _n) or \
            any(g in fraga for g in ("μέχρι", "μεχρι", "απομέν", "απομεν")):
        return None
    # interrogativ ("how many" m.fl.) ELLER grekiskt "ποσ..."
    _hur_manga = bool(re.search(
        r"how many|hur manga|combien|cuant|quant|wie viele|hoeveel|ile\b|"
        r"kuinka mon|quantos|quantas", _n)) or ("ποσ" in fraga.lower())
    _dagord = re.search(r"\b(days?|dagar|dag|tage|jours?|dias|giorni|"
                        r"paiv\w*|dage|dager)\b", _n) or \
        any(g in fraga for g in ("μέρες", "ημέρες", "μερες", "ημερες"))
    _tim = re.search(r"\b(hours?|timmar|stunden|heures|horas|ore|uur|"
                     r"godzin|tuntia)\b", _n) or ("ώρες" in fraga or "ωρες" in fraga)
    _veckord = re.search(r"\b(weeks?|veckor|wochen|semaines?|semanas|"
                         r"settimane|weken|tygodni|viikko\w*)\b", _n) or \
        any(g in fraga for g in ("εβδομάδες", "εβδομαδες"))
    _manord = re.search(r"\b(months?|manader|monate|mois|meses|mesi|"
                        r"maanden|miesi\w*|kuukau\w*)\b", _n) or \
        any(g in fraga for g in ("μήνες", "μηνες"))
    _ar_period = re.search(r"\b(year|ar|jahr|annee?|ano|anno|jaar|rok|"
                           r"vuo\w*)\b", _n) or \
        any(g in fraga for g in ("χρόνο", "χρονο", "έτος", "ετος", "χρόνος"))
    _vecko_period = re.search(r"\b(week|vecka|woche|semaine|semana|"
                              r"settimana|week|tydzie\w*|viikko\w*)\b", _n) or \
        any(g in fraga for g in ("εβδομάδα", "εβδομαδα"))
    _dag_period = re.search(r"\b(day|dag|tag|jour|dia|giorno|paiva\w*)\b", _n) or \
        any(g in fraga for g in ("μέρα", "μερα", "ημέρα", "ημερα"))
    _tim_period = re.search(r"\b(hour|timme|stunde|heure|hora|ora|uur|"
                            r"godzina|tunti)\b", _n) or ("ώρα" in fraga or "ωρα" in fraga)
    if not _hur_manga:
        # utan tydlig "hur manga"-fras kraver vi tvasidigt (enhet + period)
        # for att undvika falsklarm pa lopande text.
        pass
    # dagar pa ett ar
    if _dagord and _ar_period:
        _m_ar = re.search(r"\b(1[5-9]\d\d|20\d\d|21\d\d)\b", _n)
        if _m_ar:
            _y = int(_m_ar.group(1))
            _skott = (_y % 4 == 0 and (_y % 100 != 0 or _y % 400 == 0))
            return (f"The year {_y} has {366 if _skott else 365} days"
                    + (" (it is a leap year)." if _skott else "."))
        return "A year has 365 days (366 in a leap year)."
    if _dagord and _vecko_period:
        return "A week has 7 days."
    if _tim and _dag_period:
        return "A day has 24 hours."
    if _manord and _ar_period:
        return "A year has 12 months."
    if _veckord and _ar_period:
        return "A year has 52 weeks (52 weeks and 1 day)."
    if _dagord and _dag_period:
        return None  # tvetydigt
    if _tim and _tim_period:
        return None
    return None


# ============================================================
# "GÖR INGEN SKADA" (ur A/B-testet mot naken Qwen)
# Fix 1: korta, icke-aktuella faktafragor pa ett ICKE en/sv-sprak
# besvaras NATIVT av modellen, sa de slipper appens engelska retrieval-
# omvag som annars ger exonymer ("Rom" i st.f. "Roma", "Lissabon" i st.f.
# "Lisboa"). Basmodellen ar redan stark pa detta (A/B: vanilla 100 %).
# Begrepps-/radfragor pa dessa sprak fangas redan nativt av Grepp 5, sa
# detta traffar framst enkel faktatrivia (huvudstader, enkla fakta).
# ============================================================
def _svara_nativ_enkel(fraga, svarssprak):
    try:
        _s = (svarssprak or "").strip().lower()
        if _s in ("", "english", "swedish"):
            return None  # en/sv gar redan nativt utan omvag
        if snabbcheck_aktuell_info(fraga):
            return None  # aktuell info -> maste till retrieval
        if len(fraga.split()) > 12:
            return None  # bara KORTA fragor (trivial fakta)
        _sys = ("You are a precise, knowledgeable assistant. Answer the "
                "user's question directly and correctly in ONE short "
                "sentence. Use the natural NATIVE spelling of any place, "
                "person or thing in the user's own language (e.g. 'Roma', "
                "not 'Rome' or 'Rom'). Do not add anything extra.")
        _sys += _sprak_direktiv(svarssprak)
        _p = bygg_llama_prompt(_sys, fraga)
        _ut = _generera(_p, temperature=0.2, repeat_penalty=1.1)
        _ut = (_ut or "").strip()
        if _ut and len(_ut) >= 2 and not ar_svaret_trasigt(_ut):
            _debug_logg("nativ-enkel flerspraksfraga -> lokalt svar")
            return _ut
    except Exception as _ne:
        _debug_logg(f"nativ-enkel fel: {_ne!r}")
    return None


# Fix 2: nar en AKTUELL-fraga gav HELT tomt retrieval (ingen wiki, sok
# eller API) - ge modellens basta MINNESSVAR med ett tydligt forbehall,
# i stallet for ett platt "jag vet inte". Blir battre nar innehavaren
# INTE bytts (t.ex. Frankrikes president = Macron), och forblir arligt
# tack vare forbehallet. Kors ENDAST vid tomt underlag - retrieval-svar
# ror vi aldrig.
def _svara_ur_minne_med_forbehall(fraga, svarssprak, historik):
    try:
        _sys = ("Answer the user's question using your own knowledge. Give "
                "your best, most likely answer DIRECTLY - do NOT refuse and "
                "do NOT say you have no information. Then add ONE short final "
                "sentence noting this is based on your training data, may be "
                "outdated, and should be verified against a current source.")
        _sys += _sprak_direktiv(svarssprak)
        _p = bygg_llama_prompt_med_historik(_sys, historik or [], fraga)
        _ut = _generera(_p, temperature=0.3, repeat_penalty=1.1)
        _ut = (_ut or "").strip()
        if _ut and len(_ut) >= 5 and not ar_svaret_trasigt(_ut):
            _debug_logg("tomt-uppslag -> minnessvar med forbehall")
            return _ut
    except Exception as _me:
        _debug_logg(f"minnesfallback fel: {_me!r}")
    return None


def _svara_datumfraga(fraga, svarssprak):
    try:
        _n = _normalisera_for_matchning(fraga)
        _fakta = _datum_fakta(fraga, _n)
        if not _fakta:
            return None
        _sys = ("Restate the following fact as ONE short, natural sentence and "
                "nothing else - no caveats, no extra information, no preamble.\n"
                "FACT: " + _fakta)
        _sys += _sprak_direktiv(svarssprak)
        _p = bygg_llama_prompt(_sys, fraga)
        _ut = _generera(_p, temperature=0.0, repeat_penalty=1.1)
        _ut = (_ut or "").strip()
        _debug_logg(f"datumhjalpare: {_fakta!r} -> {_ut[:80]!r}")
        return _ut or _fakta
    except Exception as _de:
        _debug_logg(f"datumhjalpare fel: {_de!r}")
        return None


def _svara_kvantitativt(fraga, historik, svarssprak):
    try:
        _sys = ("Answer the user's follow-up by REASONING and COMPUTING from "
                "the numbers already discussed in this conversation. Do the "
                "arithmetic carefully and state the resulting number clearly. "
                "NEVER say you cannot find the answer - it follows from the "
                "conversation, not from any search.")
        _sys += _sprak_direktiv(svarssprak)
        _prompt = bygg_llama_prompt_med_historik(_sys, historik, fraga)
        _ut = _generera(_prompt, temperature=0.0, repeat_penalty=1.1)
        if _ut and len(_ut.strip()) >= 2 and not ar_svaret_trasigt(_ut):
            _debug_logg("kvantitativ foljdfraga -> lokalt svar")
            return _ut.strip()
    except Exception as _qe:
        _debug_logg(f"kvantitativ foljdfraga fel: {_qe!r}")
    return None


def _fraga_ai_inner(fraga, status_callback=None, historik=None,
                    token_callback=None):
    historik = historik or []
    fraga_orig = _ratta_vanliga_stavfel(fraga)
    # Spekulativ avsikt fangas fran ORIGINALET (fore ev. foljdfrage-
    # omskrivning) - korta foljdfragor som "who do you think?" / "if you had
    # to guess" tappar annars gissnings-framingen nar de skrivs om.
    _spek_intent = _ar_spekulativ_fraga(_normalisera_for_matchning(fraga_orig))
    # KOD-ROUTER: kodfragor besvaras av 9B-modellen med en kod-inriktad
    # systemprompt (_KOD_SYSTEM) - ingen modellvaxling langre. Hysteres: en
    # kort foljdfraga efter en kodfraga stannar i kod-lage. Vid
    # genereringsfel faller vi tillbaka pa den vanliga vagen.
    global _SENAST_KOD
    if _ar_kodfraga(fraga_orig) or (_SENAST_KOD
                                    and _ar_kod_foljdfraga(fraga_orig)):
        _SENAST_KOD = True
        try:
            return _svara_kod(fraga_orig, historik,
                              detektera_svarssprak_namn(fraga_orig))
        except Exception as _kve:
            _debug_logg(f"kod-lage misslyckades, faller tillbaka: {_kve!r}")
            _SENAST_KOD = False
    else:
        _SENAST_KOD = False
    # A5: los upp tillbakasyftande foljdfragor ('its game by wishfully') mot
    # historiken INNAN sprak/gate/sok - sa retrieval far en fraga med tydlig
    # entitet istallet for att soka pa ett losryckt fragment.
    if _ar_foljdfraga(fraga_orig, historik):
        _om = _skriv_om_foljdfraga(fraga_orig, historik)
        if _om and _om.strip().lower() != fraga_orig.strip().lower():
            _debug_logg(f"A5 foljdfraga: {fraga_orig!r} -> {_om!r}")
            fraga_orig = _om
    # Explicit oversattningsfraga ('oversatt X till Y') -> dedikerad vag.
    # Svaret ar sjalva oversattningen pa malspraket (ingen spegling).
    if _ar_oversattningsfraga(_normalisera_for_matchning(fraga_orig)):
        _ov = _direkt_oversattning(fraga_orig)
        if _ov:
            _debug_logg(f"direkt oversattning -> {_ov!r}")
            return _ov
    # Anvandarens sprak (for att spegla SVARET tillbaka) - multisprakigt,
    # ren heuristik. Detta ar det ENDA stallet spraket spelar roll.
    svarssprak = detektera_svarssprak_namn(fraga_orig)
    # Ren rakneuppgift ('137 gånger 24', '15% av 850') -> berakna EXAKT i
    # Python. 3B-modellen raknar ofta fel (gav 137*24 = 3296, ratt ar 3288).
    if _ar_raknefraga(fraga_orig.lower()):
        _rak = _berakna(fraga_orig, svarssprak)
        if _rak is not None:
            _visning, _res = _rak
            _debug_logg(f"exakt raknare: {_visning} = {_res}")
            return f"{_visning} = {_formatera_tal(_res)}"
    # A5: deterministisk kalender-/datumhjalpare (veckodagsrakning +
    # period-konstanter). _datum_fakta returnerar None snabbt for
    # icke-datumfragor, sa detta kostar inget for vanliga fragor.
    _dat = _svara_datumfraga(fraga_orig, svarssprak)
    if _dat:
        return _dat
    # GREPP 4: syntes-tur ('sammanfatta / satt ihop / gor en pitch / en plan')
    # i ett pagaende samtal -> svara ENBART fran historiken, ALDRIG via sok.
    # Loser slutturs-kollapsen (C1/C3 avstar, C5 hallucinerar "vertical
    # gardening") - retrieval har inget nytt att hamta och stjalper svaret.
    if _ar_syntesfraga(fraga_orig, historik):
        _syn = _svara_syntes(fraga_orig, historik, svarssprak)
        if _syn:
            return _syn
    # GREPP 3: kvantitativ foljdfraga i samtal ('hur mycket pa ett ar?') ->
    # rakna lokalt fran historiken i stallet for att soka och avsta.
    if _ar_kvantitativ_foljdfraga(fraga_orig, historik):
        _kv = _svara_kvantitativt(fraga_orig, historik, svarssprak)
        if _kv:
            return _kv
    # PERSONLIG/KANSLOMASSIG VAG: kansliga/relationella meddelanden gar
    # DIREKT till ett varmt lokalt svar - ingen sok, ingen pafosad fakta
    # (som annars ger "kallorna bekraftar ..."-soppa). Blir svaret
    # tomt/trasigt -> falla igenom till vanliga vagen.
    if _ar_personlig_fraga(fraga_orig):
        _debug_logg("personlig/kanslomassig -> varmt lokalt svar (ingen sok)")
        _p_sprak = detektera_sprak(fraga_orig)
        _p_sys = bygg_system_prompt(_p_sprak) + _PERSONLIG_SYS_EXTRA
        _p_sys += _sprak_direktiv(svarssprak)
        _p_prompt = bygg_llama_prompt_med_historik(_p_sys, historik, fraga_orig)
        _p_svar = _generera(_p_prompt, temperature=0.5, repeat_penalty=1.1)
        if _p_svar and not ar_svaret_trasigt(_p_svar):
            return _p_svar
        _debug_logg("personligt svar tomt/trasigt -> faller igenom")
    # AMBETSINNEHAVARE (deterministiskt Wikidata P6/P35): loser 9B:ans
    # rollforvaxling (regeringschef vs statschef, t.ex. Meloni vs Mattarella)
    # och ger DAGENS innehavare. Kors FORE ovriga uppslag och fore sok.
    _amb = _amboetsinnehavare_uppslag(fraga_orig, svarssprak)
    if _amb:
        _debug_logg(f"ambetsinnehavare-svar -> {_amb!r}")
        return _amb
    # FIX 1 (gör ingen skada): korta icke-aktuella faktafragor pa ett
    # icke-en/sv-sprak besvaras NATIVT - FORE det strukturerade Wikidata-
    # uppslaget, som annars ger exonymer via fel sprak-label ("Rom" i st.f.
    # "Roma", "Lissabon" i st.f. "Lisboa"). Aktuella ledarfragor har redan
    # hanterats av ambetsinnehavare ovan; en/sv gar vidare till strukturerat
    # som forr (nativ_enkel avstar for dem).
    _niv = _svara_nativ_enkel(fraga_orig, svarssprak)
    if _niv:
        return _niv
    # STRUKTURERAT WIKIDATA-UPPSLAG: skapare av verk (P170/P50/...) och
    # huvudstad (P36) -> exakt svar direkt fran Wikidata. Fixar t.ex. "vem
    # malade Sixtinska kapellet" / "vem skrev Faust" / "huvudstad i X" som
    # annars garblades i fritextsoket. Gatas av regex sa vanliga fragor
    # inte betalar Wikidata-overhead.
    _strukt = _strukturerad_faktauppslag(fraga_orig, detektera_sprak(fraga_orig))
    if _strukt:
        _debug_logg(f"strukturerat wikidata-svar -> {_strukt!r}")
        return _strukt
    # Utokat strukturerat uppslag: datum (fodd/dod/grundad/slappt), valuta
    # och officiellt sprak -> exakt Wikidata-svar.
    _strukt_x = _strukturerat_extra_uppslag(fraga_orig,
                                            detektera_sprak(fraga_orig))
    if _strukt_x:
        _debug_logg(f"strukturerat-extra wikidata-svar -> {_strukt_x!r}")
        return _strukt_x
    # VEM VANN: fokuserad vinnar-extraktion istallet for att dumpa
    # tavlingens beskrivning ("vem vann senaste Serie A / Ballon d'Or ...").
    _mast = _mastare_uppslag(fraga_orig)
    if _mast:
        _debug_logg(f"mastare-svar -> {_mast!r}")
        return _mast
    # GREPP 5: AVSIKTSROUTING. Nar de deterministiska faktavagarna ovan
    # inte gav svar och fragan INTE kraver aktuell info: begreppsfragor
    # ('forklara hur X fungerar') och rad/asiktsfragor ('borde jag ...')
    # besvaras lokalt ur modellens kunnande - FORE retrieval. Annars
    # moter begreppsfragan system-promptens regel/siffer-vagran (C2) och
    # radsfragan drunknar i forbehall (C6/C1). Aktuella fragor gatas bort
    # av snabbcheck_aktuell_info inne i klassificerarna och gar till sok.
    if _ar_begreppsfraga(fraga_orig):
        _beg = _svara_begrepp(fraga_orig, historik, svarssprak)
        if _beg:
            return _beg
    if _ar_radsfraga(fraga_orig):
        _rad = _svara_rad(fraga_orig, historik, svarssprak)
        if _rad:
            return _rad
    # Grinda pa ORIGINALFRAGAN (Tier 1 lokalt snabbt, annars AI-beslut).
    kravs_aktuell = behover_aktuell_info(fraga_orig)
    # Spekulativ avsikt (aven fran en omskriven foljdfraga) -> tvinga sok sa
    # vi har kandidater/kallor att grunda gissningen pa.
    if _spek_intent:
        kravs_aktuell = True
    # UNIVERSELL FLERSPRAKIG RETRIEVAL. Retrieval-lagret (Wikipedia/
    # Wikidata-uppslagen) laser bara tva sprak direkt: svenska och
    # engelska. For ALLA ANDRA EU-sprak (tyska, nederlandska, franska,
    # italienska, spanska, portugisiska ...) oversatter vi fragan till
    # ENGELSKA och kor SAMMA palitliga pipeline - sa de far identiskt bra
    # uppslag som engelskan. Svaret speglas sedan tillbaka till
    # anvandarens eget sprak (_sprak_direktiv + _oversatt_fran_engelska).
    # Svenska/engelska gar kvar pa sin NATIVA vag (ingen oversattning) -
    # att runda dem via engelska tillfor bara oversattningsfel
    # (t.ex. "statsminister" -> fel ord) och langre svarstid.
    _sv_namn = (svarssprak or "").strip().lower()
    # B2: nativ Wikipedia-sprakkod for nisch-recall (wikipedia_sok_uppslag).
    _native_wiki = _SPRAKNAMN_TILL_WIKI.get(_sv_namn, "en")
    # ENGELSKA ar det enda spraket retrieval-lagret + fakta-extraktionen
    # kors pa internt. ALLA andra sprak (aven svenska!) oversatts till
    # engelska for SJALVA SOKNINGEN och far tillbaka en engelsk fakta -
    # sedan speglas slutsvaret till anvandarens sprak (originalfragan +
    # _sprak_direktiv). Att ha svenska "nativt" i sok-grenen gav blandade
    # svar (engelsk fakta -> engelskt svar trots svensk fraga); att kora
    # svenska genom engelskan ger samma tillforlitliga vag som tyskan m.fl.
    # LOKALA (icke-sok) svar genereras dock pa anvandarens eget sprak.
    _ar_engelska = _sv_namn == "english"
    # spegel: mala for att spegla FASTA (engelska) standardsvar tillbaka.
    # For engelska behovs ingen spegling.
    spegel = None if _ar_engelska else svarssprak
    # LOKALT-FORST: for fragor som INTE bedomdes krava aktuell info later
    # vi basmodellen forsoka svara ur eget minne. Kan den svara palitligt
    # -> snabbt lokalt svar (klart). Kan den INTE (self-report/klassad som
    # icke-svar) -> eskalera till webbsok istallet for att ge ett "vet
    # inte". Uppenbart lokala fragor (matte, kod, oversattning, kreativt)
    # hoppar over detta helt - de ska aldrig ga till sok.
    # Satts True om entitets-klassificeraren flaggar en nisch-entitet -
    # styr nisch-snabbvagen i sok-grenen (hoppar strukturerade uppslag).
    _ar_nischentitet = False
    if not kravs_aktuell and not _ar_uppenbart_lokal(
            _normalisera_for_matchning(fraga_orig)):
        if _bor_soka_entitet(fraga_orig):
            # Specifik/nisch namngiven entitet -> sok DIREKT. Llama
            # hallucinerar sadant sjalvsakert (t.ex. "bobergs pa
            # hamngarden = ett par skor"), sa lokalt-forst-kontrollen
            # (som litar pa ett sjalvsakert svar) fangar det inte.
            kravs_aktuell = True
            _ar_nischentitet = True
            _debug_logg("entitetsfraga -> sok direkt (hoppar over lokalt)")
        else:
            _lokal_sprak = "sv" if _sv_namn == "swedish" else "en"
            # Kor det lokala forsoket TYST (ingen live-stromning). Det ar
            # spekulativt och kan forkastas (sentinel/icke-svar/trasigt) och
            # da eskalera till webbsok - streamade vi det skulle anvandaren
            # se ett svar dyka upp och sedan ryckas bort nar vi soker.
            # Visa istallet bara en "tanker"-status; det FARDIGA svaret
            # skrivs ut nar det ar klart (lokalt ELLER efter sok).
            if status_callback:
                status_callback("tanker", _lokal_sprak)
            _lokalt_svar = _forsok_lokalt_svar(
                fraga_orig, svarssprak, _lokal_sprak, historik)
            if _lokalt_svar is not None:
                return _lokalt_svar
            # Modellen kunde inte svara lokalt -> behandla som sok-behov.
            kravs_aktuell = True
            _debug_logg("lokalt svar osakert -> eskalerar till webbsok")
    if kravs_aktuell and not _ar_engelska:
        # Sok-gren, icke-engelska (aven svenska): oversatt till engelska
        # for retrieval. Slutsvaret speglas till anvandarens sprak via
        # originalfragan + _sprak_direktiv.
        fraga = _oversatt_till_engelska(fraga_orig)
        sprak = "en"
        _debug_logg(f"retrieval-oversattning: {fraga_orig!r} -> {fraga!r}")
    else:
        # Engelska (alltid), eller icke-sok-gren: ingen oversattning.
        fraga = fraga_orig
        sprak = "sv" if _sv_namn == "swedish" else "en"
    system_msg = bygg_system_prompt(sprak)
    _debug_logg(f"FRAGA={fraga!r} svarssprak={svarssprak} sprak={sprak} "
                f"kravs_aktuell={kravs_aktuell} spegel={spegel}")
    sok_kontext = None
    fakta = None
    wiki_fakta = None
    ar_ja_nej_oklart = False
    ar_extraktion_oklar = False

    fraga_norm = _normalisera_for_matchning(fraga)
    # P2: fragetyp (ja/nej vs vad-ar) avgors pa ORIGINALFRAGAN, inte pa den
    # ev. engelska oversattningen. Ett svenskt stavfel ("varm ar X") kunde
    # annars oversattas till "Is warm the X..." och felaktigt routas till
    # ja/nej-analysen. "vad ar X" ska ALDRIG bli en ja/nej-fraga.
    fraga_norm_orig = _normalisera_for_matchning(fraga_orig)
    # Flerlandsfragor ska BARA raknas som "for komplext" om de aven
    # innehaller ett rollord (t.ex. "president X OCH statsminister Y") -
    # dar risken finns att blanda ihop fel land/roll. Fragor om t.ex.
    # krig/matcher MELLAN tva lander (utan rollord) ar ETT faktum och
    # ska fa ga vidare till normal sokning som vanligt.
    ar_flerdels_rollfraga = (
        _ar_flerlandsfraga(fraga_norm) and _hitta_roll_sv_nyckel(fraga_norm)
    )
    ar_komplex_flerdelsfraga = (
        kravs_aktuell and
        (_ar_historikfraga(fraga_norm) or ar_flerdels_rollfraga)
    )

    if ar_komplex_flerdelsfraga:
        # Bade Wikipedia-uppslaget OCH sok+AI-extraktionen ar byggda for
        # "en fraga, ett aktuellt faktum" - att tvinga fram ett svar ur
        # dem for historik- eller flerdelsfragor riskerar att blanda
        # ihop eller hitta pa fel. Var hellre arlig om begransningen.
        _debug_logg("komplex flerdelsfraga - hoppar over sok/wiki helt")
        return _oversatt_fran_engelska(
            _str("fallback_flertal_fragor", sprak), spegel)

    # Regelbaserade begrepp (rantor, skatter, loner, aktiekurser/bolags-
    # varde): webb-extraktion av exakta AKTUELLA siffror ar opalitlig
    # (en kvartalsrapport-snutt ger sallan ett rent borsvarde). Ge hellre
    # direkt ett arligt "kolla officiell/finans-kalla" an att lita pa en
    # snutt eller gissa med foraldrade siffror. Direktreturn INNAN sok.
    # NYA API:er: riktig vaderprognos (open-meteo) och riktig aktie-/krypto-
    # kurs (Yahoo Finance) - ersatter det gamla arliga "vet inte" for dessa
    # tva fragetyper NAR API:et svarar. Missar API:et faller vi tillbaka pa
    # vanlig sok / arligt svar.
    _api_fakta = None
    if _ar_vaderfraga(fraga_norm_orig):
        _api_fakta = vader_uppslag(fraga_orig, svarssprak)
        _debug_logg(f"vader_api={bool(_api_fakta)}")
    elif _ar_aktiefraga(fraga_norm_orig):
        _api_fakta = _hamta_aktie(fraga_orig)
        _debug_logg(f"aktie_api={bool(_api_fakta)}")

    # Regelbaserade begrepp (styrranta, moms, visum ...): forr en
    # direktreturn "kolla officiell kalla" REDAN har, for att modellen
    # gissade fel ur snuttar. MEN med djup-vagen (hamta-och-las) kan vi
    # faktiskt HAMTA siffran fran sjalva sidan - sa lat dessa flöda vidare
    # till sok + djup-vag. Sakerhetsnatet (samma direktreturn) ligger kvar
    # EFTER extraktionen om ingen verifierad fakta hittas.
    if _kraver_kalla(fraga_norm) and not _api_fakta and retrieval_deep is None:
        _debug_logg("regelbaserad begrepp - direktreturn (fore sok)")
        return _oversatt_fran_engelska(_str("regelfix_svar", sprak), spegel)

    if _api_fakta:
        # Vader-/kursfakta fran API - hoppa over wiki/brave-retrieval.
        wiki_fakta = _api_fakta
        _debug_logg("api-fakta anvands (vader/kurs)")
    elif kravs_aktuell:
        if status_callback:
            status_callback("sok", sprak)
        if _ar_nischentitet:
            # NISCH-SNABBVAG: entitets-klassificeraren har redan sagt att
            # detta ar en specifik/nisch namngiven entitet. De strukturerade
            # mainstream-uppslagen (roller/utmarkelser/sport/mjukvara/
            # wikidata) missar nastan alltid har och hoppas helt.
            # B2/A2: forst Wikipedias sok-API pa NATIV sprak + engelska -
            # nischade lokala entiteter (klubbar, museer, hus, band) finns
            # ofta bara pa sitt eget sprak, och ratt titel hittas utan att
            # Llama gissar. Ren HTTP, ingen model-kostnad.
            # UNDANTAG: attributfragor ('vilken division/liga', 'nar', 'hur
            # manga') - Wikipedias inledande summary saknar ofta det exakta
            # attributet (t.ex. aktuell division), sa dem gar direkt till
            # Brave vars snuttar har det farska svaret.
            if not _ar_attributfraga(fraga_norm_orig):
                wiki_fakta = wikipedia_sok_uppslag(fraga_orig, _native_wiki, fraga)
            _debug_logg(f"nisch wikipedia_sok={bool(wiki_fakta)} "
                        f"attribut={_ar_attributfraga(fraga_norm_orig)}")
            if not wiki_fakta:
                # Ingen Wikipedia-sida (eller attributfraga) -> Brave
                # (originalsprak + engelska SAMTIDIGT).
                if fraga.strip().lower() != fraga_orig.strip().lower():
                    sok_kontext = _sok_brave_parallellt(fraga_orig, fraga)
                else:
                    sok_kontext = sok_pa_natet(fraga_orig, oversatt=False)
                _debug_logg(f"nisch-snabbvag sok_kontext={str(sok_kontext)[:200]!r}")
        else:
            # De FYRA oversta uppslagen ar rena HTTP-anrop (ingen anropar
            # GPU-modellen) och ligger contiguous overst i prioordningen -
            # kor dem SAMTIDIGT och valj hogst prioriterade traff. Identiskt
            # resultat som den gamla kedjan, men ~4 rundturer -> ~1.
            wiki_fakta = _forsta_uppslag_parallellt(fraga, sprak)
            _debug_logg(f"strukturerat_parallellt_fakta={wiki_fakta!r}")
            # Aterstaende uppslag anvander modellen (generell, wikidata_direkt)
            # eller ligger under dem i prio (universell) - kor sekventiellt i
            # OFORANDRAD ordning som fallback.
            if not wiki_fakta:
                wiki_fakta = wikipedia_generell_uppslag(fraga, sprak=sprak)
                _debug_logg(f"wikipedia_generell_fakta=({len(wiki_fakta) if wiki_fakta else 0} tecken)")
            if not wiki_fakta:
                wiki_fakta = wikidata_direktuppslag(fraga, sprak=sprak)
                _debug_logg(f"wikidata_fakta={wiki_fakta!r}")
            if not wiki_fakta:
                wiki_fakta = wikidata_universell_uppslagning(fraga, sprak=sprak)
                _debug_logg(f"wikidata_universell_fakta={wiki_fakta!r}")
            if not wiki_fakta:
                # P1: sok pa ORIGINALFRAGAN + engelska SAMTIDIGT. Originalet
                # bevarar entiteter (akronymer som "tga", egennamn, lokala
                # namn), engelskan ger bredare/internationella traffar - de
                # slas ihop. Nu parallellt istallet for tva sekventiella
                # rundturer.
                if fraga.strip().lower() != fraga_orig.strip().lower():
                    sok_kontext = _sok_brave_parallellt(fraga_orig, fraga)
                else:
                    sok_kontext = sok_pa_natet(fraga_orig, oversatt=False)
                _debug_logg(f"sok_kontext (forsta 200 tecken)={str(sok_kontext)[:200]!r}")

    if (kravs_aktuell and not _api_fakta and fakta is None
            and retrieval_deep is not None
            and not _ar_historisk_fraga(fraga_norm_orig)
            and (_RESULTAT_DETALJ.search(fraga_norm_orig)
                 or _ar_sport_resultat(fraga_norm_orig))):
        # Wiki-INGRESSEN racker inte for detalj-/resultat-/sportfragor:
        # svaret ligger i en tabell langre ned, ELLER ingressen ar bara en
        # DEFINITION av tavlingen ("Ballon d'Or ar ett arligt pris ...") i
        # stallet for vinnaren. Kor djup-vagen (hamta-och-las) pa en farsk
        # sokning. For sport-/utmarkelseresultat STYR vi om sokningen mot
        # sjalva VINNAREN. Fyrar aven utan wiki-ingress (fakta is None).
        _ar_sport = _ar_sport_resultat(fraga_norm_orig)
        _steer = ("winner champion final result standings"
                  if _ar_sport else "")
        if status_callback:
            status_callback("sok", sprak)
        _dj_fakta = _djup_pa_farsk_sok(fraga, steer=_steer, sasong=_ar_sport)
        if _dj_fakta:
            _debug_logg(f"djup-eskalering -> {_dj_fakta[:120]!r}")
            wiki_fakta = None
            fakta = _dj_fakta
            sok_kontext = None
            system_msg += _str("prompt_sok_fakta", sprak,
                               datum=_DATUM_IDAG, fakta=fakta)

    if wiki_fakta:
        fakta = wiki_fakta
        ar_varfor = _ar_varforfraga(fraga_norm)
        if ar_varfor:
            system_msg += _str("prompt_wiki_varfor", sprak,
                                datum=_DATUM_IDAG, fakta=fakta)
        else:
            system_msg += _str("prompt_wiki_vanlig", sprak,
                                datum=_DATUM_IDAG, fakta=fakta)
        if _api_fakta:
            # Vader-/kursdata ar tillforlitlig och aktuell. Undvik preamble
            # ('Har ar oversattningen'), hedge ('I'm unsure') och
            # oversattnings-meta - svara direkt och naturligt pa anvandarens
            # sprak.
            system_msg += ("\n\nThe data above is reliable and current. "
                           "Answer the question directly and naturally in one "
                           "short paragraph. Do NOT add any preamble or "
                           "heading, do NOT mention translation, and do NOT "
                           "say you are unsure - the data is trustworthy.")

    # A1: relevansgrind - salla bort orelaterade kall-block ur den raa
    # sok-kontexten INNAN den konsumeras (spekulativ-vag, ja/nej,
    # fakta-extraktion, slutsvar). Stoppar retrieval-blodning, t.ex. ett
    # cancer-vaccin-stycke som annars kan smita in i ett energisvar.
    if sok_kontext:
        sok_kontext = _relevansfiltrera_kontext(sok_kontext, fraga)

    if ((_ar_spekulativ_fraga(fraga_norm_orig) or _spek_intent)
            and not _ar_historisk_fraga(fraga_norm_orig)):
        # SPEKULATIV/ASIKTS-VAG: anvandaren vill ha en KVALIFICERAD gissning,
        # inte en vagran. VIKTIGT: hamta FARSK kontext om sjalva racet/
        # kandidatfaltet - inte den nuvarande innehavaren (annars gissar
        # modellen pa sittande person aven nar den ar avgaende/term-limited,
        # t.ex. "Trump vinner 2028"). Genererar och returnerar direkt, forbi
        # de hardare guardsen som annars tvingar fram "gar inte att forutsaga".
        # STEERAD SOKFRAGA: den raa fragan ("who will be the next president
        # of usa") ger snippets om SITTANDE innehavaren. Vi lagger till
        # race-/kandidat-nyckelord sa Brave i stallet returnerar sjalva
        # valet/kandidatfaltet (odds, opinionsmatningar, troliga kandidater).
        _spek_sok_q = (fraga.strip()
                       + " next election candidates frontrunners "
                         "polls odds who is likely to run")
        _spek_ctx = None
        try:
            _spek_ctx = sok_pa_natet(_spek_sok_q, oversatt=False)
        except Exception:
            _spek_ctx = None
        # Djup hamta-och-las pa toppresultaten -> rikare kandidatlista an
        # bara snippets (om djup-vagen finns tillganglig).
        if retrieval_deep is not None and _spek_ctx:
            try:
                _spek_urls = re.findall(
                    r"\(K[aä]lla:\s*(https?://[^\s)]+)\)", _spek_ctx)
                if _spek_urls:
                    _spek_traffar = [{"url": _u} for _u in _spek_urls]
                    _spek_djup = retrieval_deep.djup_kontext(
                        _spek_sok_q, _spek_traffar, n_sidor=3, topp=6)
                    if _spek_djup:
                        _spek_ctx = _spek_djup
            except Exception as _e:
                _debug_logg(f"spek djup-kontext fel: {_e!r}")
        if not _spek_ctx:
            _spek_ctx = sok_kontext or wiki_fakta
        # DJUP-VAG steg 1b - DUBBEL HAMTNING: race-/kandidatmaterialet ovan
        # innehaller sallan NULAGET (sittande regering, partiernas AKTUELLA
        # ledare, farsk opinion). Utan det faller bade utkastet OCH
        # verifieringen tillbaka pa modellens foraldrade minne (fel
        # partiledare, omvand opinionsordning). Hamta darfor aven en
        # nulages-kontext och lagg den FORST i faktabladet.
        _nulage_ctx = None
        try:
            _nulage_q = (fraga.strip() + " current incumbent government "
                         "prime minister party leaders latest opinion poll "
                         "standings largest party")
            _nulage_ctx = sok_pa_natet(_nulage_q, oversatt=False)
            if retrieval_deep is not None and _nulage_ctx:
                _n_urls = re.findall(
                    r"\(K[aä]lla:\s*(https?://[^\s)]+)\)", _nulage_ctx)
                if _n_urls:
                    _n_traffar = [{"url": _u}
                                  for _u in dict.fromkeys(_n_urls)]
                    _n_djup = retrieval_deep.djup_kontext(
                        _nulage_q, _n_traffar, n_sidor=3, topp=6)
                    if _n_djup:
                        _nulage_ctx = _n_djup
        except Exception as _ne:
            _debug_logg(f"nulage-kontext fel: {_ne!r}")
        if _nulage_ctx and _spek_ctx:
            _spek_ctx = ("CURRENT FACTS (incumbent government, each party's "
                         "current leader, latest opinion-poll standings):\n"
                         + _nulage_ctx + "\n\n---\n\nRACE / CANDIDATES:\n"
                         + _spek_ctx)
            _debug_logg("spek: dubbel hamtning (nulage + race) sammanslagen")
        elif _nulage_ctx:
            _spek_ctx = _nulage_ctx
        if _spek_ctx:
            _spek_sys = bygg_system_prompt(sprak) + (
                "\n\nThe user is explicitly asking for your OPINION / best "
                "guess about an uncertain FUTURE matter (often an upcoming "
                "election or contest). Using the contenders named in the "
                "search results below, answer DIRECTLY: name the single most "
                "likely person/outcome (plus one or two runners-up) with a "
                "brief reason (standing, polling, prominence). CRITICAL: the "
                "question asks who will be NEXT - this is a FUTURE contest, "
                "not the person holding the office right now. Do NOT answer "
                "with the CURRENT/sitting office-holder unless the sources "
                "clearly show they are eligible AND actively running again; "
                "if they are term-limited or not listed among the contenders, "
                "pick from the CANDIDATES named in the sources instead. If the "
                "sources only describe the current holder and name no future "
                "candidates, say plainly that no clear front-runner has "
                "emerged yet and name the most-discussed potential contenders "
                "you can. End with ONE short caveat that nothing is decided "
                "yet. Do NOT refuse and do NOT say it is impossible to "
                "predict.\n\n"
                "GROUNDING RULES (critical - this is where you usually go "
                "wrong):\n"
                "- Party names, each party's CURRENT leader, the incumbent "
                "government, which party is largest and which leads the "
                "polls: state these ONLY as given in the CURRENT FACTS part "
                "of the sources. NEVER supply a party leader, poll ranking or "
                "office-holder from your own memory.\n"
                "- If a needed current fact (a party's current leader, who "
                "currently leads the polls, the sitting PM) is NOT in the "
                "sources, do not guess it - either omit it or say plainly you "
                "don't have a current figure for it.\n"
                "- If the question names no country, assume Sweden.\n"
                "\nSources:\n" + _spek_ctx)
            _spek_sys += _sprak_direktiv(svarssprak)
            _spek_prompt = bygg_llama_prompt_med_historik(
                _spek_sys, historik, fraga_orig)
            _spek_svar = _generera(_spek_prompt, temperature=0.4,
                                   repeat_penalty=1.1)
            if _spek_svar and not ar_svaret_trasigt(
                    _spek_svar, sprak=("en" if _ar_engelska else "sv")):
                _debug_logg("spekulativ-vag: direkt kvalificerad gissning")
                # DJUP-VAG steg 2: verifiera gissningen mot kallorna. Just
                # spekulativa svar ("nasta regering/president") ar dar 9B:an
                # oftast blandar ihop namn/roller (t.ex. fel partiledare) -
                # och denna gren returnerar annars forbi de vanliga guardsen.
                if status_callback:
                    status_callback("tanker", sprak)
                _spek_svar = _verifiera_svar(_spek_svar, _spek_ctx, svarssprak, fraga_orig)
                return _stada_hallucinerad_landsparentes(_spek_svar, sprak)
            _debug_logg("spekulativt svar trasigt -> faller igenom")
    if (sok_kontext and _ar_ja_nej_fraga(fraga_norm_orig)
            and _ar_akta_ja_nej(fraga_orig)):
        # B1/A1: regexen flaggade en ja/nej-fraga OCH LLM-intent bekraftade
        # den (vetar falska utländska 'was ist X'-traffar).
        # Specialiserad ja/nej-analys - tvingar fram ett uttryckligt
        # JA/NEJ-resonemang istallet for att lata modellen fritt
        # formulera ett svar (som visat sig kunna dra fel slutsats
        # aven med korrekta sokresultat framfor sig).
        ja_nej_resultat = extrahera_ja_nej(fraga, sok_kontext)
        _debug_logg(f"ja_nej_resultat={ja_nej_resultat!r}")
        if ja_nej_resultat:
            beslut, motivering = ja_nej_resultat
            # Engelsk fakta (som all annan retrieval-fakta) -> speglas till
            # anvandarens sprak av _sprak_direktiv, inte ekad rakt av.
            fakta = f"The answer is {beslut}. {motivering}".strip()
            system_msg += _str("prompt_janej_med_analys", sprak,
                                datum=_DATUM_IDAG, fakta=fakta)
        else:
            # Aven den specialiserade analysen var osaker - var arlig.
            fakta = None
            ar_ja_nej_oklart = True
            system_msg += _str("prompt_janej_oklart", sprak)
    elif sok_kontext and _ar_listfraga(fraga_norm_orig):
        # P5: listfraga - extrahera FLERA poster istallet for ett faktum.
        lista = extrahera_lista(fraga, sok_kontext)
        _debug_logg(f"lista={lista!r}")
        if _fakta_ar_oklar(lista):
            fakta = None
            ar_extraktion_oklar = True
            system_msg += _str("prompt_extraktion_oklar", sprak)
        else:
            fakta = lista
            system_msg += _str("prompt_sok_fakta", sprak,
                                datum=_DATUM_IDAG, fakta=fakta)
    elif sok_kontext:
        fakta_raw = extrahera_fakta(fraga, sok_kontext)
        if _fakta_ar_oklar(fakta_raw):
            # Sokningen gav inget tydligt svar (t.ex. for tvetydig fraga,
            # som "vem ar statsminister" utan land) - var arlig istallet
            # for att tvinga fram ett svar ur en icke-fakta.
            fakta = None
            ar_extraktion_oklar = True
            system_msg += _str("prompt_extraktion_oklar", sprak)
        else:
            fakta = fakta_raw
            system_msg += _str("prompt_sok_fakta", sprak,
                                datum=_DATUM_IDAG, fakta=fakta)
    elif kravs_aktuell and not wiki_fakta:
        system_msg += _str("prompt_krav_aktuell_utan_wiki", sprak)

    # ===== DJUP VAG: HAMTA-OCH-LAS + EMBEDDING-RERANKER =====
    # Nar snippet-extraktionen blev OKLAR men vi HAR sokkallor: hamta
    # sjalva sidorna, ranka passager semantiskt mot fragan och extrahera
    # igen. Fakta ligger ofta i tabeller/brodtext (medaljligor, rantor,
    # vinnare) som Braves korta snuttar missar. Kors bara vid behov (kostar
    # ~10-20 s), sa den snabba vagen ar oforandrad for enkla fragor.
    if (retrieval_deep is not None and sok_kontext and fakta is None
            and (ar_extraktion_oklar or ar_ja_nej_oklart)):
        _urls = re.findall(r"\(K[aä]lla:\s*(https?://[^\s)]+)\)", sok_kontext)
        _traffar = [{"url": _u} for _u in dict.fromkeys(_urls)]
        if _traffar:
            if status_callback:
                status_callback("sok", sprak)
            try:
                _djup = retrieval_deep.djup_kontext(fraga, _traffar,
                                                    n_sidor=4, topp=6)
            except Exception as _de:
                _djup = None
                _debug_logg(f"deep-path fel: {_de!r}")
            if _djup:
                if _ar_listfraga(fraga_norm_orig):
                    _ny = extrahera_lista(fraga, _djup)
                else:
                    _ny = extrahera_fakta(fraga, _djup)
                _debug_logg(f"deep-path re-extraktion -> {str(_ny)[:160]!r}")
                if _ny and not _fakta_ar_oklar(_ny):
                    fakta = _ny
                    ar_extraktion_oklar = False
                    ar_ja_nej_oklart = False
                    sok_kontext = _djup
                    system_msg = bygg_system_prompt(sprak) + _str(
                        "prompt_sok_fakta", sprak, datum=_DATUM_IDAG,
                        fakta=fakta)

    # Fix 3: regelbaserade begrepp (visum, styrranta, moms, etc.) -
    # om vi inte hittat verifierad fakta har Llama tenderat svara
    # sjalvsakert med foraldrad/felaktig info. Skydda anvandaren genom
    # att direktreturnera "kolla officiell kalla" istallet.
    if not fakta and _kraver_kalla(fraga_norm):
        _debug_logg("regelbaserad begrepp utan fakta - direktreturn")
        return _oversatt_fran_engelska(_str("regelfix_svar", sprak), spegel)

    system_msg += _sprak_direktiv(svarssprak)
    # VIKTIGT: generera svaret utifran ORIGINALFRAGAN (fraga_orig), inte
    # den ev. engelska retrieval-oversattningen. Modellen ser da anvandarens
    # EGNA sprak i prompten och svarar naturligt pa samma sprak - aven nar
    # sprakdetektorn var osaker (None) och ingen explicit sprakdirektiv
    # gavs. Retrieval/fakta togs redan fram pa engelska; har anvander vi
    # bara faktan + originalfragan for sjalva formuleringen.
    prompt = bygg_llama_prompt_med_historik(system_msg, historik, fraga_orig)
    hade_fakta = bool(fakta)
    _debug_logg(f"fakta={fakta!r} hade_fakta={hade_fakta}")
    # Lagre temperatur nar vi har en fakta att halla oss till - gor
    # modellen mer benagen att faktiskt anvanda den istallet for att
    # falla tillbaka pa sitt eget (foraldrade) minne.
    temp1 = 0.15 if hade_fakta else 0.4
    temp2 = 0.05 if hade_fakta else 0.25
    # SAMPLING-TUNING: hogt repeat_penalty gav ofta grumlig/onaturlig text
    # (sarskilt fritt formulerade svar pa svenska). Lagre penalty = naturligare
    # sprak. Med en verifierad FAKTA att halla sig till behovs lite mer
    # aterhallsamhet; FRITT formulerade svar (chat/kreativt/asikt) far lagst
    # penalty for basta sprakflyt. (Fore: 1.15 / 1.3 rakt over.)
    rp1 = 1.12 if hade_fakta else 1.08
    rp2 = 1.20 if hade_fakta else 1.12
    # For "varfor"-fragor (t.ex. "varfor avgick X") vill vi INTE tvinga
    # fram namnet i svaret - ett arligt "jag har ingen info om det" ar
    # ett giltigt svar aven om det inte namner faktans namn ordagrant.
    krav_pa_namnmatch = hade_fakta and not _ar_varforfraga(fraga_norm)
    # Fix 1b: nar fragan kravde aktuell info och vi inte hittade nagon
    # fakta (ingen wiki, ingen sok, ingen extraherad fakta) - Llama
    # borde nu erkanna osakerhet. Om inte -> hallucination. Bevaka
    # med hedge-check tillsammans med de andra trasighetscheckerna.
    kravdes_osaker_svar = kravs_aktuell and not hade_fakta and not ar_ja_nej_oklart
    # Om var specialiserade ja/nej-analys sa OKLART, men modellen anda
    # svarar sjalvsakert "Ja"/"Nej" - det ar precis den hallucination vi
    # forsoker forhindra. Tvinga fram ett nytt forsok/arligt svar da.
    _SJALVSAKERT_JA_NEJ = re.compile(r"^(ja|nej|yes|no)\b[,.]?", re.IGNORECASE)

    def _bryter_mot_oklart(svar_text):
        return ar_ja_nej_oklart and bool(_SJALVSAKERT_JA_NEJ.match(svar_text.strip()))

    def _ej_grundat(svar_text):
        return ar_extraktion_oklar and not _svar_ar_grundat(svar_text, sok_kontext)

    def _saknar_kravd_hedge(svar_text):
        # Fix 1b-check: kravs hedge BARA nar vi helt saknar underlag. Om
        # en websokning faktiskt gav kontext (sok_kontext) far modellen
        # svara ur den aven om den strukturerade extraktionen missade -
        # da ska vi INTE tvinga fram ett "vet inte". Grundningskollen
        # (_ej_grundat) sakerstaller anda att svaret stods av kontexten,
        # sa hallucinationer fangas fortfarande.
        if sok_kontext:
            return False
        return kravdes_osaker_svar and not _svar_har_hedge(svar_text)

    def _ar_hallucinerat(svar_text):
        # Fix 2-check: om faktan har ett ar och svaret har ett annat
        return hade_fakta and not _fakta_ar_bevarat(fakta, svar_text)

    def _pastar_framtida_ar_ogrundat(svar_text):
        # Universell hallucinationsspärr: nar fragan kravde verifierbar
        # info men vi INTE hittade nagon kalla ar ett sjalvsakert
        # FRAMTIDA artal (efter innevarande ar) i svaret nastan alltid en
        # gissning - man kan inte veta framtida datum. Sprak- och
        # kategorioberoende (bara siffror), fangar t.ex. pahittade
        # slappdatum for kommande mjukvaruversioner/evenemang.
        if not kravdes_osaker_svar:
            return False
        try:
            idag_ar = int(_DATUM_IDAG[:4])
        except Exception:
            return False
        return any(int(m) > idag_ar
                   for m in re.findall(r"\b20\d{2}\b", svar_text))

    _API_VAGRAN = re.compile(
        r"\b(unsure|not sure|cannot provide|can'?t provide|do(n't| not) have|"
        r"no (current |real-?time )?(information|data)|osaker|inte saker|"
        r"kan inte (ge|svara|hjalpa)|har inte tillgang)\b", re.IGNORECASE)

    def _api_vagran(svar_text):
        # Vader-/kursdatan ar redan verifierad. Vagrar/hedgar modellen anda
        # ("I'm unsure") ar det fel - fanga det sa vi faller tillbaka pa
        # sjalva API-datan istallet for att skicka en tom hedge.
        return bool(_api_fakta) and bool(_API_VAGRAN.search(svar_text))

    # TYST generering: aven detta forsta sok-svar kan underkannas av en guard
    # och ersattas (svar2/fallback). Streama det inte live - skriv ut forst
    # det FARDIGA, godkanda svaret sa inget svar rycks bort framfor anvandaren.
    # Sprakkod for trasig-kontrollens sprakkonsistens-guard: den ska styras
    # av SVARETS malsprak (spegel/_ar_engelska), INTE av det interna sok-
    # spraket (sprak, som blir 'en' sa fort en fraga oversatts for webbsok).
    # Annars flaggades KORREKTA svenska svar som trasiga bara for att de
    # innehaller a-ring -> appen foll tillbaka pa ravsokdata istallet.
    _svarsprak_kod = "en" if _ar_engelska else "sv"
    # FIX 2 (gör ingen skada): fragan kravde aktuell info men retrieval kom
    # HELT tomt (ingen wiki, sok eller API). I stallet for ett platt "jag
    # vet inte" (samre an minnet nar innehavaren inte bytts, t.ex.
    # Frankrikes president) - ge modellens basta minnessvar MED forbehall.
    # Ror ALDRIG fall dar vi faktiskt har fakta/kontext.
    if (kravs_aktuell and not hade_fakta and not sok_kontext
            and not _api_fakta and not ar_ja_nej_oklart
            and not ar_extraktion_oklar
            and not _FRAMTID_ELLER_ATTRIBUTION.search(fraga_norm_orig)):
        # (framtids-/slappdatumfragor uteslutna - dar ska ett tomt uppslag
        # forbli ett arligt "vet ej", inte en gissad framtida-datum-siffra.)
        _mem = _svara_ur_minne_med_forbehall(fraga_orig, svarssprak, historik)
        if _mem:
            return _stada_hallucinerad_landsparentes(_mem, sprak)
    svar = _generera(prompt, temperature=temp1, repeat_penalty=rp1)
    svar1_trasigt = (ar_svaret_trasigt(svar, hade_fakta=hade_fakta, sprak=_svarsprak_kod) or
                      (krav_pa_namnmatch and not _fakta_matchar_svar(fakta, svar)) or
                      _bryter_mot_oklart(svar) or _ej_grundat(svar) or
                      _saknar_kravd_hedge(svar) or _ar_hallucinerat(svar) or
                      _pastar_framtida_ar_ogrundat(svar) or _api_vagran(svar))
    _debug_logg(f"forsok1 svar={svar!r} trasigt={svar1_trasigt}")
    if not svar1_trasigt:
        # SVARSKONTRAKT (Grepp 1): granska mot fraga + kallor - fanga
        # "definition-dumpar" (Wikipedia-ingress i st.f. svaret), faktafel
        # och fel sprak. Kors pa BADE fritt webbmaterial (sok_kontext) OCH
        # strukturerad wiki-fakta (dar dumparna faktiskt uppstod). Vader-/
        # kursdata (_api_fakta) ar redan verifierat och hoppas over.
        _kx = sok_kontext or (fakta if not _api_fakta else None)
        if _kx:
            if status_callback:
                status_callback("tanker", sprak)
            svar = _verifiera_svar(svar, _kx, svarssprak, fraga_orig)
        return _stada_hallucinerad_landsparentes(svar, sprak)

    svar2 = _generera(prompt, temperature=temp2, repeat_penalty=rp2)
    svar2_trasigt = (ar_svaret_trasigt(svar2, hade_fakta=hade_fakta, sprak=_svarsprak_kod) or
                      (krav_pa_namnmatch and not _fakta_matchar_svar(fakta, svar2)) or
                      _bryter_mot_oklart(svar2) or _ej_grundat(svar2) or
                      _saknar_kravd_hedge(svar2) or _ar_hallucinerat(svar2) or
                      _pastar_framtida_ar_ogrundat(svar2) or _api_vagran(svar2))
    _debug_logg(f"forsok2 svar={svar2!r} trasigt={svar2_trasigt}")
    if not svar2_trasigt:
        _kx2 = sok_kontext or (fakta if not _api_fakta else None)
        if _kx2:
            if status_callback:
                status_callback("tanker", sprak)
            svar2 = _verifiera_svar(svar2, _kx2, svarssprak, fraga_orig)
        return _stada_hallucinerad_landsparentes(svar2, sprak)

    if ar_extraktion_oklar:
        # Bada forsoken namnde nagot som inte finns i sokresultaten -
        # det ar en hallucination, inte en giltig gissning.
        return _oversatt_fran_engelska(_str("fallback_extraktion_oklar", sprak), spegel)

    if ar_ja_nej_oklart:
        # Tva forsok, bada sjalvsakra trots att vi inte kunde faststalla
        # ett sakert ja/nej - var hellre tydligt arlig an att chansa.
        return _oversatt_fran_engelska(_str("fallback_janej_oklart", sprak), spegel)

    if hade_fakta:
        # Llama ignorerade fakta tva ganger - lita hellre pa den
        # verifierade faktan direkt an att riskera fel/hedge-svar igen.
        return _oversatt_fran_engelska(fakta, spegel)

    if kravdes_osaker_svar:
        # Fix 1b-fallback: bada forsoken saknade hedge (Llama envisades
        # med att hallucinera istallet for att erkanna osakerhet). Ge
        # ett arligt standardsvar istallet for att chansa.
        return _oversatt_fran_engelska(_str("fallback_hedge_kravdes", sprak), spegel)

    return _oversatt_fran_engelska(_str("fallback_trasigt", sprak), spegel)


# ============ BILDIGENKANNING (Ollama qwen2.5vl:3b, lokalt/offline) ============
# Textpipelinen ovan ar HELT oforandrad (samma Qwen3-4B, samma sok, samma
# rutter). Bildfragor gar en separat vag: en fristaende vision-modell
# (Qwen2.5-VL 3B) kors via Ollamas lokala API pa localhost:11434.
# Inget lamnar datorn - allt sker offline pa GPU:n.
_OLLAMA_URL = "http://localhost:11434"
_VISION_MODELL = "qwen2.5vl:3b"


def _ollama_svarar():
    try:
        r = requests.get(_OLLAMA_URL + "/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _sakerstall_ollama():
    """Ser till att Ollama-servern kor. Startar 'ollama serve' tyst i
    bakgrunden om den inte redan svarar. Returnerar True om servern ar uppe."""
    if _ollama_svarar():
        return True
    try:
        import subprocess
        flaggor = 0x08000000  # CREATE_NO_WINDOW - inget konsolfonster blinkar
        subprocess.Popen(["ollama", "serve"], creationflags=flaggor,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return False
    for _ in range(20):
        time.sleep(1)
        if _ollama_svarar():
            return True
    return False


def beskriv_bild(bildvag, fraga=""):
    """Skickar en bild (och ev. anvandarens fraga) till den lokala vision-
    modellen och returnerar svaret pa anvandarens sprak. Kor helt lokalt."""
    import base64
    if not bildvag or not os.path.exists(bildvag):
        return "Kunde inte hitta bildfilen."
    if not _sakerstall_ollama():
        return ("Bildmodellen (Ollama) svarar inte. Kontrollera att Ollama "
                "ar installerat och igang, och forsok igen.")
    try:
        with open(bildvag, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as e:
        return "Kunde inte lasa bilden: " + str(e)

    fraga = (fraga or "").strip()
    if fraga:
        instruktion = fraga
        try:
            sprak_namn = detektera_svarssprak_namn(fraga)
        except Exception:
            sprak_namn = "English"
    else:
        instruktion = ("Describe this image in detail. If it contains any "
                       "text, read it out.")
        sprak_namn = "Swedish"
    instruktion += ("\n\nAnswer in " + sprak_namn + ". Only describe what you "
                    "can actually see in the image; do not invent details "
                    "that are not visible.")

    payload = {
        "model": _VISION_MODELL,
        "messages": [{"role": "user", "content": instruktion, "images": [b64]}],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    try:
        r = requests.post(_OLLAMA_URL + "/api/chat", json=payload, timeout=300)
        j = r.json()
        svar = (j.get("message", {}) or {}).get("content", "").strip()
        return svar or "Fick inget svar fran bildmodellen."
    except Exception as e:
        return "Fel vid bildanalys: " + str(e)


# ============ CHAT HANDLERS ============
# Multi-chatt: varje chatt har namn, egen historik (kontext) och en logg
# (renderbara meddelanden). Allt bara i RAM.
_MAX_HISTORIK_VARV = 6
_CHATTAR = []
_AKTIV = {"chatt": None}


def _aktiv_chatt():
    return _AKTIV["chatt"]


def _kort_titel(text, n=30):
    t = " ".join((text or "").split())
    return (t[: n - 1] + "…") if len(t) > n else (t or "New chat")


def hamta_svar(fraga, status_label, chatt):
    def _status(typ, sprak="sv"):
        if typ == "tanker":
            text = "Thinking..."
        elif typ == "sok":
            text = "Searching the web..."
        elif typ == "vaxlar_code":
            text = "Switching to code model..."
        elif typ == "vaxlar_chat":
            text = "Switching to chat model..."
        else:
            return

        def _visa_status():
            try:
                if status_label.winfo_exists():
                    status_label.config(text=text)
            except Exception:
                pass
        root.after(0, _visa_status)

    try:
        svar = fraga_ai(fraga, status_callback=_status,
                        historik=chatt["historik"])
    except Exception as e:
        svar = f"[Error: {e}]"
    else:
        chatt["historik"].append({"fraga": fraga, "svar": svar})
        del chatt["historik"][:-_MAX_HISTORIK_VARV]
    chatt["logg"].append(("assistant", svar))

    def _visa():
        try:
            status_label.master.destroy()
        except Exception:
            pass
        tw = lagg_till_assistentmeddelande("")
        _satt_svarstext(tw, svar)
    root.after(0, _visa)


def hamta_bildsvar(bildvag, fraga, status_label, chatt):
    def _visa_status():
        try:
            if status_label.winfo_exists():
                status_label.config(text="Analysing image...")
        except Exception:
            pass
    root.after(0, _visa_status)
    try:
        svar = beskriv_bild(bildvag, fraga)
    except Exception as e:
        svar = "[Error: " + str(e) + "]"
    else:
        chatt["historik"].append({"fraga": (fraga or "[image]"), "svar": svar})
        del chatt["historik"][:-_MAX_HISTORIK_VARV]
    chatt["logg"].append(("assistant", svar))

    def _visa():
        try:
            status_label.master.destroy()
        except Exception:
            pass
        tw = lagg_till_assistentmeddelande("")
        _satt_svarstext(tw, svar)
    root.after(0, _visa)


def skicka():
    global _bifogad_bild
    if llm is None or llm_sok is None:
        return
    fraga = entry.get()
    bild = _bifogad_bild
    if fraga.strip() == "" and not bild:
        return
    entry.delete(0, tk.END)
    chatt = _aktiv_chatt()
    if chatt is None:
        chatt = _ny_chatt(vaxla=True)
    _dolj_tomt_lage()
    if chatt.get("auto") and not chatt["logg"]:
        chatt["namn"] = _kort_titel(fraga if fraga.strip() else "Image")
        chatt["auto"] = False
        _rita_chattlista()
    if bild:
        rubrik = os.path.basename(bild)
        bubbeltext = "🖼 " + rubrik + (("\n" + fraga.strip())
                                        if fraga.strip() else "")
        chatt["logg"].append(("user", bubbeltext))
        lagg_till_anvandarbubbla(bubbeltext)
        _rensa_bifogad_bild()
        status_label = lagg_till_statusrad()
        threading.Thread(target=hamta_bildsvar,
                         args=(bild, fraga.strip(), status_label, chatt),
                         daemon=True).start()
        return
    chatt["logg"].append(("user", fraga))
    lagg_till_anvandarbubbla(fraga)
    status_label = lagg_till_statusrad()
    threading.Thread(target=hamta_svar, args=(fraga, status_label, chatt),
                     daemon=True).start()


# ============ UI: LJUST TEMA, TYPSNITT ============
# Ljust, professionellt tema med mjuk smaragdaccent - inspirerat av
# Claude/ChatGPT (mycket vitrum, runda horn, subtila kanter).
BG_MAIN      = "#eae8e1"   # chatt-ytan (varm mjuk grabeige)
SIDEBAR_BG   = "#eae8e1"   # sidopanel (varm mjuk grabeige)
SIDEBAR_HOVER = "#f0f2f1"   # hover (ljus neutral)
SIDEBAR_ACTIVE = "#e6f3ec"  # aktiv chatt (ljus gronton)
BG_BUBBLE_USER = "#e7f4ee"  # anvandarbubbla (ljus gronton)
BG_INPUT     = "#f6f5f1"
INPUT_BORDER = "#d3d7d4"
BORDER_COL   = "#dcdcd6"
BORDER_SOFT  = "#e1e0da"
CARD_BG      = "#f6f5f1"
FG_TEXT      = "#1b1f23"
FG_MUTED     = "#606a72"
FG_DIM       = "#98a1a8"
GREEN        = "#0e9f6e"   # smaragdaccent
GREEN_DK     = "#0b8259"
GREEN_SOFT   = "#e3f3ec"
CODE_FG      = "#0a7d57"
CODE_BG      = "#f3f5f4"
SEL_BG       = "#cfeadd"
ACCENT       = GREEN
ACCENT_TEXT  = "#ffffff"

FONT_MAIN   = ("Segoe UI", 13)
FONT_STATUS = ("Segoe UI", 11)
FONT_SIDE   = ("Segoe UI", 11)
FONT_H      = ("Segoe UI Semibold", 15)
FONT_SMALL  = ("Segoe UI", 10)


# ============ ROT-FONSTER + LAYOUT ============
root = tk.Tk()
root.title("Offline AI — Qwen3.5 9B (v7)")
try:
    root.iconbitmap(os.path.join(BASKATALOG, "ikon.ico"))
except Exception:
    pass
root.configure(bg=BG_MAIN)
try:
    root.state("zoomed")
except tk.TclError:
    root.attributes("-zoomed", True)

SIDO_BREDD = 272
root.grid_columnconfigure(0, weight=0, minsize=SIDO_BREDD)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)


_RR_CACHE = {}
_RR_ORD = []


def _rundad_rekt(canvas, x1, y1, x2, y2, radie=18, fill="", outline="",
                 width=1, tags=None, **kwargs):
    """Anti-aliasad rundad rektangel: renderas supersamplat i PIL och laggs
    som bild (Tkinter-canvasens polygoner/ovaler ar kantiga utan AA)."""
    w = int(round(x2 - x1))
    h = int(round(y2 - y1))
    if w < 2 or h < 2:
        return None
    key = (w, h, int(radie), fill or "", outline or "", int(width))
    ph = _RR_CACHE.get(key)
    if ph is None:
        try:
            f = 4
            im = Image.new("RGBA", (w * f, h * f), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            r = max(0, min(int(radie) * f, w * f // 2, h * f // 2))
            d.rounded_rectangle(
                [f, f, w * f - f - 1, h * f - f - 1], radius=r,
                fill=(fill or None), outline=(outline or None),
                width=(max(1, int(width)) * f) if outline else 0)
            ph = ImageTk.PhotoImage(im.resize((w, h), Image.LANCZOS))
        except Exception:
            p = [x1+radie, y1, x2-radie, y1, x2, y1, x2, y1+radie,
                 x2, y2-radie, x2, y2, x2-radie, y2, x1+radie, y2,
                 x1, y2, x1, y2-radie, x1, y1+radie, x1, y1]
            kk = {}
            if fill:
                kk["fill"] = fill
            if outline:
                kk["outline"] = outline
            if tags:
                kk["tags"] = tags
            return canvas.create_polygon(p, smooth=True, **kk)
        _RR_CACHE[key] = ph
        _RR_ORD.append(key)
        if len(_RR_ORD) > 500:
            _RR_CACHE.pop(_RR_ORD.pop(0), None)
    opts = {"image": ph, "anchor": "nw"}
    if tags:
        opts["tags"] = tags
    return canvas.create_image(int(x1), int(y1), **opts)


_DOT_CACHE = {}


def _dot_img(d, outer, inner=None):
    """Anti-aliasad cirkel (diameter d). Med 'inner' -> ring (halet i
    'inner'-farg). Anvands for grona prickar/avatarer i stallet for
    create_oval (som ar kantig)."""
    key = (d, outer, inner)
    ph = _DOT_CACHE.get(key)
    if ph is None:
        f = 4
        im = Image.new("RGBA", (d * f, d * f), (0, 0, 0, 0))
        dr = ImageDraw.Draw(im)
        dr.ellipse([f, f, d * f - f - 1, d * f - f - 1], fill=outer)
        if inner:
            m = int(d * f * 0.30)
            dr.ellipse([m, m, d * f - m - 1, d * f - m - 1], fill=inner)
        ph = ImageTk.PhotoImage(im.resize((d, d), Image.LANCZOS))
        _DOT_CACHE[key] = ph
    return ph


def _rita_dot(canvas, cx, cy, d, outer, inner=None):
    return canvas.create_image(cx, cy, image=_dot_img(d, outer, inner),
                               anchor="center")


def _rund_knapp(parent, text, cmd, fyll, txtfarg, hover=None, ram=None,
                hojd=38, radie=11, font=("Segoe UI Semibold", 11),
                align="center", ikon="", bredd=None):
    c = tk.Canvas(parent, height=hojd, bg=parent.cget("bg"),
                  highlightthickness=0, cursor="hand2")
    if bredd:
        c.configure(width=bredd)
    st = {"fill": fyll}

    def _draw(fill):
        c.delete("all")
        w = c.winfo_width() or (bredd or 150)
        _rundad_rekt(c, 1, 1, w - 1, hojd - 1, radie=radie, fill=fill,
                     outline=(ram or fill))
        lbl = (ikon + "   " + text) if ikon else text
        if align == "left":
            c.create_text(18, hojd / 2, text=lbl, anchor="w",
                          fill=txtfarg, font=font)
        else:
            c.create_text(w / 2, hojd / 2, text=lbl, fill=txtfarg, font=font)
    c.bind("<Configure>", lambda e: _draw(st["fill"]))
    c.bind("<Enter>", lambda e: _draw(hover or fyll))
    c.bind("<Leave>", lambda e: _draw(st["fill"]))
    c.bind("<Button-1>", lambda e: cmd())
    return c


def _rund_kort(parent, pad=18):
    holder = tk.Canvas(parent, bg=parent.cget("bg"), highlightthickness=0)
    holder.pack(fill="x", pady=(0, 14))
    inner = tk.Frame(holder, bg=CARD_BG)
    win = holder.create_window(pad, pad, window=inner, anchor="nw")

    def _fit(_e=None):
        w = holder.winfo_width()
        if w <= 1:
            return
        holder.itemconfig(win, width=w - 2 * pad)
        inner.update_idletasks()
        h = inner.winfo_reqheight() + 2 * pad
        holder.configure(height=h)
        holder.delete("bg")
        _rundad_rekt(holder, 1, 1, w - 2, h - 2, radie=16, fill=CARD_BG,
                     outline=BORDER_COL, tags="bg")
        holder.tag_lower("bg")
    holder.bind("<Configure>", _fit)
    inner.bind("<Configure>", lambda e: holder.after(1, _fit))
    return inner


main_holder = tk.Frame(root, bg=BG_MAIN)
main_holder.grid(row=0, column=1, sticky="nsew")
main_holder.grid_rowconfigure(0, weight=1)
main_holder.grid_columnconfigure(0, weight=1)

chat_view = tk.Frame(main_holder, bg=BG_MAIN)
chat_view.grid(row=0, column=0, sticky="nsew")
chat_view.grid_columnconfigure(0, weight=1)
chat_view.grid_columnconfigure(1, weight=0, minsize=740)
chat_view.grid_columnconfigure(2, weight=1)
chat_view.grid_rowconfigure(0, weight=1)
chat_view.grid_rowconfigure(1, weight=0)

settings_view = tk.Frame(main_holder, bg=BG_MAIN)
settings_view.grid(row=0, column=0, sticky="nsew")


def _visa_chatt():
    chat_view.tkraise()
    try:
        entry.focus_set()
    except Exception:
        pass


def _visa_installningar():
    _bygg_installningar()
    settings_view.tkraise()


# ============ SIDOPANEL ============
sidebar = tk.Frame(root, bg=SIDEBAR_BG, width=SIDO_BREDD)
sidebar.grid(row=0, column=0, sticky="nsew")
sidebar.grid_propagate(False)
sidebar.grid_columnconfigure(0, weight=1)
sidebar.grid_rowconfigure(3, weight=1)

# -- Varumarke --
brand = tk.Frame(sidebar, bg=SIDEBAR_BG)
brand.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 12))
brand_dot = tk.Canvas(brand, width=24, height=24, bg=SIDEBAR_BG,
                      highlightthickness=0)
brand_dot.pack(side="left")
_rita_dot(brand_dot, 12, 12, 22, GREEN, inner=SIDEBAR_BG)
tk.Label(brand, text="Offline AI", bg=SIDEBAR_BG, fg=FG_TEXT,
         font=("Segoe UI Semibold", 14)).pack(side="left", padx=(10, 0))

# -- Ny chatt (rundad knapp) --
_nyk = _rund_knapp(sidebar, "New chat", lambda: _ny_chatt(vaxla=True),
                   fyll="#f2f4f3", txtfarg=FG_TEXT, hover=SIDEBAR_HOVER,
                   ikon="＋", align="left", hojd=42, radie=13)
_nyk.grid(row=1, column=0, sticky="ew", padx=14, pady=(2, 14))

tk.Label(sidebar, text="CHATS", bg=SIDEBAR_BG, fg=FG_DIM,
         font=("Segoe UI Semibold", 9)).grid(row=2, column=0, sticky="w",
                                             padx=22, pady=(2, 6))

lista_holder = tk.Frame(sidebar, bg=SIDEBAR_BG)
lista_holder.grid(row=3, column=0, sticky="nsew", padx=10)
chattlista = tk.Frame(lista_holder, bg=SIDEBAR_BG)
chattlista.pack(fill="both", expand=True)

# -- Kontoknapp (canvas, rundad) --
konto = tk.Canvas(sidebar, height=60, bg=SIDEBAR_BG, highlightthickness=0,
                  cursor="hand2")
konto.grid(row=4, column=0, sticky="ew", padx=12, pady=(6, 14))


def _draw_konto(fill):
    konto.delete("all")
    w = konto.winfo_width() or 248
    _rundad_rekt(konto, 2, 3, w - 2, 57, radie=14, fill=fill,
                 outline=BORDER_COL)
    _rita_dot(konto, 30, 32, 32, GREEN)
    konto.create_text(30, 32, text="J", fill="#ffffff",
                      font=("Segoe UI Semibold", 14))
    konto.create_text(58, 24, text="Johan", anchor="w", fill=FG_TEXT,
                      font=("Segoe UI Semibold", 11))
    konto.create_text(58, 41, text="Personal · Free trial", anchor="w",
                      fill=FG_DIM, font=FONT_SMALL)
    konto.create_text(w - 18, 30, text="⚙", anchor="e", fill=FG_MUTED,
                      font=("Segoe UI", 14))


konto.bind("<Configure>", lambda e: _draw_konto(BG_MAIN))
konto.bind("<Enter>", lambda e: _draw_konto(SIDEBAR_HOVER))
konto.bind("<Leave>", lambda e: _draw_konto(BG_MAIN))
konto.bind("<Button-1>", lambda e: _visa_installningar())


# ============ CHATT: SKROLLBAR MEDDELANDEYTA ============
yttre_ram = tk.Frame(chat_view, bg=BG_MAIN)
yttre_ram.grid(row=0, column=1, sticky="nsew")
yttre_ram.grid_rowconfigure(0, weight=1)
yttre_ram.grid_columnconfigure(0, weight=1)

meddelande_canvas = tk.Canvas(yttre_ram, bg=BG_MAIN, highlightthickness=0,
                              borderwidth=0)
meddelande_canvas.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=22)

SCROLLBAR_BREDD = 7
scrollbar_canvas = tk.Canvas(yttre_ram, width=SCROLLBAR_BREDD, bg=BG_MAIN,
                             highlightthickness=0, borderwidth=0)
scrollbar_canvas.grid(row=0, column=1, sticky="ns", padx=(2, 3), pady=22)
_scroll_thumb = scrollbar_canvas.create_rectangle(
    0, 0, SCROLLBAR_BREDD, 40, fill="#c9cdd1", outline="")


def _rita_scroll_thumb(*_):
    hojd = scrollbar_canvas.winfo_height()
    forsta, sista = meddelande_canvas.yview()
    if sista - forsta >= 0.999:
        scrollbar_canvas.itemconfig(_scroll_thumb, state="hidden")
        return
    scrollbar_canvas.itemconfig(_scroll_thumb, state="normal")
    scrollbar_canvas.coords(_scroll_thumb, 0, hojd * forsta,
                            SCROLLBAR_BREDD, hojd * sista)


meddelande_canvas.configure(yscrollcommand=lambda f, s: _rita_scroll_thumb())
scrollbar_canvas.bind("<Configure>", _rita_scroll_thumb)


def _scroll_drag(event):
    hojd = scrollbar_canvas.winfo_height()
    if hojd > 0:
        meddelande_canvas.yview_moveto(max(0, min(1, event.y / hojd)))


scrollbar_canvas.bind("<Button-1>", _scroll_drag)
scrollbar_canvas.bind("<B1-Motion>", _scroll_drag)

meddelanden_frame = tk.Frame(meddelande_canvas, bg=BG_MAIN)
_frame_id = meddelande_canvas.create_window((0, 0), window=meddelanden_frame,
                                            anchor="nw")


def _pa_konfigurera_ram(event):
    meddelande_canvas.configure(scrollregion=meddelande_canvas.bbox("all"))
    meddelande_canvas.itemconfig(_frame_id, width=meddelande_canvas.winfo_width())


meddelanden_frame.bind("<Configure>", _pa_konfigurera_ram)
meddelande_canvas.bind("<Configure>",
    lambda e: meddelande_canvas.itemconfig(_frame_id, width=e.width))


def _mushjul(event):
    meddelande_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


meddelande_canvas.bind_all("<MouseWheel>", _mushjul)


def _scrolla_ner():
    root.after(10, lambda: meddelande_canvas.yview_moveto(1.0))


# ============ MEDDELANDEN ============
def lagg_till_anvandarbubbla(text):
    rad = tk.Frame(meddelanden_frame, bg=BG_MAIN)
    rad.pack(fill="x", pady=(8, 8))
    bubbla = tk.Canvas(rad, bg=BG_MAIN, highlightthickness=0, borderwidth=0)
    bubbla.pack(side="right", padx=(100, 6))
    _mat = tk.Label(bubbla, text=text, font=FONT_MAIN, wraplength=500,
                    justify="left", padx=16, pady=11)
    _mat.update_idletasks()
    bredd = _mat.winfo_reqwidth()
    hojd = _mat.winfo_reqheight()
    _mat.destroy()
    bubbla.configure(width=bredd, height=hojd)
    # Rita den rundade bubblan direkt pa canvasen och lagg texten ovanpa
    # med create_text - sa syns de runda hornen (en Label ovanpa skulle
    # tacka dem med sin fyrkantiga bakgrund).
    _rundad_rekt(bubbla, 1, 1, bredd - 1, hojd - 1, radie=18,
                 fill=BG_BUBBLE_USER, outline=BG_BUBBLE_USER)
    bubbla.create_text(16, 11, text=text, font=FONT_MAIN, fill=FG_TEXT,
                       anchor="nw", width=bredd - 32, justify="left")
    _scrolla_ner()


# ---- Markdown-rendering (assistentsvar) ----
_MD_INLINE = re.compile(r"(\*\*.+?\*\*|__.+?__|\*[^*\n]+?\*|_[^_\n]+?_|`[^`\n]+?`)")


def _konfigurera_md_taggar(t):
    t.tag_configure("p", spacing1=1, spacing3=9)
    t.tag_configure("bold", font=("Segoe UI Semibold", 13))
    t.tag_configure("italic", font=("Segoe UI", 13, "italic"))
    t.tag_configure("code", font=("Consolas", 12), foreground=CODE_FG)
    t.tag_configure("h1", font=("Segoe UI Semibold", 18), spacing1=10, spacing3=6)
    t.tag_configure("h2", font=("Segoe UI Semibold", 15), spacing1=9, spacing3=5)
    t.tag_configure("h3", font=("Segoe UI Semibold", 13), spacing1=8, spacing3=4)
    t.tag_configure("li", lmargin1=22, lmargin2=44, spacing3=5)
    t.tag_configure("codeblock", font=("Consolas", 12), background=CODE_BG,
                    foreground="#0f172a", lmargin1=16, lmargin2=16,
                    rmargin=16, spacing1=8, spacing3=8)


def _md_inline(t, s, extra=()):
    pos = 0
    for m in _MD_INLINE.finditer(s):
        if m.start() > pos:
            t.insert("end", s[pos:m.start()], extra)
        tok = m.group(1)
        if tok.startswith("**") or tok.startswith("__"):
            t.insert("end", tok[2:-2], extra + ("bold",))
        elif tok.startswith("`"):
            t.insert("end", tok[1:-1], extra + ("code",))
        else:
            t.insert("end", tok[1:-1], extra + ("italic",))
        pos = m.end()
    if pos < len(s):
        t.insert("end", s[pos:], extra)


def _rendera_markdown(t, md):
    md = (md or "").replace("\r\n", "\n").replace("\r", "\n")
    rader = md.split("\n")
    i = 0
    while i < len(rader):
        rad = rader[i]
        if re.match(r"^\s*```", rad):
            i += 1
            kod = []
            while i < len(rader) and not re.match(r"^\s*```", rader[i]):
                kod.append(rader[i]); i += 1
            i += 1
            t.insert("end", "\n".join(kod) + "\n", ("codeblock",))
            continue
        if rad.strip() == "":
            i += 1
            continue
        hm = re.match(r"^\s*(#{1,3})\s+(.*)$", rad)
        if hm:
            _md_inline(t, hm.group(2).strip(), ("h%d" % len(hm.group(1)),))
            t.insert("end", "\n"); i += 1
            continue
        bm = re.match(r"^\s*[-*•]\s+(.*)$", rad)
        if bm:
            t.insert("end", "•   ", ("li",))
            _md_inline(t, bm.group(1).strip(), ("li",))
            t.insert("end", "\n"); i += 1
            continue
        nm = re.match(r"^\s*(\d+)[.)]\s+(.*)$", rad)
        if nm:
            t.insert("end", nm.group(1) + ".   ", ("li",))
            _md_inline(t, nm.group(2).strip(), ("li",))
            t.insert("end", "\n"); i += 1
            continue
        para = [rad]; i += 1
        while (i < len(rader) and rader[i].strip() != ""
               and not re.match(r"^\s*(#{1,3}\s|[-*•]\s|\d+[.)]\s|```)",
                                rader[i])):
            para.append(rader[i]); i += 1
        _md_inline(t, " ".join(s.strip() for s in para), ("p",))
        t.insert("end", "\n")


def _auto_hojd(t):
    try:
        t.update_idletasks()
        n = t.count("1.0", "end-1c", "displaylines")
        if isinstance(n, (tuple, list)):
            n = n[0]
        t.configure(height=max(1, int(n or 1)))
    except Exception:
        try:
            t.configure(height=max(1, int(t.index("end-1c").split(".")[0])))
        except Exception:
            t.configure(height=4)


def _satt_svarstext(t, md):
    try:
        t.delete("1.0", "end")
    except Exception:
        return
    _rendera_markdown(t, md)
    _auto_hojd(t)
    root.after(30, lambda: _auto_hojd(t))
    _scrolla_ner()


def lagg_till_assistentmeddelande(text):
    rad = tk.Frame(meddelanden_frame, bg=BG_MAIN)
    rad.pack(fill="x", pady=(8, 8))
    huv = tk.Frame(rad, bg=BG_MAIN)
    huv.pack(side="top", fill="x", anchor="w")
    markor = tk.Canvas(huv, width=24, height=24, bg=BG_MAIN,
                       highlightthickness=0)
    markor.pack(side="left", padx=(2, 10), anchor="n")
    _rita_dot(markor, 12, 12, 22, GREEN, inner=BG_MAIN)
    txt = tk.Text(huv, bg=BG_MAIN, fg=FG_TEXT, font=FONT_MAIN, wrap="word",
                  relief="flat", borderwidth=0, highlightthickness=0,
                  padx=0, pady=0, cursor="arrow", height=1, insertwidth=0,
                  selectbackground=SEL_BG, spacing1=1, spacing3=5)
    txt.pack(side="left", fill="x", expand=True, anchor="w")
    _konfigurera_md_taggar(txt)
    txt.bind("<Key>", lambda e: "break")

    def _fwd_wheel(e):
        meddelande_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        return "break"
    txt.bind("<MouseWheel>", _fwd_wheel)
    txt.bind("<Configure>", lambda e: _auto_hojd(txt))
    if text:
        _satt_svarstext(txt, text)

    def _kopiera(event=None):
        root.clipboard_clear()
        root.clipboard_append(txt.get("1.0", "end-1c"))
        kopiera_knapp.config(text="Copied")
        root.after(1200, lambda: kopiera_knapp.config(text="Copy"))

    kopiera_knapp = tk.Label(rad, text="Copy", bg=BG_MAIN, fg=FG_DIM,
                             font=FONT_SMALL, cursor="hand2")
    kopiera_knapp.pack(side="top", padx=(36, 0), pady=(3, 0), anchor="w")
    kopiera_knapp.bind("<Button-1>", _kopiera)
    kopiera_knapp.bind("<Enter>", lambda e: kopiera_knapp.config(fg=FG_MUTED))
    kopiera_knapp.bind("<Leave>", lambda e: kopiera_knapp.config(fg=FG_DIM))
    _scrolla_ner()
    return txt


def lagg_till_statusrad():
    rad = tk.Frame(meddelanden_frame, bg=BG_MAIN)
    rad.pack(fill="x", pady=(8, 8))
    prick = tk.Canvas(rad, width=24, height=24, bg=BG_MAIN,
                      highlightthickness=0)
    prick.pack(side="left", padx=(2, 10), anchor="n")
    _rita_dot(prick, 12, 12, 22, GREEN, inner=BG_MAIN)
    etikett = tk.Label(rad, text="Thinking...", bg=BG_MAIN, fg=FG_MUTED,
                       font=FONT_STATUS, justify="left", anchor="w")
    etikett.pack(side="left", fill="x")
    _scrolla_ner()
    return etikett


# ---- Tomt lage ----
_tomt_ref = {"w": None}


def _visa_tomt_lage():
    ram = tk.Frame(meddelanden_frame, bg=BG_MAIN)
    ram.pack(fill="x", pady=(160, 0))
    tk.Label(ram, text="How can I help?", bg=BG_MAIN, fg=FG_TEXT,
             font=("Segoe UI Light", 30)).pack()
    tk.Label(ram, text="Runs on your machine · private · offline-first",
             bg=BG_MAIN, fg=FG_DIM, font=FONT_STATUS).pack(pady=(12, 0))
    _tomt_ref["w"] = ram


def _dolj_tomt_lage():
    w = _tomt_ref["w"]
    try:
        if w is not None and w.winfo_exists():
            w.destroy()
    except Exception:
        pass
    _tomt_ref["w"] = None


# ============ CHATT-HANTERING ============
def _rensa_meddelanden():
    for w in meddelanden_frame.winfo_children():
        w.destroy()


def _ladda_chatt_i_vy(chatt):
    _rensa_meddelanden()
    _tomt_ref["w"] = None
    if not chatt["logg"]:
        _visa_tomt_lage()
    else:
        for roll, text in chatt["logg"]:
            if roll == "user":
                lagg_till_anvandarbubbla(text)
            else:
                tw = lagg_till_assistentmeddelande("")
                _satt_svarstext(tw, text)
    _scrolla_ner()


def _byt_chatt(chatt):
    _AKTIV["chatt"] = chatt
    _ladda_chatt_i_vy(chatt)
    _rita_chattlista()
    _visa_chatt()


def _ny_chatt(vaxla=True, namn=None):
    chatt = {"namn": namn or "New chat", "auto": namn is None,
             "historik": [], "logg": []}
    _CHATTAR.insert(0, chatt)
    if vaxla:
        _byt_chatt(chatt)
    else:
        _rita_chattlista()
    return chatt


def _radera_chatt(chatt):
    if chatt in _CHATTAR:
        _CHATTAR.remove(chatt)
    if not _CHATTAR:
        _ny_chatt(vaxla=True)
        return
    if _AKTIV["chatt"] is chatt:
        _byt_chatt(_CHATTAR[0])
    else:
        _rita_chattlista()


def _starta_rename(chatt, c):
    e = tk.Entry(c, bg=CARD_BG, fg=FG_TEXT, font=FONT_SIDE, relief="flat",
                 insertbackground=FG_TEXT, highlightthickness=1,
                 highlightbackground=GREEN, highlightcolor=GREEN)
    c.create_window(10, 20, window=e, anchor="w",
                    width=max(60, c.winfo_width() - 44))
    e.insert(0, chatt["namn"])
    e.focus_set()
    e.select_range(0, tk.END)

    def _commit(_=None):
        ny = e.get().strip()
        if ny:
            chatt["namn"] = ny
            chatt["auto"] = False
        _rita_chattlista()
    e.bind("<Return>", _commit)
    e.bind("<FocusOut>", _commit)
    e.bind("<Escape>", lambda _e: _rita_chattlista())


def _chattrad(chatt):
    c = tk.Canvas(chattlista, height=40, bg=SIDEBAR_BG, highlightthickness=0,
                  cursor="hand2")
    c.pack(fill="x", pady=1)
    state = {"hover": False}

    def _draw():
        c.delete("all")
        w = c.winfo_width() or 250
        akt = chatt is _AKTIV["chatt"]
        if akt:
            _rundad_rekt(c, 2, 3, w - 2, 37, radie=11, fill=SIDEBAR_ACTIVE,
                         outline=SIDEBAR_ACTIVE)
        elif state["hover"]:
            _rundad_rekt(c, 2, 3, w - 2, 37, radie=11, fill=SIDEBAR_HOVER,
                         outline=SIDEBAR_HOVER)
        fg = FG_TEXT if akt else FG_MUTED
        c.create_text(14, 20, text=_kort_titel(chatt["namn"], 25),
                      anchor="w", fill=fg, font=FONT_SIDE)
        if akt or state["hover"]:
            c.create_text(w - 14, 20, text="✕", anchor="e", fill=FG_DIM,
                          font=FONT_SMALL)

    def _enter(_e):
        state["hover"] = True
        _draw()

    def _leave(_e):
        state["hover"] = False
        _draw()

    def _click(e):
        if (state["hover"] or chatt is _AKTIV["chatt"]) and \
                e.x >= (c.winfo_width() - 30):
            _radera_chatt(chatt)
            return
        _byt_chatt(chatt)

    c.bind("<Configure>", lambda e: _draw())
    c.bind("<Enter>", _enter)
    c.bind("<Leave>", _leave)
    c.bind("<Button-1>", _click)
    c.bind("<Double-Button-1>", lambda e: _starta_rename(chatt, c))
    _draw()


def _rita_chattlista():
    for w in chattlista.winfo_children():
        w.destroy()
    for chatt in _CHATTAR:
        _chattrad(chatt)


# ============ INPUTFALT ============
input_yttre = tk.Frame(chat_view, bg=BG_MAIN)
input_yttre.grid(row=1, column=1, sticky="ew", pady=(0, 16))
input_yttre.grid_columnconfigure(0, weight=1)

_bifogad_bild = None
bifoga_info = tk.Label(input_yttre, text="", bg=BG_MAIN, fg=FG_MUTED,
                       font=FONT_STATUS, anchor="w", cursor="hand2")
bifoga_info.grid(row=0, column=0, sticky="w", padx=(10, 0), pady=(0, 4))
bifoga_info.grid_remove()

input_canvas = tk.Canvas(input_yttre, bg=BG_MAIN, highlightthickness=0,
                         borderwidth=0, height=72)
input_canvas.grid(row=1, column=0, sticky="ew")
input_yttre.grid_rowconfigure(1, minsize=72)

hint = tk.Label(input_yttre,
                text="Offline AI can make mistakes. Your chats stay on this device.",
                bg=BG_MAIN, fg=FG_DIM, font=("Segoe UI", 9))
hint.grid(row=2, column=0, pady=(7, 0))


def _valj_bild(event=None):
    global _bifogad_bild
    vag = filedialog.askopenfilename(
        title="Choose an image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"),
                   ("All files", "*.*")])
    if vag:
        _bifogad_bild = vag
        namn = os.path.basename(vag)
        if len(namn) > 42:
            namn = namn[:39] + "..."
        bifoga_info.config(text="🖼  " + namn + "      ✕  remove")
        bifoga_info.grid()


def _rensa_bifogad_bild(event=None):
    global _bifogad_bild
    _bifogad_bild = None
    bifoga_info.config(text="")
    bifoga_info.grid_remove()


bifoga_info.bind("<Button-1>", _rensa_bifogad_bild)


def _rita_inputbakgrund(event=None):
    input_canvas.delete("bg")
    bredd = input_canvas.winfo_width()
    _rundad_rekt(input_canvas, 2, 2, bredd - 2, 70, radie=18,
                 fill=BG_INPUT, outline=INPUT_BORDER, tags="bg")
    input_canvas.tag_lower("bg")
    input_canvas.coords(bifoga_knapp, 34, 36)
    input_canvas.coords(entry_window, 60, 36)
    input_canvas.itemconfig(entry_window, width=bredd - 124)
    _rita_skicka_knapp(bredd - 46, 36)


input_canvas.bind("<Configure>", _rita_inputbakgrund)

entry = tk.Entry(input_canvas, bg=BG_INPUT, fg=FG_TEXT, font=FONT_MAIN,
                 insertbackground=GREEN, relief="flat", borderwidth=0,
                 highlightthickness=0)
entry_window = input_canvas.create_window(20, 36, window=entry, anchor="w")
entry.bind("<Return>", lambda event: skicka())


def _skapa_skicka_knapp_bild(diameter=34, faktor=8):
    stor = diameter * faktor
    img = Image.new("RGBA", (stor, stor), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([0, 0, stor - 1, stor - 1], fill=ACCENT)
    cx, cy = stor / 2, stor / 2
    s = stor * 0.27
    d.polygon([(cx, cy - s), (cx - s * 0.55, cy + s * 0.12),
               (cx + s * 0.55, cy + s * 0.12)], fill=ACCENT_TEXT)
    d.rectangle([cx - s * 0.16, cy + s * 0.12 - 2, cx + s * 0.16, cy + s],
                fill=ACCENT_TEXT)
    img = img.resize((diameter, diameter), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


_skicka_knapp_bild = _skapa_skicka_knapp_bild()
skicka_knapp = input_canvas.create_image(0, 0, image=_skicka_knapp_bild,
                                         tags="knapp")


def _skapa_bifoga_knapp_bild(diameter=28, faktor=8):
    stor = diameter * faktor
    img = Image.new("RGBA", (stor, stor), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ring = int(stor * 0.05)
    d.ellipse([ring, ring, stor - 1 - ring, stor - 1 - ring],
              outline="#9aa0a6", width=ring)
    cx, cy = stor / 2, stor / 2
    arm = stor * 0.22
    tj = int(stor * 0.06)
    d.line([(cx - arm, cy), (cx + arm, cy)], fill="#9aa0a6", width=tj)
    d.line([(cx, cy - arm), (cx, cy + arm)], fill="#9aa0a6", width=tj)
    img = img.resize((diameter, diameter), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


_bifoga_knapp_bild = _skapa_bifoga_knapp_bild()
bifoga_knapp = input_canvas.create_image(0, 0, image=_bifoga_knapp_bild,
                                         tags="bifoga")
input_canvas.tag_bind("bifoga", "<Button-1>", _valj_bild)
input_canvas.tag_bind("bifoga", "<Enter>",
                      lambda e: input_canvas.config(cursor="hand2"))
input_canvas.tag_bind("bifoga", "<Leave>",
                      lambda e: input_canvas.config(cursor=""))


def _rita_skicka_knapp(x, y):
    input_canvas.coords(skicka_knapp, x, y)


input_canvas.tag_bind("knapp", "<Button-1>", lambda e: skicka())
input_canvas.tag_bind("knapp", "<Enter>",
                      lambda e: input_canvas.config(cursor="hand2"))
input_canvas.tag_bind("knapp", "<Leave>",
                      lambda e: input_canvas.config(cursor=""))


# ============ INSTALLNINGAR (Account / Billing / Usage — UX-mock) ============
_settings = {"byggd": False, "flik": "Account", "content": None, "navc": {}}


def _falt(parent, etikett, varde, knapp=None):
    rad = tk.Frame(parent, bg=CARD_BG)
    rad.pack(fill="x", pady=6)
    tk.Label(rad, text=etikett, bg=CARD_BG, fg=FG_MUTED, font=FONT_SMALL,
             width=17, anchor="w").pack(side="left")
    box = tk.Canvas(rad, height=36, bg=CARD_BG, highlightthickness=0)
    box.pack(side="left", fill="x", expand=True)

    def _bx(_e=None):
        box.delete("all")
        w = box.winfo_width() or 220
        _rundad_rekt(box, 1, 1, w - 1, 35, radie=10, fill="#f5f6f6",
                     outline=BORDER_COL)
        box.create_text(13, 18, text=varde, anchor="w", fill=FG_TEXT,
                        font=FONT_SIDE)
    box.bind("<Configure>", _bx)
    if knapp:
        tk.Label(rad, text=knapp, bg=CARD_BG, fg=GREEN, font=FONT_SMALL,
                 cursor="hand2").pack(side="left", padx=(12, 0))


def _knapp(parent, text, primar=True, sida="left"):
    br = max(112, len(text) * 9 + 42)
    if primar:
        b = _rund_knapp(parent, text, lambda: None, fyll=GREEN,
                        txtfarg="#ffffff", hover=GREEN_DK, hojd=40, radie=11,
                        bredd=br, font=("Segoe UI Semibold", 11))
    else:
        b = _rund_knapp(parent, text, lambda: None, fyll=CARD_BG,
                        txtfarg=FG_TEXT, hover=SIDEBAR_HOVER, ram=BORDER_COL,
                        hojd=40, radie=11, bredd=br,
                        font=("Segoe UI Semibold", 11))
    b.pack(side=sida, padx=(0, 10))
    return b


def _byt_flik(namn):
    _settings["flik"] = namn
    for fn in _settings["navc"].values():
        fn()
    for w in _settings["content"].winfo_children():
        w.destroy()
    if namn == "Account":
        _panel_account(_settings["content"])
    elif namn == "Billing":
        _panel_billing(_settings["content"])
    else:
        _panel_usage(_settings["content"])


def _navitem(parent, namn, ikon):
    c = tk.Canvas(parent, height=42, bg=BG_MAIN, highlightthickness=0,
                  cursor="hand2")
    c.pack(fill="x", pady=2)

    def _draw():
        c.delete("all")
        w = c.winfo_width() or 190
        akt = _settings["flik"] == namn
        if akt:
            _rundad_rekt(c, 1, 2, w - 1, 40, radie=11, fill=SIDEBAR_ACTIVE,
                         outline=SIDEBAR_ACTIVE)
        fg = GREEN_DK if akt else FG_MUTED
        fam = "Segoe UI Semibold" if akt else "Segoe UI"
        c.create_text(16, 21, text=ikon + "    " + namn, anchor="w",
                      fill=fg, font=(fam, 11))
    c.bind("<Configure>", lambda e: _draw())
    c.bind("<Button-1>", lambda e: _byt_flik(namn))
    _settings["navc"][namn] = _draw
    _draw()


def _bygg_installningar():
    if _settings["byggd"]:
        return
    _settings["byggd"] = True
    settings_view.grid_rowconfigure(1, weight=1)
    settings_view.grid_columnconfigure(0, weight=1)

    head = tk.Frame(settings_view, bg=BG_MAIN)
    head.grid(row=0, column=0, sticky="ew", padx=40, pady=(28, 10))
    tk.Label(head, text="Settings", bg=BG_MAIN, fg=FG_TEXT,
             font=("Segoe UI Semibold", 22)).pack(side="left")
    _stang = _rund_knapp(head, "Close", lambda: _visa_chatt(), fyll=BG_MAIN,
                         txtfarg=FG_MUTED, hover=SIDEBAR_HOVER, ram=BORDER_COL,
                         ikon="✕", hojd=36, radie=10, bredd=104,
                         font=("Segoe UI", 10))
    _stang.pack(side="right")

    body = tk.Frame(settings_view, bg=BG_MAIN)
    body.grid(row=1, column=0, sticky="nsew", padx=40, pady=(6, 28))
    body.grid_columnconfigure(0, weight=0, minsize=200)
    body.grid_columnconfigure(1, weight=1)
    body.grid_rowconfigure(0, weight=1)

    nav = tk.Frame(body, bg=BG_MAIN)
    nav.grid(row=0, column=0, sticky="nw")
    for namn, ikon in (("Account", "👤"), ("Billing", "💳"), ("Usage", "📊")):
        _navitem(nav, namn, ikon)

    cwrap = tk.Frame(body, bg=BG_MAIN)
    cwrap.grid(row=0, column=1, sticky="nsew", padx=(28, 0))
    content = tk.Frame(cwrap, bg=BG_MAIN)
    content.pack(fill="both", expand=True)
    _settings["content"] = content
    _byt_flik("Account")


def _korttitel(inner, text):
    tk.Label(inner, text=text, bg=CARD_BG, fg=FG_TEXT,
             font=("Segoe UI Semibold", 12)).pack(anchor="w", pady=(0, 12))


def _panel_account(parent):
    inner = _rund_kort(parent)
    _korttitel(inner, "Profile")
    top = tk.Frame(inner, bg=CARD_BG)
    top.pack(fill="x", pady=(0, 14))
    av = tk.Canvas(top, width=60, height=60, bg=CARD_BG, highlightthickness=0)
    av.pack(side="left")
    _rita_dot(av, 30, 30, 56, GREEN)
    av.create_text(30, 30, text="J", fill="#ffffff",
                   font=("Segoe UI Semibold", 24))
    nm = tk.Frame(top, bg=CARD_BG)
    nm.pack(side="left", padx=(16, 0))
    tk.Label(nm, text="Johan", bg=CARD_BG, fg=FG_TEXT,
             font=("Segoe UI Semibold", 15)).pack(anchor="w")
    tk.Label(nm, text="johan@pfannenstill.se", bg=CARD_BG, fg=FG_MUTED,
             font=FONT_SIDE).pack(anchor="w")
    _falt(inner, "Display name", "Johan", knapp="Edit")
    _falt(inner, "Email", "johan@pfannenstill.se", knapp="Edit")
    _falt(inner, "Password", "••••••••••", knapp="Change")

    pref = _rund_kort(parent)
    _korttitel(pref, "Preferences")
    _falt(pref, "Language", "English  ▾")
    _falt(pref, "Theme", "Light  ▾")
    rad = tk.Frame(pref, bg=CARD_BG)
    rad.pack(fill="x", pady=8)
    tk.Label(rad, text="Allow web search", bg=CARD_BG, fg=FG_MUTED,
             font=FONT_SMALL, width=17, anchor="w").pack(side="left")
    sw = tk.Canvas(rad, width=46, height=26, bg=CARD_BG, highlightthickness=0)
    sw.pack(side="left")
    _rundad_rekt(sw, 2, 2, 44, 24, radie=12, fill=GREEN, outline="")
    _rita_dot(sw, 34, 13, 18, "#ffffff")

    br = tk.Frame(parent, bg=BG_MAIN)
    br.pack(fill="x", pady=(2, 0))
    _knapp(br, "Save changes", True)
    _knapp(br, "Sign out", False)


def _panel_billing(parent):
    inner = _rund_kort(parent)
    _korttitel(inner, "Current plan")
    top = tk.Frame(inner, bg=CARD_BG)
    top.pack(fill="x")
    left = tk.Frame(top, bg=CARD_BG)
    left.pack(side="left")
    tk.Label(left, text="Personal", bg=CARD_BG, fg=FG_TEXT,
             font=("Segoe UI Semibold", 16)).pack(side="left")
    chip = tk.Canvas(left, width=150, height=24, bg=CARD_BG,
                     highlightthickness=0)
    chip.pack(side="left", padx=(10, 0))
    _rundad_rekt(chip, 1, 1, 149, 23, radie=11, fill=GREEN_SOFT,
                 outline=GREEN_SOFT)
    chip.create_text(75, 12, text="Free trial · 12 days left", fill=GREEN_DK,
                     font=FONT_SMALL)
    tk.Label(top, text="€9 / month", bg=CARD_BG, fg=FG_TEXT,
             font=("Segoe UI Semibold", 16)).pack(side="right")
    tk.Label(inner, text="Full hybrid retrieval · all 12 languages · coding "
             "model · priority updates.", bg=CARD_BG, fg=FG_MUTED,
             font=FONT_SIDE, anchor="w", justify="left").pack(fill="x",
                                                              pady=(10, 14))
    kr = tk.Frame(inner, bg=CARD_BG)
    kr.pack(fill="x")
    _knapp(kr, "Manage plan", True)
    _knapp(kr, "Upgrade to Pro", False)

    pm = _rund_kort(parent)
    _korttitel(pm, "Payment method")
    rad = tk.Frame(pm, bg=CARD_BG)
    rad.pack(fill="x")
    vc = tk.Canvas(rad, width=42, height=27, bg=CARD_BG, highlightthickness=0)
    vc.pack(side="left")
    _rundad_rekt(vc, 1, 1, 41, 26, radie=6, fill="#1a1f71", outline="")
    vc.create_text(21, 13, text="VISA", fill="#ffffff",
                   font=("Segoe UI Semibold", 9))
    tk.Label(rad, text="  Visa ending in 4242", bg=CARD_BG, fg=FG_TEXT,
             font=FONT_SIDE).pack(side="left", padx=(6, 0))
    tk.Label(rad, text="Expires 08/28", bg=CARD_BG, fg=FG_MUTED,
             font=FONT_SMALL).pack(side="left", padx=(14, 0))
    tk.Label(rad, text="Update", bg=CARD_BG, fg=GREEN, font=FONT_SMALL,
             cursor="hand2").pack(side="right")

    hist = _rund_kort(parent)
    _korttitel(hist, "Billing history")
    for datum, txt, belopp, status in (
            ("Aug 1, 2026", "Personal — monthly", "€9.00", "Trial"),
            ("Jul 1, 2026", "Personal — monthly", "€9.00", "Paid"),
            ("Jun 1, 2026", "Personal — monthly", "€9.00", "Paid")):
        r = tk.Frame(hist, bg=CARD_BG)
        r.pack(fill="x", pady=5)
        tk.Label(r, text=datum, bg=CARD_BG, fg=FG_MUTED, font=FONT_SMALL,
                 width=14, anchor="w").pack(side="left")
        tk.Label(r, text=txt, bg=CARD_BG, fg=FG_TEXT, font=FONT_SIDE,
                 anchor="w").pack(side="left", fill="x", expand=True)
        tk.Label(r, text=status, bg=CARD_BG, fg=FG_MUTED,
                 font=FONT_SMALL).pack(side="right", padx=(0, 16))
        tk.Label(r, text=belopp, bg=CARD_BG, fg=FG_TEXT, font=FONT_SIDE,
                 width=8, anchor="e").pack(side="right")


def _panel_usage(parent):
    tiles = tk.Frame(parent, bg=BG_MAIN)
    tiles.pack(fill="x", pady=(0, 16))
    for _i in range(4):
        tiles.grid_columnconfigure(_i, weight=1, uniform="tile")

    def _tile(col, varde, etikett):
        c = tk.Canvas(tiles, height=86, bg=BG_MAIN, highlightthickness=0)
        c.grid(row=0, column=col, sticky="ew",
               padx=(0 if col == 0 else 6, 0 if col == 3 else 6))

        def _d(_e=None):
            c.delete("all")
            w = c.winfo_width() or 150
            _rundad_rekt(c, 1, 1, w - 1, 84, radie=16, fill=CARD_BG,
                         outline=BORDER_COL)
            c.create_text(18, 34, text=varde, anchor="w", fill=FG_TEXT,
                          font=("Segoe UI Semibold", 20))
            c.create_text(18, 60, text=etikett, anchor="w", fill=FG_MUTED,
                          font=FONT_SMALL)
        c.bind("<Configure>", _d)
    _tile(0, "1,284", "Messages this month")
    _tile(1, "342", "Web searches")
    _tile(2, "6.9s", "Avg response time")
    _tile(3, "100%", "Processed on-device")

    ch = _rund_kort(parent)
    _korttitel(ch, "Messages per day — last 7 days")
    cv = tk.Canvas(ch, bg=CARD_BG, height=150, highlightthickness=0)
    cv.pack(fill="x")
    varden = [64, 92, 51, 120, 87, 143, 108]
    dagar = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def _stap(_=None):
        cv.delete("all")
        b = cv.winfo_width() or 560
        h = 150
        n = len(varden)
        marg = 26
        bw = (b - 2 * marg) / n
        mx = max(varden)
        for i, v in enumerate(varden):
            x = marg + i * bw
            bh = (v / mx) * (h - 46)
            _rundad_rekt(cv, x + 9, h - 24 - bh, x + bw - 9, h - 24,
                         radie=6, fill=GREEN, outline="")
            cv.create_text(x + bw / 2, h - 12, text=dagar[i], fill=FG_MUTED,
                           font=("Segoe UI", 8))
            cv.create_text(x + bw / 2, h - 30 - bh, text=str(v),
                           fill=FG_MUTED, font=("Segoe UI", 8))
    cv.bind("<Configure>", _stap)

    info = _rund_kort(parent)
    _korttitel(info, "Model & hardware")
    for k, v in (("Chat & code model", "Qwen3.5 9B — Q4_K_M (local)"),
                 ("Acceleration", "Vulkan · NVIDIA RTX 2060 (6 GB)"),
                 ("Context window", "4096 tokens"),
                 ("Data location", "This device only — nothing leaves")):
        r = tk.Frame(info, bg=CARD_BG)
        r.pack(fill="x", pady=5)
        tk.Label(r, text=k, bg=CARD_BG, fg=FG_MUTED, font=FONT_SMALL,
                 width=17, anchor="w").pack(side="left")
        tk.Label(r, text=v, bg=CARD_BG, fg=FG_TEXT, font=FONT_SIDE,
                 anchor="w").pack(side="left")


# ============ START ============
_laddstatus = tk.Label(input_yttre, text="", bg=BG_MAIN, fg=FG_MUTED,
                       font=FONT_STATUS)
_laddstatus.grid(row=3, column=0, pady=(4, 0))

# Tunn avdelare mellan (vit) sidopanel och (vit) chattyta. Skapas sist
# sa den ligger overst och doljs inte av att vyerna hojs (tkraise).
_divider = tk.Frame(root, bg=BORDER_SOFT, width=1)
_divider.place(x=SIDO_BREDD, y=0, relheight=1.0)

_ny_chatt(vaxla=True)
_visa_chatt()
entry.focus_set()

threading.Thread(target=ladda_modell, args=(_laddstatus,), daemon=True).start()

root.mainloop()
