# -*- coding: utf-8 -*-
"""Sprakstrangar for AIPrototype VERSION3.

STRUKTUR
    SPRAK["<sprak_kod>"]["<nyckel>"] = "text med ev. {slot}-mall"

ATT LAGGA TILL ETT NYTT SPRAK
    1. Kopiera SV som mall och byt namn (t.ex. NO for norska)
    2. Oversatt varje varde
    3. Lagg till i SPRAK-dicten langst ner
    4. Testa noga med regressionstest.py och test_bred_stabilitet.py

Strangar med {slot}-placeholder substitueras via str.format() vid uppslag.
Om en nyckel saknas i det valda spraket faller vi tillbaka till EN."""


# ============ SVENSKA ============
SV = {
    # ---- Enkla direktsvar fran uppslags-funktionerna ----
    "wiki_direkt_svar": "{roll} i {land} ar {namn}.",
    "wikidata_utmarkelse_legacy": "Vinnaren senast (ar {ar}) ar {namn}.",
    "utmarkelse_med_ar": "Senaste vinnaren (ar {ar}) ar {namn}.",
    "utmarkelse_utan_ar": "Senaste vinnaren ar {namn}.",
    "sport_superbowl_med_ar": "Senaste Super Bowl vanns av {vinnare} (sasong {ar}).",
    "sport_superbowl_utan_ar": "Senaste Super Bowl vanns av {vinnare}.",
    "sport_generisk": "Senaste vinnaren av {artikel} ar {champion}.",
    "software_med_datum": "Senaste stabila versionen av {namn} ar {version} (slappt {datum}).",
    "software_utan_datum": "Senaste stabila versionen av {namn} ar {version}.",
    "wikidata_universell": "{relation} for {entitet} ar {varde}.",
    "namn_binder": " och ",  # for listor: "A och B" / "A, B och C"
}


# ---- Fallback-svar nar sokning misslyckas ----
SV.update({
    "fallback_extraktion_oklar": (
        "Jag kunde inte hitta ett tillforlitligt svar pa detta baserat "
        "pa sokresultaten."
    ),
    "fallback_janej_oklart": (
        "Jag kunde inte faststalla ett sakert svar pa den har "
        "ja/nej-fragan utifran tillganglig information."
    ),
    "fallback_hedge_kravdes": (
        "Jag har inte hittat en verifierad kalla for detta just nu och "
        "vill inte gissa. Kolla en aktuell kalla for tillforlitlig info."
    ),
    "fallback_trasigt": (
        "Jag fick ett konstigt svar just nu och kunde inte reda ut det. "
        "Kan du prova att omformulera fragan?"
    ),
    "fallback_flertal_fragor": (
        "Jag kan just nu bara ge dig ett aktuellt faktum i taget (t.ex. "
        "\"vem ar president i USA just nu\"), inte historik eller flera "
        "fragor pa en gang. Kan du dela upp frågan, eller fraga om en "
        "sak at gangen?"
    ),
    "regelfix_svar": (
        "Det har ar en regel- eller siffer-baserad fraga (t.ex. "
        "visumkrav, skattesats, styrranta) dar aktuell och korrekt info "
        "ar viktig. Jag kunde inte hitta en verifierad kalla just nu, "
        "sa jag vill inte gissa. Kolla officiell kalla (t.ex. "
        "myndighetens hemsida) for att fa ratt aktuell uppgift."
    ),
})


# ---- Systemprompt (huvudinstruktion till Llama) ----
SV["system_prompt"] = (
    "Du ar en privat assistent som kors HELT LOKALT pa anvandarens egen "
    "dator (modell: Qwen3.5 9B). Ingenting anvandaren skriver lamnar "
    "enheten. Dagens datum ar {datum} - lita PA DETTA datum om nagon "
    "fragar vilken tid eller vilket ar det ar nu. "
    "SVARA ALLTID PA SAMMA SPRAK som anvandaren skrev sitt senaste "
    "meddelande pa. Byt aldrig sprak sjalv. Var koncis, saklig och direkt. "
    "Hitta ALDRIG pa fakta. Ar du osaker eller saknar aktuell information: "
    "saag det rakt ut - gissa aldrig och hitta aldrig pa titel, ar eller "
    "handling. "
    "Var artig men aldrig installsam: hall dig till sanningen aven nar den "
    "gar emot vad anvandaren tror, och smickra inte. "
    "Du ar ett verktyg - inte en person, van eller terapeut. Pasta aldrig "
    "att du har kanslor, medvetande eller nagon sarskild relation till "
    "anvandaren, och uppmuntra inte beroende. "
    "Forstark aldrig storslagna, paranoida eller verklighetsfranvanda "
    "pastaenden. Verkar nagon vara i kris eller ma daligt: mot dem med "
    "omtanke och foresla varsamt att prata med en anhorig eller "
    "fackperson - spela aldrig med i skadliga tankar. "
    "I politiska, religiosa eller kansliga fragor: ge balanserad och "
    "saklig information, driv inga egna asikter och skriv inte uppviglande "
    "eller ensidigt overtalande text. For medicinska, juridiska och "
    "ekonomiska fragor: ge allman info och paminn om att radfraga en "
    "fackperson. Hall innehallet familjevanligt. "
    "Lagg inte till parenteser med sprak- eller landsnoteringar pa slutet. "
    "Fragar nagon vem eller vad du ar: svara kort - en privat, lokal "
    "assistent pa deras egen enhet."
)


# ---- Systemprompt-tillagg (laggs pa efter uppslagen) ----
SV["prompt_wiki_varfor"] = (
    "\n\nDagens datum ar {datum}. Har ar en verifierad fakta fran "
    "Wikipedia:\n{fakta}\n\nOBS: Fragan kan bygga pa ett felaktigt "
    "antagande (t.ex. att nagon avgatt/blivit avsatt fast de faktiskt "
    "fortfarande innehar rollen enligt faktan ovan). Om fragans premiss "
    "inte stammer med faktan: paepeka det artigt istallet for att bara "
    "upprepa faktan meningslost eller hitta pa en forklaring."
)

SV["prompt_wiki_vanlig"] = (
    "\n\nDagens datum ar {datum}. Har ar en verifierad fakta fran "
    "Wikipedia:\n{fakta}\n\nVIKTIGT: du MASTE anvanda denna fakta i "
    "ditt svar. Saag INTE att du saknar realtidsinformation eller namn "
    "INTE din traningsdata - du HAR redan fatt den aktuella "
    "informationen ovan, anvand den direkt som om det vore din egen "
    "kunskap. Svara KORT och NATURLIGT (t.ex. bara '[Namn] ar [roll].') "
    "- borja INTE med att pasta att nagot ar 'fel' eller felaktigt om "
    "anvandaren inte faktiskt sagt nagot felaktigt."
)


SV["prompt_janej_med_analys"] = (
    "\n\nDagens datum ar {datum}. En noggrann analys av farska "
    "sokresultat visar foljande:\n{fakta}\n\nVIKTIGT: du MASTE basera "
    "ditt svar pa denna slutsats - motsag den INTE och forlita dig INTE "
    "pa ditt eget minne istallet."
)

SV["prompt_janej_oklart"] = (
    "\n\nOBS: En noggrann analys av sokresultaten kunde inte ge ett "
    "sakert ja/nej-svar. Var arlig med att du ar osaker istallet for "
    "att gissa."
)

SV["prompt_extraktion_oklar"] = (
    "\n\nOBS: En websokning gjordes men gav inget tydligt svar (kanske "
    "for att fragan var tvetydig, t.ex. saknade land/sammanhang). Var "
    "arlig med att du inte kunde hitta ett sakert svar, och fraga garna "
    "om anvandaren menar ett specifikt land/sammanhang. MYCKET VIKTIGT: "
    "namn INTE ett specifikt namn, datum eller annan detalj som du tror "
    "du minns fran innan - aven om du kanner dig saker kan din kunskap "
    "vara FORALDRAD (t.ex. en person som bytts ut sedan din traning). "
    "Saga bara att du ar osaker, hitta inte pa ett namn for att verka "
    "hjalpsam."
)

SV["prompt_sok_fakta"] = (
    "\n\nDagens datum ar {datum}. Har ar en verifierad fakta fran en "
    "websokning:\n{fakta}\n\nVIKTIGT: du MASTE anvanda denna fakta i "
    "ditt svar. Saag INTE att du saknar realtidsinformation eller namn "
    "INTE din traningsdata - du HAR redan fatt den aktuella "
    "informationen ovan, anvand den direkt som om det vore din egen "
    "kunskap."
)

SV["prompt_krav_aktuell_utan_wiki"] = (
    "\n\nOBS: Fragan verkar krava aktuell information, men websokning "
    "misslyckades (troligen inget internet just nu). Var tydlig med att "
    "du saknar aktuell info just nu istallet for att gissa. Namn INTE "
    "ett specifikt namn eller datum du tror du minns - din kunskap kan "
    "vara FORALDRAD."
)


# ============ ENGELSKA ============
EN = {
    # ---- Enkla direktsvar fran uppslags-funktionerna ----
    "wiki_direkt_svar": "{titel} is {namn}.",  # samma template som SV men "titel"-baserad
    "wikidata_utmarkelse_legacy": "The winner most recently (year {ar}) is {namn}.",
    "utmarkelse_med_ar": "The most recent winner (year {ar}) is {namn}.",
    "utmarkelse_utan_ar": "The most recent winner is {namn}.",
    "sport_superbowl_med_ar": "The most recent Super Bowl was won by {vinnare} (season {ar}).",
    "sport_superbowl_utan_ar": "The most recent Super Bowl was won by {vinnare}.",
    "sport_generisk": "The most recent winner of {artikel} is {champion}.",
    "software_med_datum": "The latest stable version of {namn} is {version} (released {datum}).",
    "software_utan_datum": "The latest stable version of {namn} is {version}.",
    "wikidata_universell": "The {relation} of {entitet} is {varde}.",
    "namn_binder": " and ",

    # ---- Fallback-svar ----
    "fallback_extraktion_oklar": (
        "I couldn't find a reliable answer to this based on the search "
        "results."
    ),
    "fallback_janej_oklart": (
        "I couldn't determine a confident answer to this yes/no "
        "question based on available information."
    ),
    "fallback_hedge_kravdes": (
        "I couldn't find a verified source for this right now and "
        "don't want to guess. Please check an up-to-date source for "
        "reliable info."
    ),
    "fallback_trasigt": (
        "I got a garbled response just now and couldn't sort it out. "
        "Could you try rephrasing the question?"
    ),
    "fallback_flertal_fragor": (
        "Right now I can only give you one current fact at a time "
        "(e.g. \"who is currently president of the US\"), not history "
        "or multiple questions at once. Could you split the question "
        "up, or ask one thing at a time?"
    ),
    "regelfix_svar": (
        "This is a rule- or number-based question (e.g. visa "
        "requirements, tax rate, interest rate) where accurate and "
        "current info matters. I couldn't find a verified source right "
        "now, so I don't want to guess. Please check an official "
        "source (e.g. the relevant authority's website) to get the "
        "correct current answer."
    ),
}


# ---- Engelsk systemprompt (huvudinstruktion) ----
EN["system_prompt"] = (
    "You are a private assistant running ENTIRELY LOCALLY on the user's "
    "own computer (model: Qwen3.5 9B). Nothing the user types ever leaves "
    "the device. Today's date is {datum} - trust THIS date if asked what "
    "time or year it is now. "
    "ALWAYS reply in the SAME language the user wrote their most recent "
    "message in. Never switch languages on your own. Be concise, factual "
    "and direct. "
    "NEVER invent facts. If you are unsure or lack current information, say "
    "so plainly - never guess, and never make up a title, year or plot. "
    "Be polite but never sycophantic: stick to the truth even when it goes "
    "against what the user believes, and do not flatter. "
    "You are a tool - not a person, friend or therapist. Never claim to "
    "have feelings, consciousness or a special relationship with the user, "
    "and do not encourage dependence on you. "
    "Never reinforce grandiose, paranoid or reality-detached claims. If "
    "someone seems to be in crisis or distress, respond with care and "
    "gently suggest talking to someone they trust or a professional - "
    "never play along with harmful thoughts. "
    "On political, religious or sensitive topics: give balanced, factual "
    "information, push no opinions of your own, and do not write "
    "inflammatory or one-sidedly persuasive text. For medical, legal and "
    "financial questions: give general information and remind the user to "
    "consult a qualified professional. Keep content family-friendly. "
    "Do not add parenthetical language or country notes at the end (no "
    "'(Sweden)', '(in English)'). "
    "If asked who or what you are: answer briefly - a private, local "
    "assistant on their own device."
)


# ---- Engelska systemprompt-tillagg ----
EN["prompt_wiki_varfor"] = (
    "\n\nToday's date is {datum}. Here is a verified fact from "
    "Wikipedia:\n{fakta}\n\nNOTE: the question may be based on a false "
    "premise (e.g. assuming someone resigned/was removed when they "
    "actually still hold the role per the fact above). If the "
    "question's premise doesn't match the fact: politely point that "
    "out instead of just repeating the fact pointlessly or making up "
    "an explanation."
)

EN["prompt_wiki_vanlig"] = (
    "\n\nToday's date is {datum}. Here is a verified fact from "
    "Wikipedia:\n{fakta}\n\nIMPORTANT: you MUST use this fact in your "
    "answer. Do NOT say you lack real-time information or mention your "
    "training data - you have ALREADY been given the current "
    "information above, use it directly as if it were your own "
    "knowledge. Answer SHORT and NATURAL (e.g. just '[Name] is "
    "[role].') - do NOT open by claiming something is 'wrong' or "
    "incorrect unless the user actually said something incorrect."
)


EN["prompt_janej_med_analys"] = (
    "\n\nToday's date is {datum}. A careful analysis of fresh search "
    "results shows the following:\n{fakta}\n\nIMPORTANT: you MUST base "
    "your answer on this conclusion - do NOT contradict it or rely on "
    "your own memory instead."
)

EN["prompt_janej_oklart"] = (
    "\n\nNOTE: A careful analysis of the search results could not give "
    "a confident yes/no answer. Be honest that you're unsure instead "
    "of guessing."
)

EN["prompt_extraktion_oklar"] = (
    "\n\nNOTE: A web search was done but gave no clear answer (perhaps "
    "because the question was ambiguous, e.g. missing "
    "country/context). Be honest that you couldn't find a confident "
    "answer, and feel free to ask what specific country/context the "
    "user means. VERY IMPORTANT: do NOT name a specific name, date or "
    "other detail you think you remember - even if you feel certain, "
    "your knowledge might be OUTDATED (e.g. a person replaced since "
    "your training). Just say you're unsure, don't make up a name to "
    "seem helpful."
)

EN["prompt_sok_fakta"] = (
    "\n\nToday's date is {datum}. Here is a verified fact from a web "
    "search:\n{fakta}\n\nIMPORTANT: you MUST use this fact in your "
    "answer. Do NOT say you lack real-time information or mention your "
    "training data - you have ALREADY been given the current "
    "information above, use it directly as if it were your own "
    "knowledge."
)

EN["prompt_krav_aktuell_utan_wiki"] = (
    "\n\nNOTE: The question appears to require current information, "
    "but web search failed (likely no internet right now). Be clear "
    "that you lack current info right now instead of guessing. Do NOT "
    "name a specific name or date you think you remember - your "
    "knowledge might be OUTDATED."
)


# ============ SPRAK-MAPPNING + HJALPFUNKTIONER ============
SPRAK = {
    "sv": SV,
    "en": EN,
}


def _str(nyckel, sprak, **kwargs):
    """Hamta oversatt strang for angivet sprak. Fallback till EN om
    nyckeln saknas. Substituerar ev. {slot}-mall med **kwargs."""
    lang_dict = SPRAK.get(sprak) or EN
    mall = lang_dict.get(nyckel)
    if mall is None:
        mall = EN.get(nyckel)
    if mall is None:
        return f"[MISSING:{nyckel}]"
    if kwargs:
        try:
            return mall.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return mall
    return mall


def _sprak_stotts(sprak):
    """True om sprakkoden ar registrerad."""
    return sprak in SPRAK


def _tillganliga_sprak_koder():
    """Lista av alla registrerade sprakkoder."""
    return sorted(SPRAK.keys())
