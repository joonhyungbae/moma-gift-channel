# -*- coding: utf-8 -*-
"""
Revision analysis (referees' round-two concern): artists with no nationality in
MoMA's metadata.

For every artist in the gift channel whose MoMA record carries no nationality,
the script looks the artist up in Wikidata and, when one person can be
identified, records the country of the artist's place of birth (the rule the
Tate data already use). The country is mapped to a MoMA nationality term, which
01_data_preprocessing.py then fills into the artist's empty nationality slot.

Identity (anything else stays uncoded):
  * the Wikidata item MoMA links to the constituent (Wiki QID in Artists_full.csv), or
  * the single Wikidata item that carries the constituent's MoMA artist ID (P2174), or
  * otherwise, a full-text name search, accepting the only candidate that
      - is a human (P31 = Q5),
      - has a label or alias matching the MoMA name (surname and first initial,
        after normalising transliteration), and
      - was born within one year of MoMA's birth year when MoMA records one, or,
        when it does not, has an English description naming an occupation in the
        arts, letters, design, architecture, photography, or printing.
Country: the present-day country of the place of birth (P19 -> P17 statements
without an end date; where historical and present-day states are both given, the
present-day one). Without a place of birth, the countries of citizenship (P27),
if they give one nationality term. Searches and entity data go through the
Wikidata Query Service (the MediaWiki search API is called from inside SPARQL).

Network access is used only to build data/wikidata_origin_cache.csv (searches are
cached in data/wikidata_origin_search.jsonl); pass --offline to read the cache alone.

Output: data/wikidata_origin_cache.csv
"""

import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher

import pandas as pd

from config import PROJECT_ROOT
from rev_utils import artist_names, constituents, load_gifts

SPARQL = "https://query.wikidata.org/sparql"
UA = {"User-Agent": "collection-origin-check/1.0 (research replication script)"}
CACHE = os.path.join(PROJECT_ROOT, "data", "wikidata_origin_cache.csv")
SEARCH = os.path.join(PROJECT_ROOT, "data", "wikidata_origin_search.jsonl")
PAUSE = 2.0  # seconds between query-service requests
BATCH_SEARCH, BATCH_ENTITY = 20, 80

ART_OCCUPATION_WORDS = ("artist", "painter", "illustrator", "designer", "graphic", "poet", "writer", "author", "photograph",
                        "architect", "printmaker", "engraver", "lithograph", "typograph", "sculptor", "caricatur", "cartoon",
                        "draughts", "drafts", "set designer", "scenograph", "book", "publisher", "editor", "critic",
                        "film", "theatre", "theater", "composer", "playwright", "translator", "journalist", "art")

# Country labels (English, as Wikidata gives them) -> MoMA nationality term.
COUNTRY_TERM = {
    "Russia": "Russian", "Russian Empire": "Russian", "Soviet Union": "Russian", "Russian Soviet Federative Socialist Republic": "Russian",
    "Ukraine": "Ukrainian", "Ukrainian Soviet Socialist Republic": "Ukrainian", "Belarus": "Belarusian", "Georgia": "Georgian",
    "Armenia": "Armenian", "Azerbaijan": "Azerbaijani", "Latvia": "Latvian", "Lithuania": "Lithuanian", "Estonia": "Estonian",
    "Moldova": "Moldovan", "Kazakhstan": "Kazakhstani", "Uzbekistan": "Uzbekistani", "Kyrgyzstan": "Kyrgyz", "Tajikistan": "Tajik",
    "Turkmenistan": "Turkmen", "Poland": "Polish", "Czech Republic": "Czech", "Czechia": "Czech", "Slovakia": "Slovak",
    "Hungary": "Hungarian", "Romania": "Romanian", "Bulgaria": "Bulgarian", "Serbia": "Serbian", "Croatia": "Croatian",
    "Slovenia": "Slovenian", "Bosnia and Herzegovina": "Bosnian", "North Macedonia": "Macedonian", "Albania": "Albanian",
    "Montenegro": "Montenegrin", "Kosovo": "Kosovar",
    "United States": "American", "United States of America": "American", "Canada": "Canadian", "United Kingdom": "British",
    "France": "French", "Germany": "German", "Italy": "Italian", "Spain": "Spanish", "Netherlands": "Dutch", "Kingdom of the Netherlands": "Dutch",
    "Belgium": "Belgian", "Switzerland": "Swiss", "Austria": "Austrian", "Sweden": "Swedish", "Norway": "Norwegian",
    "Denmark": "Danish", "Finland": "Finnish", "Portugal": "Portuguese", "Greece": "Greek", "Ireland": "Irish",
    "Iceland": "Icelandic", "Luxembourg": "Luxembourger", "Australia": "Australian", "New Zealand": "New Zealander",
    "Israel": "Israeli", "Turkey": "Turkish", "Iran": "Iranian", "Iraq": "Iraqi", "Egypt": "Egyptian", "Lebanon": "Lebanese",
    "Syria": "Syrian", "Japan": "Japanese", "People's Republic of China": "Chinese", "China": "Chinese", "Taiwan": "Taiwanese",
    "South Korea": "Korean", "North Korea": "Korean", "India": "Indian", "Pakistan": "Pakistani", "Bangladesh": "Bangladeshi",
    "Indonesia": "Indonesian", "Philippines": "Filipino", "Thailand": "Thai", "Vietnam": "Vietnamese",
    "Mexico": "Mexican", "Brazil": "Brazilian", "Argentina": "Argentine", "Chile": "Chilean", "Colombia": "Colombian",
    "Cuba": "Cuban", "Venezuela": "Venezuelan", "Peru": "Peruvian", "Uruguay": "Uruguayan", "South Africa": "South African",
    "Nigeria": "Nigerian", "Ghana": "Ghanaian", "Senegal": "Senegalese", "Ivory Coast": "Ivorian", "Kenya": "Kenyan",
    "Ethiopia": "Ethiopian", "Morocco": "Moroccan", "Algeria": "Algerian", "Tunisia": "Tunisian",
}
PRESENT_STATES = set(COUNTRY_TERM) - {"Russian Empire", "Soviet Union", "Russian Soviet Federative Socialist Republic",
                                      "Ukrainian Soviet Socialist Republic"}
COUNTRY_TERM.update({
    # historical states
    "Russian Republic": "Russian", "Russian Socialist Federative Soviet Republic": "Russian",
    "Congress Poland": "Polish", "Second Polish Republic": "Polish", "Polish People's Republic": "Polish",
    "German Empire": "German", "Weimar Republic": "German", "Nazi Germany": "German", "West Germany": "German",
    "East Germany": "German", "Kingdom of Prussia": "German", "Kingdom of Italy": "Italian", "French Third Republic": "French",
    "Czechoslovakia": "Czechoslovakian", "Yugoslavia": "Yugoslavian", "Kingdom of Yugoslavia": "Yugoslavian",
    "Socialist Federal Republic of Yugoslavia": "Yugoslavian", "Kingdom of Hungary": "Hungarian", "Kingdom of Romania": "Romanian",
    "Empire of Japan": "Japanese", "Kingdom of Great Britain": "British", "United Kingdom of Great Britain and Ireland": "British",
    "Ottoman Empire": "Turkish", "Mandatory Palestine": "Palestinian", "Qing dynasty": "Chinese", "Republic of China (1912–1949)": "Chinese",
    "Korean Empire": "Korean", "British Raj": "Indian", "Kingdom of the Netherlands": "Dutch", "Kingdom of Denmark": "Danish",
    "Tsardom of Russia": "Russian", "Russian SFSR": "Russian", "Ukrainian SSR": "Ukrainian", "Byelorussian Soviet Socialist Republic": "Belarusian",
    "Mauritius": "Mauritian", "Palestine": "Palestinian", "State of Palestine": "Palestinian",
})


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"['’`ʹʼ]", "", s)
    s = re.sub(r"[^a-z ]", " ", s)
    for a, b in (("iia ", "ia "), ("ii ", "y "), ("iy ", "y "), ("yi ", "y "), ("ij ", "y "), ("ei", "ey"), ("x", "ks"),
                 ("ia", "ya"), ("iu", "yu"), ("ie", "ye"), ("j", "y"), ("w", "v"), ("ff", "f"), ("kh", "h")):
        s = (s + " ").replace(a, b).strip()
    return re.sub(r"\s+", " ", s).strip()


def name_variants(display):
    d = str(display).strip()
    out = [d]
    m = re.match(r"^(.*?)\s*\[(.*?)\]\s*$", d)
    if m:
        out = [m.group(1).strip(), m.group(2).strip()]
    out += [re.sub(r"['’ʹʼ]", "", v) for v in out]
    return list(dict.fromkeys(v for v in out if v))


def names_match(moma, cand):
    a, b = norm(moma).split(), norm(cand).split()
    if len(a) < 2 or len(b) < 2:
        return False
    if SequenceMatcher(None, a[-1], b[-1]).ratio() < 0.84:
        return False
    return a[0][0] == b[0][0]


def uncoded_artists():
    g = load_gifts()
    from config import get_processed_path
    if "Nationality_MoMA" in pd.read_csv(get_processed_path(), nrows=0).columns:
        g["Nationality"] = g["Nationality_MoMA"]  # MoMA's own field, before any fill
    cons = constituents(g, artist_names())
    cons = cons[~cons["placeholder"]].copy()
    cons["unit"] = g["Unit"].reindex(cons["rec"]).to_numpy()
    n_rec = cons.groupby("cid")["rec"].nunique()
    roth = set(cons.loc[cons["unit"] == "Judith Rothschild Foundation", "cid"])
    art = cons.drop_duplicates("cid").set_index("cid")
    art = art[art["origin"].isna()]
    meta = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "Artists_full.csv"), dtype={"ConstituentID": str}).set_index("ConstituentID")
    rows = []
    for cid in art.index:
        m = meta.loc[cid] if cid in meta.index else None
        b = m["BeginDate"] if m is not None else None
        rows.append({"cid": cid, "name": (m["DisplayName"] if m is not None else art.loc[cid, "name"]),
                     "moma_birth": int(b) if b is not None and pd.notna(b) and int(b) > 0 else None,
                     "moma_qid": (m["Wiki QID"] if m is not None and isinstance(m["Wiki QID"], str) else None),
                     "n_records": int(n_rec.get(cid, 0)), "rothschild": cid in roth})
    return pd.DataFrame(rows).sort_values(["rothschild", "n_records"], ascending=False).reset_index(drop=True)


def person_like(name):
    """Two or more name tokens and no firm, place, or collection words."""
    if re.search(r",|\b(inc|co|corp|company|ltd|gmbh|a\.?g|s\.?a|school|collection|studio|press|album|products|"
                 r"associates|group|workshop|society|museum|gallery|editions|unknown|anonymous)\b", name, re.I):
        return False
    return len(norm(name).split()) >= 2


def sparql(query, tries=6):
    """One SPARQL request to the Wikidata Query Service, with retries."""
    data = urllib.parse.urlencode({"query": query, "format": "json"}).encode()
    for attempt in range(tries):
        try:
            req = urllib.request.Request(SPARQL, data=data, headers={**UA, "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)["results"]["bindings"]
        except urllib.error.HTTPError as e:
            if attempt == tries - 1:
                raise
            wait = e.headers.get("Retry-After")
            time.sleep(float(wait) if wait and wait.isdigit() else 30 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == tries - 1:
                raise
            time.sleep(30)
    raise RuntimeError("Wikidata Query Service unavailable")


def qid(uri):
    return uri.rsplit("/", 1)[-1]


def search_all(todo):
    """Full-text search (list=search, humans only) per name variant, run through the
    query service in batches; results cached in SEARCH (one JSON line per query)."""
    done = {}
    if os.path.exists(SEARCH):
        for line in open(SEARCH, encoding="utf-8"):
            r = json.loads(line)
            done[r["q"]] = r["titles"]
    queries = [v for _, r in todo.iterrows() for v in name_variants(r["name"]) if person_like(v)]
    queries = [q for q in dict.fromkeys(queries) if q not in done]
    print(f"{len(queries)} searches to run", flush=True)
    with open(SEARCH, "a", encoding="utf-8") as f:
        for i in range(0, len(queries), BATCH_SEARCH):
            chunk = queries[i:i + BATCH_SEARCH]
            vals = " ".join(json.dumps(q + " haswbstatement:P31=Q5") for q in chunk)
            rows = sparql(f"""SELECT ?s ?title WHERE {{ VALUES ?s {{ {vals} }}
              SERVICE wikibase:mwapi {{ bd:serviceParam wikibase:endpoint "www.wikidata.org"; wikibase:api "Search";
                mwapi:srsearch ?s; mwapi:srlimit "7"; wikibase:limit "once". ?title wikibase:apiOutput mwapi:title. }} }}""")
            hits = {}
            for b in rows:
                hits.setdefault(b["s"]["value"].replace(" haswbstatement:P31=Q5", ""), []).append(b["title"]["value"])
            for q in chunk:
                done[q] = hits.get(q, [])
                f.write(json.dumps({"q": q, "titles": done[q]}) + "\n")
            f.flush()
            print(min(i + BATCH_SEARCH, len(queries)), "searches", flush=True)
            time.sleep(PAUSE)
    return done


def moma_links(cids):
    """Wikidata items that carry the MoMA artist ID (P2174) of each constituent; cached in SEARCH."""
    done = {}
    if os.path.exists(SEARCH):
        for line in open(SEARCH, encoding="utf-8"):
            r = json.loads(line)
            if r["q"].startswith("P2174="):
                done[r["q"][6:]] = r["titles"]
    todo = [c for c in cids if c not in done]
    with open(SEARCH, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), 200):
            chunk = todo[i:i + 200]
            vals = " ".join(json.dumps(c) for c in chunk)
            rows = sparql(f"SELECT ?id ?item WHERE {{ VALUES ?id {{ {vals} }} ?item wdt:P2174 ?id }}")
            hits = {}
            for b in rows:
                hits.setdefault(b["id"]["value"], []).append(qid(b["item"]["value"]))
            for c in chunk:
                done[c] = hits.get(c, [])
                f.write(json.dumps({"q": "P2174=" + c, "titles": done[c]}) + "\n")
            time.sleep(PAUSE)
    return done


def entities(ids):
    """Human flag, English label, aliases and description, birth year, place of
    birth with its current countries (no end date), and countries of citizenship."""
    ids = [i for i in dict.fromkeys(ids) if isinstance(i, str) and re.fullmatch(r"Q\d+", i)]
    out = {}
    for i in range(0, len(ids), BATCH_ENTITY):
        vals = " ".join("wd:" + q for q in ids[i:i + BATCH_ENTITY])
        rows = sparql(f"""SELECT ?item ?human ?label ?desc ?birth
              (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?aliases)
              (GROUP_CONCAT(DISTINCT ?place; separator="|") AS ?places)
              (GROUP_CONCAT(DISTINCT ?pc; separator="|") AS ?pcs)
              (GROUP_CONCAT(DISTINCT ?pcLabel; separator="|") AS ?pcLabels)
              (GROUP_CONCAT(DISTINCT ?citLabel; separator="|") AS ?citLabels) WHERE {{
          VALUES ?item {{ {vals} }}
          BIND(EXISTS {{ ?item wdt:P31 wd:Q5 }} AS ?human)
          OPTIONAL {{ ?item rdfs:label ?label FILTER(LANG(?label) = "en") }}
          OPTIONAL {{ ?item schema:description ?desc FILTER(LANG(?desc) = "en") }}
          OPTIONAL {{ ?item skos:altLabel ?alias FILTER(LANG(?alias) = "en") }}
          OPTIONAL {{ ?item wdt:P569 ?b BIND(YEAR(?b) AS ?birth) }}
          OPTIONAL {{ ?item wdt:P19 ?place .
                      OPTIONAL {{ ?place p:P17 ?st . ?st ps:P17 ?pc . ?st wikibase:rank ?rk
                                  FILTER(?rk != wikibase:DeprecatedRank) FILTER NOT EXISTS {{ ?st pq:P582 ?end }}
                                  ?pc rdfs:label ?pcLabel FILTER(LANG(?pcLabel) = "en") }} }}
          OPTIONAL {{ ?item wdt:P27 ?cit . ?cit rdfs:label ?citLabel FILTER(LANG(?citLabel) = "en") }}
        }} GROUP BY ?item ?human ?label ?desc ?birth""")
        for b in rows:
            q = qid(b["item"]["value"])
            v = lambda k: b.get(k, {}).get("value", "")
            e = out.setdefault(q, {"human": v("human") == "true", "labels": set(), "desc": v("desc"), "birth": None,
                                   "places": set(), "place_countries": set(), "citizenships": set()})
            e["labels"].update(x for x in [v("label")] + v("aliases").split("|") if x)
            if v("birth") and e["birth"] is None:
                e["birth"] = int(float(v("birth")))
            e["places"].update(qid(x) for x in v("places").split("|") if x)
            e["place_countries"].update(x for x in v("pcLabels").split("|") if x)
            e["citizenships"].update(x for x in v("citLabels").split("|") if x)
        time.sleep(PAUSE)
    return out


def decide(row, cands, ents):
    passing = []
    for q in cands:
        e = ents.get(q)
        if not e or not e["human"]:
            continue
        if isinstance(row["link"], str):
            passing.append(q)
            continue
        if not any(names_match(v, lab) for v in name_variants(row["name"]) for lab in e["labels"]):
            continue
        if pd.notna(row["moma_birth"]):
            if e["birth"] is None or abs(e["birth"] - int(row["moma_birth"])) > 1:
                continue
        else:
            if not any(w in e["desc"].lower() for w in ART_OCCUPATION_WORDS):
                continue
        passing.append(q)
    return passing


def pick(countries):
    """Present-day states among the countries if any, else all; one term or none."""
    now = {c for c in countries if c in PRESENT_STATES}
    terms = {COUNTRY_TERM.get(c) for c in (now or countries)} - {None}
    return terms.pop() if len(terms) == 1 else None


def country_of(e):
    """Nationality term from the place of birth; else from the citizenships."""
    if len(e["places"]) == 1 and e["place_countries"]:
        t = pick(e["place_countries"])
        if t:
            return "|".join(sorted(e["place_countries"])), "place of birth", t
    if e["citizenships"]:
        t = pick(e["citizenships"])
        if t:
            return "|".join(sorted(e["citizenships"])), "citizenship", t
    return ("|".join(sorted(e["place_countries"] or e["citizenships"])) or None), None, None


def main():
    offline = "--offline" in sys.argv
    if offline:
        out = pd.read_csv(CACHE, dtype={"cid": str})
    else:
        todo = uncoded_artists()
        todo["moma_qid"] = [x if isinstance(x, str) else None for x in todo["moma_qid"]]
        todo["moma_birth"] = pd.Series([int(x) if pd.notna(x) else None for x in todo["moma_birth"]], index=todo.index, dtype=object)
        links = moma_links(todo["cid"].tolist())
        todo["link"] = [r["moma_qid"] if isinstance(r["moma_qid"], str) else
                        (links[r["cid"]][0] if len(links.get(r["cid"], [])) == 1 else None) for _, r in todo.iterrows()]
        todo["link_source"] = ["MoMA Wiki QID" if isinstance(r["moma_qid"], str) else
                               ("Wikidata MoMA artist ID" if isinstance(r["link"], str) else None) for _, r in todo.iterrows()]
        found = search_all(todo[todo["link"].isna()])
        cands = {r["cid"]: ([r["link"]] if isinstance(r["link"], str) else
                            list(dict.fromkeys(t for v in name_variants(r["name"]) for t in found.get(v, []))))
                 for _, r in todo.iterrows()}
        ents = entities([q for c in cands.values() for q in c])
        rows = []
        for _, r in todo.iterrows():
            p = decide(r, cands[r["cid"]], ents)
            status = "matched" if len(p) == 1 else ("ambiguous" if p else ("no match" if cands[r["cid"]] else "no candidate"))
            row = {**r.to_dict(), "status": status, "n_candidates": len(cands[r["cid"]]), "n_passing": len(p)}
            if len(p) == 1:
                e = ents[p[0]]
                c, rule, term = country_of(e)
                row.update({"qid": p[0], "wd_label": sorted(e["labels"])[0] if e["labels"] else None, "wd_birth": e["birth"],
                            "country": c, "rule": rule, "nationality_term": term})
            rows.append(row)
        out = pd.DataFrame(rows)
        out.to_csv(CACHE, index=False)
    print(out["status"].value_counts().to_string())
    m = out[out["status"] == "matched"]
    roth = out["rothschild"].astype(bool)
    print("with a nationality term:", int(m["nationality_term"].notna().sum()), "of", len(out),
          "| Rothschild:", int(m.loc[roth[m.index], "nationality_term"].notna().sum()), "of", int(roth.sum()))
    print("countries without a term:", sorted(set(m.loc[m["nationality_term"].isna(), "country"].dropna())))
    print(m["nationality_term"].value_counts().head(25).to_string())


if __name__ == "__main__":
    main()
