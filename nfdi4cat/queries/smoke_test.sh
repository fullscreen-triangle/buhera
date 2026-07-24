#!/usr/bin/env bash
# Exercise a running instance of the SPARQL API. Assumes the service is up at
# $BASE (default http://localhost:8000). Prints each query and its result.
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"

echo "== service info =="
curl -sf "$BASE/" | python -m json.tool | head -n 20
echo

echo "== all reactions (SPARQL JSON) =="
Q='PREFIX ta: <https://w3id.org/nfdi4cat/transaminase-kg/ontology#> SELECT ?r WHERE { ?r a ta:Transamination }'
curl -sf --get "$BASE/sparql" --data-urlencode "query=$Q" | python -m json.tool
echo

echo "== reactions using PLP (through the ChEBI cross-reference) =="
Q='PREFIX ta: <https://w3id.org/nfdi4cat/transaminase-kg/ontology#>
   PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
   PREFIX obo: <http://purl.obolibrary.org/obo/>
   SELECT ?r WHERE { ?r ta:catalyzedBy ?e . ?e ta:hasCofactor ?c . ?c skos:exactMatch obo:CHEBI_18405 . }'
curl -sf --get "$BASE/sparql" --data-urlencode "query=$Q" | python -m json.tool

echo
echo "OK — endpoint responding."
