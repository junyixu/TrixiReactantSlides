#!/bin/bash
# Build every deck (<dir>/main.typ) into _site/<dir>/: slides.pdf, one SVG per page and the
# presentation viewer; _site/index.html lists the decks. Only the fonts in fonts/ and Typst's
# embedded fonts are used, so CI and local builds look the same.
# usage: ./build.sh            then  python3 -m http.server -d _site
set -euo pipefail
cd "$(dirname "$0")"
rm -rf _site
mkdir -p _site
typst_compile() { typst compile --font-path fonts --ignore-system-fonts "$@"; }

items=""
for typ in */main.typ; do
    deck=$(dirname "$typ")
    out=_site/$deck
    mkdir -p "$out/slides"
    typst_compile "$typ" "$out/slides.pdf"
    typst_compile "$typ" "$out/slides/{p}.svg"
    pages=$(find "$out/slides" -name '*.svg' | wc -l)
    title=$(grep -oP 'title: \[\K[^\]]+' "$typ" | head -1)
    subtitle=$(grep -oP 'subtitle: \[\K[^\]]+' "$typ" | head -1 || true)
    printf '{"title": "%s", "pages": %d, "pdf": "slides.pdf"}\n' "$title" "$pages" > "$out/pages.json"
    cp web/viewer.html "$out/index.html"
    items+="<li><a class=\"deck\" href=\"$deck/\">$title</a><span>$subtitle · $deck · $pages slides</span><a href=\"$deck/slides.pdf\">PDF</a></li>"
    echo "built $deck: $pages slides"
done
DECKS=$items python3 -c 'import os, sys; sys.stdout.write(open("web/index.html").read().replace("$DECKS", os.environ["DECKS"]))' > _site/index.html
