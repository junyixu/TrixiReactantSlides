# Trixi.jl on TPUs via Reactant.jl — slides

Progress reports (Typst + [Touying](https://touying-typ.github.io/)) of a summer research project
on running [Trixi.jl](https://github.com/trixi-framework/Trixi.jl) on TPUs through
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl).

**Present online:** https://junyixu.github.io/TrixiReactantSlides/ — arrow keys / space / click to
move, `f` full screen, `o` overview, `#n` in the URL links to slide *n*; each deck also has a PDF.

## Layout

- `<deck>/main.typ` — one directory per deck (e.g. `2026-10-progress/`)
- `fonts/` — the fonts the decks use (Adwaita Sans, Fira Code; SIL OFL 1.1, licences included),
  so local and CI builds are identical (`--ignore-system-fonts`)
- `build.sh` — builds `_site/<deck>/` (`slides.pdf`, one SVG per page, `web/viewer.html`) and
  `_site/index.html`
- `.github/workflows/pages.yml` — runs `build.sh` with Typst 0.15.1 and deploys to GitHub Pages
  on every push to `main`

## Build locally

```bash
./build.sh && python3 -m http.server -d _site      # then open http://localhost:8000
# or just the PDF:
typst watch --font-path fonts --ignore-system-fonts 2026-10-progress/main.typ
```
