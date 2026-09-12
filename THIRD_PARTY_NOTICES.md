# Third-party notices

Project-owned code is licensed under AGPL-3.0-only. The following bundled
third-party material retains its upstream license and attribution.

## omml2latex 0.1.1

The vendored parser originates from the PyPI `omml2latex==0.1.1` wheel, SHA-256
`b24a13f4a14791662972d1b6716e5a1f9114891ccf1a96622f61dbfb1c300bd4`.
It is licensed under Apache-2.0. BeMarkdown changed one Python 3.11-incompatible
f-string into an equivalent local assignment; the source carries this notice.

License: `SOURCE/bemarkdown/src/bemarkdown/_vendor/OMML2LATEX_LICENSE.txt`.
The license also remains inside the distributed wheel.

## MathLive 0.110.0

Copyright (c) 2017 - present Arno Gourdol. MIT License.
Upstream: https://github.com/arnog/mathlive and npm `mathlive@0.110.0`.

The bundled browser runtime, CSS and related review assets retain
`SOURCE/bemarkdown/src/bemarkdown/static/formula_review/mathlive/LICENSE.txt`.

## KaTeX fonts

Copyright (c) 2018 Khan Academy. MIT License.
Upstream font project: https://github.com/KaTeX/katex-fonts.

The font notice is provided in `licenses/KaTeX-fonts-MIT.txt` and next to the
bundled WOFF2 fonts in the corresponding source and wheel.

## External downloads

Model files and Python runtime dependencies are downloaded separately, not
embedded into this GitHub folder. Their licenses remain applicable. Exact model
revisions and file hashes are in `scripts/model-downloads.json`; dependency pins
are in `scripts/requirements-pypi.lock` and the Tool runtime locks. See
`docs/LICENSE_AUDIT.md` for NVIDIA proprietary components and the PyMuPDF AGPL
license route.

The public distribution includes no private textbook/exam test corpus, student
records, internal task transcripts, Python environment, driver, or model weight.
