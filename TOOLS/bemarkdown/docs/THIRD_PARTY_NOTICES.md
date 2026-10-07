# Third-party notices

## MathLive 0.110.0

Copyright (c) Arno Gourdol and MathLive contributors.

MathLive is distributed under the MIT License. The locally bundled review UI
includes the minified browser runtime, required KaTeX-compatible WOFF2 fonts,
sound assets, CSS, and the upstream `LICENSE.txt`. Source:
<https://github.com/arnog/mathlive>, npm package `mathlive@0.110.0`, integrity
`sha512-UOpJsQ6h1eeN0xZULGTl1MwUB/lZLCDuzKeHvKEh7Zra8U/rtgDNrssj5PZdJtBTIV48AfEhtv6rv47AeMOxJA==`.

## omml2latex 0.1.1 vendored parser

`src/bemarkdown/_vendor/omml2latex_parser.py` is derived from the
`omml2latex 0.1.1` PyPI wheel:

- Wheel SHA-256: `b24a13f4a14791662972d1b6716e5a1f9114891ccf1a96622f61dbfb1c300bd4`
- Original `_parser.py` SHA-256:
  `1ea42deb02b417573872a448d836c283e964a963a4d546a627c1761b1ecf26f1`
- PyPI source archive SHA-256:
  `eda9ff25c881f753357c325ab7a63ba33ad807f0ec8293fff39c8b370ff9084a`

BeMarkdown changes exactly one source statement: the Python 3.11-invalid
f-string expression containing `\\hat` is split into a local `accent`
assignment followed by an equivalent f-string. The vendored file carries a
prominent modification notice. No other upstream source is reformatted or
refactored. PyPI reports 0.1.1 as the latest release and the package metadata
does not advertise an upstream repository or fixed commit.

The upstream Apache License 2.0 is preserved at
`src/bemarkdown/_vendor/OMML2LATEX_LICENSE.txt` and included as package data.

## mathtypejx test fixture

`tests/conftest.py` contains a gzip/base64 representation of the public
`tests/fixtures/oleObject1.bin` fixture from `mathtypejx`, commit
`7d90e7274c85cf56ac28d4d15e593044693d7e70`:

https://github.com/a917470154/mathtypejx

Copyright (c) mathtypejx contributors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Production runtime and model inventory

The DOCX Tool RC freezes the following direct production identities. Transitive
installed distributions and their package-metadata license fields are captured
in Phase 4B `dependency_inventory.json`.

| Component | Version/revision | Source | License status | Bundled in Tool wheel |
|---|---|---|---|---|
| Beautiful Soup | 4.15.0 | PyPI project metadata | MIT, verified from installed metadata | No |
| lxml | 6.1.2 | <https://lxml.de/> | BSD-3-Clause, verified from installed metadata | No |
| mathml2latex | 0.2.12 | <https://github.com/KiaismAgre/mathml2latex> | MIT, verified from installed metadata | No |
| mathtypejx | `7d90e7274c85cf56ac28d4d15e593044693d7e70` | <https://github.com/a917470154/mathtypejx> | MIT, verified from upstream source and installed metadata | No |
| olefile | 0.47 | <https://www.decalage.info/python/olefileio> | BSD, verified from installed metadata | No |
| Pillow | 12.1.0 | <https://python-pillow.org/> | License files supplied by distribution; SPDX field absent in installed metadata | No |
| PaddlePaddle GPU | 3.2.2 cu126 | Paddle official package index | Apache Software License, verified from installed metadata | No |
| PaddleX | 3.7.2 | PyPI project metadata | Apache-2.0, verified from installed metadata | No |
| NVIDIA CUDA/cuDNN Python runtime packages | CUDA 12.6 / cuDNN 9.9.0.52 | NVIDIA Python packages | Proprietary; redistribution authorization is not established by this project | No |
| PP-FormulaNet_plus-L | `0809597a77f735bfb35354edb632f2e6dff606f3` | `PaddlePaddle/PP-FormulaNet_plus-L` | Apache-2.0 declared in the actual revision's model-card metadata | Separate `MODELS` asset, never bundled in Tool wheel |

The model-card identity is covered by the model manifest's full inventory and
directory fingerprint. The RC model payload contains the upstream model card;
the Tool does not download it at runtime.

The repository currently has no project-level license file declaring terms for
BeMarkdown's own code. Its publication license is therefore
`LICENSE_STATUS_UNVERIFIED`. This and NVIDIA runtime redistribution review are
publication gates; the RC must not invent or infer permission.
