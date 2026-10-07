"""Versioned positive/negative image retrieval; similarity never grants exclusion."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def rank_queries(query_vectors, templates, *, model_fingerprint, cosine_min=.90, margin_min=.05):
    """Return nearest positive AND protected template for each asset SHA.

    Vectors must be produced by the same verified model/preprocessing. Callers
    must keep development templates separate from evaluation truth.
    """
    import numpy as np
    if not isinstance(model_fingerprint,str) or len(model_fingerprint)!=64:
        raise ValueError('Expected model fingerprint SHA256')
    if not math.isfinite(cosine_min) or not -1 <= cosine_min <= 1 or not math.isfinite(margin_min) or not 0 <= margin_min <= 2:
        raise ValueError('Invalid retrieval thresholds')
    if len({r['template_id'] for r in templates})!=len(templates):
        raise ValueError('Duplicate template identity')
    if any(r['role'] not in {'DECORATION','PROTECTED'} for r in templates):
        raise ValueError('Unknown template role')
    if any(r['model_fingerprint']!=model_fingerprint for r in templates):
        raise ValueError('Model fingerprint mismatch')
    positive=[i for i,r in enumerate(templates) if r['role']=='DECORATION']
    negative=[i for i,r in enumerate(templates) if r['role']=='PROTECTED']
    if not positive or not negative:raise ValueError('Both positive and protected templates are required')
    matrix=np.asarray([r['vector'] for r in templates],dtype=np.float32)
    if matrix.ndim!=2 or not np.isfinite(matrix).all():raise ValueError('Invalid template vectors')
    lengths=np.linalg.norm(matrix,axis=1)
    if (lengths<=0).any():raise ValueError('Zero template vector')
    matrix=matrix/lengths[:,None]
    results={}
    for identity,value in query_vectors.items():
        vector=np.asarray(value,dtype=np.float32)
        if vector.shape!=(matrix.shape[1],) or not np.isfinite(vector).all() or np.linalg.norm(vector)<=0:
            raise ValueError('Invalid query vector')
        scores=matrix@(vector/np.linalg.norm(vector))
        pos=positive[int(scores[positive].argmax())];neg=negative[int(scores[negative].argmax())]
        best=float(scores[pos]);against=float(scores[neg]);margin=best-against
        results[identity]=dict(model_fingerprint=model_fingerprint,positive_template=templates[pos]['template_id'],
            negative_template=templates[neg]['template_id'],positive_cosine=best,negative_cosine=against,
            margin=margin,cosine_min=cosine_min,margin_min=margin_min,
            template_candidate=best>=cosine_min and margin>=margin_min,authorized_to_exclude=False)
    return results


def load_extraction(directory, expected_manifest_sha256):
    """Read sealed extraction output; reject tampered or incomplete vectors."""
    import numpy as np
    directory=Path(directory).resolve(strict=True)
    sealed=json.loads((directory/'checksums.json').read_text('utf-8'))['files']
    for relative,expected in sealed.items():
        path=(directory/relative).resolve(strict=True)
        if not path.is_relative_to(directory) or sha(path)!=expected:raise ValueError('Extraction seal mismatch')
    results=json.loads((directory/'results.json').read_text('utf-8'))
    if results['status']!='SUCCESS' or results['manifest_sha256']!=expected_manifest_sha256:
        raise ValueError('Extraction identity mismatch')
    rows={};vectors={}
    for r in results['records']:
        if r['sample_id'] in rows:raise ValueError('Duplicate extraction sample')
        path=(directory/r['vector_path']).resolve(strict=True)
        if not path.is_relative_to(directory) or sha(path)!=r['vector_sha256']:
            raise ValueError('Vector identity mismatch')
        rows[r['sample_id']]=r
        vectors[r['sample_id']]=np.load(path,allow_pickle=False)
    return results,rows,vectors
