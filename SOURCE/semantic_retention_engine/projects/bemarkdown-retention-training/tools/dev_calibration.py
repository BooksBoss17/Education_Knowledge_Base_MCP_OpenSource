"""Dev-only threshold replay: new complete prediction freeze precedes every score."""

from contracts import digest

def policy_fingerprint(threshold):
    return digest({'d_threshold': threshold, 'aggregation': 'K>U>D', 'missing_source_D_to_U': True,
                   'unknown_D_confidence_to_U': True, 'all_reference_action_after_asset_aggregation': True})
