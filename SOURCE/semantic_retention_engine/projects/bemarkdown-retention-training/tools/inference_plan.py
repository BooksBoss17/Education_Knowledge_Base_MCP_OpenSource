"""Label-blind request-local groups, with global coverage verified before any discard."""

import copy

from contracts import validate_input, visible_input, digest, asset_plan_coverage, aggregate, source_complete

def use_set(row, uses):
    result = copy.deepcopy(row)
    result['scope'] = 'use_set'
    result['uses'] = copy.deepcopy(uses)
    result['coverage'] = {'expected_use_count': len(uses), 'covered_use_ids': [u['use_id'] for u in uses], 'enumeration_complete': True}
    validate_input(result)
    return result

def plan_groups(row, length, policy, source_inputs_sha256, max_tokens=3072):
    """length measures untruncated native input plus output reserve; it must not read labels."""
    validate_input(row)
    groups = []
    def visit(uses):
        group = use_set(row, uses)
        measured = length(group)
        if measured > max_tokens and len(uses) > 1:
            mid = len(uses) // 2
            visit(uses[:mid])
            visit(uses[mid:])
        else:
            groups.append({'input': group, 'tokens': measured, 'over_budget': measured > max_tokens})
    visit(row['uses'])
    plan = {'schema_version': '1.0', 'asset_id': row['sample_id'], 'image_sha256': row['image']['sha256'],
            'source_inventory_sha256': source_inputs_sha256, 'global_use_ids': [u['use_id'] for u in row['uses']],
            'enumeration_complete': row['coverage']['enumeration_complete'],
            'groups': [{'group_id': 'g' + str(i + 1), 'use_ids': [u['use_id'] for u in g['input']['uses']],
                        'input_fingerprint': digest(visible_input(g['input'], policy))} for i, g in enumerate(groups)]}
    assert asset_plan_coverage(plan) == row['coverage']['enumeration_complete']
    return plan, groups

def final_group(raw, confidence_d, group_input, threshold):
    if raw == 'D' and (not source_complete(group_input) or confidence_d is None or confidence_d < threshold):
        return 'U'
    return raw

def aggregate_asset(plan, responses, threshold):
    """Only one asset action is returned: K/U protect every reference, even a group that said D."""
    covered = {r['group_id'] for r in responses}
    expected = {g['group_id'] for g in plan['groups']}
    assert len(covered) == len(responses), 'DUPLICATE_GROUP_RESPONSE'
    assert covered <= expected, 'UNKNOWN_GROUP_RESPONSE'
    complete = asset_plan_coverage(plan) and covered == expected
    raw = aggregate([r['raw'] for r in responses], coverage_complete=complete) if responses else 'U'
    final = aggregate([r['final'] for r in responses], coverage_complete=complete) if responses else 'U'
    return {'raw': raw, 'final': final, 'all_references_action': 'discard' if final == 'D' else 'preserve',
            'coverage_complete': complete, 'threshold': threshold}
