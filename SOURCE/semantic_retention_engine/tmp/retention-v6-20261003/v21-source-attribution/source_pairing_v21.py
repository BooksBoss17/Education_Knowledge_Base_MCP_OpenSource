"""Navigation only: printed identity, geometrical groups, and other-page references."""

import re

FIGURE=re.compile(r'^\s*图\s*(\d+(?:[.．－—–-]\d+)+)(?:\s|[\u3400-\u9fff]|$)')

def norm(n):return re.sub('[．－—–-]','.',n)

def intersection(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))

def attribute(number,target,visible_text,records,neighbor_captions):
 own=FIGURE.match(visible_text or '')
 if own and norm(own.group(1))==number:return {'assignment':'OWN_PRINTED_CAPTION','evidence':'frozen visible_text starts with the figure number; identity only'}
 owners={o['image_id']:o for r in records for o in r['owners']}
 if owners:
  intersections=[intersection(target,o['bbox']) for o in owners.values()]
  if max(intersections)==0:label='OTHER_IMAGE_GROUP' if len(owners)>1 else 'OTHER_IMAGE_REGION'
  elif len(owners)>1:
   area=max(.001,(target[2]-target[0])*(target[3]-target[1]));label='TARGET_GROUP_CANDIDATE' if max(intersections)/area>=.999 else 'UNKNOWN'
  else:
   area=max(.001,(target[2]-target[0])*(target[3]-target[1]));label='TARGET_REGION_OR_COMPONENT' if max(intersections)/area>=.8 else 'UNKNOWN'
  return {'assignment':label,'unique_owners':list(owners.values()),'caption_record_count':len(records),'evidence':'geometry only; group/containment is not scientific role evidence'}
 if neighbor_captions:return {'assignment':'OTHER_PAGE_REFERENCE','neighbor_captions':neighbor_captions,'evidence':'same-page owner absent; MD-verified caption exists on adjacent page, not bound to target'}
 return {'assignment':'UNKNOWN','evidence':'no supported identity/owner/adjacent-page caption in code scope'}
