"""Positive evidence for image-valued multiple-choice options from source geometry."""

import re

def option_witnesses(words, occurrences):
    labels=[]
    for w in words:
        token=str(w[4]).strip().rstrip('.．、')
        if re.fullmatch('[ABCD]',token):labels.append((token,list(w[:4])))
    found={}
    for a in [x for x in labels if x[0]=='A']:
        group=[a]
        for letter in 'BCD':
            choices=[x for x in labels if x[0]==letter and abs(x[1][1]-a[1][1])<=3
                     and x[1][0]>group[-1][1][2]+10]
            if len(choices)!=1:break
            group.append(choices[0])
        if len(group)!=4:continue
        matched=[]
        for letter,box in group:
            center=(box[0]+box[2])/2
            candidates=[]
            for image in occurrences:
                b=image['bbox']
                if b[2]-b[0]<4 or b[3]-b[1]<4:continue
                gap=box[1]-b[3]
                if -6<=gap<=30 and b[0]<=center<=b[2] and b[1]<box[1]:
                    candidates.append((abs(gap),abs((b[0]+b[2])/2-center),image))
            candidates.sort(key=lambda x:x[:2])
            if not candidates:break
            if len(candidates)>1 and candidates[0][:2]==candidates[1][:2]:break
            matched.append((letter,box,candidates[0][2]))
        if len(matched)!=4 or len({x[2]['id'] for x in matched})!=4:continue
        if any(matched[i][2]['bbox'][2]>matched[i+1][2]['bbox'][0]+2 for i in range(3)):continue
        for letter,box,image in matched:
            found[image['id']]={'kind':'FOUR_DISTINCT_IMAGE_VALUED_OPTIONS','label':letter,
                'label_bbox':box,'target_bbox':image['bbox'],
                'row':[{'label':c,'label_bbox':b,'image_bbox':im['bbox'],'occurrence_id':im['id']} for c,b,im in matched]}
    return found
