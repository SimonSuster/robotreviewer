from collections import Counter

from robotreviewer.util import load_json

d = load_json("/home/simon/Apps/robotreviewer/eval_data/robotreviewer_eval_data.json")
topics = load_json("/home/simon/Apps/robotreviewer/eval_data/robotreviewer_topics.json")
criteria = ["allo_conceal", "outcome_blinding", "blinding", "rand_seq_gen"]
counts = []
overall_count = Counter()

for i, anno_list in d.items():
    if "Public health" not in topics[i]:
        continue
    for anno in anno_list:
        labels = [anno.get(criterion, None) for criterion in criteria]
        counts.append(Counter(labels))
        overall_count.update(labels)

print(f'N of times (prop of insts) when all criteria are low-risk: {sum([counter["Low risk"]==4 for counter in counts])} ({sum([counter["Low risk"]==4 for counter in counts]) / len(counts)})')
print(f'N of times (prop of insts) when all but one criterion are low-risk: {sum([counter["Low risk"]==3 for counter in counts])} ({sum([counter["Low risk"]==3 for counter in counts]) / len(counts)})')
print(f'N of times (prop of insts) when no criterion is low-risk: {sum([counter["Low risk"]==0 for counter in counts])} ({sum([counter["Low risk"]==0 for counter in counts]) / len(counts)})')

print(overall_count)

