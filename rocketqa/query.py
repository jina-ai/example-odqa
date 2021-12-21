from jina import Flow, Document


def print_answers(resp):
    for d in resp.docs:
        for m in d.matches:
            score = m.scores['relevance_score'].value
            ans = m.text
            title = m.tags['title']
            para = m.tags['para']
            print(f'Answer (score: {score:.4f}): {ans}')
            print(f'Support: {title} [SEP] {para}')
            print('-'*20)
        print('\n')


if __name__ == '__main__':
    f = (Flow()
         .add(uses='jinahub://RocketQADualEncoder',
              uses_with={'use_cuda': False})
         .add(uses='jinahub://SimpleIndexer/v0.10',
              uses_metas={'workspace': 'workspace_rocketqa'},
              uses_with={'match_args': {'limit': 3}})
         .add(uses='jinahub://RocketQAReranker',
              uses_with={'model': 'v1_marco_ce', 'use_cuda': False})
         .add(uses='jinahub+docker://DPRReaderRanker/v0.3',
              uses_with={'title_tag_key': 'title', 'num_spans_per_match': 1},
              volumes='.cache:/root/.cache/huggingface'))

    with f:
        while True:
            q = input('Question?: ')
            if not q:
                break
            f.post(on='/search', inputs=Document(text=q), on_done=print_answers)
