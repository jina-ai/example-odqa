from jina import Flow, Document


def print_answers(resp):
    for d in resp.docs:
        for m in d.matches:
            score = m.scores['relevance_score'].value
            ans = m.text
            print(f'Answer (score: {score:.4f}): {ans}')
            print('-'*20)
        print('\n')


if __name__ == '__main__':
    f = (Flow()
         .add(uses='jinahub+docker://DPRTextEncoder',
              volumes='.cache:/root/.cache/huggingface',
              uses_with={
                  'encoder_type': 'question',
                  'traversal_paths': 'r',
                  'pretrained_model_name_or_path': 'facebook/dpr-question_encoder-single-nq-base'})
         .add(uses='jinahub://SimpleIndexer',
              uses_metas={'workspace': 'workspace_dpr'},
              uses_with={'match_args': {'limit': 3}})
         .add(uses='jinahub+docker://DPRReaderRanker/v0.3',
              uses_with={'title_tag_key': 'title', 'num_spans_per_match': 1},
              volumes='.cache:/root/.cache/huggingface'))

    with f:
        while True:
            q = input('Question?: ')
            if not q:
                break
            f.post(on='/search', inputs=Document(text=q), on_done=print_answers)
