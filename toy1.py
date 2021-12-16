from jina import Document, Flow

fn = 'toy_data/marco.tp.1k'


def get_doc(fn, model='rocketqa'):
    with open(fn, 'r') as f:
        for idx, l in enumerate(f):
            try:
                title, para = l.strip().split('\t')
                if model == 'rocketqa':
                    doc = Document(tags={'title': title, 'para': para})
                elif model == 'dpr':
                    doc = Document(text=para, tags={'title': title})
                else:
                    continue
                yield doc
            except:
                print(f'skip line {idx}')
                continue


# f = (Flow()
#      .add(uses='jinahub://RocketQADualEncoder', uses_with={'use_cuda': False}, install_requirements=True)
#      .add(uses='jinahub://SimpleIndexer', uses_metas={'workspace': 'workspace_rocketqa'}))

f = (Flow()
     .add(uses='jinahub+docker://DPRTextEncoder',
          volumes='.cache/huggingface:/root/.cache/huggingface',
          uses_with={
              'encoder_type': 'context',
              'traversal_paths': 'r',
              'pretrained_model_name_or_path': 'facebook/dpr-ctx_encoder-single-nq-base'})
     .add(uses='jinahub://SimpleIndexer',
          uses_metas={
              'workspace': 'workspace_dpr',
              'title_tag_key': 'title'}))

with f:
    f.post(on='/index', inputs=get_doc(fn, model='dpr'))