from jina import Document, Flow


def get_doc(fn):
    with open(fn, 'r') as fh:
        for idx, l in enumerate(fh):
            if idx >= 10:
                break
            title, para = l.strip().split('\t')
            doc = Document(text=para, tags={'title': title})
            yield doc


def main():
    fn = '../toy_data/marco.tp.1k'
    f = (Flow()
         .add(uses='jinahub+docker://DPRTextEncoder',
              volumes='.cache:/root/.cache/huggingface',
              uses_with={
                  'encoder_type': 'context',
                  'traversal_paths': 'r',
                  'pretrained_model_name_or_path': 'facebook/dpr-ctx_encoder-single-nq-base'})
         .add(uses='jinahub://SimpleIndexer',
              uses_metas={
                  'workspace': 'workspace_dpr',
                  'title_tag_key': 'title'}))

    with f:
        f.post(on='/index', inputs=get_doc(fn))


if __name__ == '__main__':
    main()
