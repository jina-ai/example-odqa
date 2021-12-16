from jina import Document, Flow

fn = 'toy_data/marco.tp.1k'


def get_doc(fn):
    with open(fn, 'r') as f:
        for idx, l in enumerate(f):
            try:
                title, para = l.strip().split('\t')
                doc = Document(tags={'title': title, 'para': para})
                yield doc
            except:
                print(f'skip line {idx}')
                continue


f = (Flow()
     .add(uses='jinahub://RocketQADualEncoder', uses_with={'use_cuda': False}, install_requirements=True)
     .add(uses='jinahub://SimpleIndexer'))


with f:
    f.post(on='/index', inputs=get_doc(fn))