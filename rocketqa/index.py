from jina import Document, Flow


def get_doc(fn):
    with open(fn, 'r') as fh:
        for idx, l in enumerate(fh):
            if idx >= 10:
                break
            title, para = l.strip().split('\t')
            doc = Document(tags={'title': title, 'para': para})
            yield doc


def main():
    fn = '../toy_data/marco.tp.1k'
    f = (Flow()
         .add(
        uses='jinahub+docker://RocketQADualEncoder/latest',
        volumes='.rocketqa:/root/.rocketqa',
        uses_with={'use_cuda': False})
         .add(
        uses='jinahub://SimpleIndexer',
        install_requirements=True,
        uses_metas={'workspace': 'workspace_rocketqa'}))

    with f:
        f.post(on='/index', inputs=get_doc(fn))


if __name__ == '__main__':
    main()
