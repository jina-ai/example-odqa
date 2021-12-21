from jina import Document, Flow

title = 'test'
para = 'test'

doc = Document(tags={'title': title, 'para': para})

f = (Flow()
     .add(
    uses='jinahub://RocketQADualEncoder',
    uses_with={'use_cuda': False},
    install_requirements=True))
    #  .add(
    # uses='jinahub://SimpleIndexer',
    # uses_metas={'workspace': 'workspace_rocketqa'}))

with f:
    f.post(on='/index', inputs=[doc,])