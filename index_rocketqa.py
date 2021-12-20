from jina import Document, Flow

title = ''
para = ''

doc = Document(tags={'title': title, 'para': para})

f = (Flow()
     .add(
    uses='jinahub+docker://RocketQADualEncoder',
    uses_with={'use_cuda': False})
     .add(
    uses='jinahub://SimpleIndexer',
    uses_metas={'workspace': 'workspace_rocketqa'}))

with f:
    f.post(on='/index', inputs=[doc,])