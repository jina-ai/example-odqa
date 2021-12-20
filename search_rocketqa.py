from jina import Flow

f = (Flow(use_cors=True, protocol='http', port_expose=45698)
     .add(uses='jinahub://RocketQADualEncoder',
          uses_with={'use_cuda': False})
     .add(uses='jinahub://SimpleIndexer',
          uses_metas={'workspace': 'workspace_rocketqa'},
          uses_with={'match_args': {'limits': 10}})
     .add(uses='jinahub://RocketQAReranker',
          uses_with={'model': 'v1_marco_ce', 'use_cuda': False}))

with f:
    f.block()