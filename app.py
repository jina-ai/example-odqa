from jina import Flow

# f = (Flow(use_cors=True, protocol='http', port_expose=45678)
#      .add(uses='jinahub://RocketQADualEncoder', uses_with={'use_cuda': False})
#      .add(uses='jinahub://SimpleIndexer', uses_metas={'workspace': 'workspace_rocketqa'})
#      .add(uses='jinahub://RocketQAReranker', uses_with={'model': 'v1_marco_ce', 'use_cuda': False}))

f = (Flow(use_cors=True, protocol='http', port_expose=45688)
     .add(uses='jinahub+docker://DPRTextEncoder',
          volumes='.cache/huggingface:/root/.cache/huggingface',
          uses_with={
              'encoder_type': 'question',
              'traversal_paths': 'r',
              'pretrained_model_name_or_path': 'facebook/dpr-question_encoder-single-nq-base'})
     .add(uses='jinahub://SimpleIndexer', uses_metas={'workspace': 'workspace_dpr'}))

with f:
    f.block()