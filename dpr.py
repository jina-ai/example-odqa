from jina import Flow

f = (Flow(use_cors=True, protocol='http', port_expose=45688)
     .add(uses='jinahub+docker://DPRTextEncoder',
          volumes='.cache/huggingface:/root/.cache/huggingface',
          uses_with={
              'encoder_type': 'question',
              'traversal_paths': 'r',
              'pretrained_model_name_or_path': 'facebook/dpr-question_encoder-single-nq-base'})
     .add(uses='jinahub://SimpleIndexer',
          uses_metas={'workspace': 'workspace_dpr'},
          uses_with={'match_args': {'limits': 10}}))

with f:
    f.block()