from pydantic import BaseModel
class DatasetConfig(BaseModel):
    stride:int=1
    max_context_length:int=1024
    
class DataLoaderConfig(BaseModel):
    batch_size:int=32
    shuffle:bool=False
    drop_last:bool=False
    num_workers:int=0

class EmbeddingLayerConfig(BaseModel):
    embedding_dim:int=256

class PosEmbeddingLayerConfig(BaseModel):
    embedding_dim:int=256


