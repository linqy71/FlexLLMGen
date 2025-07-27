import torch 
from lsh import LSH

def load_hash_value(layer_id, request_id=0):
    path = "/HOME/nsccgz_zgchen/nsccgz_zgchen_3/OLD_HOME/lqy/llm_infer/snapshots/"
    filename = path + "hash_value." + str(layer_id) + "." + str(request_id) + ".pt"
    hash_value = torch.load(filename)
    return hash_value
  
def load_hash_indicies(layer_id, request_id=0):
    path = "/HOME/nsccgz_zgchen/nsccgz_zgchen_3/OLD_HOME/lqy/llm_infer/snapshots/"
    filename = path + "hash_indices." + str(layer_id) + "." + str(request_id) + ".pt"
    hash_indices = torch.load(filename)
    return hash_indices
  
def load_query(layer_id):
    path = "/HOME/nsccgz_zgchen/nsccgz_zgchen_3/OLD_HOME/lqy/llm_infer/snapshots/"
    filename = path + "query." + str(layer_id) + ".pt"
    return torch.load(filename)

def test(layer_id):
    lsh = LSH()
    lsh.alloc(K=10, L=150, num_layers=32, num_attention_heads=32, num_key_value_heads=8, batch_size=1, max_length=8192)
    
    hash_values = load_hash_value(layer_id=layer_id)
    hash_indices = load_hash_indicies(layer_id=layer_id)
    
    lsh.fill(layer_id, 0, hash_values, hash_indices)
    
    query = load_query(layer_id=layer_id)
    results = torch.zeros((32, 8192), dtype=torch.int32)
    nnz = torch.zeros((32), dtype=torch.int32)
    
    lsh.batch_retrieve(layer_id, query, results, nnz)
    
    print(results)
    
    ## match result and table0