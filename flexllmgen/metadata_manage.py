import dataclasses
from typing import List,Dict
import unittest
from itertools import count

""" 
maintain token's longest common prefix

RadixTree接口:
def insert(self,key:List[int]) 插入token序列,后续还可能要插入kv_ptr和重要性(注意一个性质,NR整体在最后一个新生成节点)
def search(self,key:List[int])->List[RadixToken] 查找最长公共前缀,返回数据类型为RadixToken。
"""
@dataclasses.dataclass
class CachePointer:
    # 通过chunk_id定位chunk，通过offset定位在chunk内的位置
    # chunk中的kv缓存为一个TorchTensor,存了chunk_size个token的KV缓存, shape为(chunk_size, b * n_head, head_dim)
    # offset表明该token的KV张量为 (offset:offset+1, 0:b * n_head, 0:head_dim)
    chunk_id: int
    offset: int

class RadixToken:
    token_count = count()
    def __init__(self, token_id:int, token_name = None, importance:int = 0, kv_ptr:List[CachePointer] = None, probe_ptr:List[CachePointer] = None):
        self.token_id = token_id
        
        self.kv_ptr = kv_ptr  # 为该token所有层的chunk_id. kv_ptr[i]表示第i层的缓存的CachePointer
        self.probe_ptr = probe_ptr
        self.importance = importance

        self.token_name = token_name or RadixToken.next_token_name()

        self.layer_importance = [0 for _ in range(len(kv_ptr))]

    def __repr__(self):
        return f"(id={self.token_id},imp={self.importance})"
        #return f"(id={self.token_id}, probe_ptr={self.probe_ptr})\n"

    @classmethod
    def next_token_name(cls):
        return next(cls.token_count)

class RadixTreeNode:
    def __init__(self,tokens:List[RadixToken], mapping_list=None):
        # tokens始终保留原来的顺序,重排只需要修改mapping_list,要获取重排后的顺序使用mapping_list
        # 即 tokens[mapping_list[i]] 表示按照重要性排序后的, 第i个token
        self.tokens = tokens
        if mapping_list is not None:
            self.mapping_list = mapping_list
        else:
            self.mapping_list = [i for i in range(len(tokens))]
        
        self.children:Dict[int,RadixTreeNode] = {} 
    
    def __repr__(self):
        return f"Node(token={self.get_all_sorted_token()})"
    
    def sort_by_importance(self):
        self.mapping_list.sort(key=lambda i: self.tokens[i].importance, reverse=True)

    def get_sorted_token(self, idx):
        return self.tokens[self.mapping_list[idx]]

    def get_all_sorted_token(self):
        return [self.tokens[i] for i in self.mapping_list]    

    
    @classmethod
    def create_from_token_id(cls,tokens:List[int], kv_ptr=None, probe_ptr=None):
        if kv_ptr is None:
            tokens = [RadixToken(token_id=x) for x in tokens]
        else:
            tokens = [RadixToken(token_id=tokens[i], kv_ptr=kv_ptr[i], probe_ptr=probe_ptr[i]) for i in range(len(tokens))]
        return cls(tokens)


    
class RadixTree:
    def __init__(self):
        # root is an empty node
        self.root = RadixTreeNode(tokens=[])
    
    # 计算两个序列的最长公共前缀长度
    @staticmethod
    def common_prefix_length(a:List, b:List[RadixToken]):
        min_len = min(len(a),len(b))
        for i in range (min_len):
            if a[i] != b[i].token_id:
                return i;
        return min_len
    
    # 插入新的序列,同时插入新节点的kv_ptr
    # kv_ptr:List[List[CachePointer]]  kv_ptr[i]表示NR中第i个token在每一层的缓存的位置
    def insert(self, key:List[int], kv_ptr = None, probe_ptr = None):
        if len(key) == 0:
            return
        current = self.root
        remaining = key

        while remaining:
            next_token = remaining[0]
            if next_token in current.children:
                child = current.children[next_token]
                common_len = self.common_prefix_length(remaining,child.tokens)

                if common_len == len(child.tokens):
                    remaining = remaining[common_len:]
                    current = child
                else:
                    # split node
                    new_node = RadixTreeNode(tokens=child.tokens[:common_len])
                    new_node.mapping_list = [x for x in child.mapping_list if x < common_len]
                    
                    child.tokens = child.tokens[common_len:]
                    child.mapping_list = [x-common_len for x in child.mapping_list if x >= common_len]

                    current.children[next_token] = new_node
                    new_node.children[child.tokens[0].token_id] = child

                    remaining = remaining[common_len:]
                    current = new_node
            else:
                new_node = RadixTreeNode.create_from_token_id(remaining, kv_ptr, probe_ptr)
                current.children[next_token] = new_node
                break

    def search(self, key:List[int]):
        current = self.root
        remaining = key
        ans = list()
        while remaining:
            next_token = remaining[0]
            if next_token in current.children:
                child = current.children[next_token]
                common_len = self.common_prefix_length(remaining,child.tokens)
                ans.extend(
                    #[t.kv_ptr for t in child.tokens[:common_len]]
                    [t for t in child.tokens[:common_len]]
                )
                if common_len != len(child.tokens):
                    break
                remaining = remaining[common_len:]
                current = child
            else:
                break
        # type = List[List[CachePointer]]
        return ans

    def delete(self, key:List):
        pass

    def visualize(self, node=None, depth=0):
        """for debug"""
        if node is None:
            node = self.root
            print("root:")
        else:
            indent = "  " * depth
            print(f"{indent}{node}")
        
        for child in node.children.values():
            self.visualize(child, depth + 1)


if __name__ == "__main__":
    # from test_module import TestRadixTree
    # import unittest
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestRadixTree)
    # runner = unittest.TextTestRunner(verbosity=2)
    # runner.run(suite)
    tree  = RadixTree()
    keys = [
        [1,5,7,4,2,7],
        [1,5,5,3,7],
    ]
    kv_ptrs=[
        [CachePointer(1,2),CachePointer(1,2),CachePointer(1,2),CachePointer(1,2),CachePointer(1,2),CachePointer(1,2)],
        [CachePointer(1,2),CachePointer(1,2),CachePointer(1,2),CachePointer(1,2),CachePointer(1,2)],
    ]
    for i in range(len(keys)):
        key = keys[i]
        kv_ptr = kv_ptrs[i]
        tree.insert(key,kv_ptr)
        tree.root.children[1].sort_by_importance()
    #tree.visualize()

    prefix = list()
    inputs=[[1,5,5],[1,5,7,2]]
    seq = inputs[0]
    prefix = tree.search(seq)
    prefix[0].kv_ptr.chunk_id = 1000
    print(prefix)
    print('=' * 20)
    tree.visualize()


    # prefix_kv_ptr = []
    # inputs=[[1,5,5],[1,5,7,2]]
    # for seq in inputs:
    #     prefix_kv_ptr.append(tree.search(seq))
    # print(prefix_kv_ptr)