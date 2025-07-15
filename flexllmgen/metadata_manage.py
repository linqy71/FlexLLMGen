from typing import List,Dict
import unittest
from itertools import count

""" 
maintain token's longest common prefix

RadixTree接口:
def insert(self,key:List[int]) 插入token序列,后续还可能要插入kv_ptr和重要性(注意一个性质,NR整体在最后一个新生成节点)
def search(self,key:List[int])->List[RadixToken] 查找最长公共前缀,返回数据类型为RadixToken。
"""
class CachePointer:
    def __init__(self, chunk_id:int = None, token_name:int = None):
        self.chunk_id = chunk_id
        self.token_name = token_name

class RadixToken:
    # 节点内token的唯一标识
    token_count = count()
    def __init__(self, token_id:int, token_name = None, importance:int = 0):
        self.token_id = token_id
        
        self.k_ptr = None # 该token所有层的chunk_id
        self.v_ptr = None # 该token所有层的chunk_id
        
        self.importance = importance

        self.token_name = token_name or RadixToken.next_token_name()

    def __repr__(self):
        return f"(id={self.token_id},imp={self.importance})"

    
    @classmethod
    def next_token_name(cls):
        return next(cls.token_count)

class RadixTreeNode:
    def __init__(self,tokens:List[RadixToken],mapping_list=None):
        #tokens始终保留原来的顺序,重排只需要修改mapping_list,要获取重排后的顺序使用mapping_list
        self.tokens = tokens
        if mapping_list is not None:
            self.mapping_list = mapping_list
        else:
            self.mapping_list = [i for i in range(len(tokens))]
        
        self.children:Dict[int,RadixTreeNode] = {} 
    
    def is_leaf(self)->bool:
        return len(self.children) == 0

    def __repr__(self):
        return f"Node(token={self.tokens}, mapping_list={self.mapping_list})"
    
    def sort_by_importance(self):
        self.mapping_list.sort(key=lambda i: self.tokens[i].importance, reverse=True)

    def get_sorted_tokens(self):
        return [self.tokens[i] for i in self.mapping_list]
    
    @classmethod
    def create_from_int(cls,tokens:List[int]):
        tokens = [RadixToken(token_id=x) for x in tokens]
        return cls(tokens)


    
class RadixTree:
    def __init__(self):
        # root is an empty node
        self.root = RadixTreeNode(tokens=[])
    
    @staticmethod
    def common_prefix_length(a:List,b:List[RadixToken]):
        min_len = min(len(a),len(b))
        for i in range (min_len):
            if a[i] != b[i].token_id:
                return i;
        return min_len
    
    def insert(self,key:List[int]):
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
                    new_node.children[child.tokens[0]] = child

                    remaining = remaining[common_len:]
                    current = new_node
            else:
                new_node = RadixTreeNode.create_from_int(remaining)
                current.children[next_token] = new_node
                break

    
    def search(self,key:List[int]):
        current = self.root
        remaining = key
        ans = list()
        while remaining:
            next_token = remaining[0]
            if next_token in current.children:
                child = current.children[next_token]
                common_len = self.common_prefix_length(remaining,child.tokens)
                ans.extend(
                    [t for t in child.tokens[:common_len]]
                )
                if common_len != len(child.tokens):
                    break
                remaining = remaining[common_len:]
                current = child
            else:
                break
        # type = List[RadixToken]
        return ans

    def delete(self,key:List):
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
    for key in keys:
        tree.insert(key)
        tree.root.children[1].sort_by_importance()
    tree.visualize()