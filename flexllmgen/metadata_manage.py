from typing import List,Dict
import unittest

# class Token:
#     def __init__(self,tokenID):
#         self.tokenID = tokenID

""" 
maintain token's longest common prefix
"""
class RadixTreeNode:
    def __init__(self,tokens:List[int],mapping_list=None):
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
        return f"Node(token={self.tokens},mapping_list={self.mapping_list})"
    
    def sort_by_importance(self):
        self.mapping_list.sort(key=lambda i: self.tokens[i], reverse=True)

    def get_sorted_tokens(self):
        return [self.tokens[i] for i in self.mapping_list]
    
class RadixTree:
    def __init__(self):
        # root is an empty node
        self.root = RadixTreeNode(tokens=[])
    
    @staticmethod
    def common_prefix_length(a:List,b:List):
        min_len = min(len(a),len(b))
        for i in range (min_len):
            if a[i] != b[i]:
                return i;
        return min_len
    
    def insert(self,key:List):
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
                new_node = RadixTreeNode(remaining)
                current.children[next_token] = new_node
                break

    
    def search(self,key:List):
        current = self.root
        remaining = key
        ans = list()
        while remaining:
            next_token = remaining[0]
            if next_token in current.children:
                child = current.children[next_token]
                common_len = self.common_prefix_length(remaining,child.tokens)
                ans.extend(child.tokens[:common_len])

                if common_len != len(child.tokens):
                    break
                remaining = remaining[common_len:]
                current = child
            else:
                break
        return ans

    def delete(self,key:List):
        pass

    def visualize(self, node=None, depth=0):
        """for debug"""
        if node is None:
            node = self.root
            print(f"root:")
        else:
            indent = "  " * depth
            print(f"{indent}{node}")
        
        for child in node.children.values():
            self.visualize(child, depth + 1)


if __name__ == "__main__":
    from test_module import TestRadixTree
    import unittest
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRadixTree)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
