import unittest
from metadata_manage import RadixTree,RadixTreeNode


class TestRadixTree(unittest.TestCase):
    def setUp(self):
        self.tree = RadixTree()
    
    def test_single_insert(self):
        """测试单个键插入"""
        key = [1, 2, 3]
        self.tree.insert(key)
        
        # 验证根节点有一个子节点
        self.assertEqual(len(self.tree.root.children), 1)
        
        # 验证子节点内容
        child = self.tree.root.children[1]
        self.assertEqual(child.tokens, [1, 2, 3])
        self.assertEqual(child.mapping_list, [0, 1, 2])
    
    def test_common_prefix_insert(self):
        """测试有公共前缀的键插入"""
        self.tree.insert([1, 2, 3])
        self.tree.insert([1, 2, 4])
        
        # 验证根节点有一个子节点
        self.assertEqual(len(self.tree.root.children), 1)
        
        # 验证公共前缀节点
        prefix_node = self.tree.root.children[1]
        self.assertEqual(prefix_node.tokens, [1, 2])
        self.assertEqual(prefix_node.mapping_list, [0, 1])
        
        # 验证子节点
        self.assertEqual(len(prefix_node.children), 2)
        
        child3 = prefix_node.children[3]
        self.assertEqual(child3.tokens, [3])
        self.assertEqual(child3.mapping_list, [0])
        
        child4 = prefix_node.children[4]
        self.assertEqual(child4.tokens, [4])
        self.assertEqual(child4.mapping_list, [0])
    
    def test_partial_prefix_insert(self):
        """测试部分公共前缀的键插入"""
        self.tree.insert([1, 2, 3])
        self.tree.insert([1, 3, 4])
        
        # 验证根节点有一个子节点
        self.assertEqual(len(self.tree.root.children), 1)
        
        # 验证公共前缀节点
        prefix_node = self.tree.root.children[1]
        self.assertEqual(prefix_node.tokens, [1])
        self.assertEqual(prefix_node.mapping_list, [0])
        
        # 验证子节点
        self.assertEqual(len(prefix_node.children), 2)
        
        child2 = prefix_node.children[2]
        self.assertEqual(child2.tokens, [2, 3])
        self.assertEqual(child2.mapping_list, [0, 1])
        
        child3 = prefix_node.children[3]
        self.assertEqual(child3.tokens, [3, 4])
        self.assertEqual(child3.mapping_list, [0, 1])
    
    def test_no_common_prefix_insert(self):
        """测试无公共前缀的键插入"""
        self.tree.insert([1, 2, 3])
        self.tree.insert([4, 5, 6])
        
        # 验证根节点有两个子节点
        self.assertEqual(len(self.tree.root.children), 2)
        
        # 验证子节点
        child1 = self.tree.root.children[1]
        self.assertEqual(child1.tokens, [1, 2, 3])
        self.assertEqual(child1.mapping_list, [0, 1, 2])
        
        child4 = self.tree.root.children[4]
        self.assertEqual(child4.tokens, [4, 5, 6])
        self.assertEqual(child4.mapping_list, [0, 1, 2])
    
    def test_search_exact_match(self):
        """测试精确匹配搜索"""
        keys = [
            [1, 2, 3],
            [1, 2, 4],
            [1, 3, 4],
            [4, 5, 6]
        ]
        
        for key in keys:
            self.tree.insert(key)
        
        # 测试搜索
        result = self.tree.search([1, 2, 3])
        result = [x[1] for x in result]
        self.assertEqual(result, [1, 2, 3])
        
        result = self.tree.search([1, 2, 4])
        result = [x[1] for x in result]
        self.assertEqual(result, [1, 2, 4])
        
        result = self.tree.search([1, 3, 4])
        result = [x[1] for x in result]
        self.assertEqual(result, [1, 3, 4])
        
        result = self.tree.search([4, 5, 6])
        result = [x[1] for x in result]
        self.assertEqual(result, [4, 5, 6])
    
    def test_search_partial_match(self):
        """测试部分匹配搜索"""
        keys = [
            [1, 2, 3],
            [1, 2, 4],
            [1, 3, 4],
            [4, 5, 6]
        ]
        
        for key in keys:
            self.tree.insert(key)
        
        # 测试部分匹配
        result = self.tree.search([1, 2])
        result = [x[1] for x in result]
        self.assertEqual(result, [1, 2])
        
        result = self.tree.search([1])
        result = [x[1] for x in result]
        self.assertEqual(result, [1])
        
        result = self.tree.search([4, 5])
        result = [x[1] for x in result]
        self.assertEqual(result, [4, 5])
    
    def test_search_no_match(self):
        """测试无匹配搜索"""
        keys = [
            [1, 2, 3],
            [1, 2, 4],
            [1, 3, 4],
            [4, 5, 6]
        ]
        
        for key in keys:
            self.tree.insert(key)
        
        # 测试无匹配
        result = self.tree.search([2, 3, 4])
        result = [x[1] for x in result]
        self.assertEqual(result, [])
        
        result = self.tree.search([1, 4, 5])
        result = [x[1] for x in result]
        self.assertEqual(result, [1])  # 只匹配第一个token
    
    def test_mapping_list_after_split(self):
        """测试未排序分裂后 mapping_list 的正确性"""
        self.tree.insert([3, 1, 2])  # 插入键 [3, 1, 2]
        self.tree.insert([3, 2, 4])  # 插入键 [3, 2, 4]，导致分裂
        # 验证根节点有一个子节点
        self.assertEqual(len(self.tree.root.children), 1)
        root_child = self.tree.root.children[3]
        
        # 验证公共前缀节点
        self.assertEqual(root_child.tokens, [3])
        self.assertEqual(root_child.mapping_list, [0])
        
        # 验证子节点
        self.assertEqual(len(root_child.children), 2)
        
        child1 = root_child.children[1]
        self.assertEqual(child1.tokens, [1, 2])
        self.assertEqual(child1.mapping_list, [0, 1])
        
        child2 = root_child.children[2]
        self.assertEqual(child2.tokens, [2, 4])
        self.assertEqual(child2.mapping_list, [0, 1])
        
        # 验证原始顺序
        self.assertEqual(root_child.get_sorted_tokens(), [3])
        self.assertEqual(child1.get_sorted_tokens(), [1, 2])
        self.assertEqual(child2.get_sorted_tokens(), [2, 4])
    
    def test_token_reordering(self):
        """测试 tokens 重排序后的行为"""
        # 插入初始键
        key = [1,5,7,4,2,7]
        self.tree.insert(key)  
        
        # 获取节点
        node = self.tree.root.children[1]
        
        # 重排序 tokens (按值排序)
        sorted_indices = sorted(range(len(node.tokens)), key=lambda i: node.tokens[i], reverse=True)
        sorted_tokens = sorted(node.tokens,reverse=True)
        node.sort_by_importance()
        
        # 验证重排序后
        self.assertEqual(node.tokens, key)
        self.assertEqual(node.mapping_list, sorted_indices)
        self.assertEqual(node.get_sorted_tokens(), sorted_tokens)
        
        # 插入新键（应正确处理原始顺序）
        self.tree.insert([1,5,5,3,7])
        
        # 验证树结构
        root_child = self.tree.root.children[1]
        self.assertEqual(root_child.tokens, [1,5])
        self.assertEqual(root_child.mapping_list, [1,0])
        
        # 验证子节点
        self.assertEqual(len(root_child.children), 2)
        
        child1 = root_child.children[7]
        self.assertEqual(child1.tokens, [7,4,2,7])
        self.assertEqual(child1.mapping_list, [0,3,1,2])
        
        child2 = root_child.children[5]
        self.assertEqual(child2.tokens, [5, 3, 7])
        self.assertEqual(child2.mapping_list, [0, 1, 2])
        
