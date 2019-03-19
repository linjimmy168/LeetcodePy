from TreeNode
class BST:
    """
    114. Flatten Binary Tree to Linked List
    """
    def flatten1(self,root):
        self.flattenTree(root)

    def flattenTree(self,TreeNode):
        if not TreeNode:
            return
        
        if not TreeNode.left and not TreeNode.right:
            return TreeNode
        
        if not TreeNode.left:
            return self.flattenTree(TreeNode.right)
        leftEnd = self.flattenTree(TreeNode.left)
        leftEnd.right = TreeNode.right
        rightEnd = self.flattenTree(TreeNode.right)
        TreeNode.right = TreeNode.left
        TreeNode.left = None
        return rightEnd if rightEnd else leftEnd

    def __init__(self,root):
        self.prev = None
    
    def flatten2(self,node):
        if node is None:
            return
        self.flatten2(node.right)
        self.flatten2(node.left)
        node.right = self.prev
        node.left = None
        self.prev = node

    """
    94. Binary Tree Inorder Traversal
    """
    def inorderTraversal(self, root):
        self.result = []
        def helper(node):
            if node is None:
                return
            helper(node.left)
            self.result.append(node.val)
            helper(node.right)
        helper(root)
        return self.result
    """
    94. Binary Tree Inorder Traversal
    """
    def inorderTraversalIteral(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        result = []
        if not root:
            return result
        stack = []
        curr = root
        while curr or stack:
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            result.append(curr.val)
            curr = curr.right
        return result

    """
    145. Binary Tree Postorder Traversal
    """
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        result = []
        if not root:
            return result
        stack = []
        stack.append(root)
        while stack:
            node = stack.pop()
            result.append(node.val)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return result[::-1]

    """
    235. Lowest Common Ancestor of a Binary Search Tree
    """
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root.val > max(p.val, q.val):
            return self.lowestCommonAncestor(root.left, p, q)
        elif root.val < min(p.val, q.val):
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return root


    """
    109. Convert Sorted List to Binary Search Tree
    """
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        if not head:
            return None
        return self.sortedListToBSTHelper(head, None)
    
    def sortedListToBSTHelper(self, head, tail):
        fast = slow = head
        if head == tail:
            return None
        while fast != tail and fast.next != tail:
            slow = slow.next
            fast = fast.next.next
        root = TreeNode(slow.val)
        root.left = self.sortedListToBSTHelper(head, slow)
        root.right = self.sortedListToBSTHelper(slow.next,tail)
        return root


            