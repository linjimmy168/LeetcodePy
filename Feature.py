
"""
Delete Node in a BST
"""
def deleteNode(self, root, key):
    """
    :type root: TreeNode
    :type key: int
    :rtype: TreeNode
    """
    if not root: # if root doesn't exist, just return it
        return root
    if root.val > key: # if key value is less than root value, find the node in the left subtree
        root.left = self.deleteNode(root.left, key)
    elif root.val < key: # if key value is greater than root value, find the node in right subtree
        root.right= self.deleteNode(root.right, key)
    else: #if we found the node (root.value == key), start to delete it
        if not root.right: # if it doesn't have right children, we delete the node then new root would be root.left
            return root.left
        if not root.left: # if it has no left children, we delete the node then new root would be root.right
            return root.right
               # if the node have both left and right children,  we replace its value with the minmimum value in the right subtree and then delete that minimum node in the right subtree
        temp = root.right
        mini = temp.val
        while temp.left:
            temp = temp.left
            mini = temp.val
        root.val = mini # replace value
        root.right = self.deleteNode(root.right,root.val) # delete the minimum node in right subtree
    return root



"""
21. Merge Two Sorted Lists
"""
def mergeTwoLists(self, l1, l2):
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """
    if not l1 and not l2:
        return None
    
    result = curr = ListNode(0)
    while l1 and l2:
        if l1.val < l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    if not l1:
        curr.next = l2
    if not l2:
        curr.next = l1
    return result.next
        
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.minVal = float('inf')
        

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        if x <= self.minVal:
            self.stack.append(self.minVal)
            self.minVal = x
        self.stack.append(x)
        

    def pop(self):
        """
        :rtype: void
        """
        if self.stack.pop() == self.minVal:
            self.minVal = self.stack.pop()
        

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1]
        

    def getMin(self):
        """
        :rtype: int
        """
        return self.minVal