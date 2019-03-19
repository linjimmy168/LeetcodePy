a = ['1','2','3','4','5','6','7']
a = a[:2][::-1] + a[2:]
print(a)

"""
N-ary Tree Preorder Traversal
"""
def preorder(self, root):
    """
    :type root: Node
    :rtype: List[int]
    """
    stack, result = [root], []
    while stack:
        temp = stack.pop()
        if temp:
            result.append(temp.val)
            for i in range(len(temp.children)-1,-1,-1):
                stack.append(temp.children[i])
    return result

"""
N-ary Tree Postorder Traversal
"""
def postorder(self, root):
    """
    :type root: Node
    :rtype: List[int]
    """
    stack, result = [root], []
    while any(stack):
        node = stack.pop()
        result.append(node.val)
        stack += [child for child in node.children if child]
    return result[::-1]

"""
226. Invert Binary Tree
"""
def invertTree(self, root):
    """
    :type root: TreeNode
    :rtype: TreeNode
    """
    if not root:
        return None
    temp = root.left
    root.left = self.invertTree(root.right)
    root.right = self.invertTree(temp)
    return root


"""
617. Merge Two Binary Trees
"""
def mergeTrees(self, t1, t2):
    """
    :type t1: TreeNode
    :type t2: TreeNode
    :rtype: TreeNode
    """
    if not t1:
        return t2
    if not t2:
        return t1
    n1 = t1.val if t1 else 0
    n2 = t2.val if t2 else 0
    node = TreeNode(n1 + n2)
    node.left = self.mergeTrees(t1.left, t2.left)
    node.right = self.mergeTrees(t1.right, t2.right)
    return node

"""
110. Balanced Binary Tree
TimeO = Big(n) not nlogn
"""

def isBalanced(self, root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    height =self.getTreeHeight(root)
    return height != -1

def getTreeHeight(self, node):
    if not node:
        return 0
    l = self.getTreeHeight(node.left)
    r = self.getTreeHeight(node.right)
    if abs(l-r) > 1 or l == -1 or r == -1:
        return -1
    return max(self.getTreeHeight(node.left),self.getTreeHeight(node.right)) + 1


"""
111. Minimum Depth of Binary Tree
"""
def minDepth(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if not root:
        return 0
    if root.left == None or root.right == None:
        return max(self.minDepth(root.left), self.minDepth(root.right)) + 1
    return min(self.minDepth(root.left), self.minDepth(root.right)) + 1

"""
490. The Maze
"""
def hasPath(self, maze, start, destination):
    """
    :type maze: List[List[int]]
    :type start: List[int]
    :type destination: List[int]
    :rtype: bool
    """
    self.m, self.n = len(maze), len(maze[0])
    return self.hasPathHelper(maze, start[0], start[1], destination, set())


def hasPathHelper(self, maze, x, y, destination, stop):
    if (x, y) in stop:
        return False
    stop.add((x, y))
    if [x, y] == destination:
        return True
    for dirX, dirY in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        newX, newY = x, y     # use newX, newY to avoid over write the value.
        while 0 <= newX + dirX < self.m and 0 <= newY + dirY < self.n and maze[newX + dirX][newY + dirY] != 1: # I should not check the visited because we may pass it but not stop
            newX += dirX
            newY += dirY
        if self.hasPathHelper(maze, newX, newY, destination, stop):
            return True
    return False

"""
112. Path Sum
"""
def hasPathSum(self, root, sum):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: bool
    """
    return self.hasPathSumHelper(root, sum)

def hasPathSumHelper(self, node, sum):
    if not node:
        return False
    
    if not node.left and not node.right and sum - node.val == 0:
        return True
    return self.hasPathSumHelper(node.left, sum - node.val) or self.hasPathSumHelper(node.right, sum - node.val)

"""
199. Binary Tree Right Side View
"""
def rightSideView(self, root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    if not root:
        return []
    queue = collections.deque()
    queue.append(root)
    result = []
    while queue:
        lvl = len(queue)
        result.append(queue[-1].val)
        for _ in range(lvl):
            temp = queue.popleft()
            if temp.left:
                queue.append(temp.left)
            if temp.right:
                queue.append(temp.right)
    return result
                

    """
    297. Serialize and Deserialize Binary Tree
    """
    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        result = []
        if not root:
            return result
        que = collections.deque()
        que.append(root)
        while que:
            temp = que.popleft()
            if temp:
                result.append(str(temp.val))
            else:
                result.append('#')
                continue
            que.append(temp.left)
            que.append(temp.right)
        return ' '.join(result)

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if not data:
            return None
        ite = iter(data.split(' '))
        root = TreeNode(next(ite))
        que = collections.deque()
        que.append(root)
        while que:
            temp = que.popleft()
            if not temp:
                continue
            val = next(ite)
            temp.left = TreeNode(int(val)) if val != '#' else None
            que.append(temp.left)
            val = next(ite)
            temp.right = TreeNode(int(val)) if val != '#' else None
            que.append(temp.right)
        return root



"""
113. Path Sum II
"""
def pathSum(self, root, sum):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: List[List[int]]
    """
    result = []
    self.helper(root, sum, result, [])
    return result

def helper(self, node, sum, result, tempList):
    if not node:
        return
    tempList.append(node.val)
    if not node.left and not node.right and sum == node.val:
        result.append(tempList[:])
        tempList.pop()
        return
    else:
        self.helper(node.left, sum - node.val, result, tempList)
        self.helper(node.right, sum - node.val, result, tempList)
    tempList.pop()
