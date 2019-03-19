class BFS:
    """
    127. Word Ladder
    """
    def ladderLength(self, beginWord, endWord, wordList):
        wordList = set(wordList)
        import collections
        import string
        q = collections.deque([(beginWord, 1)])
        while q:
            word, level = q.popleft()
            if word == endWord:
                return level
            for i in range(len(word)):
                for c in string.ascii_lowercase:
                    tmp = word[:i] + c + word[i+1:]
                    if tmp in wordList:
                        q.append((tmp, level + 1))
                        wordList.remove(tmp)
        return 0 

    """
    126. Word Ladder II
    """
    def findLadders(self, beginWord: 'str', endWord: 'str', wordList: 'List[str]') -> 'List[List[str]]':
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: List[List[str]]

        1) BFS starting at beginWord by transforming every letter to see if the next word is in the wordList, if so, put in queue.
        2) During BFS, maintain a graph of {word:nextWord} for all valid next wods
        3) When a nextWord reaches endWord, do a backtracking DFS (pre-order traversal) on the graph to get all paths.

        Time: O(26*L*N + N), where L is average length of each word, and N is the number of words in the wordList.
        Worst case here is every word transformed happens to be in the list, so each transformation needs 26 * length of word.

        The DFS part is O(N).
        Space: O(N)
        """
        graph = collections.defaultdict(set)
        wordSet = set(wordList)
        que = set([beginWord])
        while que:
            newQue = set()
            for word in que:
                if word in wordSet:
                    wordSet.remove(word)
                if word == endWord:
                    result = []
                    self.getAllPath(result, [beginWord], graph, beginWord, endWord)
                    return result
            for word in que:
                for i in range(len(word)):
                    for c in string.ascii_lowercase:
                        newWord = word[:i] + c + word[i + 1:]
                        if newWord in wordSet:
                            newQue.add(newWord)
                            graph[word].add(newWord)
            que = newQue
        return []

    def getAllPath(self, result, tempList, graph, beginWord, endWord):
        if beginWord == endWord:
            result.append(tempList)
            return
        for word in graph[beginWord]:
            self.getAllPath(result, tempList + [word], graph, word, endWord)

    def findLadders(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: List[List[str]]
        """
        wordSet = set(wordList)
        level = set([beginWord])
        
        parents = collections.defaultdict(set)
        
        while level and endWord not in parents:
            next_level = collections.defaultdict(set)
            for word in level:
                for i in range(len(beginWord)):
                    for c in string.ascii_lowercase:
                        newWord = word[:i] + c + word[i+1:]
                        if newWord in wordSet and newWord not in parents:
                            next_level[newWord].add(word)
            level = next_level
            parents.update(next_level)
        
        res = [[endWord]]
        while res and res[0][0] !=beginWord:
            # res = [[p] + r for r in res for p in parents[r[0]]]
            next_res = []
            for r in res:
                for p in parents[r[0]]:
                    next_res.append([p] + r)
            res = next_res
        return res

    

    """
    127. Topological Sorting (BFS)
    """
    def topSortBFS(self, graph):
        import collections
        indegree = self.get_indegree(graph)
        result = []
        que = collections.deque([x for x in graph if indegree[x] == 0])
        while que:
            node = que.popleft()
            result.append(node)
            for neighbor in node.neighbors:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    que.append(neighbor)
        return result

    def get_indegree(self,graph):
        indegree = {x: 0 for x in graph}
        for node in graph:
            for neighbor in node.neighbors:
                indegree[neighbor] += 1
        return indegree

    
"""
137. Clone Graph
"""
def cloneGraph(self, node):
    if not node:
        return
    nodeCopy = UndirectedGraphNode(node.label)
    dic = {node: nodeCopy}
    que = collections.deque([node])
    while que:
        node = que.popleft()
        for neighbor in node.neighbors:
            if neighbor not in dic:
                neighborCopy = UndirectedGraphNode(neighbor.label)
                dic[neighbor] = neighborCopy
                que.append(neighbor)
            dic[node].neighbors.append(dic[neighbor])
    return nodeCopy

    """
    131. Palindrome Partitioning
    """
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        result = []
        self.helper(s, result, [])
        return result
        
    def helper(self, s, result, tempList):
        if not s:
            result.append(tempList[:])
            return 
        for i in range(1, len(s)+1):
            tempS = s[:i]
            if self.isPalindrome(tempS):
                self.helper(s[i:], result, tempList + [tempS])
            
    def isPalindrome(self, s):
        return s == s[::-1]


class DirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []


class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []
