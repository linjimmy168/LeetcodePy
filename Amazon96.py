# 185. Department Top Three Salaries
# SELECT dp.Name as Department, emp.Name as Employee, emp.Salary
# FROM Employee emp
# INNER JOIN Department dp
# ON emp.DepartmentId = dp.Id
# WHERE (SELECT count(Distinct(Salary)) 
#       FROM Employee emp2 
#       WHERE emp2.DepartmentId = emp.DepartmentId AND emp.Salary < emp2.Salary) < 3

#175. Combine Two Tables
# SELECT P.FirstName, P.LastName, A.City, A.State
# FROM Person P
# LEFT JOIN Address A
# ON P.PersonId = A.PersonId

#579. Find Cumulative Salary of an Employee
# SELECT A.Id, Max(B.Month) as Month , Sum(B.Salary) as Salary
# FROM Employee A, Employee B
# WHERE A.Id = B.Id AND B.Month Between(A.Month - 3) AND (A.Month - 1)
# Group by A.Id, A.Month
# Order By Id, Month DESC

# 579. Find Cumulative Salary of an Employee
# SELECT e1.Id, Max(e2.Month) as Month, Sum(e2.Salary) as Salary
# FROM Employee e1, Employee e2
# where e1.Id = e2.Id and e2.Month between (e1.Month - 3) and (e1.Month - 1)
# group by e1.Id, e1.Month
# order by Id, Month desc

# 176. Second Highest Salary
# select max(Salary) as SecondHighestSalary
# from employee 
# where salary < (select max(salary) from employee)
import collections


class Solution:


def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
    total = currSum = start = 0
    for i in range(len(gas)):
        total += gas[i] - cost[i]
        currSum += gas[i] - cost[i]
        if currSum < 0:
            start = i + 1
            currSum = 0
    return -1 if total < 0 else start


"""
129. Sum Root to Leaf Numbers
"""
def sumNumbers(self, root: TreeNode) -> int:
    return self.sumHelper(root, 0)
def sumHelper(self, node, resNum):
    if not node:
        return 0
    if node.left or node.right:
        return self.sumHelper(node.left, node.val + resNum * 10) + self.sumHelper(node.right,
                                                                                  node.val + resNum * 10)
    return resNum * 10 + node.val


"""
369. Plus One Linked List
"""
def plusOne(self, head: ListNode) -> ListNode:
    if not head:
        return None
    dummy = ListNode(-1)
    num = 0
    while head:
        num *= 10
        num += head.val
        head = head.next
    num += 1
    while num != 0:
        newNode = ListNode(num%10)
        temp = dummy.next
        dummy.next = newNode
        newNode.next = temp
        num //= 10
    return dummy.next

def findItinerary(self, tickets: 'List[List[str]]') -> 'List[str]':
    graph = collections.defaultdict(list)
    for f, t in sorted(tickets, reverse = True):
        graph[f].append(t)
    result = []
    route, stack = [], ['JFK']
    while stack:
        while graph[stack[-1]]:
            stack.append(graph[stack[-1]].pop())
        route.append(stack.pop())
    return route[::-1]


"""
753. Cracking the Safe
"""
def crackSafe(self, n: int, k: int) -> str:
    ans = "0" * (n - 1)
    visits = set()
    for _ in range(k ** n):
        current = ans[-n+1:] if n > 1 else '' # avoid n = 1; k = 2 => out put 11
        for y in range(k - 1, -1, -1):
            if current + str(y) not in visits:
                visits.add(current + str(y))
                ans += str(y)
                break
    return ans


"""
472. Concatenated Words
"""
def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
    words = sorted(words, key=lambda x:len(x))
    wordSet = set()
    def isQualify(w):
        if w in wordSet:
            return True
        for i in range(1, len(w)):
            if w[:i] in wordSet and isQualify(w[i:]):
                return True
        return False
    result = []
    for word in words:
        if isQualify(word):
            result.append(word)
        wordSet.add(word)
    return result

"""
403. Frog Jump
"""
def canCross(self, stones: 'List[int]') -> 'bool':
    target, stones, memo = stones[-1], set(stones), set()
    return self.canCrossHelper(stones, 1, 1, target, memo)

def canCrossHelper(self, stones, pos, jump, target, memo):
    if (pos, jump) in memo:
        return False
    if pos == target:
        return True
    if jump <= 0 or pos not in stones:
        return False
    for i in [jump - 1, jump, jump + 1]:
        if self.canCrossHelper(stones, pos + i, i, target, memo):
            return True
    memo.add((pos, jump))
    return False



"""
117. Populating Next Right Pointers in Each Node II
"""
def connect(self, root: 'Node') -> 'Node':
    result = root
    tail = dummy = TreeLinkNode(0)
    while root:
        tail.next = root.left
        if tail.next:
            tail = tail.next
        tail.next = root.right
        if tail.next:
            tail = tail.next
        root = root.next
        if not root:
            tail = dummy
            root = dummy.next
    return result

"""
51. N-Queens

diagno:
210
321
432

"""
def solveNQueens(self, n: 'int') -> 'List[List[str]]':
    self.result = []
    self.board = [['.'] * n for _ in range(n)]
    self.cols = [0] * n
    self.diag, self.anti_diag = [0] * (n * 2 - 1), [0] * (n * 2 - 1)
    self.dfs(n, 0)
    return self.result


def updateBoard(self, x, y, n, isPut):
    self.cols[x] = isPut
    self.diag[x + y] = isPut
    self.anti_diag[x - y + n - 1] = isPut
    self.board[y][x] = 'Q' if isPut else '.'


def valid(self, x, y, n):
    return not self.cols[x] and not self.diag[x + y] and not self.anti_diag[x - y + n - 1]


def dfs(self, n, y):
    if y == n:
        temp = []
        for i in range(n):
            newLine = []
            for j in range(n):
                newLine.append(self.board[i][j])
            temp.append(''.join(newLine))
        self.result.append(temp)
        return
    for x in range(n):
        if not self.valid(x, y, n):
            continue
        self.updateBoard(x, y, n, True)
        self.dfs(n, y + 1)
        self.updateBoard(x, y, n, False)

"""
149. Max Points on a Line

"""
    def maxPoints(self, points: 'List[Point]') -> 'int':
        maxNum = 0
        for i in range(len(points)):
            count = collections.defaultdict(int)
            samePoint, maxPoints = 1, 0
            for j in range(i + 1, len(points)):
                p1 = points[i]
                p2 = points[j]
                if p1.x == p2.x and p1.y == p2.y:
                    samePoint += 1
                else:
                    count[self.getSlope(p1, p2)] += 1
                    maxPoints = max(maxPoints, count[self.getSlope(p1, p2)])
            maxNum = max(maxNum, samePoint + maxPoints)
        return maxNum

    def getSlope(self, p1, p2):
        def gcd(m, n):
            return m if n == 0 else gcd(n, m % n)

        dx, dy = p2.x - p1.x, p2.y - p1.y
        if dy == 0:
            return (p1.y, 0)
        elif dx == 0:
            return (0, p1.x)
        d = gcd(dx, dy)
        return (dy / d, dx / d)



"""
20. Valid Parentheses
"""
def isValid(self, s: 'str') -> 'bool':
    dic = {')' : '(', '}' : '{', ']' : '['}
    stack = []
    for c in s:
        if c not in dic:
            stack.append(c)
        else:
            if stack:
                if dic[c] != stack.pop():
                    return False
            else:
                return False
    return len(stack) == 0


"""
315. Count of Smaller Numbers After Self
TC(nlogn)
SC(k)  unique element
finwck update: parent bit + lowerest digit
"""
def countSmaller(self, nums: 'List[int]') -> 'List[int]':
    rank, N, res = {}, len(nums), []
    finwickTree = [0] * (N + 1)
    for i, v in enumerate(sorted(nums)):
        rank[v] = i + 1

    def update(i):
        while i <= N:
            finwickTree[i] += 1
            i += (i & -i)

    def getSum(i):
        s = 0
        while i:
            s += finwickTree[i]
            i -= (i & -i)
        return s

    for i in reversed(nums):
        res += [getSum(rank[i] - 1)]
        update(rank[i])
    return res[::-1]

"""
25. Reverse Nodes in k-Group
"""
def reverseKGroup(self, head: 'ListNode', k: 'int') -> 'ListNode':
    count, node = 0, head
    while node and count < k:
        node = node.next
        count += 1
    if count < k:
        return head
    newHead, prev = self.reverseGroup(head, count)
    head.next = self.reverseKGroup(newHead, k) #head will be the last one
    return prev

def reverseGroup(self, head, count):
    prev, curr = None, head
    while count > 0:
        temp = curr.next
        curr.next = prev
        prev = curr
        curr = temp
        count -= 1
    return (curr, prev)

"""
505. The Maze II
TC(mnlog(mn))
SC(mn)
"""
def shortestDistance(self, maze: 'List[List[int]]', start: 'List[int]', destination: 'List[int]') -> 'int':
    m, n, heap, stopped = len(maze), len(maze[0]), [(0, start[0], start[1])], {(start[0], start[1]) : 0}
    while heap:
        dist, x, y = heapq.heappop(heap)
        if [x, y] == destination:
            return dist
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            newX, newY, d = x, y, 0
            while 0 <= newX + dx < m and 0 <= newY + dy < n and maze[newX + dx][newY + dy] != 1:
                newX += dx
                newY += dy
                d += 1
            if (newX, newY) not in stopped or dist + d < stopped[(newX, newY)]:
                stopped[(newX, newY)] = d + dist
                heapq.heappush(heap, (dist + d, newX, newY))
    return -1


"""
46. Permutations
"""
def permute(self, nums: 'List[int]') -> 'List[List[int]]':
    self.result = []
    self.permuteHelper(nums, [])
    return self.result

def permuteHelper(self, nums, tempList):
    #base case
    if len(tempList) == len(nums):
        self.result.append(tempList)
        return

    for i in range(len(nums)):
        if nums[i] in tempList:
            continue
        self.permuteHelper(nums, tempList + [nums[i]])

"""
Find the Celebrity
"""
def findCelebrity(self, n):
    """
    :type n: int
    :rtype: int
    """
    if n < 1:
        return 0
    candidate = 0
    for i in range(1, n):
        if not knows(i, candidate):
            candidate = i
    for i in range(n):
        if i == candidate:
            continue
        if not knows(i, candidate) or knows(candidate, i):
            return -1
    return candidate

"""
503. Next Greater Element II
"""
def nextGreaterElements(self, nums: 'List[int]') -> 'List[int]':
    result, stack, dic = [], [], {}
    for i, n in enumerate(nums):
        while stack and stack[-1][1] < n:
            idx, val = stack.pop()
            dic[idx] = n
        stack.append((i, n))
    for i, n in enumerate(nums):
        while stack and stack[-1][1] < n:
            idx, val = stack.pop()
            dic[idx] = n
        stack.append((i, n))
    for i, n in enumerate(nums):
        result.append(dic.get(i, -1))
    return result

"""
556. Next Greater Element III
"""
def nextGreaterElement(self, n: 'int') -> 'int':
    digits = list(str(n))
    smallIdx = largeIdx = -1
    for i in range(len(digits) - 2, -1, -1):
        if digits[i] < digits[i + 1]:
            smallIdx = i
            break
    if smallIdx == -1: return -1
    for i in range(len(digits) - 1, -1, -1):
        if digits[smallIdx] < digits[i]:
            largeIdx = i
            break
    digits[smallIdx], digits[largeIdx] = digits[largeIdx], digits[smallIdx]
    res = int(''.join(digits[:smallIdx + 1] + digits[smallIdx + 1:][::-1]))
    if res >= 2 ** 31 or res == n:
        return -1
    return res

"""
480. Sliding Window Median
Insertion sort
remove: Binary search O(logk) + shift O(k)
insert: Binary search O(logk) + shift O(k)
Total O(k)

TC((n-k+1)*k)
SC(k)
"""
def medianSlidingWindow(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: List[float]
    """
    if k == 0: return []
    ans = []
    window = sorted(nums[:k])
    for i in range(k, len(nums) + 1):  # len(nums) + 1 => for last calcaulate
        ans.append((window[k // 2] + window[(k - 1) // 2]) / 2.0)  # if odd got same result, even got middle part
        if i == len(nums): break
        index = bisect.bisect_left(window, nums[i - k]) # index of the num which should be pop out
        window.pop(index)
        bisect.insort_left(window, nums[i])
    return ans

"""
628. Maximum Product of Three Numbers
Q: Time complexity?
A: Let me make it clear.
My solution is O(NlogK), where K = 3.
So I said it's O(N)
The best solution can be O(N+KlogK).
====
https://hg.python.org/cpython/file/2.7/Lib/heapq.py
heapq.nlargest(n, it), where it is an iterable with m elements. 
It first constructs a min heap with the first n elements of it. 
Then, for the rest m-n elements, if they are bigger than the root, it takes the root out, puts the new element, and shifts it down.
"""
def maximumProduct(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    large = heapq.nlargest(3, nums)
    small = heapq.nsmallest(2, nums)
    return max(large[0] * large[1] * large[2], large[0] * small[0] * small[1])

"""
227. Basic Calculator II
flooring always moves to the lower integer value
so -3 // 2 == -2
"""
def calculate(self, s):
    num, stack, sign = 0, [], "+"
    for i in range(len(s)):
        if s[i].isdigit():
            num = num * 10 + int(s[i])
        if s[i] in "+-*/" or i == len(s) - 1:
            if sign == "+":
                stack.append(num)
            elif sign == "-":
                stack.append(-num)
            elif sign == "*":
                stack.append(stack.pop()*num)
            else:
                stack.append(int(stack.pop()/num))  # 14 - 3 / 2 => we though (-3)/2 =>  -3 //2 = -2 ; -3/2 = -1.5
            num = 0
            sign = s[i]
    return sum(stack)


"""
678. Valid Parenthesis String
The number of open parenthesis is in a range [cmin, cmax]
cmax counts the maximum open parenthesis, 
which means the maximum number of unbalanced '(' that COULD be paired.

cmin counts the minimum open parenthesis, 
which means the number of unbalanced '(' that MUST be paired.

The string is valid for 2 condition:

cmax will never be negative.
cmin is 0 at the end.
"""
def checkValidString(self, s):
    """
    :type s: str
    :rtype: bool
    """
    cmin = cmax = 0
    for i in s:
        if i == '(':
            cmax += 1
            cmin += 1
        elif i == ')':
            cmax -= 1
            cmin = max(cmin - 1, 0)  # there may be couples of **  so we make sure the cmin will not be negative
        elif i == '*':
            cmax += 1
            cmin = max(cmin - 1, 0)
        if cmax < 0:
            return False
    return cmin == 0

"""
532. K-diff Pairs in an Array
"""
def findPairs(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    result = 0
    counter = collections.Counter(nums)
    for n in counter:
        if k > 0 and n + k in counter or k == 0 and counter[n] > 1:   # counter[n] > 1 e.g. dic[3] = 2 => 3-3 == 0
            result += 1
    return result



"""
366. Find Leaves of Binary Tree
"""
def findLeaves(self, root):
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    def order(node, dic):
        if not node:
            return 0
        left = order(node.left, dic)
        right = order(node.right, dic)
        lvl = max(left, right) + 1
        dic[lvl].append(node.val)
        return lvl

    dic = collections.defaultdict(list)
    order(root, dic)
    return dic.values()


"""
268. Missing Number
"""
def missingNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    n = len(nums)
    return (n * (n+1))//2 - sum(nums)

"""
100. Same Tree
"""
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

"""
236. Lowest Common Ancestor of a Binary Tree
"""
def lowestCommonAncestor(self, root, p, q):
    """
    :type root: TreeNode
    :type p: TreeNode
    :type q: TreeNode
    :rtype: TreeNode
    """
    if not root:
        return None
    if root == p or root == q:
        return root
    left = self.lowestCommonAncestor(root.left, p, q)
    right = self.lowestCommonAncestor(root.right, p, q)
    if left and right:
        return root
    return right if right else left



"""
233. Number of Digit One
1. 
10 number 1 one at one place   k = 1
100 number 10 one at ten place k = 10
1000 number 100 one at hundred place k = 100
=>  r = n / k  =>  r / 10 * k

2.
how to fix special number. (10)
=> r/ 10 * k + (m + 1 if r % 10 == 1 else 0)

So far, everything looks good, but we need to fix those special rows, how?
We can use the mod. Take 10, 11, and 12 for example, 
if n is 10, we get (n / 1) / 10 * 1 = 1 ones at ones's place, perfect, 
but for tens' place, we get (n / 10) / 10 * 10 = 0, that's not right, there should be a one at tens' place! 
Calm down, from 10 to 19, we always have a one at tens's place, let m = n % k, the number of ones at this special place is m + 1, 
so let's fix the formula to be:


3.  handle 10 ~ 19
once digit is larger than 2 we should add 10 more ones to the tens' place.
=> ((r + 8) / 10) *k + (m + 1 if r % 10 == 1 else 0)


Wait, how about 20, 21 and 22?
Let's say 20, use the above formula we get 0 ones at tens' place, 
but it should be 10! How to fix it? We know that once the digit is larger than 2, 
we should add 10 more ones to the tens' place, a clever way to fix is to add 8 to r, 
so our final formula is:
(r + 8) / 10 * k + (r % 10 == 1 ? m + 1 : 0)
As you can see, it's all about how we fix the formula. Really hope that makes sense to you.

"""
def countDigitOne(self, n):
    """
    :type n: int
    :rtype: int
    """
    k, count = 1, 0
    while k <= n: # every while loop is from digit one, ten , hundred,.... and so on
        r = n // k  #save previous num
        m = n % k   # m used to calculated 1 eg. 51 ~ 59 not to 60; 61 ~ 69 not to 70
        count += ((r + 8) // 10) * k + (m + 1 if r % 10 == 1 else 0)   # (m + 1 if r % 10 == 1 else 0) only handle the one digit part only
        k *= 10
    return count


"""
11. Container With Most Water
"""
def maxArea(self, height: 'List[int]') -> 'int':
    left, right = 0, len(height) - 1
    area = 0
    while left < right:
        area = max(area, (right - left) * min(height[left], height[right]))
        if height[left] > height[right]:
            right -= 1
        else:
            left += 1
    return area

"""
763. Partition Labels
"""
    def partitionLabels(self, S):
        """
        :type S: str
        :rtype: List[int]
        """
        dic = {}
        for idx, val in enumerate(S):
            dic[val] = idx

        result = []
        maxHere, preIdx = 0, 0
        for idx, val in enumerate(S):
            maxHere = max(maxHere, dic[val])
            if maxHere == idx:
                result.append(maxHere - preIdx + 1)
                preIdx = maxHere + 1 # this part is the last alph. So we have to add one index
        return result

    """
    Longest Substring with At Most K Distinct Characters
    """
    def lengthOfLongestSubstringKDistinct(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        start_indx, res, = 0, 0
        dic = {}
        for idx, c in enumerate(s):
            dic[c] = idx
            if len(dic) > k:
                start_indx = min(dic.values())
                del dic[s[start_indx]]
                start_indx += 1
            res = max(res, idx - start_indx + 1)
        return res

    """
    Add Bold Tag in String
    """

    def addBoldTag(self, s, dict):
        """
        :type s: str
        :type dict: List[str]
        :rtype: str
        """
        status = [False] * len(s)
        result = ''
        for word in dict:
            start = s.find(word)
            last = len(word)
            while start != -1:
                for i in range(start, last + start):
                    status[i] = True
                start = s.find(word, start + 1)  # this part supports the overlap
        i = 0
        while i < len(s):
            if status[i]:
                result += '<b>'
                while i < len(s) and status[i]:
                    result += s[i]
                    i += 1
                result += '</b>'
            else:
                result += s[i]
                i += 1
        return result

    """
    Max Consecutive Ones
    """

    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max_consecutive, total_here = 0, 0
        for i in nums:
            if i == 1:
                total_here += 1
            else:
                total_here = 0
            max_consecutive = max(total_here, max_consecutive)
        return max_consecutive

    """
    Max Consecutive Ones II
    """

    def findMaxConsecutiveOnesII(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        pre, curr, maxlen = -1, 0, 0
        for n in nums:
            if n == 0:
                pre, curr = curr, 0
            else:
                curr += 1
            maxlen = max(maxlen, pre + 1 + curr)
        return maxlen

    """
    399. Evaluate Division
    """

    # This version time O(e + q*e), space O(e)
    # e = equation, q= query
    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        """

        import collections
        def divide(x, y, visited):
            if x == y:
                return 1.0  # it is himself
            visited.add(x)
            for n in g[x]:
                if n in visited:
                    continue
                d = divide(n, y, visited)
                if d > 0:
                    return d * g[x][n]  # don't understand
            return -1.0

        g = collections.defaultdict(dict)  # make a graph
        for (x, y), v in zip(equations, values):
            g[x][y] = v
            g[y][x] = 1.0 / v
        ans = []
        for x, y in queries:
            if x in g and y in g:
                ans.append(divide(x, y, set()))
            else:
                ans.append(-1)
        # ans = [divide(x,y, set()) if x in g and y inb g else -1 for x, y in queries]
        return ans

    # Union Find  Time: O(e + q) Space: O(e)
    def calcEquationUnionFindVersion(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        """
        parents = {}  # {Child:Parent}, eg {'a':'b'}
        weights = {}  # {Node: float}, eg{'a': 1.0}
        result = []

        for (eq1, eq2), value in zip(equations, values):
            if eq1 not in parents:
                parents[eq1] = eq1
                weights[eq1] = 1.0
            if eq2 not in parents:
                parents[eq2] = eq2
                weights[eq2] = 1.0
            self.calUnion(eq1, eq2, parents, weights, value)

        for q1, q2 in queries:
            if q1 not in parents or q2 not in parents:
                result.append(-1.0)
            else:
                parent1 = self.calFind(q1, parents, weights)
                parent2 = self.calFind(q2, parents, weights)
                if parent1 != parent2:
                    result.append(-1.0)
                else:
                    result.append(weights[q1] / weights[q2])

        return result

    def calUnion(self, node1, node2, parents, weights, value):
        parent1 = self.calFind(node1, parents, weights)
        parent2 = self.calFind(node2, parents, weights)
        if parent1 != parent2:
            parents[parent1] = parent2
            weights[parent1] = value * (weights[node2] / weights[
                node1])  # IMPORTANT: Node1 may already be compressed: its weight could be the product of all weights up to parent1
            # the value of compression will not be higher than

    def calFind(self, node, parents, weights):
        # Find parent node of a given node, doing path compression while doing so (set the node's parent to its root and multiply all weights along the way.)
        if parents[node] != node:
            p = parents[node]
            parents[node] = self.calFind(parents[node], parents, weights)
            weights[node] = weights[node] * weights[p]
        return parents[node]

    """
    Redundant Connection II
    """

    def findRedundantDirectedConnection(self, edges):
        """
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        import collections
        node_parent = collections.defaultdict(int)
        parents = {}
        ans1, ans2 = None, None
        remove_edge = None

        def find(x):
            while x != parents[x]:
                x = parents[x]
            return parents[x]

        # step 1 find 2 parents
        for n1, n2 in edges:
            if node_parent[n2] != 0:
                ans1, ans2 = [node_parent[n2], n2], [n1, n2]
                remove_edge = [n1, n2]

            node_parent[n2] = n1

        # step 2 find cycle
        for n1, n2 in edges:
            if [n1, n2] == remove_edge:
                continue
            if n1 not in parents: parents[n1] = n1
            if n2 not in parents: parents[n2] = n2
            p1 = find(n1)
            p2 = find(n2)
            if p1 == p2:
                if ans1 is None:
                    return [n1, n2]
                else:
                    return ans1
            parents[p2] = p1

        return ans2

    """
    Word Search II
    """

    def findWords(self, board, words):
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """
        m, n = len(board), len(board[0])
        self.visited = [[False] * n for _ in range(m)]
        self.board = board
        self.result = set()
        self.buildTrie(words)
        for i in range(len(board)):
            for j in range(len(board[0])):
                self.findWordsDfs(i, j, '', self.trie)
        return list(self.result)

    def findWordsDfs(self, x, y, word, trie):
        if '#' in trie:
            self.result.add(word)
        if 0 <= x < len(self.board) and 0 <= y < len(self.board[0]) and self.visited[x][y] != True and self.board[x][
            y] in trie:
            self.visited[x][y] = True
            for newX, newY in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                self.findWordsDfs(newX, newY, word + self.board[x][y], trie[self.board[x][y]])
            self.visited[x][y] = False

    def buildTrie(self, words):
        self.trie = {}
        for word in words:
            p = self.trie
            for c in word:
                if c not in p:
                    p[c] = {}
                p = p[c]
            p['#'] = '#'


"""
House Robber
"""


def rob(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:
        return 0
    if len(nums) < 2:
        return nums[0]

    dp = [0] * len(nums)
    for i in range(len(nums)):
        dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])

    return max(dp[len(dp) - 1], dp[len(dp) - 2])


"""
House Robber II
"""


def rob2(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums: return 0
    if len(nums) <= 2: return max(nums)
    return max(self.rob_row(nums[1:]), self.rob_row(nums[:-1]))


def rob_row(self, nums):
    res = [0] * len(nums)
    res[0], res[1] = nums[0], max(nums[0], nums[1])

    for i in range(2, len(nums)):
        res[i] = max(res[i - 1], res[i - 2] + nums[i])
    return res[-1]


"""
346. Moving Average from Data Stream
"""


class MovingAverage:

    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self.vector, self.size, self.sum, self.idx = [0] * size, size, 0, 0

    def next(self, val):
        """
        :type val: int
        :rtype: float
        """
        self.idx += 1
        self.sum -= self.vector[self.idx % self.size]
        self.vector[self.idx % self.size] = val
        self.sum += val
        return self.sum / float(min(self.idx, self.size))


"""
694. Number of Distinct Islands
"""
def numDistinctIslands(self, grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    self.step = ''
    record = set()
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                self.dfs(i, j, grid, 's')
                record.add(self.step)
                self.step = ''
    return len(record)

def dfs(self, x, y, grid, direct):
    if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 1:
        self.step += direct
        grid[x][y] = 0
        for newX, newY, newDirect in [(x + 1, y, 'd'), (x - 1, y, 'u'), (x, y + 1, 'r'), (x, y - 1, 'l')]:
            self.dfs(newX, newY, grid, newDirect)
        self.step += 'f'  # if we don't have 'f'; [[1,1,0],[0,1,1],[0,0,0],[1,1,1],[0,1,0]] we will got same result


"""
17. Letter Combinations of a Phone Number
"""
def letterCombinations(self, digits):
    """
    :type digits: str
    :rtype: List[str]
    """
    if not digits:
        return []
    self.alphbat = ['','','abc','def','ghi','jkl','mno','pqrs','tuv','wxyz']
    result = [] 
    self.dfs(digits,result,'',0)
    return result

def dfs(self, digits, result, tempStr,idx):
    if len(tempStr) == len(digits):
        result.append(tempStr)
        return
    alphbats = self.alphbat[int(digits[idx])]
    for c in alphbats:
        self.dfs(digits,result,tempStr+c,idx+1)


"""
252. Meeting Rooms
"""
def canAttendMeetings(self, intervals):
    """
    :type intervals: List[Interval]
    :rtype: bool
    """
    intervals.sort(key=lambda x:x.start)
    for i in range(0,len(intervals)-1):
        if intervals[i].end > intervals[i+1].start:
            return False
    return True

"""
253. Meeting Rooms II
"""
    def minMeetingRooms(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: int
        """
        startArray, endArray = [], []
        for v in intervals:
            startArray.append(v.start)
            endArray.append(v.end)
            
        startArray.sort()
        endArray.sort()
        
        room = endIdx = 0
        for i in range(len(intervals)):
            if startArray[i] < endArray[endIdx]:
                room += 1
            else:
                endIdx += 1
        return room



"""
127. Word Ladder
"""
def ladderLength(self, beginWord, endWord, wordList):
    """
    :type beginWord: str
    :type endWord: str
    :type wordList: List[str]
    :rtype: int
    """
    import string
    wordSet = set(wordList)
    que = collections.deque()
    que.append((beginWord,1))
    while que:
        word, lvl = que.popleft()
        if word == endWord:
            return lvl
        for i in range(len(word)):
            for c in string.ascii_lowercase:
                tempWord = word[:i] + c + word[i+1:]
                if tempWord in wordSet:
                    que.append((tempWord, lvl + 1))
                    wordSet.remove(tempWord)
    return 0

"""
447. Number of Boomerangs
For every i, we capture the number of points equidistant from i. Now for this i, 
we have to calculate all possible permutations of (j,k) from these equidistant points.
Total number of permutations of size 2 from n different points is nP2 = n!/(n-2)! = n * (n-1). hope this helps.

permutation = p!/(p-r)!
n!/(n-2)! = n * (n - 1)  => n candidates 2 position can be placed
"""
def numberOfBoomerangs(self, points):
    """
    :type points: List[List[int]]
    :rtype: int
    """
    result = 0
    for p_a in points:
        dic = {}
        for p_b in points:
            dist = (p_a[0] - p_b[0]) ** 2 + (p_a[1] - p_b[1]) ** 2
            dic[dist] = dic.get(dist, 0) + 1
        for n in dic.values():
            result += n * (n - 1)
    return result

"""
973. K Closest Points to Origin
"""
import heapq
class Solution:
    def kClosest(self, points: 'List[List[int]]', K: 'int') -> 'List[List[int]]':
        heap, result = [], []
        for point in points:
            heapq.heappush(heap, (point[0] ** 2 + point[1] ** 2, point))
        for _ in range(K):
            result.append(heapq.heappop(heap)[1])
        return result
    

"""
417. Pacific Atlantic Water Flow
"""


def pacificAtlantic(self, matrix: 'List[List[int]]') -> 'List[List[int]]':
    if not matrix:
        return []
    self.matrix = matrix
    pacific, atlantic = [[False] * len(matrix[0]) for _ in range(len(matrix))], [[False] * len(matrix[0]) for _ in
                                                                                 range(len(matrix))]
    result = []

    for i in range(len(matrix)):
        self.pacificAtlanticDfs(i, 0, i, 0, pacific)
        self.pacificAtlanticDfs(i, len(matrix[0]) - 1, i, len(matrix[0]) - 1, atlantic)

    for i in range(len(matrix[0])):
        self.pacificAtlanticDfs(0, i, 0, i, pacific)
        self.pacificAtlanticDfs(len(matrix) - 1, i, len(matrix) - 1, i, atlantic)

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if pacific[i][j] and atlantic[i][j]:
                result.append([i, j])
    return result


def pacificAtlanticDfs(self, x, y, prex, prey, visited):
    if 0 <= x < len(self.matrix) and 0 <= y < len(self.matrix[0]) and not visited[x][y] and self.matrix[x][y] >= \
            self.matrix[prex][prey]:
        visited[x][y] = True
        for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            self.pacificAtlanticDfs(nx, ny, x, y, visited)

"""
105. Construct Binary Tree from Preorder and Inorder Traversal
"""
def buildTree(self, preorder, inorder):
    """
    :type preorder: List[int]
    :type inorder: List[int]
    :rtype: TreeNode
    """
    if inorder:
        ind = inorder.index(preorder.pop(0))
        root = TreeNode(inorder[ind])
        root.left = self.buildTree(preorder, inorder[0:ind])
        root.right = self.buildTree(preorder, inorder[ind+1:])
        return root


"""
50. Pow(x, n)
2^3 = (2^0 * 2^0 * 2^1) * (2^0 * 2^0 * 2^1) * (2^0 * 2^0 * 2^1)
"""
def myPow(self, x, n):
    """
    :type x: float
    :type n: int
    :rtype: float
    """
    if not n:
        return 1
    if n < 0:
        return 1/self.myPow(x,-n)
    if n%2 == 1:
        return x* self.myPow(x,n-1)
    return self.myPow(x*x,n/2)
nextPermutation
"""
116. Populating Next Right Pointers in Each Node
"""
def connect(self, root: 'Node') -> 'Node':
    result = root
    while root and root.left:
        next = root.left
        while root:
            root.left.next = root.right
            if root.next:
                root.right.next = root.next.left
            else:
                root.right.next = None
            root = root.next
        root = next
    return result


"""
787. Cheapest Flights Within K Stops
TC(O( Knlogn))
"""
def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, K: int) -> int:
    graph = collections.defaultdict(dict)
    heap = [(0, src, 0)]
    for start, dest, prices in flights:
        graph[start][dest] = prices
    while heap:
        price, src, k = heapq.heappop(heap)
        if src == dst:
            return price
        if k <= K:
            for stop in graph[src]:
                heapq.heappush(heap, (price + graph[src][stop], stop, k + 1))
    return -1


"""
286. Walls and Gates
"""
def wallsAndGates(self, rooms: 'List[List[int]]') -> 'None':
    """
    Do not return anything, modify rooms in-place instead.
    """
    if not rooms: return
    m, n = len(rooms), len(rooms[0])

    def dfs(x, y, step):
        if 0 <= x < m and 0 <= y < n and rooms[x][y] >= step:
            rooms[x][y] = step
            for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                dfs(nx, ny, rooms[x][y] + 1)

    for i in range(m):
        for j in range(n):
            if rooms[i][j] == 0:
                dfs(i, j, 0)
        

"""
221. Maximal Square
"""
def maximalSquare(self, matrix):
    """
    :type matrix: List[List[str]]
    :rtype: int
    """
    if not matrix:
        return 0
    dp = [[0 if matrix[i][j] == '0' else 1 for j in range(len(matrix[0]))] for i in range(len(matrix))]
    for i in range(1, len(matrix)):
        for j in range(1, len(matrix[0])):
            if matrix[i][j] == '1':
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    maxNum = float('-inf')
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            maxNum = max(maxNum, dp[i][j])
    return maxNum ** 2 if maxNum != float('-inf') else 0
                


"""
221. Maximal Square
"""
def maximalSquare(self, matrix):
    """
    :type matrix: List[List[str]]
    :rtype: int
    """
    if not matrix: 
        return 0
    m,n = len(matrix), len(matrix[0])
    dp = [[0 if matrix[i][j] == '0' else 1 for j in range(n)] for i in range(m)]
    
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == '1':
                dp[i][j] = min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1]) + 1
            else:
                dp[i][j] == 0
    result = max(max(row) for row in dp)
    return result**2


"""
652. Find Duplicate Subtrees
"""
def findDuplicateSubtrees(self, root):
    """
    :type root: TreeNode
    :rtype: List[TreeNode]
    """
    self.res = []
    self.dic = {}
    self.dfs(root)
    return self.res


def dfs(self, root):
    if not root:
        return '#'
    tree = self.dfs(root.left) + self.dfs(root.right) + str(root.val)
    if tree in self.dic and self.dic[tree] == 1:
        self.res.append(root)
    self.dic[tree] = self.dic.get(tree,0) + 1
    return tree

"""
957. Prison Cells After N Days
"""
def prisonAfterNDays(self, cells, N):
    """
    :type cells: List[int]
    :type N: int
    :rtype: List[int]
    """
    N %= 14
    if N == 0:
        N = 14
    for _ in range(N):
        for i in range(1, len(cells) - 1):
            cells[i] = cells[i] + 2 if cells[i - 1] & 1 == cells[i + 1] & 1 else cells[i]
        for i in range(len(cells)):
            cells[i] >>= 1
    return cells

"""
13. Roman to Integer
"""
def romanToInt(self, s):
    """
    :type s: str
    :rtype: int
    """
    dic = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    for i in range(len(s) - 1):
        if dic[s[i]] < dic[s[i + 1]]:
            total -= dic[s[i]]
        else:
            total += dic[s[i]]
    total += dic[s[-1]]
    return total

"""
323. Number of Connected Components in an Undirected Graph
TC(E + n) => edges + n
every node through 'e' edges
"""
def countComponents(self, n, edges):
    """
    :type n: int
    :type edges: List[List[int]]
    :rtype: int
    """
    self.parent = [i for i in range(n)]
    result = n
    for i, j in edges:
        if self.isUnion(self.parent[i], self.parent[j]):
            result -= 1
    return result


def isUnion(self, i, j):
    sid = self.find(i)
    eid = self.find(j)
    if sid == eid:
        return False
    else:
        self.parent[sid] = eid
        return True


def find(self, node):
    while self.parent[node] != node:
        self.parent[node] = self.parent[self.parent[node]]
        node = self.parent[node]
    return node


"""
163. Missing Ranges
"""
def findMissingRanges(self, nums, lower, upper):
    """
    :type nums: List[int]
    :type lower: int
    :type upper: int
    :rtype: List[str]
    """
    result = []
    nums.append(upper + 1)
    pre = lower - 1
    for n in nums:
        if n == pre + 2:
            result.append(str(n-1))
        elif n > pre +2:
            result.append(str(pre+1) + '->' + str(n-1))
        pre = n
    return result

"""
76. Minimum Window Substring
"""
def minWindow(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: str
    """
    need = collections.Counter(t)            #hash table to store char frequency
    missing = len(t)                         #total number of chars we care
    start, end = 0, 0
    i = 0                                    # i will always stop at the leftest char in t
    for j, char in enumerate(s, 1):          #index j from 1
        if need[char] > 0:
            missing -= 1
        need[char] -= 1
        if missing == 0:                     #match all chars
            while i < j and need[s[i]] < 0:  #remove chars to find the real start, j will be next start; why : need[s[i]] < 0 => because we would like to find the smallest one
                need[s[i]] += 1             #The middle part may contain lots of need word so we have to add it back
                i += 1
            if end == 0 or j-i < end-start:  #update window
                start, end = i, j
            need[s[i]] += 1                  #make sure the first appearing char satisfies need[char]>0
            missing += 1                     #we missed this first char, so add missing by 1
            i += 1                           #i + 1 because j start from 1
    return s[start:end]

"""
289. Game of Life
"""
def gameOfLife(self, board):
    """
    :type board: List[List[int]]
    :rtype: void Do not return anything, modify board in-place instead.
    """
    n, m = len(board), len(board[0])

    def update(i, j):
        live = 0
        for x, y in [(i - 1, j - 1), (i - 1, j + 0), (i - 1, j + 1), (i + 0, j - 1), (i + 0, j + 1), (i + 0, j + 0),
                     (i + 1, j - 1), (i + 1, j + 0), (i + 1, j + 1)]:
            if 0 <= x < n and 0 <= y < m:
                live += board[x][y] & 1
        if (board[i][j] & 1 == 1) and (live - board[i][j]) in [2, 3]:  # if you add 2, when you //2, you will get one. So the purpose is helping to survive.
            board[i][j] += 2
        if (board[i][j] & 1 == 0) and (live == 3):
            board[i][j] += 2

    if not board and not board[0]:
        return
    for i in range(n):
        for j in range(m):
            update(i, j)
    for i in range(n):
        for j in range(m):
            board[i][j] >>= 1


"""
733. Flood Fill
"""
def floodFill(self, image, sr, sc, newColor):
    """
    :type image: List[List[int]]
    :type sr: int
    :type sc: int
    :type newColor: int
    :rtype: List[List[int]]
    """
    if not image or image[sr][sc] == newColor:
        return image
    n,m = len(image), len(image[0])
    def dfs(x, y, oldColor, color):
        if 0 <= x < n and 0 <= y < m and image[x][y] == oldColor:
            image[x][y] = newColor
            for newX, newY in [(x+1, y),(x-1,y),(x,y+1),(x,y-1)]:
                dfs(newX, newY, oldColor, color)
    dfs(sr, sc, image[sr][sc], newColor)
    return image
        

"""
317. Shortest Distance from All Buildings
"""
def shortestDistance(self, grid: List[List[int]]) -> int:
    if not grid or not grid[0]:
        return -1
    n, m = len(grid), len(grid[0])
    buildingNum = sum(val for line in grid for val in line if val == 1)
    hit, distance = [[0] * m for _ in range(n)], [[0] * m for _ in range(n)]

    def bfs(x, y):
        visited = [[0] * m for _ in range(n)]
        visited[x][y], count1, que = True, 1, collections.deque([(x, y)])
        dist = 0
        while que:
            size = len(que)
            dist += 1
            for _ in range(size):
                tx, ty = que.popleft()
                for i, j in [(tx + 1, ty), (tx - 1, ty), (tx, ty + 1), (tx, ty - 1)]:
                    if 0 <= i < n and 0 <= j < m and not visited[i][j]:
                        visited[i][j] = True
                        if grid[i][j] == 0:
                            que.append((i, j))
                            hit[i][j] += 1
                            distance[i][j] += dist
                        elif grid[i][j] == 1:
                            count1 += 1
        return count1 == buildingNum

    for i in range(n):
        for j in range(m):
            if grid[i][j] == 1:  # calculate every building to every ground's distance
                if not bfs(i, j):
                    return -1

    shortDistance = float('inf')
    for i in range(n):
        for j in range(m):
            if grid[i][j] == 0 and hit[i][j] == buildingNum:
                shortDistance = min(distance[i][j], shortDistance)
    return -1 if shortDistance == float('inf') else shortDistance


"""
827. Making A Large Island
"""


def largestIsland(self, grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    areaDic = {}

    def move(i, j):   # I have to make sure I provide valid move so I am going to check after before I return it
        for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
                yield x, y

    def dfs(i, j, idx):
        area = 1
        grid[i][j] = idx
        for x, y in move(i, j):
            if grid[x][y] == 1:
                area += dfs(x, y, idx)
        return area

    index = 1
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                index += 1
                areaDic[index] = dfs(i, j, index)

    result = max(areaDic.values() or [0]) # avoid only area one
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                possible = set(grid[x][y] for x, y in move(i, j) if grid[x][y] > 1)
                result = max(result, sum(areaDic[index] for index in possible) + 1)
    return result

"""
373. Find K Pairs with Smallest Sums
when j bigger; restart from i
"""
def kSmallestPairs(self, nums1, nums2, k):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :type k: int
    :rtype: List[List[int]]
    """
    que, pairs = [], []
    def push(i, j):
        if i < len(nums1) and j < len(nums2):
            heapq.heappush(que, [nums1[i] + nums2[j], i, j])
    push(0,0)
    while que and len(pairs) < k:
        _, i, j = heapq.heappop(que)
        pairs.append([nums1[i],nums2[j]])
        push(i, j+1)
        if j == 0:
            push(i + 1, 0)
    return pairs

"""
545. Boundary of Binary Tree
"""
def boundaryOfBinaryTree(self, root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    if not root:
        return []
    self.root = root
    self.result = [root.val]
    self.leftBoundary(root.left)
    self.leavesBoundary(root)
    self.rightBoundary(root.right)
    return self.result


def leftBoundary(self, node):
    if not node or not node.left and not node.right:
        return
    self.result.append(node.val)
    if node.left:
        self.leftBoundary(node.left)
    else:
        self.leftBoundary(node.right)


def rightBoundary(self, node):
    if not node or not node.left and not node.right:
        return
    if node.right:
        self.rightBoundary(node.right)
    else:
        self.rightBoundary(node.left)
    self.result.append(node.val)


def leavesBoundary(self, node):
    if not node:
        return
    if node != self.root and not node.left and not node.right:
        self.result.append(node.val)
    if node.left:
        self.leavesBoundary(node.left)
    if node.right:
        self.leavesBoundary(node.right)



def largestNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: str
    """
    self.quickSort(nums, 0, len(nums)-1)
    return str(int(''.join(map(str, nums))))

def quickSort(self, nums, l, r):
    if l >= r:
        return
    pos = self.partition(nums, l, r)
    self.quickSort(nums, l, pos-1)
    self.quickSort(nums, pos+1, r)

def partition(self, nums, l, r):
    i = l - 1
    pivot = nums[r]
    for j in range(l, r):
        if self.compare(nums[j], pivot):
            i = i + 1
            nums[i], nums[j] = nums[j], nums[i]
    nums[i + 1], nums[r] = nums[r], nums[i + 1]
    return i+1
def compare(self, n1, n2):
    return str(n1) + str(n2) > str(n2) + str(n1)


#left difit small than right digit = right - left => IV => 5 - 1 = 4
def intToRoman(self, num):
    """
    :type num: int
    :rtype: str
    """
    numerals = ['M','CM','D','CD','C','XC','L','XL','X','IX','V','IV','I']
    values = [1000,900,500,400,100,90,50,40,10,9,5,4,1]
    result = []
    for i in range(len(numerals)):
        while num >= values[i]:
            result.append(numerals[i])
            num -= values[i]
    return ''.join(result)

"""
159. Longest Substring with At Most Two Distinct Characters
"""
def lengthOfLongestSubstringTwoDistinct(self, s):
    """
    :type s: str
    :rtype: int
    """
    left, longest, dic = 0, 0, {}
    maxDistance = 2
    for i, c in enumerate(s):
        dic[c] = i
        if len(dic) > maxDistance:
            removeIdx = min(dic.values())
            del dic[s[removeIdx]]
            left = removeIdx + 1
        longest = max(longest, i - left + 1)
    return longest


"""
909. Snakes and Ladders
"""
def snakesAndLadders(self, board):
    """
    :type board: List[List[int]]
    :rtype: int
    """
    N = len(board)
    def get(s):
        if s % N == 0:  # find is final of the row(6,12,18)
            row = N - (s // N)  # from left or right (odd or even)
            if (s // N) % 2 == 0:  # should not use row to judge the left or right. we should use N to calcualte, the N is odd or even the result will be difference.
                return (row, 0)  # it is at the last of the row; so we have to minus 1 (s // N - 1)
            else:
                return (row, N - 1)  # total row - current row; odd row: num left to right; even: right to left
        else:
            row = N - 1 - s // N
            if (s // N) % 2 == 0:
                return (row, s % N - 1)
            else:
                return (row, N - s % N)  # left to right; N - curr col

    dist = {1: 0}  # 用于记录  How many steps can be used to arrive at the place.
    queue = collections.deque([1])
    while queue:
        s = queue.popleft()  # 出队
        if s == N*N:
            return dist[s]
        for s2 in range(s+1, min(s+6, N*N) + 1):  # 向下扩展子节点，判断入队
            r, c = get(s2)
            if board[r][c] != -1:
                s2 = board[r][c]
            if s2 not in dist:
                dist[s2] = dist[s] + 1
                queue.append(s2)  # 入队
    return -1

"""
783. Minimum Distance Between BST Nodes
use inorder to iterate all the node. 
"""
def minDiffInBST(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    self.res = float('inf')
    self.prev = -float('inf')
    self.minDiffInBSTDfs(root)
    return self.res

def minDiffInBSTDfs(self, node):
    if node.left:
        self.minDiffInBSTDfs(node.left)
    self.res = min(self.res, node.val - self.prev)
    self.prev = node.val
    if node.right:
        self.minDiffInBSTDfs(node.right)

"""
518. Coin Change 2
Explain: https://www.youtube.com/watch?v=DJ4a7cmjZY0
"""
def change(self, amount: 'int', coins: 'List[int]') -> 'int':
    dp = [[0] * (amount + 1) for _ in range(len(coins) + 1)]
    dp[0][0] = 1
    for i in range(1, len(dp)):
        dp[i][0] = 1
        for j in range(1, len(dp[0])):    #dp[i-1][j] how previous coins to total the amount
            dp[i][j] = dp[i - 1][j] + (dp[i][j - coins[i - 1]] if j - coins[i - 1] >= 0 else 0) #  dp[i][j-coins[i-1]] j=> amount ; i is what kind of currency
    return dp[-1][-1]


"""
518. Coin Change 2 - 2
"""
def change(self, amount, coins):
    """
    :type amount: int
    :type coins: List[int]
    :rtype: int
    """
    dp = [0] * (amount+1)
    dp[0] = 1
    for coin in coins:
        for i in range(coin,len(dp)):
            dp[i] += dp[i - coin]
    return dp[-1]


"""
322. Coin Change
"""
def coinChange(self, coins, amount):
    coins.sort(reverse=True)
    INVALID = 10**10
    self.ans = INVALID
    self.coinChangeDfs(0, amount, 0,coins)
    return -1 if self.ans == INVALID else self.ans



def coinChangeDfs(self,idx, amount, count,coins):      
    if amount == 0:
        self.ans = count
        return
    if idx == len(coins): return
    coin = coins[idx]
    for k in range(amount // coin, -1, -1):
        if count + k >= self.ans: break    # now is the best solution because bigger dimension one cannot be pass. The rest of the small dimension will create more coins.
        self.coinChangeDfs(idx + 1, amount - k * coin, count + k,coins)


"""
339. Nested List Weight Sum
"""
def depthSum(self, nestedList):
    """
    :type nestedList: List[NestedInteger]
    :rtype: int
    """
    result = 0
    que = collections.deque([(1, lis) for lis in nestedList])
    while que:
        weight, lis = que.popleft()
        if lis.isInteger():
            result += lis.getInteger() * weight
        else:
            for child in lis.getList():
                que.append((weight+1, child))
    return result

"""
496. Next Greater Element I
"""
def nextGreaterElement(self, findNums, nums):
    """
    :type findNums: List[int]
    :type nums: List[int]
    :rtype: List[int]
    """
    dic ={}
    stack, ans = [], []
    for num in nums:
        while len(stack) and stack[-1] < num:
            dic[stack.pop()] = num
        stack.append(num)
    for x in findNums:
        ans.append(dic.get(x,-1))
    return ans


"""
658. Find K Closest Elements
"""
def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
    left, right = 0, len(arr) - k
    while left < right:
        mid = (left + right) // 2
        if x - arr[mid] > arr[mid + k] - x:
            left = mid + 1
        else:
            right = mid
    return arr[left:left + k]



"""
543. Diameter of Binary Tree
"""
def diameterOfBinaryTree(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    self.long = float('-inf')
    self.calculate(root)
    return 0 if self.long == float('-inf') else self.long


def calculate(self, root):
    if not root:
        return 0
    left = self.calculate(root.left)
    right = self.calculate(root.right)
    self.long = max(self.long, left + right)
    return max(left, right) + 1

"""
191. Number of 1 Bits
"""
def hammingWeight(self, n):
    """
    :type n: int
    :rtype: int
    """
    count = 0
    for _ in range(32):
        count += n & 1
        n >>= 1
    return count

"""
14. Longest Common Prefix
"""
def longestCommonPrefix(self, strs):
    """
    :type strs: List[str]
    :rtype: str
    """
    if not strs:
        return ""
    minStr = min(strs, key = len)
    for i, c in enumerate(minStr):
        for str in strs:
            if str[i] != c:
                return minStr[:i]
    return minStr

"""
71. Simplify Path
"""
def simplifyPath(self, path):
    """
    :type path: str
    :rtype: str
    """
    paths = [p for p in path.split('/') if p != '.' and p!= '']  # split('/') => /home/ => ['','home',''] so we have to ignore the ''
    stack = []
    for p in paths:
        if p == '..':
            if stack:
                stack.pop()
        else:
            stack.append(p)
    return '/' + '/'.join(stack)

"""
64. Minimum Path Sum
"""
def minPathSum(self, grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if i == 0 and j == 0:
                continue
            if i == 0:
                grid[i][j] += grid[i][j-1]
            elif j == 0:
                grid[i][j] += grid[i-1][j]
            else:
                grid[i][j] += min(grid[i-1][j],grid[i][j-1])
    return grid[-1][-1]

"""
63. Unique Paths II
"""
def uniquePathsWithObstacles(self, obstacleGrid):
    """
    :type obstacleGrid: List[List[int]]
    :rtype: int
    """
    dp = [0] * len(obstacleGrid[0])
    dp[0] = 1
    for i in range(len(obstacleGrid)):
        for j in range(len(obstacleGrid[0])):
            if obstacleGrid[i][j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j-1]
    return dp[-1]

    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])

        # If the starting cell has an obstacle, then simply return as there would be
        # no paths to the destination.
        if obstacleGrid[0][0] == 1:
            return 0

        # Number of ways of reaching the starting cell = 1.
        obstacleGrid[0][0] = 1

        # Filling the values for the first column
        for i in range(1,m):
            obstacleGrid[i][0] = int(obstacleGrid[i][0] == 0 and obstacleGrid[i-1][0] == 1)

        # Filling the values for the first row
        for j in range(1, n):
            obstacleGrid[0][j] = int(obstacleGrid[0][j] == 0 and obstacleGrid[0][j-1] == 1)

        # Starting from cell(1,1) fill up the values
        # No. of ways of reaching cell[i][j] = cell[i - 1][j] + cell[i][j - 1]
        # i.e. From above and left.
        for i in range(1,m):
            for j in range(1,n):
                if obstacleGrid[i][j] == 0:
                    obstacleGrid[i][j] = obstacleGrid[i-1][j] + obstacleGrid[i][j-1]
                else:
                    obstacleGrid[i][j] = 0

        # Return value stored in rightmost bottommost cell. That is the destination.
        return obstacleGrid[m-1][n-1]
"""
62. Unique Paths
Draw the pic you will know how to decrease the dimension
"""
def uniquePaths(self, m, n):
    """
    :type m: int
    :type n: int
    :rtype: int
    """
    dp = [0] * n
    dp[0] = 1
    for i in range(m):
        for j in range(1,n):
            dp[j] += dp[j-1]
    return dp[-1]

"""
59. Spiral Matrix II
"""
def generateMatrix(self, n: 'int') -> 'List[List[int]]':
    matrix = [[0] * n for _ in range(n)]
    if not n:
        return matrix
    num = 1
    rowStart, rowEnd, colStart, colEnd = 0, n - 1, 0, n - 1
    while num <= n * n:
        for i in range(colStart, colEnd + 1):
            matrix[rowStart][i] = num
            num += 1
        rowStart += 1

        for i in range(rowStart, rowEnd + 1):
            matrix[i][colEnd] = num
            num += 1
        colEnd -= 1

        if rowStart <= rowEnd:
            for i in range(colEnd, colStart - 1, -1):
                matrix[rowEnd][i] = num
                num += 1
        rowEnd -= 1

        if colStart <= colEnd:
            for i in range(rowEnd, rowStart - 1, -1):
                matrix[i][colStart] = num
                num += 1
        colStart += 1
    return matrix

"""
937. Reorder Log Files
"""

def reorderLogFiles(self, logs):
    """
    :type logs: List[str]
    :rtype: List[str]
    """
    nums, letters = [], []
    for log in logs:
        logsplit = log.split()
        if logsplit[1].isalpha():
            letters.append((" ".join(logsplit[1:]), logsplit[0]))
        else:
            nums.append(log)
    letters.sort()
    return [letter[1] + ' ' + letter[0] for letter in letters] + nums


"""
140. Word Break II
fibonacci 可以解釋這件事
"""
def wordBreak(self, s, wordDict):
    """
    :type s: str
    :type wordDict: List[str]
    :rtype: List[str]
    """
    return self.wordBreakDfs(s, wordDict, {})


def wordBreakDfs(self, s, wordDict, memo):
    if s in memo:
        return memo[s]
    if not s:
        return []
    result = []
    for word in wordDict:
        if not s.startswith(word):
            continue
        if len(word) == len(s):
            result.append(word)
        else:
            resultOfTheRest = self.wordBreakDfs(s[len(word):],wordDict, memo)
            for item in resultOfTheRest:
                tempStr  = word + ' ' + item
                result.append(tempStr)
    memo[s] = result
    return result

"""
84. Largest Rectangle in Histogram
"""
def largestRectangleArea(self, heights):
    """
    :type heights: List[int]
    :rtype: int
    """
    heights.append(0)
    stack = [-1]
    ans = 0
    for i in range(len(heights)):
        while heights[i] < heights[stack[-1]]: #calculate the area which current high lower than previous
            h = heights[stack.pop()]
            w = i - stack[-1] - 1
            ans = max(ans, h * w)
        stack.append(i)
    return ans


"""
852. Peak Index in a Mountain Array
"""
def peakIndexInMountainArray(self, A):
    """
    :type A: List[int]
    :rtype: int
    """
    left, right = 0, len(A) - 1
    while left < right:
        mid = (left + right)//2
        if A[mid] < A[mid -1]:
            right = mid
        elif A[mid] < A[mid + 1]:
            left = mid 
        else:
            return mid

"""
174. Dungeon Game
"""
def calculateMinimumHP(self, dungeon):
    """
    :type dungeon: List[List[int]]
    :rtype: int
    """
    dp = [[float('inf')] * (len(dungeon[0]) + 1) for _ in range(len(dungeon) + 1)] # I would like to reiterate.
    dp[-1][-2] = dp[-2][-1] = 1
    for i in range(len(dungeon) - 1, -1, -1):
        for j in range(len(dungeon[0]) - 1, -1, -1):
            dp[i][j] = max(1, min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j])
    return dp[0][0]


"""
124. Binary Tree Maximum Path Sum
"""
def maxPathSum(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    self.res = -float('inf')
    self.maxPathSumDFS(root)
    return self.res

def maxPathSumDFS(self, node):
    if not node:
        return 0
    left = self.maxPathSumDFS(node.left)
    right = self.maxPathSumDFS(node.right)
    self.res = max(self.res, node.val + left + right)  # the node decide to connect left or right
    cur = max(left, right) + node.val # the side (left or right side)
    return cur if cur > 0 else 0

"""
819. Most Common Word
"""
def mostCommonWord(self, paragraph: 'str', banned: 'List[str]') -> 'str':
    ban = set(banned)
    words = re.findall('\w+', paragraph.lower())
    counter = collections.Counter(w for w in words if w not in banned)
    return counter.most_common()[0][0]


"""
675. Cut Off Trees for Golf Event
"""
def cutOffTree(self, forest):
    """
    :type forest: List[List[int]]
    :rtype: int
    """
    m, n = len(forest), len(forest[0])
    trees = []
    for i in range(m):
        for j in range(n):
            if forest[i][j] != 0:
                trees.append((forest[i][j], i, j))
    trees.sort()
    count = 0
    sx, sy = 0, 0
    for _, x, y in trees:
        step = self.cutOffTreeBFS(forest, sx, sy, x, y)
        if step == -1:
            return -1
        else:
            count += step
            forest[x][y] = 1
            sx, sy = x, y
    return count


def cutOffTreeBFS(self, forest, sx, sy, tx, ty):
    que = collections.deque([(sx, sy)])
    visited = [[False] * len(forest[0]) for _ in range(len(forest))]
    step = -1
    while que:
        size = len(que)
        step += 1
        for _ in range(size):
            x, y = que.popleft()
            visited[x][y] = True
            if x == tx and y == ty:
                return step
            for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if 0 <= nx < len(forest) and 0 <= ny < len(forest[0]) and forest[nx][ny] != 0 and not visited[nx][ny]:
                    que.append((nx, ny))
    return -1

"""
243. Shortest Word Distance
"""
def shortestDistance(self, words, word1, word2):
    """
    :type words: List[str]
    :type word1: str
    :type word2: str
    :rtype: int
    """
    w1 = w2 = -1
    minDis = len(words)
    for i,v in enumerate(words):
        if v == word1:
            w1 = i
        elif v == word2:
            w2 = i
            
        if w1 != -1 and w2 != -1:
            minDis = min(abs(w1 - w2), minDis)
    return minDis
            

"""
682. Baseball Game
"""
def calPoints(self, ops):
    """
    :type ops: List[str]
    :rtype: int
    """
    stack = []
    for val in ops:
        if val == 'C' and stack:
            stack.pop()
        elif val == 'D':
            stack.append(stack[-1] * 2)
        elif val == '+':
            stack.append(stack[-1] + stack[-2])
        else:
            stack.append(int(val))
    return sum(stack)


"""
48. Rotate Image
"""


def rotate(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: void Do not return anything, modify matrix in-place instead.
    """
    if not matrix:
        return
    self.m, self.n = len(matrix), len(matrix[0])
    for i in range(self.m // 2):
        for j in range(self.n):
            matrix[i][j], matrix[self.m - i - 1][j] = matrix[self.m - i - 1][j], matrix[i][j]

    for i in range(self.m):
        for j in range(self.n):
            if j > i:
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


"""
73. Set Matrix Zeroes
"""
def setZeroes(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: void Do not return anything, modify matrix in-place instead.
    """
    isRowAffect = isColAffect = False
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            if matrix[row][col] == 0:
                if row == 0: 
                    isRowAffect = True
                if col == 0:
                    isColAffect = True
                matrix[0][col] = 0
                matrix[row][0] = 0
                
    for i in range(1, len(matrix)):
        for j in range(1, len(matrix[0])):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    
    if isRowAffect:
        for i in range(len(matrix[0])):
            matrix[0][i] = 0
            
    if isColAffect:
        for i in range(len(matrix)):
            matrix[i][0] = 0

"""
119. Pascal's Triangle II
"""
def getRow(self, rowIndex):
    """
    :type rowIndex: int
    :rtype: List[int]
    """
    result = [0] * (rowIndex+1)
    result[0] = 1
    for i in range(rowIndex):
        for j in range(i+1, 0, -1):  # i+1, it can make sure that the I add previous number
            result[j] = result[j] + result[j-1] 
    return result


"""
alien Dictionary
"""
def alienOrder(self, words: 'List[str]') -> 'str':
    graph = {}
    for word in words:
        for c in word:
            key = ord(c) - ord('a')
            graph[key] = set()
    indegree = [0] * 26
    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]
        minLength = min(len(word1), len(word2))
        for cIdx in range(minLength):
            if word1[cIdx] != word2[cIdx]:
                word1Idx, word2Idx = ord(word1[cIdx]) - ord('a'), ord(word2[cIdx]) - ord('a')
                if word2Idx not in graph[word1Idx]:
                    indegree[word2Idx] += 1
                    graph[word1Idx].add(word2Idx)
                break
    que = collections.deque()
    res = ''

    for i in range(len(indegree)):
        if indegree[i] == 0 and i in graph:
            que.append(i)
    while que:
        nextup = que.popleft()
        res += chr((nextup) + ord('a'))
        for neighbor in graph[nextup]:
            indegree[neighbor] -= 1
            if (indegree[neighbor] == 0):
                que.append(neighbor)
    return res if len(graph) == len(res) else ''


"""
8. String to Integer (atoi)
"""
class Solution:
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        ls = list(str.strip())
        if not ls:
            return 0
        sign = -1 if ls[0] == '-' else 1
        if ls[0] in ['+','-']:
            del ls[0]
        res, i = 0, 0
        while i < len(ls) and ls[i].isdigit():
            res = res * 10 + ord(ls[i]) - ord('0')
            i += 1
        return max(-2**31, min(2**31 - 1, sign * res))

"""
78. Subsets
Time: 	O(2 ^ n) 
Space:  O(n)

https://docs.google.com/document/d/1nX7fjp3t3umBhwLt1jaXctpcFrjyaT135gz1DpZQeSw/edit
"""
def subsets(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    result = []
    self.subsetHelper(nums, result, [], 0)
    return result
    
def subsetHelper(self, nums, result, tempStr, startIdx):
    result.append(list(tempStr))
    
    for i in range(startIdx, len(nums)):
        self.subsetHelper(nums, result, tempStr + [nums[i]], i + 1)



def groupAnagrams(self, strs: 'List[str]') -> 'List[List[str]]':
    """
    :type strs: List[str]
    :rtype: List[List[str]]
    TO(nmlogm) => n is num of words, m is leng of each word
    you can use counting sort to reduce
    """
    dic = collections.defaultdict(list)
    for str in strs:
        tempStr = ''.join(sorted(str))
        dic[tempStr].append(str)
    return list(dic.values())


"""
Word Break
"""
def wordBreak(self, s, wordDict):
    """
    :type s: str
    :type wordDict: List[str]
    :rtype: bool
    """
    dp = [False] * (len(s) + 1)
    dp[0] = True
    wordSet = set(wordDict)
    for i in range(1, len(dp)):
        for j in range(i):
            if dp[j] and s[j:i] in wordSet:
                dp[i] = True
                break
    return dp[len(dp) - 1]


def numJewelsInStones(self, J, S):
    """
    :type J: str
    :type S: str
    :rtype: int
    """
    jSet = set(J)
    return math.sum(s in jSet for s in S)
        

"""
Reverse Linked List
"""
def reverseList(self, head: ListNode) -> ListNode:
    prev = None
    while head:
        temp = head.next
        head.next = prev
        prev = head
        head = temp
    return prev





"""
Copy List with Random Pointer
"""
def copyRandomList(self, head):
    """
    :type head: RandomListNode
    :rtype: RandomListNode
    """
    dic= {}
    dic[None] = None
    curr = head
    while curr:
        dic[curr] = RandomListNode(curr.label)
        curr = curr.next
    curr =head
    while curr:
        dic[curr].next = dic[curr.next]
        dic[curr].random = dic[curr.random]
        curr = curr.next
    return dic[head]
            

"""
Longest Substring Without Repeating Characters
"""
def lengthOfLongestSubstring(self, s: 'str') -> 'int':
    dic = collections.defaultdict(int)
    startIdx, longest = 0, 0
    for i, c in enumerate(s):
        if c in dic:
            startIdx = max(startIdx, dic[c] + 1)  # if you don't use max() => 'abba' will be wrong. the last of the 'a' prevent it go back
        dic[c] = i
        longest = max(longest, i - startIdx + 1)
    return longest

"""
Maximum Subarray
"""
def maxSubArray(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums: return 0
    
    maxSum, sumSoFar = nums[0], nums[0]
    for i in range(1, len(nums)):
        sumSoFar = max(sumSoFar + nums[i], nums[i])
        maxSum = max(sumSoFar, maxSum)
    return maxSum

"""
Best Time to Buy and Sell Stock
"""
def maxProfit(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    if not prices: return 0
    maxBenefit, buyPrice = 0, prices[0]
    for v in prices:
        if v < buyPrice:
            buyPrice = v
        maxBenefit = max(maxBenefit, v - buyPrice)
    return maxBenefit

"""
Max Heap vs. Min Heap
哪个算法更加好？

Max: Time: O(n + klog(n)) | Space: O(n)
Min: Time: O(k) + O((n-k) * logk) | Space: O(K)

如果考虑k无限接近n
Max: O(n + nlog(n)) ~= O(nlogn)
Min: O(n + logk) ~= O(n)

如果考虑k = 0.5n
Max: O(n + nlogn)
Min: O(n + nlogn)

如果考虑n 无限大
Max: O(constant * n) 为什么是constant * n，参考
Min: O(log(k) * n)

Kth Largest Element in an Array
Max Heap
Time: O(n + klog(n)) | Space: O(n)   log(1) = 0 ; log(2) 0.30
"""
def findKthLargest(self, nums, k):
    nums = [-num for num in nums]
    heapq.heapify(nums)
    res = float('inf')
    for _ in range(k):
        res = heapq.heappop(nums)
    return -res



    def findKthLargest(self, nums, k):
        min_heap = nums[:k]
        heapq.heapify(min_heap)
        for i in range(k, len(nums)):
            if nums[i] > min_heap[0]:
                heapq.heappop(min_heap)
                heapq.heappush(min_heap, nums[i])
        return min_heap[0]

"""
Kth Largest Element in an Array
Min Heap
Time: O(k) + O(n * logk) | Space: O(K)
"""
def findKthLargest(self, nums, k):
    min_heap = [float('-inf')] * k
    for num in nums:
        heapq.heappushpop(min_heap, num)
    return min_heap[0]

"""
Combination Sum
"""
def combinationSum(self, candidates: 'List[int]', target: 'int') -> 'List[List[int]]':
    result = []
    self.combinationSumDfs(result, candidates, target, [], 0)
    return result


def combinationSumDfs(self, result, candidates, target, tempList, idx):
    if target == 0:
        result.append(tempList)
        return
    elif target < 0:
        return
    for i in range(idx, len(candidates)):
        self.combinationSumDfs(result, candidates, target - candidates[i], tempList + [candidates[i]], i)


"""

"""
def letterCombinations(self, digits):
    """
    :type digits: str
    :rtype: List[str]
    """
    if not digits: return []
    self.digitArray = ['','','abc','def','ghi','jkl','mno','pqrs','tuv','wxyz']
    self.result = []
    self.letterHelper('', 0, digits)
    return self.result

def letterHelper(self, tempString, pos, digits):
    if len(tempString) == len(digits):
        self.result.append(tempString)
        return
    num = int(digits[pos])
    for c in self.digitArray[num]:
        tempString += c
        self.letterHelper(tempString, pos + 1, digits)
        tempString = tempString[:-1]


"""
Graph Valid Tree
"""
def validTree(self, n, edges):
    """
    :type n: int
    :type edges: List[List[int]]
    :rtype: bool
    """
    if len(edges) != n - 1:
        return False
    graph = collections.defaultdict(list)
    for x, y in edges:
        graph[x].append(y)
        graph[y].append(x)
    visited = set()
    que = collections.deque([0])
    while que:
        node = que.popleft()
        visited.add(node)
        for v in graph[node]:
            if v not in visited:
                visited.add(v)
                que.append(v)
                
    return len(visited) == n

"""
261 Graph Valid Tree (union find version)
"""
def validTreeUnionFind(self, n, edges):
    """
    :type n: int
    :type edges: List[List[int]]
    :rtype: bool
    """
    parent = list(range(n))
    def find(x):
        return x if x == parent[x] else find(parent[x])
    def union(xy):
        x, y = map(find, xy)
        parent[x] = y
        return x != y
    return len(edges) == n - 1 and all(map(union, edges))


"""
Number of Islands
"""
def numIslands(self, grid):
    """
    :type grid: List[List[str]]
    :rtype: int
    """
    if not grid:
        return 0
    island = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                island += 1
                self.islandDfs(i, j, grid)
    return island


def islandDfs(self, i, j, grid):
    if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == '1':
        grid[i][j] = '0'
        for x, y in [(i+1, j),(i-1,j),(i,j+1),(i,j-1)]:
            self.islandDfs(x,y,grid)

"""
695. Max Area of Island
"""
def maxAreaOfIsland(self, grid: 'List[List[int]]') -> 'int':
    maxArea = 0
    self.m, self.n = len(grid), len(grid[0])
    self.grid = grid
    for i in range(self.m):
        for j in range(self.n):
            if self.grid[i][j] == 1:
                maxArea = max(maxArea, self.areaDfs(i, j))
    return maxArea


def areaDfs(self, x, y):
    area = 1
    self.grid[x][y] = 0
    for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
        if 0 <= nx < self.m and 0 <= ny < self.n and self.grid[nx][ny] == 1:
            area += self.areaDfs(nx, ny)
    return area


"""
Inorder Successor in BST
"""
def inorderSuccessor(self, root, p):
    """
    :type root: TreeNode
    :type p: TreeNode
    :rtype: TreeNode
    """
    if not root: return None
    if root.val <= p.val:
        return self.inorderSuccessor(root.right, p)
    else:
        left = self.inorderSuccessor(root.left, p)
        return left if left else root

"""
Binary Tree Vertical Order Traversal
"""
def verticalOrder(self, root):
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    import collections
    if not root: return []
    
    cols = collections.defaultdict(list)
    queue = collections.deque([(root,0)])
    
    while queue:
        node, col = queue.popleft()
        cols[col].append(node.val)
        if node.left:
            queue.append((node.left, col - 1))
        if node.right:
            queue.append((node.right, col + 1))
    return [cols[i] for i in sorted(cols)]
            


"""
Binary Tree Level Order Traversal
"""
def levelOrder(self, root):
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    from collections import deque
    
    if not root: return []
    result = []
    que = deque([root])
    while que:
        lvl = len(que)
        tempLvl = []
        for _ in range(lvl):
            tempNode = que.popleft()
            tempLvl.append(tempNode.val)
            if tempNode.left:
                que.append(tempNode.left)
            if tempNode.right:
                que.append(tempNode.right)
        result.append(tempLvl)
    return result

"""
Subtree of Another Tree
"""
def isSubtree(self, s, t):
    """
    :type s: TreeNode
    :type t: TreeNode
    :rtype: bool
    """
    if not s: return False
    if self.isSame(s, t): return True
    return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)

def isSame(self, s, t):
    if not s and not t: return True
    if not s or not t: return False
    if s.val != t.val: return False
    return self.isSame(s.left, t.left) and self.isSame(s.right,t.right)


"""
Two Sum IV - Input is a BST
"""
def findTarget(self, root, k):
    """
    :type root: TreeNode
    :type k: int
    :rtype: bool
    """
    wordSet = set()
    return self.findTargetHelper(wordSet , root, k)

def findTargetHelper(self, wordSet, node, k):
    if not node: return False
    
    target = k - node.val
    
    if target in wordSet:
        return True
    wordSet.add(node.val)
    return self.findTargetHelper(wordSet, node.left,k) or self.findTargetHelper(wordSet, node.right,k)

"""
  Maximum Binary Tree
"""
def constructMaximumBinaryTree(self, nums):
    """
    :type nums: List[int]
    :rtype: TreeNode
    """
    return self.constructMaximumBinaryTreeHelper(nums, 0, len(nums) -1)

def constructMaximumBinaryTreeHelper(self, nums, start, end):
    if start > end:
        return None
    maxIdx = start
    for i in range(start+1, end+1):
        if nums[i] > nums[maxIdx]:
            maxIdx = i
    root = TreeNode(nums[maxIdx])
    root.left = self.constructMaximumBinaryTreeHelper(nums, start, maxIdx-1)
    root.right = self.constructMaximumBinaryTreeHelper(nums, maxIdx+1, end)
    return root

"""
Sum Root to Leaf Numbers
"""
def sumNumbers(self, root: TreeNode) -> int:
    return self.sumNumbersHelper(root, 0)

def sumNumbersHelper(self, node, res):
    if not node:
        return 0
    left = self.sumNumbersHelper(node.left, res * 10 + node.val)
    right = self.sumNumbersHelper(node.right, res * 10 + node.val)
    if left or right:
        return left + right
    return res * 10 + node.val

"""
Closest Binary Search Tree Value
"""
def closestValue(self, root, target):
    """
    :type root: TreeNode
    :type target: float
    :rtype: int
    """
    result = root.val
    while root:
        if abs(result - target) > abs(root.val - target):
            result = root.val
        elif root.val >= target:
            root = root.left
        else:
            root = root.right
    return result


"""
Symmetric Tree
"""
def isSymmetric(self, root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    return not root or self.isSymmetricHelper(root.left, root.right)

def isSymmetricHelper(self, left, right):
    if not left or not right:
        return left == right
    
    if left.val != right.val:
        return False
    return self.isSymmetricHelper(left.left, right.right) and self.isSymmetricHelper(left.right,right.left)

"""
Validate Binary Search Tree
"""
def isValidBST(self, root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    return self.isValidHelper(root, None, None)


def isValidHelper(self, node, maxNode, minNode):
    if not node: 
        return True
    if maxNode and node.val >= maxNode.val: return False
    if minNode and node.val <= minNode.val: return False
    return self.isValidHelper(node.left, node, minNode) and self.isValidHelper(node.right, maxNode, node)


"""
Merge k Sorted Lists
"""
def mergeKLists(self, lists):
    """
    :type lists: List[ListNode]
    :rtype: ListNode
    """
    return self.mergeList(lists, 0, len(lists)-1)

def mergeList(self, lists, low, high):
    if low > high: return None
    if low == high: return lists[low]
    mid = low + (high - low)//2
    leftList = self.mergeList(lists, low, mid)
    rightList = self.mergeList(lists, mid+1, high)
    return self.merge(leftList, rightList)

def merge(self, leftList, rightList):
    result = ListNode(0)
    curr = result
    while leftList and rightList:
        if leftList.val > rightList.val:
            curr.next = ListNode(rightList.val)
            curr = curr.next
            rightList = rightList.next
        else:
            curr.next = ListNode(leftList.val)
            curr = curr.next
            leftList = leftList.next
    if rightList:
        curr.next = rightList
    if leftList:
        curr.next = leftList
    return result.next

"""
 Intersection of Two Linked Lists
"""
def getIntersectionNode(self, headA, headB):
    curA,curB = headA,headB
    lenA,lenB = 0,0
    while curA is not None:
        lenA += 1
        curA = curA.next
    while curB is not None:
        lenB += 1
        curB = curB.next
    curA,curB = headA,headB
    if lenA > lenB:
        for i in range(lenA-lenB):
            curA = curA.next
    elif lenB > lenA:
        for i in range(lenB-lenA):
            curB = curB.next
    while curB != curA:
        curB = curB.next
        curA = curA.next
    return curA

"""

"""
def minMeetingRooms(self, intervals):
    """
    :type intervals: List[Interval]
    :rtype: int
    """
    startArray, endArray = [], []
    for v in intervals:
        startArray.append(v.start)
        endArray.append(v.end)
        
    startArray.sort()
    endArray.sort()
    
    room = endIdx = 0
    for i in range(len(intervals)):
        if startArray[i] < endArray[endIdx]:
            room += 1
        else:
            endIdx += 1
    return room

"""
 Intersection of Two Linked Lists  (TLE)
"""
def getIntersectionNodeTLE(self, headA, headB):
    """
    :type head1, head1: ListNode
    :rtype: ListNode
    """
    if not headA or not headB: return None
    currA = headA
    currB = headB
    while currA != currB:
        currA = currA.next if currA else headA
        currB = currB.next if currB else headB
    return currA


"""
Add Two Numbers
"""
def addTwoNumbers(self, l1, l2):
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """
    carry = total = 0
    result = ListNode(0)
    curr = result
    while l1 or l2 or carry != 0:
        l1_val = l2_val = 0
        
        if l1:
            l1_val = l1.val
            l1 = l1.next
        if l2:
            l2_val = l2.val
            l2 = l2.next
        total = l1_val + l2_val + carry
        carry = total // 10
        total = total % 10
        curr.next = ListNode(total)
        curr = curr.next
    return result.next

"""
Compare Version Numbers
"""
def compareVersion(self, version1, version2):
    """
    :type version1: str
    :type version2: str
    :rtype: int
    """
    v1 = [int(v) for v in version1.split('.')]
    v2 = [int(v) for v in version2.split('.')]
    
    for i in range(max(len(v1),len(v2))):
        v1_int = v1[i] if i < len(v1) else 0
        v2_int = v2[i] if i < len(v2) else 0
        if v1_int > v2_int:
            return 1
        elif v1_int < v2_int:
            return -1
    return 0

"""
Longest Palindromic Substring
"""
def longestPalindrome(s):
    """
    :type s: str
    :rtype: str
    """
    max_length = start_index = 0
    dp = [[False] * (len(s)) for _ in range(len(s))]
    for i in range(len(s)):
        for j in range(i+1):
            if s[i] == s[j]:
                leng = i - j + 1
                if leng < 3 or dp[i-1][j+1]:  # leng small than 3 or previous string is True
                    dp[i][j] = True
                    if leng > max_length:
                        max_length = leng
                        start_index = j
    return s[start_index: start_index+max_length]

longestPalindrome('babad')

"""
Product of Array Except Self
left 等於＊
right **
"""
def productExceptSelf(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    left, right = 1,1
    result = [0] * len(nums)
    for i in range(len(nums)):
        result[i] = left
        left *= nums[i]
    for i in range(len(nums)-1,-1,-1):
        result[i] *= right
        right *= nums[i]
    return result

"""
Maximum Size Subarray Sum Equals k
"""
def maxSubArrayLen(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    maxLength, currNum = 0, 0
    dic = {0:-1}
    for i, v in enumerate(nums):
        currNum += v
        if currNum - k in dic:
            maxLength = max(maxLength, i - dic[currNum - k])
        if currNum not in dic:
            dic[currNum] = i
    return maxLength

"""
560. Subarray Sum Equals K
"""
def subarraySum(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    dic = collections.defaultdict(int)
    dic[0] = 1
    total = count = 0
    for n in nums:
        total += n
        count += dic[total - k]
        dic[total] = dic[total] + 1
    return count


"""
First Unique Character in a String
"""
def firstUniqChar(self, s):
    """
    :type s: str
    :rtype: int
    """
    if not s: return -1
    from collections import Counter
    dic = dict(Counter(s))
    
    for idx, c in enumerate(s):
        if dic[c] == 1:
            return idx
    return -1

"""
Reverse Words in a String II
"""
def reverseWords(self, str):
    """
    :type str: List[str]
    :rtype: void Do not return anything, modify str in-place instead.
    """
    str.reverse()
    
    start_idx = 0
    for i in range(len(str)):
        if str[i] == ' ':
            str[start_idx:i] = reversed(str[start_idx:i])
            start_idx = i + 1
    str[start_idx:] = reversed(str[start_idx:])



"""
Two Sum
"""
def twoSum(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    dic = {}
    for i in range(len(nums)):
        if (target - nums[i]) in dic:
            return [dic[target-nums[i]],i]
        else:
            dic[nums[i]] = i
    return [-1,-1]


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None



        
from heapq import *
class MedianFinder:  # put small to the small and put it as negative.
                     # It demonstrates the bigger one in samll heap will be smallerest one.
                     # maintain odd situtation in large heap
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.small = []
        self.large = []
        

    def addNum(self, num):
        """
        :type num: int
        :rtype: void
        """
        if len(self.small) == len(self.large):
            heappush(self.large, -heappushpop(self.small, -num))
        else:
            heappush(self.small, -heappushpop(self.large, num))

    def findMedian(self):
        """
        :rtype: float
        """
        if len(self.small) == len(self.large):
            return (self.large[0] - self.small[0]) / 2.0
        else:
            return self.large[0]
        
"""
103. Binary Tree Zigzag Level Order Traversal
"""
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        result = []
        if not root:
            return result
        que = collections.deque()
        que.append(root)
        isToRight = True 
        while que:
            count = len(que)
            tempList = []
            for i in range(count):
                tempNode = que.popleft()
                if tempNode.left:
                    que.append(tempNode.left)
                if tempNode.right:
                    que.append(tempNode.right)
                tempList.append(tempNode.val)
            if not isToRight:
                result.append(tempList[::-1])
            else:
                result.append(list(tempList))
            isToRight = not isToRight
        return result
"""
93. Restore IP Addresses
count is partition not the dot
"""
def restoreIpAddresses(self, s):
    """
    :type s: str
    :rtype: List[str]
    """
    result = []
    self.restoreIpHelper(s, result, '',0)
    return result

def restoreIpHelper(self, s, result, tempStr, count ):
    if count > 4:
        return
    if count == 4 and not s: # count == 4 prevent 1111=> 111.1. corner case
        result.append(tempStr)
        return
    for i in range(1, 4):
        if i > len(s):  #prevent add multiple result
            return
        tempS = s[:i]
        if (tempS.startswith('0') and len(tempS) > 1) or int(tempS) >= 256:
            return
        self.restoreIpHelper(s[i:],result, tempStr + tempS + ('' if count  == 3 else '.'), count+1)

    
"""
240. Search a 2D Matrix II
"""
def searchMatrix(self, matrix, target):
    """
    :type matrix: List[List[int]]
    :type target: int
    :rtype: bool
    """
    if not matrix:
        return False
    m, n = len(matrix[0]) - 1, len(matrix) - 1
    row, col = 0, m
    while row <= n and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1
        elif matrix[row][col] < target:
            row += 1
    return False

"""
239. Sliding Window Maximum
"""
def maxSlidingWindow(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: List[int]
    """
    que = collections.deque()
    result = []
    for i, v in enumerate(nums):
        while que and nums[que[-1]] < v:
            que.pop()
        que.append(i)
        if que[0] == i - k: # know the biggest number valid or not
            que.popleft()
        if i >= k - 1:  #start to add num 1 is '0' index. so we have to - it
            result.append(nums[que[0]])
    return result


# 45. Jump Game II
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        jump = curFar = curFarest = 0
        for i in range(len(nums) - 1):
            curFarest = max(curFarest, i + nums[i])
            if curFar == i:       #I cannot jump without stop to adding jump
                jump += 1
                curFar = curFarest
        return jump


"""
380. Insert Delete GetRandom O(1)
"""
class RandomizedSet:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.set = set()
        self.array = []
        

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.set:
            return False
        else:
            self.set.add(val)
            self.array.append(val)
            return True
        

    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.set:
            self.set.remove(val)
            self.array.remove(val)
            return True
        else:
            return False

    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        idx = random.randint(0,len(self.array)-1)
        return self.array[idx]

#381 Insert Delete GetRandom O(1) - Duplicates allowed
class RandomizedCollection:
    import collections
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.dic = collections.defaultdict(set)
        self.array = []

    def insert(self, val):
        """
        Inserts a value to the collection. Returns true if the collection did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        self.array.append(val)
        self.dic[val].add(len(self.array) - 1)
        return len(self.dic[val]) == 1

    def remove(self, val):
        """
        Removes a value from the collection. Returns true if the collection contained the specified element.
        :type val: int
        :rtype: bool
        """
        if self.dic[val]:  # difference between val in self.dic =>   self.dic[val] => set() only will not be true; However, val in dic will always be true
            removeIdx, lastVal = self.dic[val].pop(), self.array[-1]
            self.array[removeIdx] = lastVal
            if self.dic[lastVal]: # is not the last number. There is a potential that lastVal == array[removeIdx] so it will be empty
                self.dic[lastVal].add(removeIdx)
                self.dic[lastVal].remove(len(self.array) - 1)
            self.array.pop()
            return True
        return False

    def getRandom(self):
        """
        Get a random element from the collection.
        :rtype: int
        """
        return self.array[random.randint(0, len(self.array) - 1)]


class Solution:
    def __init__(self):
        self.less20 = ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten","Eleven","Twelve","Thirteen","Fourteen","Fifteen","Sixteen","Seventeen","Eighteen","Nineteen"]
        self.tens = ["","Ten","Twenty","Thirty","Forty","Fifty","Sixty","Seventy","Eighty","Ninety"]
        self.thousands = ["","Thousand","Million","Billion"]
    
    
    def numberToWords(self, num):
        """
        :type num: int
        :rtype: str
        """
        if num == 0:
            return self.less20[0]
        count = 0
        words = ''
        while num > 0:
            if num % 1000 != 0:
                words = self.helper(num%1000) + self.thousands[count] + ' ' + words
            num //= 1000
            count += 1
        return words.strip()
    
    def helper(self, num):
        if num == 0:
            return ''
        elif num < 20:
            return self.less20[num] + ' '
        elif num < 100:
            return self.tens[num//10] + ' ' + self.helper(num%10)
        else:
            return self.less20[num//100] + ' Hundred ' + self.helper(num%100)
        
class LFUCache:
    def __init__(self, capacity):
        self.remain = capacity
        self.nodesForFrequency = collections.defaultdict(collections.OrderedDict) #FreqLists
        self.leastFrequency = 1
        self.nodeForKey = {}  # HashTable to control the nodes

    def _update(self, key, newValue=None):
        value, freq = self.nodeForKey[key]
        if newValue is not None: value = newValue
        self.nodesForFrequency[freq].pop(key)
        if len(self.nodesForFrequency[self.leastFrequency]) == 0: 
            self.leastFrequency += 1
        self.nodesForFrequency[freq+1][key] = (value, freq+1)
        self.nodeForKey[key] = (value, freq+1)

    def get(self, key):
        if key not in self.nodeForKey: return -1
        self._update(key)
        return self.nodeForKey[key][0]

    def put(self, key, value):
        if key in self.nodeForKey: 
            self._update(key, value)
        else:
            self.nodeForKey[key] = (value,1)
            self.nodesForFrequency[1][key] = (value,1)
            if self.remain == 0:
                removed = self.nodesForFrequency[self.leastFrequency].popitem(last=False)
                self.nodeForKey.pop(removed[0])
            else: self.remain -= 1
            self.leastFrequency = 1 # should be one after adding a new item


"""
703. Kth Largest Element in a Stream
TC: n log k
"""
class KthLargest:
    def __init__(self, k, nums):
        """
        :type k: int
        :type nums: List[int]
        """
        self.heap = [float('-inf')] * k
        for n in nums:
            heapq.heappushpop(self.heap, n)

    def add(self, val):
        """
        :type val: int
        :rtype: int
        """
        heapq.heappushpop(self.heap, val)
        return self.heap[0]


class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        def preOrder(node):
            if node:
                result.append(str(node.val))
                preOrder(node.left)
                preOrder(node.right)
        result = []
        preOrder(root)
        return ' '.join(result)
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        preOrder= map(int, data.split())
        inOrder = sorted(preOrder)
        return self.buildTree(preOrder, inOrder)
    
    def buildTree(self, preOrder, inOrder):
        if inOrder:
            index = inOrder.index(preOrder.pop(0))
            root = TreeNode(inOrder[index])
            root.left = self.buildTree(preOrder, inOrder[:index])
            root.right = self.buildTree(preOrder, inOrder[index+1:])
            return root

"""
minHeap: O(nlogk)
maxHeap: o(nlogn)

692. Top K Frequent Words
"""
class Solution(object):
    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :type k: int
        :rtype: List[str]
        """
        heap = []
        counter = collections.Counter(words)
        for key, val in counter.items():
            heapq.heappush(heap, Element(val,key))
            if len(heap) > k:
                heapq.heappop(heap)     
        result = []
        for _ in range(len(heap)):
            result.append(heapq.heappop(heap).word)
        return result[::-1]


class Element: #heap > [. ....... (high freq, lower alphabetical]  use min heap so min will be pop out
    def __init__(self, freq, word):
        self.freq = freq
        self.word = word
    
    def __lt__(self, other):
        if self.freq == other.freq:
            return self.word > other.word  # lexicographic order and use minheap
        return self.freq < other.freq

"""
348. Design Tic-Tac-Toe
"""
class TicTacToe:

    def __init__(self, n):
        """
        Initialize your data structure here.
        :type n: int
        """
        self.row, self.col, self.diag, self.anti_diag, self.n = [0] * n, [0] * n, 0, 0, n


def move(self, row, col, player):
    """
    Player {player} makes a move at ({row}, {col}).
    @param row The row of the board.
    @param col The column of the board.
    @param player The player, can be either 1 or 2.
    @return The current winning condition, can be either:
            0: No one wins.
            1: Player 1 wins.
            2: Player 2 wins.
    :type row: int
    :type col: int
    :type player: int
    :rtype: int
    """
    offset = player * 2 - 3
    self.row[row] += offset
    self.col[col] += offset
    if col == row:
        self.diag += offset
    if row + col == self.n - 1:
        self.anti_diag += offset
    if self.n in [self.diag, self.row[row], self.col[col], self.anti_diag]:
        return 2
    if -self.n in [self.diag, self.row[row], self.col[col], self.anti_diag]:
        return 1
    return 0



"""
Number of Islands II
"""
def numIslands2(self, m, n, positions):
    """
    :type m: int
    :type n: int
    :type positions: List[List[int]]
    :rtype: List[int]
    """
    ans = []
    islands = UnionFind()
    for x, y in positions:
        islands.add((x, y))
        for nx, ny in [(x + 1, y), (x, y - 1), (x, y + 1), (x - 1, y)]:
            if (nx, ny) in islands.parents:
                islands.union((x, y), (nx, ny))
        ans += [islands.count]
    return ans

"""
Number of Islands II
"""
class UnionFind:
    def __init__(self):
        self.size, self.parents = {}, {}
        self.count = 0

    def add(self, p):
        self.parents[p] = p
        self.size[p] = 1
        self.count += 1

    def find(self, node):
        while node != self.parents[node]:
            self.parents[node] = self.parents[self.parents[node]]
            node = self.parents[node]
        return node

    def union(self, n1, n2):
        n1P, n2P = self.find(n1), self.find(n2)
        if n1P == n2P:
            return
        if self.size[n1P] > self.size[n2P]:
            n1P, n2P = n2P, n1P
        self.parents[n1P] = n2P
        self.size[n2P] += self.size[n1P]
        self.count -= 1




class Iterator:
    def __init__(self, nums):
        """
        Initializes an iterator object to the beginning of a list.
        :type nums: List[int]
        """

    def hasNext(self):
        """
        Returns true if the iteration has more elements.
        :rtype: bool
        """

    def next(self):
        """
        Returns the next element in the iteration.
        :rtype: int
        """
"""
284. Peeking Iterator
"""
class PeekingIterator:
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iterator = iterator
        self.temp = iterator.next() if iterator.hasNext() else None

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        return self.temp

    def next(self):
        """
        :rtype: int
        """
        self.res = self.temp
        self.temp = self.iterator.next() if self.iterator.hasNext() else None
        return self.res

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.temp != None


"""
895. Maximum Frequency Stack
"""
class FreqStack:

    def __init__(self):
        self.counter = collections.defaultdict(int)  # this.number's freqency
        self.freq = collections.defaultdict(list)    # frequence of the all number e.g. 3 time: [2,1,3] # [frequency]. [ sequency] =>
        self.maxNum = 0
        

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.counter[x] += 1
        self.maxNum = max(self.maxNum, self.counter[x])
        self.freq[self.counter[x]].append(x)

    def pop(self):
        """
        :rtype: int
        """
        temp = self.freq[self.maxNum].pop()
        if not self.freq[self.maxNum]:  #if this freq is empty
            self.maxNum -= 1            #freq - 1
        self.counter[temp] -=1 
        return temp


class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = {}
        

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        p = self.trie
        for c in word:
            if c not in p:
                p[c] = {}
            p = p[c]
        p['#'] = '#'

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        p = self.trie
        for c in word:
            if c not in p:
                return False
            else:
                p = p[c]
        if '#' in p:
            return True
        else:
            return False

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        p = self.trie
        for c in prefix:
            if c not in p:
                return False
            else:
                p = p[c]
        return True

"""
622. Design Circular Queue
"""
class Node:
    def __init__(self, value):
        self.val = value
        self.next = self.pre = None
        
class MyCircularQueue(object):

    def __init__(self, k):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        :type k: int
        """
        self.size=k
        self.curSize = 0
        self.head = self.tail = Node(-1)
        self.head.next = self.tail
        self.tail.pre = self.head

    def enQueue(self, value):
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        :type value: int
        :rtype: bool
        """
        if self.curSize < self.size:
            node = Node(value)
            node.pre = self.tail.pre
            node.next = self.tail
            node.pre.next = node.next.pre = node
            self.curSize += 1
            return True
        return False
        

    def deQueue(self):
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        :rtype: bool
        """
        if self.curSize > 0:
            node = self.head.next
            node.pre.next = node.next
            node.next.pre = node.pre
            self.curSize -= 1
            return True
        return False

    def Front(self):
        """
        Get the front item from the queue.
        :rtype: int
        """
        return self.head.next.val

    def Rear(self):
        """
        Get the last item from the queue.
        :rtype: int
        """
        return self.tail.pre.val

    def isEmpty(self):
        """
        Checks whether the circular queue is empty or not.
        :rtype: bool
        """
        return self.curSize == 0

    def isFull(self):
        """
        Checks whether the circular queue is full or not.
        :rtype: bool
        """
        return self.curSize == self.size

"""
642. Design Search Autocomplete System
"""
class TrieNode():
    def __init__(self):
        self.children = {}
        self.isEnd = False
        self.data = None
        self.rank = 0
        
class AutocompleteSystem(object):

    def __init__(self, sentences, times):
        """
        :type sentences: List[str]
        :type times: List[int]
        """
        self.root = TrieNode()
        self.keyword = ""
        for i, sentence in enumerate(sentences):
            self.addRecord(sentence, times[i])
        

    def input(self, c):
        """
        :type c: str
        :rtype: List[str]
        """
        results = []
        if c != '#':
            self.keyword += c
            results = self.search(self.keyword)
        else:
            self.addRecord(self.keyword, 1)
            self.keyword = ''
        return [item[1] for item in sorted(results)[:3]]
    
    def search(self, sentence):  #iterate the sentence to the last word
        p = self.root
        for c in sentence:
            if c not in p.children:
                return []
            p = p.children[c]
        return self.dfsFindEnd(p)
    
    def dfsFindEnd(self, root): # iterate the rest of the trie to find the 'End'
        result = []
        if root:
            if root.isEnd:
                result.append((root.rank, root.data))
            for child in root.children:
                result += self.dfsFindEnd(root.children[child])
        return result
    
    def addRecord(self, sentence, hot):
        p = self.root
        for c in sentence:
            if c not in p.children:
                p.children[c] = TrieNode()
            p = p.children[c]
        p.isEnd = True
        p.data = sentence
        p.rank -= hot # I would like to report popular words. I will sort it


class Codec:

    def __init__(self):
        self.url2code = {}
        self.code2url = {}
        self.alphabet = string.ascii_letters + '0123456789'

    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.

        :type longUrl: str
        :rtype: str
        """
        while longUrl not in self.url2code:
            code = ''.join(random.choice(self.alphabet) for _ in range(6))
            if code not in self.code2url:
                self.url2code[longUrl] = code
                self.code2url[code] = longUrl
        return 'http://tinyurl.com/' + self.url2code[longUrl]

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.

        :type shortUrl: str
        :rtype: str
        """
        return self.code2url[shortUrl[-6:]]

"""
353. Design Snake Game
"""
class SnakeGame(object):

    def __init__(self, width, height, food):
        """
        Initialize your data structure here.
        @param width - screen width
        @param height - screen height
        @param food - A list of food positions
        E.g food = [[1,1], [1,0]] means the first food is positioned at [1,1], the second is at [1,0].
        :type width: int
        :type height: int
        :type food: List[List[int]]
        """
        self.snake = collections.deque([[0, 0]])
        self.width = width
        self.height = height
        self.food = collections.deque(food)
        self.direct = {'U': [-1, 0], 'L': [0, -1], 'R': [0, 1], 'D': [1, 0]}

    def move(self, direction: 'str') -> 'int':
        """
        Moves the snake.
        @param direction - 'U' = Up, 'L' = Left, 'R' = Right, 'D' = Down
        @return The game's score after the move. Return -1 if game over.
        Game over when snake crosses the screen boundary or bites its body.
        """
        newHead = [self.snake[0][0] + self.direct[direction][0], self.snake[0][1] + self.direct[direction][1]]
        if 0 <= newHead[0] < self.height and 0 <= newHead[1] < self.width and (newHead not in self.snake or newHead == self.snake[-1]):
            self.snake.appendleft(newHead)
            if self.food and self.food[0] == newHead:
                self.food.popleft()
            else:
                self.snake.pop()
            return len(self.snake) - 1
        return -1


class MaxStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stk=[]
        self.maxstk=[]

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.stk.append(x)
        if not self.maxstk:
            self.maxstk.append(x)
        else:
            self.maxstk.append(max(x,self.maxstk[-1]))

    def pop(self):
        """
        :rtype: int
        """
        self.maxstk.pop()
        return self.stk.pop()
    def top(self):
        """
        :rtype: int
        """
        return self.stk[-1]

    def peekMax(self):
        """
        :rtype: int
        """
        return self.maxstk[-1]

    def popMax(self):
        """
        :rtype: int
        """
        n=self.maxstk.pop()
        tmp=[]
        while n != self.stk[-1]:
            tmp.append(self.pop())
        ret=self.stk.pop()
        for i in range(len(tmp)-1,-1,-1):
            self.push(tmp[i])
        return ret

"""
384. Shuffle an Array
"""
class Solution(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums = nums

    def reset(self):
        """
        Resets the array to its original configuration and return it.
        :rtype: List[int]
        """
        return self.nums

    def shuffle(self):
        """
        Returns a random shuffling of the array.
        :rtype: List[int]
        """
        ans = self.nums[:]
        for i in range(len(ans) - 1, -1, -1):
            j = random.randint(0, i)
            ans[i], ans[j] = ans[j], ans[i]
        return ans


class ListNode():
    def __init__(self, k, v):
        self.pair = (k, v)
        self.next = None

#Design Hit Counter
class HitCounter:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.capacity = 300
        self.times, self.hits = [0] * self.capacity, [0] * self.capacity

    def hit(self, timestamp: int) -> None:
        """
        Record a hit.
        @param timestamp - The current timestamp (in seconds granularity).
        """
        index = timestamp % self.capacity
        if self.times[index] != timestamp:
            self.times[index] = timestamp
            self.hits[index] = 1
        else:
            self.hits[index] += 1

    def getHits(self, timestamp: int) -> int:
        """
        Return the number of hits in the past 5 minutes.
        @param timestamp - The current timestamp (in seconds granularity).
        """
        return sum(self.hits[i] for i in range(300) if timestamp - self.times[i] < 300)
"""
706. Design HashMap
"""
class MyHashMap(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.capacity = 1000
        self.h = [None] * self.capacity

    def put(self, key, value):
        """
        value will always be non-negative.
        :type key: int
        :type value: int
        :rtype: void
        """
        index = key % self.capacity
        if not self.h[index]:
            self.h[index] = ListNode(key, value)
        else:
            curr = self.h[index]
            while True:
                if curr.pair[0] == key:
                    curr.pair = (key, value)
                    return
                if not curr.next:
                    break
                curr = curr.next
            curr.next = ListNode(key, value)

    def get(self, key):
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        :type key: int
        :rtype: int
        """
        index = key % self.capacity
        cur = self.h[index]
        while cur:
            if cur.pair[0] == key:
                return cur.pair[1]
            else:
                cur = cur.next
        return -1

    def remove(self, key):
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        :type key: int
        :rtype: void
        """
        index = key % self.capacity
        cur = prev = self.h[index]
        if not cur:
            return
        if cur.pair[0] == key:
            self.h[index] = cur.next
        else:
            cur = cur.next
            while cur:
                if cur.pair[0] == key:
                    prev.next = cur.next
                    break
                else:
                    cur, prev = cur.next, prev.next


"""
307. Range Sum Query - Mutable
"""
class NumArray:

    def __init__(self, nums: 'List[int]'):
        self.nums = nums
        self.finwickTree = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            index = i + 1
            self._update(index, nums[i])

    def update(self, i: 'int', val: 'int') -> 'None':
        diff, self.nums[i] = val - self.nums[i], val
        i += 1
        self._update(i, diff)

    def sumRange(self, i: 'int', j: 'int') -> 'int':
        return self._query(j + 1) - self._query(i)

    def _update(self, i, delta):
        while i < len(self.finwickTree):
            self.finwickTree[i] += delta
            i += i & -i

    def _query(self, i):
        s = 0
        while i > 0:
            s += self.finwickTree[i]
            i -= i & -i
        return s


class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = Trie()

    def addWord(self, word: str) -> None:
        """
        Adds a word into the data structure.
        """
        node = self.root
        for w in word:
            node = node.children[w]
        node.isWord = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        """
        node = self.root
        self.res = False
        self.dfs(node, word)
        return self.res

    def dfs(self, node, word):
        if not word:
            if node.isWord:
                self.res = True
            return
        if word[0] == '.':
            for n in node.children.values():
                self.dfs(n, word[1:])
        else:
            if word[0] not in node.children:
                return
            self.dfs(node.children[word[0]], word[1:])


class Trie:
    def __init__(self):
        self.children = collections.defaultdict(Trie)
        self.isWord = False