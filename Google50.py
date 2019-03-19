"""蝦皮  糙"""
import random
import collections
def generateRan(weights):
    total = 0 
    for weight in weights:
        total += weight[0]
    
    prevTotal = 0
    weightRatio = []
    for weight in weights:
        prevTotal += weight[0]
        weightRatio.append(prevTotal/total)

    seed = random.randint(1,10)
    randomRange = seed / 10
    
    for idx,val in enumerate(weightRatio):
        if randomRange <= val:
            return weights[idx][1]
            
weights = [(20,'A'),(20,'B'),(40,'C')]

dic = collections.defaultdict(int)
for _ in range(7000):
    dic[generateRan(weights)] += 1
print(dic)

"""
Game Of Life
"""
def gameOfLife(self, board):
    """
    :type board: List[List[int]]
    :rtype: void Do not return anything, modify board in-place instead.
    """
    n, m = len(board), len(board[0])
    def update(i,j):
        live = 0
        for x, y in [(i-1,j-1),(i-1,j+0),(i-1,j+1),(i+0,j-1),(i+0,j+1),(i+0,j+0),(i+1,j-1),(i+1,j+0),(i+1,j+1)]:
            if 0 <= x < n and 0 <= y < m:
                live += board[x][y] & 1
        if live == 3 or live == board[i][j] + 3:
            board[i][j] += 2
                
    if not board and not board[0]:
        return 
    for i in range(n):
        for j in range(m):
            update(i,j)
            
    for i in range(n):
        for j in range(m):
            board[i][j] >>= 1

"""
312. Burst Ballons
"""
def maxCoins(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums: return 0
    
    length = len(nums)
    nums.insert(0,1)
    nums.insert(len(nums),1)
    dp = [[0] * len(nums) for _ in range(len(nums))]
    
    for l in range(1,length+1):
        for i in range(1, length - l + 2):
            j = i + l - 1
            for k in range(i, j+1):
                dp[i][j] = max(dp[i][j], dp[i][k-1] + nums[i-1] * nums[k] * nums[j+1] + dp[k+1][j])
                
    return dp[1][length]
        
        
        
"""
336. Palindrome Pairs
"""
def palindromePairs(self, words):
    """
    :type words: List[str]
    :rtype: List[List[int]]
    """
    def isPalindrome(word):
        return word == word[::-1]
    
    wordDic = {word: i for i, word in enumerate(words)}
    result = []
    for word, idx in wordDic.items():
        n = len(word)
        for j in range(n+1):
            str1 = word[:j]
            str2 = word[j:]
            if isPalindrome(str1):
                str2Reverse = str2[::-1]
                if str2Reverse in wordDic and str2Reverse != word:
                    result.append([wordDic[str2Reverse], idx])
            if j != n and isPalindrome(str2):
                str1Reverse = str1[::-1]
                if str1Reverse in wordDic and str1Reverse != word:
                    result.append([idx, wordDic[str1Reverse]])
    return result


"""
739 Daily Temperatures
"""
def dailyTemperatures(self, T):
    """
    :type T: List[int]
    :rtype: List[int]
    """
    ans = [0] * len(T)
    stack = []
    for i, t in enumerate(T):
        while stack and T[stack[-1]] < t:
            cur = stack.pop()
            ans[cur] = i - cur
        stack.append(i)
    return ans


"""
50 Pow(x,n)
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
        return 1 / self.myPow(x, -n)  #ratio the x and then let it become positive(* -1)
    if n % 2:
        return x * self.myPow(x,n-1)  
    return self.myPow(x*x, n//2)

"""
128 Longest Consecutive Sequence
"""
def longestConsecutive(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    numbers = set(nums)
    max_len = 0
    while numbers:
        m = n = numbers.pop()
        length = 1
        while m - 1 in numbers:
            numbers.remove(m-1)
            m -= 1
            length += 1
        while n + 1 in numbers:
            numbers.remove(n+1)
            n += 1
            length += 1
        max_len = max(max_len, length)
    return max_len





"""
224 Basic Calculator
"""
def calculate(self, s):
    """
    :type s: str
    :rtype: int
    """
    res, num, sign, stack = 0, 0, 1, []
    for ss in s:
        if ss.isdigit():
            num = 10*num + int(ss)
        elif ss in ["-", "+"]:
            res += sign*num
            num = 0
            sign = 1 if ss == '+' else -1
        elif ss == "(":
            stack.append(res)
            stack.append(sign)
            sign, res = 1, 0
        elif ss == ")":
            res += sign*num
            res *= stack.pop()  # -(xxxx) this pop will pop "-"
            res += stack.pop()
            num = 0
    return res + num*sign

"""
31 Next Permutation

6,8,7,4,3,2,
1. From right to left, find the first digit(ParitionNumber) which violate the incrase trend, in this example, is 6
2. From right to left, find the first digit which is large than ParitionNumber, call it changeNumber. which is 7
3. Swap it
4. Reverse all the digit on the right of partition index.
"""
def nextPermutation(self, nums):
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    smallIdx = -1
    for i in range(len(nums) - 2, -1, -1):
        if nums[i] < nums[i + 1]:
            smallIdx = i
            break
    if smallIdx == -1:
        nums.reverse()
        return
    largeIdx = -1
    for i in range(len(nums) - 1, -1, -1):
        if nums[smallIdx] < nums[i]:
            largeIdx = i
            break
    nums[smallIdx], nums[largeIdx] = nums[largeIdx], nums[smallIdx]
    self.reverse(nums, smallIdx + 1, len(nums) - 1)


def reverse(self, nums, start, end):
    while start < end:
        nums[start], nums[end] = nums[end], nums[start]
        start += 1
        end -= 1
    

# nextPermutation([1,3,2])

"""
Longest Line of Consecutive One in Matrix
"""
def longestLine(self, M):
    """
    :type M: List[List[int]]
    :rtype: int
    """
    import collections
    temp = collections.defaultdict(dict)
    maxVal = 0
    
    for i, row in enumerate(M):
        for j, val in enumerate(row):
            if val == 0:
                temp[(i,j)] = [0,0,0,0]
                continue
            else:
                temp[(i,j)] = [1,1,1,1]
                if j > 0:
                    temp[(i,j)][0] = temp[(i,j-1)][0] + 1
                if i > 0:
                    temp[(i,j)][1] = temp[(i-1,j)][1] + 1
                if i > 0 and j > 0:
                    temp[(i,j)][2] = temp[(i-1,j-1)][2] + 1
                if i > 0 and j < len(row) -1:
                    temp[(i,j)][3] = temp[(i-1,j+1)][3] + 1
            maxVal = max(maxVal,max(temp[i,j]))
    return maxVal

"""
659 Split Array into Consecutive Subsequences
"""
def isPossible(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    import collections
    left = collections.Counter(nums)
    end = collections.Counter()
    for i in nums:
        if not left[i]: continue
        left[i] -= 1
        if end[i - 1] > 0:
            end[i-1] -= 1
            end[i] += 1
        elif left[i+1] and left[i + 2]:
            left[i+1] -= 1
            left[i+2] -= 1
            end[i+2] += 1
        else:
            return False
    return True

"""
750. Number of corner Rectangles
"""
def countCornerRectangles(grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    if not grid or not grid[0]:
        return 0
    dp_set = []
    ans = 0
    for y in range(len(grid)):
        dp_set.append(set(idx for idx,val in enumerate(grid[y]) if val))
        for prev in range(y):
            print(dp_set[y] )
            print(dp_set[prev])
            matches = len(dp_set[y] & dp_set[prev])
            print(matches)
            print(dp_set)
            print('============')
            if matches >= 2:
                ans += matches * (matches-1) // 2
        
    return ans

# grid =[[1, 0, 0, 1, 0],
#  [0, 0, 1, 0, 1],
#  [0, 0, 0, 1, 0],
#  [1, 0, 1, 0, 1]]
# countCornerRectangles(grid)

"""
Bus Routes
Tm(m*n) m: no of bus * no of route
Space(m*n)

Step1. Build paths key:val => stop:bus No.
Step2. iterate all the stops in TravelStop
Step3. Find the buss which pass by the current TravelStop and find all add all the bus stops in TravelStop

"""
def numBusesToDestination(self, routes: 'List[List[int]]', S: 'int', T: 'int') -> 'int':
    path, travelStops, travelTaken, used = collections.defaultdict(set), [S], 0, set()
    for bus, route in enumerate(routes):  # path=> stop1=>bus1
        for stop in route:  # path=> stop2=>bus1...
            path[stop].add(bus)
    while travelStops:
        new = []
        for stop in travelStops:  # get stop
            if stop == T:
                return travelTaken
            for bus in path[stop]:  # get bus
                if bus not in used:  # travel all thr rout of the bus so going to put the bus in visited
                    used.add(bus)
                    for nextStop in routes[bus]:  # get routes of the bus from routes
                        if nextStop != stop:
                            new.append(nextStop)
        travelTaken += 1
        travelStops = new
    return -1



def leetcode62Modification(m, n):
    dp = [[0] * n for _ in range(m)]
    direction = [[-1,-1],[-1,0],[-1,1]]
    dp[0][0] = 1
    for x in range(m):
        for y in range(n):
            for dir_x, dir_y in direction:
                pre_x = x + dir_x
                pre_y = y + dir_y
                if pre_x < 0 or pre_y < 0 or pre_y >= n:
                    continue
                dp[x][y] += dp[pre_x][pre_y]
    return dp[m-1][0]

# print(leetcode62Modification(4,3))

"""
489. Robot Room Cleaner
how it works:
it will trace back to previous position and turn to the same direction when it cannot move anymore.
it will turn 90 degree every time and then move.
rotated r u l d 
回字
"""
def cleanRoom(robot):
    """
    :type robot: Robot
    :rtype: None
    """
    dfs(robot, 0, 0, 0, 1, set())

def dfs(robot, x, y, direction_x, direction_y, visited):
    robot.clean()
    visited.add((x, y))
    
    for k in range(4):
        neighbor_x = x + direction_x
        neighbor_y = y + direction_y
        if (neighbor_x, neighbor_y) not in visited and robot.move():
            dfs(robot, neighbor_x, neighbor_y, direction_x, direction_y, visited)
            robot.turnLeft()
            robot.turnLeft()
            robot.move()
            robot.turnLeft()
            robot.turnLeft()
        robot.turnLeft()
        direction_x, direction_y = -direction_y, direction_x  


"""
890 Find and Replace Pattern
"""
def findAndReplacePattern(self, words, pattern):
    """
    :type words: List[str]
    :type pattern: str
    :rtype: List[str]
    """
    def isMatch(word, pattern):
        dic = {}
        for w, p in zip(word, pattern):
            if w not in dic:
                if p in dic.values():
                    return False
                dic[w] = p
            else:
                if dic[w] != p:
                    return False
        return True
    result = []
    for w in words:
        if isMatch(w, pattern):
            result.append(w)
    return result

"""
Guess the word
"""
def findSecretWord(self, wordlist, master):
    """
    :type wordlist: List[Str]
    :type master: Master
    :rtype: None
    """
	
    def pair_matches(a, b):         # count the number of matching characters
        return sum(c1 == c2 for c1, c2 in zip(a, b))
    def most_overlap_word():
        counts = [[0 for _ in range(26)] for _ in range(6)]     # counts[i][j] is nb of words with char j at index i
        for word in candidates:
            for i, c in enumerate(word):
                counts[i][ord(c) - ord("a")] += 1
        best_score = 0
        for word in candidates:
            score = 0
            for i, c in enumerate(word):
                score += counts[i][ord(c) - ord("a")]           # all words with same chars in same positions
            if score > best_score:
                best_score = score
                best_word = word
        return best_word
    candidates = wordlist[:]        # all remaining candidates, initially all words
    while candidates:
        s = most_overlap_word()     # guess the word that overlaps with most others
        matches = master.guess(s)
        if matches == 6:
            return
        candidates = [w for w in candidates if pair_matches(s, w) == matches]   # filter words with same matches

"""
Inorder Successor in BST
"""
def inorderSuccessor(self, root, p):
    """
    :type root: TreeNode
    :type p: TreeNode
    :rtype: TreeNode
    """
    if not root:
        return None
    if root.val <= p.val:
        return self.inorderSuccessor(root.right, p)
    else:
        left = self.inorderSuccessor(root.left, p)
        return root if not left else left

"""
857. Minimum Cost to Hire K Workers
"""
def mincostToHireWorkers(quality, wage, K):
    """
    :type quality: List[int]
    :type wage: List[int]
    :type K: int
    :rtype: float
    """
    import heapq
    workers = sorted([wage[i]/quality[i], quality[i]] for i in range(len(quality)))
    res,qsum = float('inf'),0
    heap = []
    for i in range(len(workers)):
    	# 选定比例 r
        r,q = workers[i]
        heapq.heappush(heap,-q)
        # qsum始终记录k个人的quality之和，乘以r即为最后结果
        qsum += q
        if len(heap) > K:
        	# 始终丢弃quality最大的人
            qsum += heapq.heappop(heap)
        if len(heap) == K:
            res = min(res, qsum * r)
    return res

mincostToHireWorkers([10,20,5],[70,50,30],2)


"""
15. 3Sum
"""
def threeSum(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    nums.sort()
    result = []
    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        target = -nums[i]
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[left] + nums[right]
            if target == total:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            else:
                if total > target:
                    right -= 1
                else:
                    left += 1
    return result

"""
Generate Parentheses
"""
def generateParenthesis(self, n):
    """
    :type n: int
    :rtype: List[str]
    """
    def helper(temp, left_side, right_side):
        if right_side > left_side:
            return 
        if left_side == 0 and right_side == 0:
            result.append(temp)
            return 
        if right_side > 0:
            helper(temp + '(', left_side, right_side - 1)
        if left_side > 0:
            helper(temp + ')', left_side - 1, right_side)
    result = []
    helper("", n ,n)
    return result

"""
 Minimum Path Sum
"""
def minPathSum(self, grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    m, n = len(grid), len(grid[0])
    dp = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                dp[i][j] = grid[i][j]
            elif i == 0:
                dp[i][j] = dp[i][j-1] + grid[i][j]
            elif j == 0:
                dp[i][j] = dp[i-1][j] + grid[i][j]
            else:
                dp[i][j] = grid[i][j] + min(dp[i][j-1], dp[i-1][j])
    return dp[m-1][n-1]

"""
Edit Distance
"""
def minDistance(self, word1, word2):
    """
    :type word1: str
    :type word2: str
    :rtype: int
    """
    m, n = len(word1), len(word2)
    dp = [[0 for _ in range(n+1)] for _ in range(m + 1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
    return dp[m][n]

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
        dp[i] = max(dp[i-2] + nums[i], dp[i-1])
    
    return max(dp[len(dp)-1], dp[len(dp)-2])

"""
House Robber II
"""
def rob2(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if not nums:return 0
    if len(nums) <= 2: return max(nums)
    return max(self.rob_row(nums[1:]),self.rob_row(nums[:-1]))

def rob_row(self, nums):
    res = [0] * len(nums)
    res[0], res[1] = nums[0], max(nums[0],nums[1])
    
    for i in range(2,len(nums)):
        res[i] = max(res[i-1],res[i-2] + nums[i])
    return res[-1]

"""
Sentence Screen Fitting
"""
def wordsTyping(self, sentence, rows, cols):
    """
    :type sentence: List[str]
    :type rows: int
    :type cols: int
    :rtype: int
    """
    s = ' '.join(sentence) + ' '
    count = 0
    length = len(s)
    for i in range(rows):
        count += cols
        if s[count % length] == ' ': #we use % to find out the word which is not long enough
            count += 1   # if the last is ' ', +1 means go to next new cycle
        else:
            while count > 0 and s[(count - 1)% length] != ' ': # because you are in the middle of the words, so you have to check previous one is space to restart
                count -= 1
    return count // length

"""
Decode Ways
"""
def numDecodings(self, s):
    """
    :type s: str
    :rtype: int
    """
    if s == '': return 0
    dp = [0 for x in range(len(s) + 1)]
    dp[0] = 1
    for i in range(1, len(s) + 1):
        if s[i-1] != '0':
            dp[i] += dp[i-1]
        if i != 1 and s[i-2:i] < '27' and s[i-2:i] > '09':
            dp[i] += dp[i-2]
    return dp[len(s)]

"""
O(1)
"""
def numDecodingsO1(self, s):
    """
    :type s: str
    :rtype: int
    """
    if not s or s.startswith('0'): return 0
    c1 = c2 = 1
    for i in range(1, len(s)):
        if s[i] == '0':
            c1 = 0
        if s[i-1] == '1' or s[i-1] == '2' and s[i] <='6':
            c1 = c1 + c2
            c2 = c1 - c2
        else:
            c2 = c1
    return c1
        

"""
Longest Increasing Path in a Matrix  V1
"""
def longestIncreasingPath(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: int
    """
    def dfs(i, j):
        if not dp[i][j]:
            val = matrix[i][j]
            dp[i][j] = 1 + max(
                dfs(i - 1, j) if i > 0 and val > matrix[i-1][j] else 0,
                dfs(i + 1, j) if i < M - 1 and val > matrix[i+1][j] else 0,
                dfs(i, j - 1) if j > 0 and val > matrix[i][j-1] else 0,
                dfs(i, j + 1) if j < N - 1 and val > matrix[i][j+1] else 0
            )
        return dp[i][j]
    
    if not matrix:
        return 0
    N, M = len(matrix[0]), len(matrix)
    dp = [[0] * N for _ in range(M)]
    return max(dfs(x,y) for x in range(M) for y in range(N))


"""
Longest Increasing Path in a Matrix  V2
"""
def longestIncreasingPath2(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: int
    """
    if not matrix:
        return 0
    self.directions = [(1,0),(-1,0),(0,1),(0,-1)]
    m, n = len(matrix), len(matrix[0])
    dp = [[-1] * n for _ in range(m)]
    res = 0
    for i in range(m):
        for j in range(n):
            cur_len = self.dfsPath2(i, j, matrix, dp, m, n)
            res = max(cur_len, res)
    return res

def dfsPath2(self, i, j, matrix, dp, m, n):
    if dp[i][j] != -1:
        return dp[i][j]
    res = 1
    for direction in self.directions:
        x, y = i + direction[0], j + direction[1]
        if x < 0 or x >= m or y < 0 or y >= n or matrix[x][y] <= matrix[i][j]:
            continue
        length = 1 + self.dfs(x,y,matrix, dp, m, n)
        res = max(length, res)
    dp[i][j] = res
    return res



"""
Kth Largest Element in an Array
"""
def findKthLargest(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    n = len(nums)
    p = self.quickSelect(nums,0, n-1, n-k + 1)
    return nums[p]

def quickSelect(self, nums, start,end, k):
    low, high, pivot = start, end, nums[end]
    while low < high:
        if nums[low] > pivot:
            high -= 1
            nums[low], nums[high] = nums[high], nums[low]
        else:
            low += 1
    nums[low],nums[end] = nums[end],nums[low]
    m = low - start + 1
    if m == k:
        return low
    elif m > k:
        return self.quickSelect(nums,start, low -1, k)
    else:
        return self.quickSelect(nums,low+1, end,k-m)
        

"""
Trapping Rain Water
"""
def trap(self, height):
    """
    :type height: List[int]
    :rtype: int
    """
    left, left_max, right_max, sum = 0, 0, 0, 0
    right = len(height) - 1
    while left < right:
        if height[left] > height[right]:
            right_max = max(right_max, height[right])
            sum += right_max - height[right]
            right -= 1
        else:
            left_max = max(left_max, height[left])
            sum += left_max - height[left]
            left += 1
    return sum

"""
  First Unique Character in a String
"""
def firstUniqChar(s):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    index = [s.index(l) for l in letters if s.count(l) == 1]
    print(index)
    return min(index) if len(index) > 0 else -1

#print(firstUniqChar('leetcode'))


def repeatedStringMatch(A,B):
    times = len(B)//len(A)
    for i in range(3):  #0, 1, 2
        if B in (A * (times + i)):
            return times + i
    return -1

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
    for i in range(1, len(dp)):
        for j in range(i):
            if dp[j] and s[j:i] in wordDict:
                dp[i] = True
                break
    return dp[-1]


def moveZeroes(self, nums):
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    non_zero_index = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[non_zero_index] = nums[i]
            non_zero_index += 1
    for i in range(non_zero_index,len(nums)):
        nums[i] = 0
"""
Remove Duplicates from Sorted Array
"""
def removeDuplicates(self, nums):
    if not nums:
        return 0
    startIndex = 0
    for i in range(1, len(nums)):
        if nums[i] != nums[startIndex]:
            startIndex += 1
            nums[startIndex] = nums[i]
    return startIndex + 1


"""
Merge k Sorted Lists
"""
def mergeKLists(self, lists: 'List[ListNode]') -> 'ListNode':
    return self.mergeSort(lists, 0, len(lists) - 1)


def mergeSort(self, lists, start, end):
    if start > end:
        return None
    if start == end:
        return lists[start]
    mid = (start + end) // 2
    left = self.mergeSort(lists, start, mid)
    right = self.mergeSort(lists, mid + 1, end)
    return self.merge(left, right)


def merge(self, l1, l2):
    dummy = ListNode(-1)
    curr = dummy
    while l1 and l2:
        if l1.val > l2.val:
            curr.next = l2
            l2 = l2.next
        else:
            curr.next = l1
            l1 = l1.next
        curr = curr.next
    if l1:
        curr.next = l1
    if l2:
        curr.next = l2
    return dummy.next

def mergeKLists(self, lists):
    """
    :type lists: List[ListNode]
    :rtype: ListNode
    """
    import queue as Q
    dummy = ListNode(None)
    curr = dummy
    q = Q.PriorityQueue()
    for node in lists:
        if node:
            q.put((node.val,node))
    while q.qsize() > 0:
        curr.next = q.get()[1]
        curr = curr.next
        if curr.next:
            q.put((curr.next.val, curr.next))
    return dummy.next


def licenseKeyFormatting(self, S, K):
    S = S.replace('-','').upper()[::-1]
    return '-'.join(S[i:i+K] for i in range(0,len(S),K))[::-1]

def nextClosestTime(self, time):
    """
    :type time: str
    :rtype: str
    """
    digit_set = set(time.replace(':',''))
    hours, minutes = time.split(':')
    while True:
        if minutes == '59':
            hours = str(int(hours) + 1)
            minutes = '00'
        else:
            minutes = str(int(minutes) + 1)
        if int(hours) > 23:
            hours = '00'
        if len(hours) == 1:
            hours = '0'+ hours
        if len(minutes) == 1:
            minutes = '0' + minutes
        if all(x in digit_set for x in hours+minutes):
            return hours+':'+minutes

"""
Longest Univalue Path
"""
def longestUnivaluePath(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    self.longest = 0
    def traverse(node):
        if not node:
            return 0
        left_len, right_len = traverse(node.left), traverse(node.right)
        left = (left_len + 1) if node.left and node.left.val == node.val else 0
        right = (right_len + 1) if node.right and node.right.val == node.val else 0
        self.longest = max(self.longest, left + right)
        return max(right,left)
    traverse(root)
    return self.longest

"""
K Empty Slots
"""
def kEmptySlots(flowers, k):
    """
    :type flowers: List[int]
    :type k: int
    :rtype: int
    """
    garden = [[i - 1, i + 1] for i in range(len(flowers))]
    garden[0][0], garden[-1][1] = None, None
    ans = -1
    for i in range(len(flowers) -1, -1, -1):
        cur = flowers[i] - 1
        left, right = garden[cur]
        if right != None and right - cur == k + 1:
            ans = i + 1
        if left != None and cur - left == k + 1:
            ans = i + 1
        if right != None:
            garden[right][0] = left
        if left != None:
            garden[left][1] = right
    return ans

"""
Spiral Matrix
"""
def spiralOrder(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: List[int]
    """
    res = []
    if len(matrix) == 0:
        return res
    
    row_start, row_end, col_start, col_end = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
    while row_start <= row_end and col_start <= col_end:
        for i in range(col_start, col_end+1):
            res.append(matrix[row_start][i])
        row_start += 1
        
        for i in range(row_start, row_end+1):
            res.append(matrix[i][col_end])
        col_end -= 1
        
        # I traverse left or up I have to check whether the row or col still
        #  exists to prevent duplicates.
        if row_start <= row_end: 
            for i in range(col_end, col_start-1, -1):
                res.append(matrix[row_end][i])
        row_end -= 1
        
        if col_start <= col_end:
            for i in range(row_end, row_start - 1, -1):
                res.append(matrix[i][col_start])
        col_start += 1        
    return res

"""
Plus One
"""
def plusOne(self, digits):
    """
    :type digits: List[int]
    :rtype: List[int]
    """
    res = []
    carry = 0
    for i in range(len(digits)-1, -1, -1):
        sum = digits[i] + carry
        if i == len(digits) -1:
            sum += 1
        carry = sum // 10
        sum = sum % 10
        res.append(sum)
    if carry != 0:
        res.append(1)
    return res[::-1]

"""
Valid Palindrome
Forgot : lower
"""
def isPalindrome(self, s):
    """
    :type s: str
    :rtype: bool
    """
    start, end = 0, len(s) - 1
    while start < end:  # if the length is odd the middle one will not be checked
        while start < end and not s[start].isalnum():
            start += 1
        while start < end and not s[end].isalnum():
            end -= 1
        if s[start].lower() != s[end].lower():
            return False
        start += 1
        end -= 1
    return True

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
        res = max(res,idx - start_indx + 1)
    return res


def isValid(self, s):
    """
    :type s: str
    :rtype: bool
    """
    dic = {')':'(','}':'{',']':'['}
    stack = []
    for c in s:
        if c not in dic:
            stack.append(c)
        else:
            if len(stack) == 0 or stack.pop() != dic[c]:
                return False
    return len(stack) == 0

"""
One Edit Distance
"""
def isOneEditDistance(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: bool
    """
    if s == t:
        return False
    ls, lt = len(s), len(t)
    if ls > lt:
        return self.isOneEditDistance(t, s)
    if lt - ls > 1:
        return False
    for i in range(ls):
        if s[i] != t[i]:
            if ls == lt:
                s = s[:i] + t[i] + s[i+1:]
            else:
                s = s[:i] + t[i] + s[i:]
            break
    return s == t or s == t[:-1]


"""
Edit Distance
"""
def minDistanceDP(word1, word2):
    """
    :type word1: str
    :type word2: str
    :rtype: int
    """
    dp = [[0] * (len(word1) + 1) for _ in range(len(word2) + 1)]
    for i in range(len(dp)):
        for j in range(len(dp[0])):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[j-1] == word2[i-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i-1][j-1],dp[i][j-1]) + 1
    return dp[len(dp)-1][len(dp[0])-1]



def isValidBST(self, root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    return self.isValidHelper(root, None, None)
    
def isValidHelper(self, node, maxNode, minNode):
    if not node: return True
    if maxNode and node.val >= maxNode.val: return False
    if minNode and node.val <= minNode.val: return False
    return self.isValidHelper(node.left, node, minNode) and self.isValidHelper(node.right, maxNode, node)
        
"""
5. Longest Palindromic Substring
"""
def longestPalindrome(self, s: 'str') -> 'str':
    maxLength = startIdx = 0
    dp = [[False] * len(s) for _ in range(len(s))]
    for i in range(len(s)):
        for j in range(i + 1):  # include i itself. so i + 1
            tempLength = i - j + 1
            if s[i] == s[j] and (tempLength < 3 or dp[i - 1][j + 1]):
                dp[i][j] = True
                if tempLength > maxLength:
                    maxLength = tempLength
                    startIdx = j
    return s[startIdx:startIdx + maxLength]

"""
Closest Binary Search Tree Value
"""
def closestValue(self, root, target):
    """
    :type root: TreeNode
    :type target: float
    :rtype: int
    """
    res = root.val
    while root is not None:
        if abs(root.val - target) < abs(res - target):
            res = root.val
        root = root.left if root.val > target else root.right
    return res

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

"""
346. Moving Average from Data Stream
"""
class MovingAverage:

    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self.vector, self.size,self.sum,self.idx = [0]*size, size, 0, 0
        

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
Add Bold Tag in String
"""
def addBoldTag(self, s, dict):
    """
    :type s: str
    :type dict: List[str]
    :rtype: str
    """
    status = [False]*len(s)
    result = ''
    for word in dict:
        start = s.find(word)
        last = len(word)
        while start != -1:
            for i in range(start, last + start):
                status[i] = True
            start = s.find(word,start+1) # this part supports the overlap
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
Intersection of Two Arrays
"""
def intersection(self, nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: List[int]
    """
    nums1.sort()
    nums2.sort()
    result = []
    i = j = 0
    while i < len(nums1) and j < len(nums2):
        if nums1[i] == nums2[j]:
            if i == 0 or (i > 0 and nums1[i] != nums1[i-1]):
                result.append(nums1[i])
            i += 1
            j += 1
        elif nums1[i] > nums2[j]:
            j += 1
        else:
            i += 1
    return result

def imageSmoother(self, M):
    """
    :type M: List[List[int]]
    :rtype: List[List[int]]
    """
    import math
    result = [[0 for _ in range(len(M[0]))] for _ in range(len(M))]
    coord = ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1))
    for i in range(len(M)):
        for j in range(len(M[0])):
            count, total = 0, 0
            for x,y in coord:
                new_x,new_y = x + i, y+j
                if new_x < 0 or new_x >= len(M) or new_y < 0 or new_y >= len(M[0]):
                    continue
                count += 1
                total += M[new_x][new_y]
            result[i][j] = math.floor(total / count)
    return result


"""
Max Consecutive Ones
"""
def findMaxConsecutiveOnes(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    max_consecutive, total_here = 0,0
    for i in nums:
        if i == 1:
            total_here += 1
        else:
            total_here = 0
        max_consecutive = max(total_here,max_consecutive)
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
Shortest Palindrome
"""
def shortestPalindrome(s):
    """
    :type s: str
    :rtype: str
    """
    r = s[::-1]
    for i in range(len(s) + 1):
        if s.startswith(r[i:]):
            return r[:i] + s


"""
Regular Expression Matching
"""
def isMatch(self, s, p):
    """
    :type s: str
    :type p: str
    :rtype: bool
    """
    dp = [[False] * (len(s) + 1) for _ in range(len(p) + 1)]
    dp[0][0] = True
    
    for i in range(2, len(p) + 1):  # in initial part if it is *, it should be reference previous one.
        if p[i-1] == '*':          
            dp[i][0] = dp[i-2][0]
    
    for i in range(1, len(p) + 1):
        for j in range(1, len(s) + 1):
            if p[i - 1] != '*':
                dp[i][j] = dp[i - 1][j - 1] and (p[i-1] == s[j-1] or p[i-1] == '.')
            else:
                dp[i][j] = dp[i-2][j] or dp[i-1][j]  # i-2 means it accept empty char. 
                                                     #so if there is no char in string. you should compare previous one
                if p[i-2] == s[j-1] or p[i-2] == '.':
                    dp[i][j] |= dp[i][j-1]
    return dp[-1][-1]


def insert(self, head, insertVal):
    """
    :type head: Node
    :type insertVal: int
    :rtype: Node
    """
    node = Node(insertVal, head)
    if not head:
        return node
    prev, cur = head, head.next
    while True:
        if prev.val <= insertVal <= cur.val:
            break
        elif prev.val > cur.val and (insertVal < cur.val or insertVal > prev.val):
            break
        prev, cur = prev.next, cur.next
        if prev == head:
            break
    prev.next = node
    node.next = cur
    return head


def findKthLargest2(self, nums, k):
  return self.findKthSmallest(nums, len(nums)+1-k)

def findKthSmallest(self, nums, k):
    if nums:
        pos = self.partition(nums, 0, len(nums)-1)
        if k > pos+1:
            return self.findKthSmallest(nums[pos+1:], k-pos-1)
        elif k < pos+1:
            return self.findKthSmallest(nums[:pos], k)
        else:
            return nums[pos]

def partition(self, nums, l, r):
    low = l
    while l < r:
        if nums[l] < nums[r]:
            nums[l], nums[low] = nums[low], nums[l]
            low += 1
        l += 1
    nums[low], nums[r] = nums[r], nums[low]
    return low

"""
399. Evaluate Division
"""
#This version time O(e + q*e), space O(e)
#e = equation, q= query
def calcEquation(self, equations, values, queries):
    """
    :type equations: List[List[str]]
    :type values: List[float]
    :type queries: List[List[str]]
    :rtype: List[float]
    """
    
    import collections
    def divide(x,y,visited):
        if x == y:
            return 1.0  #it is himself
        visited.add(x)
        for n in g[x]:
            if n in visited:
                continue
            d = divide(n,y,visited) 
            if d > 0:
                return d * g[x][n]  # don't understand
        return -1.0
    g = collections.defaultdict(dict)      # make a graph
    for (x,y), v in zip(equations, values): 
        g[x][y] = v
        g[y][x] = 1.0 / v
    ans = []
    for x,y in queries:
        if x in g and y in g:
            ans.append(divide(x,y,set()))
        else:
            ans.append(-1)
    #ans = [divide(x,y, set()) if x in g and y inb g else -1 for x, y in queries]
    return ans

#Union Find  Time: O(e + q) Space: O(e)
def calcEquationUnionFindVersion(self, equations, values, queries):
    """
    :type equations: List[List[str]]
    :type values: List[float]
    :type queries: List[List[str]]
    :rtype: List[float]
    """
    parents = {} #{Child:Parent}, eg {'a':'b'}
    weights = {} #{Node: float}, eg{'a': 1.0}
    result = []
    
    for (eq1,eq2),value in zip(equations,values):
        if eq1 not in parents:
            parents[eq1] = eq1
            weights[eq1] = 1.0
        if eq2 not in parents:
            parents[eq2] = eq2
            weights[eq2] = 1.0            
        self.calUnion(eq1,eq2,parents,weights,value)
    
    for q1,q2 in queries:
        if q1 not in parents or q2 not in parents:
            result.append(-1.0)
        else:
            parent1 = self.calFind(q1,parents,weights)
            parent2 = self.calFind(q2,parents,weights)       
            if parent1!=parent2:
                result.append(-1.0)
            else:
                result.append(weights[q1]/weights[q2])
            
    return result
        
        
def calUnion(self,node1,node2,parents,weights,value):
    parent1 = self.calFind(node1,parents,weights)
    parent2 = self.calFind(node2,parents,weights)
    if parent1 != parent2:
        parents[parent1] = parent2
        weights[parent1] = value * (weights[node2] /weights[node1]) #IMPORTANT: Node1 may already be compressed: its weight could be the product of all weights up to parent1            
                                                                    # the value of compression will not be higher than 
    
def calFind(self, node, parents,weights):
    #Find parent node of a given node, doing path compression while doing so (set the node's parent to its root and multiply all weights along the way.)     
    if parents[node] != node:
        p = parents[node]
        parents[node] = self.calFind(parents[node],parents,weights)
        weights[node] = weights[node] * weights[p]
    return parents[node]
        


"""
207. Course Schedule
O(n) time complexity
"""
def canFinish(self, numCourses: 'int', prerequisites: 'List[List[int]]') -> 'bool':
    graph = collections.defaultdict(set)
    visited = [0] * numCourses
    def isCircle(course):
        if visited[course] == 1:
            return True
        if visited[course] == -1:
            return False
        visited[course] = 1
        for nextCourse in graph[course]:
            if isCircle(nextCourse):
                return False
        visited[course] = -1
        return False

    for course, pre in prerequisites:
        graph[pre].add(course)
    for course in range(numCourses):
        if isCircle(course):
            return False
    return True

"""
Redundant Connection
"""
def findRedundantConnection(self, edges):
    """
    :type edges: List[List[int]]
    :rtype: List[int]
    """
    p = [0] * (len(edges) + 1)
    s = [1] * (len(edges) + 1)
    
    def find(u):
        while p[u] != u:
            p[u] = p[p[u]]
            u = p[u]
        return u
    for u, v in edges:
        if p[u] == 0: p[u] = u
        if p[v] == 0: p[v] = v
        
        pu, pv = find(u), find(v)
        if pu == pv: return [u, v]
        if s[pv] > s[pu]:
            u, v = v, u
        p[pv] = pu
        s[pu] += s[pv]
    return []




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

# def findRedundantDirectedConnection(self, edges):
#     """
#     :type edges: List[List[int]]
#     :rtype: List[int]
#     """
#     nodes = set()
#     for e in edges:
#         nodes |= set(e)
#     parents = {n: [] for n in nodes}
#     root = set(nodes)
#     for u, v in edges:
#         parents[v].append(u)
#         root.discard(v)
#     if root:
#         root = root.pop()
#         two_candidates = [e for e in edges if len(parents[e[1]]) > 1]
#         child = two_candidates[0][1]
#         parent = two_candidates[0][0]
#         while parent != root:
#             if parent == child:
#                 return two_candidates[0]
#             parent = parents[parent][0]
#         return two_candidates[1]
#     else: # root cycle
#         rc = [nodes.pop()]
#         while True:
#             parent = parents[rc[-1]][0]
#             if parent in rc:
#                 rc = set(rc[rc.index(parent):])
#                 break
#             else:
#                 rc.append(parent)
#         for u, v in edges:
#             rc.discard(v)
#             if len(rc) == 1:
#                 root = rc.pop()
#                 return [parents[root][0], root]

def findStrobogrammatic(n):
    """
    :type n: int
    :rtype: List[str]
    """
    evenMidCandidate = ["11","69","88","96", "00"]
    oddMidCandidate = ["0", "1", "8"]
    if n == 1:
        return oddMidCandidate
    if n == 2:
        return evenMidCandidate[:-1]
    if n % 2:
        pre, midCandidate = findStrobogrammatic(n-1), oddMidCandidate
    else: 
        pre, midCandidate = findStrobogrammatic(n-2), evenMidCandidate
    premid = (n-1)//2
    return [p[:premid] + c + p[premid:] for c in midCandidate for p in pre]


def mySqrt(self, x):
    """
    :type x: int
    :rtype: int
    """
    if x == 0:
        return 0
    if x > 0 and x < 4:
        return 1
    left, right = 1, x//2
    while True:
        mid = left + (right - left) // 2
        if mid > 0 and mid > x//mid:
            right = mid - 1
        elif (mid + 1) > x // (mid+1):
            return mid
        else:
            left = mid + 1

"""
Insert Interval
"""
def insertInterval(self, intervals, newInterval):
    """
    :type intervals: List[Interval]
    :type newInterval: Interval
    :rtype: List[Interval]
    """
    result = []
    i=0
    while i < len(intervals) and intervals[i].end < newInterval.start:
        result.append(intervals[i])
        i += 1
    while i < len(intervals) and intervals[i].start <= newInterval.end:
        newInterval.start = min(intervals[i].start, newInterval.start)
        newInterval.end = max(intervals[i].end,newInterval.end)
        i += 1
    result.append(newInterval)
    while i < len(intervals):
        result.append(intervals[i])
        i += 1
    return result


"""
Diagonal Traverse
"""
def findDiagonalOrder(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: List[int]
    """
    if not matrix or len(matrix) == 0 :
        return []
    rows,cols = len(matrix),len(matrix[0])
    res = [None] * (rows * cols)
    number_scan = rows + cols - 1
    index = 0
    for i in range(number_scan):
        if i%2 ==0:
            x = i if i < rows else rows - 1
            y = 0 if i < rows else i - (rows - 1)
            while x >= 0 and y < cols:
                res[index] = matrix[x][y]
                index += 1; y += 1; x -= 1
        else:
            x = 0 if i < cols else i - (cols - 1)
            y = i if i < cols else cols - 1
            while x < rows and y >= 0:
                res[index] = matrix[x][y]
                index += 1; x+=1; y -= 1
    return res

"""
Evaluate Reverse Polish Notation
"""
def evalRPN(self, tokens):
    """
    :type tokens: List[str]
    :rtype: int
    """
    stack = []
    for t in tokens:
        if t not in ["+", "-", "*", "/"]:
            stack.append(int(t))
        else:
            r, l = stack.pop(), stack.pop()
            if t == "+":
                stack.append(l + r)
            elif t == "-":
                stack.append(l - r)
            elif t == "*":
                stack.append(l * r)
            else:
                stack.append(int(float(l) / r))  #["10","6","9","3","+","-11","*","/","*","17","+","5","+"] we cannot pass the case. So do this tricky.
    return stack.pop()


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

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Node:
    def __init__(self, val, next):
        self.val = val
        self.next = next


class PeekingIterator:
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iter = iterator
        self.temp = self.iter.next() if self.iter.hasNext() else None

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
        ret = self.temp
        self.temp = self.iter.next() if self.iter.hasNext() else None
        return ret

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.temp is not None

class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}
        

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        p = self.root
        for c in word:
            if c not in p:
                p[c] = {}
            p = p[c]
        p['#'] = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node = self.find(word)
        return node is not None and '#' in node
        

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        return self.find(prefix) is not None
    
    
    def find(self, prefix):
        p = self.root
        for c in prefix:
            if c not in p:
                return None
            p = p[c]
        return p
    
class BSTIterator(object):
    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left
        

    def hasNext(self):
        """
        :rtype: bool
        """
        return len(self.stack) > 0 

    def next(self):
        """
        :rtype: int
        """
        node = self.stack.pop()
        temp_node = node.right
        while temp_node:
            self.stack.append(temp_node)
            temp_node = temp_node.left
        return node.val

