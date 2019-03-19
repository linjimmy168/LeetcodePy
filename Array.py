"""
122. Best Time to Buy and Sell Stock II
"""
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0
        total = 0
        for i in range(1,len(prices)):
            profit = prices[i] - prices[i-1]
            total += 0 if profit < 0 else profit
        return total


    """
    22. Generate Parentheses
    """
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        result = []
        self.generateHelper(result, '',0,0,n)
        return result
    
    def generateHelper(self, result, tempStr, open, close, n):
        if len(tempStr) == n*2:
            result.append(tempStr)
            return
        
        if open < n:
            self.generateHelper(result, tempStr + '(', open + 1,close,n)
        if close < open:
            self.generateHelper(result, tempStr + ')', open, close+1, n)

"""
33. Search in Rotated Sorted Array
"""
def search(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid

        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1

"""
75. Sort Colors
"""
def sortColors(self, nums):
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    l, r, idx = 0,len(nums)-1, 0
    while idx<=r:
        if nums[idx] == 0:
            nums[l], nums[idx] = nums[idx], nums[l]
            idx += 1
            l += 1
        elif nums[idx] == 1:
            idx += 1
        else:
            nums[idx],nums[r] = nums[r], nums[idx]
            r-=1

def strStr(self, haystack, needle):
    """
    :type haystack: str
    :type needle: str
    :rtype: int
    """
    if not needle:
        return 0
    for i in range(len(haystack) - len(needle) + 1):
        for j in range(len(needle)):
            if haystack[i+j] != needle[j]:
                break
            if j == len(needle) - 1:
                return i
    return -1

"""
153. Find Minimum in Rotated Sorted Array
"""
def findMin(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    start, end = 0, len(nums) - 1
    while start < end:
        mid = start + (end - start)//2
        if nums[mid] > nums[end]:
            start = mid + 1
        else:
            end = mid
    return nums[start]

"""
162. Find Peak Element
"""
def findPeakElement(self, nums: 'List[int]') -> 'int':
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    return left




"""
41. First Missing Positive
"""
def firstMissingPositive(self, nums: 'List[int]') -> 'int':
    for i in range(len(nums)):
        while 0 <= nums[i] - 1 < len(nums) and nums[nums[i] - 1] != nums[i]: # swap is swap current num to right place; however we may get a new valid number so we should swap it again until current is not valid   [3,4,-1,1]
            tmp = nums[i] - 1
            nums[i], nums[tmp] = nums[tmp], nums[i]
    for i in range(len(nums)):
        if nums[i] != i + 1:
            return i + 1
    return len(nums) + 1

"""
347. Top K Frequent Elements
"""
def topKFrequent(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: List[int]
    """
    counter = collections.Counter(nums)
    bucket = [[] for _ in range(len(nums) + 1)]
    result = []
    for key in counter.keys():
        bucket[counter[key]].append(key)
    for i in range(len(bucket)-1, -1, -1):
            result += bucket[i]
            if len(result) == k:
                break
    return result




"""
86. Partition List
"""
def partition(self, head: 'ListNode', x: 'int') -> 'ListNode':
    h1 = curr1 = ListNode(-1)
    h2 = curr2 = ListNode(-1)
    while head:
        if head.val < x:
            curr1.next = head
            curr1 = curr1.next
        else:
            curr2.next = head
            curr2 = curr2.next
        head = head.next
    curr2.next = None
    curr1.next = h2.next
    return h1.next