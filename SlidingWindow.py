"""
904. Fruit Into Baskets
"""
def totalFruit(self, tree):
    """
    :type tree: List[int]
    :rtype: int
    """
    dic = collections.defaultdict(int)
    maxLength, startIdx = 0, 0
    for idx, val in enumerate(tree):
        dic[val] += 1
        while len(dic) > 2:
            dic[tree[startIdx]] -= 1
            if dic[tree[startIdx]] == 0:
                del dic[tree[startIdx]]
            startIdx += 1
        maxLength = max(maxLength, idx - startIdx + 1)
    return maxLength

"""
56. Merge Intervals
"""
def merge(self, intervals):
    """
    :type intervals: List[Interval]
    :rtype: List[Interval]
    """
    result = []
    for interval in sorted(intervals, key=lambda i:i.start):
        if result and interval.start <= result[-1].end:
            result[-1].end = max(result[-1].end, interval.end)
        else:
            result.append(interval)
    return result

    
"""

"""
def maxSlidingWindow(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: List[int]
    """
    import collections
    que = collections.deque()
    result = []
    for i, v in enumerate(nums):
        while que and nums[que[-1]] < v: # maintain left side is biggest. It will be compared with the new adding value
            que.pop()
        que.append(i)
        if que[0] == i - k: #remove the biggest one because it is over the window
            que.popleft()
        if i-k+1 >= 0:  # the window is biger than k, = is for the first index
            result.append(nums[que[0]])
    return result


