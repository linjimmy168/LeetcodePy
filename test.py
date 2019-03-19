import heapq
class Solution:
    def findFrind(self, n, friendLists):
        self.parents = [i for i in range(n)]
        for i, fd in enumerate(friendLists):
            iParent = self.find(i)
            for f in fd:
                fParent = self.find(f)
                self.parents[fParent] = iParent

        numberOfGroup = set()
        for i in self.parents:
            numberOfGroup.add(i)
        return len(numberOfGroup)

    def find(self, node):
        while self.parents[node] != node:
            self.parents[node] = self.parents[self.parents[node]]
            node = self.parents[node]
        return node



if __name__ == '__main__':
    b = Solution()
    print(b.findFrind(10, [[0], [5,3,0], [8,4,0], [9, 0], [3, 0], [0], [7, 9, 0], [0], [9, 7, 0]]))

