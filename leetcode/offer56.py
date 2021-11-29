# -*- coding: UTF-8 -*-
import math
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        set1 = set([])
        set2 = set([])
        for item in nums:
            if item in set1:
                set2.add(item)
            if item not in set1:
                set1.add(item)
        # print set1
        # print set2
        return set1.difference(set2).pop()

# print 1 << 2 - 1
# print 1 << 1 - 1
# print 1 << 3 - 1
obj = Solution()
print obj.singleNumber([3,4,3,3])

