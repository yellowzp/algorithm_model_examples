class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        candi = -1
        count = 0
        for item in nums:
            if count == 0:
                if item != candi:
                    candi = item
            if item == candi:
                count += 1
            else:
                count -= 1
            # print(candi, count)
        count = 0
        for item in nums:
            if item == candi:
                count += 1
        if count * 2 > len(nums):
            return candi
        return -1

obj = Solution()
print(obj.majorityElement([3,3,4]))