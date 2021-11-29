class Solution(object):
    def multiply(self, A, B):
        """
        :type A: int
        :type B: int
        :rtype: int
        """
        def recursion(x, y):
            if x <= 0 or y <= 0:
                return 0
            if x > y:
                return x + recursion(x, y - 1)
            else:
                return y + recursion(x - 1, y)
        return recursion(A, B)

obj = Solution()
print(obj.multiply(2, 4))