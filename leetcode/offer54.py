class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def kthLargest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        def tra(sortlist, root):
            if not root:
                return
            for i in xrange(k):
                if root.val > sortlist[i]:
                    sortlist.insert(i, root.val)
                    # print sortlist
                    break
            tra(sortlist, root.left)
            tra(sortlist, root.right)
        sortlist = [0] * k
        tra(sortlist, root)
        # print sortlist
        return sortlist[k - 1]


root = TreeNode(5)
root.left = TreeNode(3)
root.right = TreeNode(6)
root.left.left = TreeNode(2)
root.left.right = TreeNode(4)
root.left.left.left = TreeNode(1)
obj = Solution()
print obj.kthLargest(root, 3)