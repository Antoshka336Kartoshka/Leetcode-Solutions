from typing import Optional, List
from random import randint
import math
import collections


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __str__(self) -> str:
        s = ''
        cursor = self
        while cursor:
            s += f'{cursor.val} -> '
            cursor = cursor.next
            if cursor is self:
                s += f'{self.val}(self)'
                return s
        s += 'None'
        return s

    @staticmethod
    def list_to_node(l: list):
        if not l:
            return ListNode()
        l.reverse()
        for i, v in enumerate(l):
            l[i] = ListNode(val=v, next=l[i - 1] if i - 1 >= 0 else None)
        return l[-1]


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def _str(self):
        """Internal method for ASCII art."""
        label = str(self.val)
        if self.left is None:
            left_lines, left_pos, left_width = [], 0, 0
        else:
            left_lines, left_pos, left_width = self.left._str()
        if self.right is None:
            right_lines, right_pos, right_width = [], 0, 0
        else:
            right_lines, right_pos, right_width = self.right._str()
        middle = max(right_pos + left_width - left_pos + 1, len(label), 2)
        pos = left_pos + middle // 2
        width = left_pos + middle + right_width - right_pos
        while len(left_lines) < len(right_lines):
            left_lines.append(' ' * left_width)
        while len(right_lines) < len(left_lines):
            right_lines.append(' ' * right_width)
        if (middle - len(label)) % 2 == 1 and len(label) < middle:
            label += '.'
        label = label.center(middle, '.')
        if label[0] == '.':
            label = ' ' + label[1:]
        if label[-1] == '.':
            label = label[:-1] + ' '
        lines = [' ' * left_pos + label + ' ' * (right_width - right_pos),
                 ' ' * left_pos + '/' + ' ' * (middle - 2) +
                 '\\' + ' ' * (right_width - right_pos)] + \
            [left_line + ' ' * (width - left_width - right_width) + right_line
             for left_line, right_line in zip(left_lines, right_lines)]
        return lines, pos, width

    def __str__(self):
        return '\n'.join(self._str()[0])

    def __repr__(self):
        return f"TreeNode({self.val!r}, {self.left!r}, {self.right!r})"

    @staticmethod
    def creatBTree(data, index=0):
        pNode = None
        if index < len(data):
            if data[index] == None:
                return
            pNode = TreeNode(data[index])
            pNode.left = TreeNode.creatBTree(
                data, 2 * index + 1)  # [1, 3, 7, 15, ...]
            pNode.right = TreeNode.creatBTree(
                data, 2 * index + 2)  # [2, 5, 12, 25, ...]
        return pNode


class Solution:

    def __init__(self):
        self.score1 = 0
        self.score2 = 0

    def twoSum(self, nums: list, target: int) -> list:
        for num in range(len(nums)):
            if target - nums[num] in nums:
                first_index = num
                second_num = target - nums[num]
                second_index = nums.index(second_num)
                if first_index == second_index:
                    continue
                return [first_index, second_index]

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        current_node_l1 = l1
        current_node_l2 = l2
        s = next_num = 0
        previous_node = None
        first_node = None
        while current_node_l1 or current_node_l2 or next_num:
            s = (current_node_l1.val if current_node_l1 else 0) + \
                (current_node_l2.val if current_node_l2 else 0)
            print('S = ', s, 'N_N = ', next_num)
            if next_num:
                s += next_num
                next_num = 0
            if s >= 10:
                next_num += s // 10
                al = ListNode(s % 10)
            else:
                al = ListNode(s)
            if not previous_node:
                previous_node = al
                first_node = al
            else:
                previous_node.next = al
                previous_node = al
            s = 0
            if current_node_l1 and current_node_l1.next:
                current_node_l1 = current_node_l1.next
            else:
                current_node_l1 = None
            if current_node_l2 and current_node_l2.next:
                current_node_l2 = current_node_l2.next
            else:
                current_node_l2 = None
        return first_node

    def lengthOfLongestSubstring(self, s: str) -> int:
        maxword = []
        maxlength = 0
        for i in s:
            if i not in maxword:
                maxword.append(i)
            else:
                maxword.append(i)
                maxword = maxword[maxword.index(i) + 1:]
            if len(maxword) > maxlength:
                maxlength = len(maxword)
        print(maxword)
        print(maxlength)
        return maxlength

    def findMedianSortedArrays(self, nums1: list, nums2: list) -> float:
        n = sorted(nums1 + nums2)
        if len(n) % 2 != 0:
            return n[len(n) // 2]
        else:
            return (n[len(n) // 2] + n[(len(n) // 2) - 1]) / 2

    def find_indexes(self, s: str):
        unique = set(s)
        indexes = {}
        for u in unique:
            indexes[u] = []
        for i in range(len(s)):
            indexes[s[i]].append(i)
        return indexes

    def longestPalindrome(self, s: str) -> str:
        indexes = self.find_indexes(s)
        maxpalidrom = ''
        for key in indexes:
            for i in range(len(indexes[key])):
                index = indexes[key][i]
                otherpart = indexes[key][i + 1:]
                if otherpart:
                    for o in otherpart:
                        word = s[index:o + 1]
                        if word == word[::-1] and len(word) > len(maxpalidrom):
                            maxpalidrom = word
        if maxpalidrom == '' and s:
            return s[0]
        else:
            return maxpalidrom

    def FasterPalidrom(s: str) -> str:
        """
        good solution(not mine)
        :return:
        """
        m = ''  # Memory to remember a palindrome
        for i in range(len(s)):  # i = start, O = n
            for j in range(len(s), i, -1):  # j = end, O = n^2
                print(i, j)
                if len(m) >= j - i:  # To reduce time
                    break
                elif s[i:j] == s[i:j][::-1]:
                    m = s[i:j]
                    break
        return m

    def reverse(self, x: int) -> int:
        s = str(x)
        if s[0] == '-':
            s = '-' + s[::-1][:-1]
        else:
            s = s[::-1]
        x = int(s)
        if x > 2147483647 or x < -2147483647:
            return 0
        return x

    def myAtoi(self, string: str) -> int:
        min_int = -2 ** 31
        max_int = 2 ** 31 - 1
        string = string.lstrip()
        s = ''
        count = 0
        for i in string:
            if i.isalpha() or i in '. ' or i in '+-' and count > 0:
                break
            s += i
            count += 1
        if not s or s in '+-':
            s = '0'
        int_string = int(s)
        if int_string not in range(min_int, max_int):
            return max_int if int_string >= max_int else min_int
        return int_string

    def isPalindrome(self, x: int) -> bool:
        x = str(x)
        if x == x[::-1]:
            return True
        else:
            return False

    def isMatch(self, s: str, p: str) -> bool:
        pass

    def maxArea(self, height: list) -> int:
        maxarea = 0
        for i in range(len(height)):
            iteration = 0
            for j in range(i + 1, len(height)):
                iteration += 1
                area = min(height[i], height[j]) * iteration
                # print('AREA=', area, 'height[i]=', height[i], 'height[j]=', height[j], 'iter=', iteration)
                if maxarea < area:
                    maxarea = area
        return maxarea

    def maxArea2(self, height: list) -> int:
        l = 0
        r = len(height) - 1
        maxarea = 0
        while l < r:
            area = min(height[l], height[r]) * (r - l)
            if maxarea < area:
                maxarea = area
            if height[l] > height[r]:
                r -= 1
            else:
                l += 1
        return maxarea

    def intToRoman(self, num):
        values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        numerals = ["M", "CM", "D", "CD", "C", "XC",
                    "L", "XL", "X", "IX", "V", "IV", "I"]
        answer = ''
        for n, v in enumerate(values):
            print(v, '--', num // v)
            answer += (num // v) * numerals[n]
            num = num % v
        return answer

    def longestCommonPrefix(self, strs: list) -> str:
        if not strs:
            return ''
        if len(strs) == 1:
            return strs[0]
        currentword = strs[0]
        answer = ''
        for nextword in strs[1:]:
            common = ''
            count = 0
            # print(currentword, nextword)
            try:
                while currentword[count] == nextword[count]:
                    # print(currentword[count], nextword[count])
                    common += currentword[count]
                    count += 1

            except IndexError:
                pass
            answer = common
            if answer == '':
                return ''
            currentword = common
        return answer

    def longestCommonPrefixNotMine(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ""
        shortest = min(strs, key=len)
        print(strs)
        print(shortest)
        for i, ch in enumerate(shortest):
            for other in strs:
                if other[i] != ch:
                    return shortest[:i]
        return shortest

    def threeSum(self, nums: list) -> list:
        print(f'Initial - {nums}')
        result = set()
        for a in range(len(nums)):
            for b in range(a + 1, len(nums)):
                # c = 0 - (a + b); a + b + c = 0 -> a + b = -c
                c = 0 - (nums[a] + nums[b])
                if c in nums:
                    ci = nums.index(c)
                    if ci != a and ci != b:
                        flist = [nums[a], nums[b], c]
                        flist.sort()
                        result.add(tuple(flist))
        return list(result)

    def letterCombinations(self, digits):
        mapping = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
                   '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        if len(digits) == 0:
            return []
        if len(digits) == 1:
            return list(mapping[digits[0]])
        prev = self.letterCombinations(digits[:-1])
        additional = mapping[digits[-1]]
        return [s + c for s in prev for c in additional]

    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        cursor = head
        l = []
        count = 0
        while True:
            count += 1
            l.append(cursor)
            if cursor.next:
                cursor = cursor.next
            else:
                break
        if count > n:
            l[-(n + 1)].next = l[-(n + 1)].next.next
            return head
        return head.next

    def isValid(self, s: str) -> bool:
        brackets = {']': '[', '}': '{', ')': '('}
        stack = []
        for i in s:
            if i in brackets.values():
                stack.append(i)
            elif i in brackets.keys():
                if stack == [] or brackets[i] != stack.pop():
                    return False
            else:
                return False
        return stack == []

    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1 and not l2:
            return l1

        def list_of_node(l: ListNode) -> list:
            lnode = []
            cursor = l
            while cursor:
                if cursor:
                    lnode.append(cursor.val)
                cursor = cursor.next
            return lnode

        l = list(reversed(sorted(list_of_node(l1) + list_of_node(l2))))
        print(l)
        answer = []
        for node in l:
            if not answer:
                answer.append(ListNode(node))
            else:
                answer.append(ListNode(node, next=answer[len(answer) - 1]))
        return answer[-1]

    def generateParenthesis(self, n: int) -> list:  # Rebuild this
        gen_list = []

        def generate_str(N: str, M: int, prefix=None):
            if M == 0:
                s = ''.join(prefix)
                if self.isValid(s):
                    gen_list.append(s)
                return
            prefix = prefix or []
            for digit in N:
                prefix.append(digit)
                generate_str(N, M - 1, prefix)
                prefix.pop()

        generate_str('()', n * 2)
        return gen_list

    def mergeKLists(self, lists: list) -> ListNode:
        if not lists:
            return ListNode().next

        def list_of_node(l: ListNode) -> list:
            lnode = []
            cursor = l
            while cursor:
                if cursor:
                    lnode.append(cursor.val)
                cursor = cursor.next
            return lnode

        l = []
        for node in lists:
            l += list_of_node(node)
        l = list(reversed(sorted(l)))
        print(l)
        answer = []
        for node in l:
            if not answer:
                answer.append(ListNode(node))
            else:
                answer.append(ListNode(node, next=answer[len(answer) - 1]))
        if not answer:
            return ListNode().next
        return answer[-1]

    def swapPairs(self, head: ListNode) -> ListNode:
        if not head:
            return ListNode().next
        if not head.next:
            return head
        first = head
        second = first.next
        answer = second
        previous = None
        next_node = second.next  # 3
        while first.next:
            second.next = first  # 2 -> 1
            first.next = next_node  # 1 -> 3
            if previous:
                previous.next = second
            previous = first
            if not next_node:
                break
            first = next_node  # first = 3
            second = next_node.next if next_node.next else second  # second = 4
            next_node = second.next  # next_node = None
        return answer

    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        def list_of_node(head, k):
            lnode = []
            cursor = head
            tl = []
            count = 0
            while cursor:
                if count == k:
                    count = 0
                    lnode += tl[::-1]
                    tl = []
                tl.append(cursor)
                count += 1
                cursor = cursor.next
            else:
                lnode += tl[::-1] if count == k else tl
            return lnode

        lnode = list_of_node(head, k)
        cursor = lnode[0]
        for node in lnode[1:]:
            cursor.next = node
            cursor = node
        else:
            cursor.next = None
        return lnode[0]

    def removeDuplicates(self, nums: list) -> int:
        for num in nums[::-1]:
            if nums.count(num) > 1:
                nums.remove(num)
        return len(nums)

    def removeElement(self, nums: list, val: int) -> int:
        for num in nums[::-1]:
            if val == num:
                nums.remove(num)
        return len(nums)

    def strStr(self, haystack: str, needle: str) -> int:
        return haystack.index(needle) if needle in haystack else -1

    def reverseWords(self, s: str) -> str:
        return ' '.join(reversed(s.split()))

    def findSubstring(self, s: str, words: list) -> list:
        if not s or not words:
            return []

        def get_list(s: str, size: int):
            word = ''
            l = []
            for alpha in s:
                word += alpha
                if len(word) == size:
                    l.append(word)
                    word = ''
            return l

        lenword = len(words[0])
        swords = ''.join(words)
        start = 0
        end = len(swords)
        answer = []
        while end <= len(s):
            if sorted(s[start:end]) == sorted(swords) and sorted(get_list(s[start:end], lenword)) == sorted(words):
                answer.append(start)
            start += 1
            end += 1
        return answer

    def myPow(self, x: float, n: int) -> float:
        return x ** n

    def longestValidParentheses(self, s: str) -> int:
        stack = []
        maxlength = 0
        stack.append(-1)
        for i, v in enumerate(s):
            if v == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    maxlength = max(maxlength, i - stack[-1])
        return maxlength

    def topKFrequent(self, nums: list, k: int) -> list:
        answer = []
        for i in set(nums):
            answer.append((nums.count(i), i))
        answer.sort()
        print(answer)
        return [i[1] for i in answer[-k:]]

    def removeElements(self, head: ListNode, val: int) -> ListNode:
        cursor = head
        dummy = ListNode(0, cursor)
        previous = dummy
        while cursor:
            if cursor.val == val:
                previous.next = cursor.next
            else:
                previous = cursor
            cursor = cursor.next
        return dummy.next

    def exist(self, board: list, word: str) -> bool:
        def find(board, word, row, col, i=0):
            if i == len(word):
                return True
            if row >= len(board) or row < 0 or col >= len(board[0]) or col < 0 or word[i] != board[row][col]:
                return False
            board[row][col] = '*'
            res = find(board, word, row + 1, col, i + 1) or find(board, word, row - 1, col, i + 1) \
                or find(board, word, row, col + 1, i + 1) or find(board, word, row, col - 1, i + 1)
            board[row][col] = word[i]
            return res

        n = len(board)
        m = len(board[0])
        for i in range(n):
            for j in range(m):
                if word[0] == board[i][j]:
                    if find(board, word, i, j):
                        return True
        return False

    def zigzagLevelOrder(self, root: TreeNode) -> list:
        if root is None:
            return None
        res = [[]]
        level = 0

        def zigzaghelper(root: TreeNode, level: int, res: list):
            if root is None:
                return None
            if len(res) < level + 1:
                res.append([])
            if level % 2 != 0:
                res[level].append(root.val)
            else:
                res[level].insert(0, root.val)
            zigzaghelper(root.right, level + 1, res)
            zigzaghelper(root.left, level + 1, res)

        zigzaghelper(root, level, res)
        return res

    def allPathsSourceTarget(self, graph: list) -> list:
        last_node = len(graph) - 1
        paths = [[0]]
        ans = []
        while paths:
            print(f'paths={paths}')
            path = paths.pop()
            print(f'path={path}, graph[path[-1]]={graph[path[-1]]}')
            for node in graph[path[-1]]:
                print(f'node={node}')
                if node == last_node:
                    ans.append(path + [node])
                    print(f'ans={ans}')
                else:
                    paths.append(path + [node])
                    print(f'paths.append={paths}')
            print('-' * 100)
        return ans

    def addDigits(self, num: int) -> int:
        l = list(str(num))
        l = list(map(int, l))
        num = sum(l)
        if len(str(num)) > 1:
            num = self.addDigits(num)
        return num

    def search(self, nums: list, target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            if nums[l] <= nums[mid]:  # sorted
                if target < nums[l] or target > nums[mid]:
                    l = mid + 1
                else:
                    r = mid - 1
            else:  # not sorted
                if target > nums[r] or target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
        return -1

    def searchRange(self, nums: list, target: int) -> list:
        def find_left(nums, target):
            index = -1
            l, r = 0, len(nums) - 1
            while l <= r:
                mid = (l + r) // 2
                if target <= nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
                if nums[mid] == target:
                    index = mid
            return index

        def find_right(nums, target):
            index = -1
            l, r = 0, len(nums) - 1
            while l <= r:
                mid = (l + r) // 2
                if target >= nums[mid]:
                    l = mid + 1
                else:
                    r = mid - 1
                if nums[mid] == target:
                    index = mid
            return index

        left, right = find_left(nums, target), find_right(nums, target)
        return [left, right]

    def searchInsert(self, nums: list, target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            if target < nums[mid]:
                r = mid - 1
            else:
                l = mid + 1
        return l

    def isValidSudoku(self, board: list) -> bool:
        def validate(l):
            l = list(filter(lambda d: d.isdigit(), l))
            return True if len(set(l)) == len(l) else False

        left = []
        mid = []
        right = []
        for i, line in enumerate(board):
            if not validate(line):
                return False
            left += board[i][:3]
            mid += board[i][3:6]
            right += board[i][6:9]
            if len(left) == 9:
                if not validate(left) or not validate(mid) or not validate(right):
                    return False
                left = []
                mid = []
                right = []
        for line in list(zip(*board)):
            if not validate(line):
                return False
        return True

    def printSudoku(self, board):
        numrow = 0
        for row in board:
            if numrow % 3 == 0 and numrow != 0:
                print(' ')
            print(row[0:3], ' ', row[3:6], ' ', row[6:9])
            numrow += 1
        return

    def solveSudoku(self, board: list) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        backtracks = 0

        def findNextCellToFill(board):
            for i in range(0, 9):
                for j in range(0, 9):
                    if board[i][j] == '.':
                        return i, j
            return -1, -1

        def isValid(board, i, j, e):
            rowOK = all([e != board[i][x] for x in range(0, 9)])
            if rowOK:
                columnOK = all([e != board[x][j] for x in range(0, 9)])
                if columnOK:
                    sectionTopX, sectionTopY = 3 * (i // 3), 3 * (j // 3)
                    for x in range(sectionTopX, sectionTopX + 3):
                        for y in range(sectionTopY, sectionTopY + 3):
                            if board[x][y] == e:
                                return False
                    return True
            return False

        def solver(board, i=0, j=0):
            nonlocal backtracks
            i, j = findNextCellToFill(board)
            if i == -1:
                return True
            for e in range(1, 10):
                if isValid(board, i, j, str(e)):
                    board[i][j] = str(e)
                    if solver(board, i, j):
                        return True

                    backtracks += 1
                    board[i][j] = '.'
            return False

        solver(board)
        self.printSudoku(board)
        print(backtracks)

    def combinationSum(self, candidates: list, target: int) -> list:  # Check
        solution_set = []
        candidates = sorted(candidates)

        def dfs(remain, stack):
            if remain == 0:
                solution_set.append(stack)
                return
            for candidate in candidates:
                if candidate > remain:
                    break
                if stack and candidate < stack[-1]:
                    continue
                else:
                    dfs(remain - candidate, stack + [candidate])

        dfs(target, [])
        return solution_set

    def firstMissingPositive(self, nums: list) -> int:
        m = max(nums) if nums else 1
        if m < 0:
            m = 1
        for n in range(1, m + 2):
            if n not in nums:
                return n

    def trap(self, height: list) -> int:
        if not height or len(height) < 3:
            return 0
        volume = 0
        left, right = 0, len(height) - 1
        l_max, r_max = height[left], height[right]
        while left < right:
            l_max, r_max = max(height[left], l_max), max(height[right], r_max)
            if l_max <= r_max:
                volume += l_max - height[left]
                left += 1
            else:
                volume += r_max - height[right]
                right -= 1
        return volume

    def subarraySum(self, nums: list, k: int) -> int:
        count = 0
        for i, vi in enumerate(nums):
            sum = 0
            for j, vj in enumerate(nums[i:]):
                sum += vj
                if sum == k:
                    count += 1
        return count

    def permute(self, nums: list) -> list:
        answer = []

        def _inner(nums: list, N=None, prefix=None):
            nonlocal answer
            N = len(nums) if N is None else N
            prefix = prefix or []
            if N == 0:
                answer.append(list(prefix))
                return
            for n in nums:
                if n not in prefix:
                    prefix.append(n)
                else:
                    continue
                _inner(nums, N - 1, prefix)
                prefix.pop()

        _inner(nums)
        return answer

    def permuteUnique(self, nums: list) -> list:
        answer = []

        def _inner(nums: list, checked=None, prefix=None):
            nonlocal answer
            prefix = prefix or []
            checked = checked or []
            if len(checked) == len(nums) and prefix not in answer:
                answer.append(list(prefix))
                return
            for i, v in enumerate(nums):
                if i in checked:
                    continue
                prefix.append(v)
                checked.append(i)
                _inner(nums, checked, prefix)
                checked.pop()
                prefix.pop()

        _inner(nums)
        return answer

    def rotate(self, matrix: list) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        l = len(matrix)
        for i in range(len(matrix) // 2):
            for j in range(i, len(matrix[i]) - i - 1):
                temporary = matrix[i][j]
                matrix[i][j] = matrix[l - 1 - j][i]
                matrix[l - 1 - j][i] = matrix[l - 1 - i][l - 1 - j]
                matrix[l - 1 - i][l - 1 - j] = matrix[j][l - 1 - i]
                matrix[j][l - 1 - i] = temporary

    def groupAnagrams(self, strs: list) -> list:
        d = {}
        for w in sorted(strs):
            key = tuple(sorted(w))
            d[key] = d.get(key, []) + [w]
        return list(d.values())

    def maxSubArray(self, nums: list) -> int:
        def first(nums):
            for i in range(1, len(nums)):
                if nums[i - 1] > 0:
                    nums[i] += nums[i - 1]
            return max(nums)

        def second(nums):
            curSum = maxSum = nums[0]
            for num in nums[1:]:
                curSum = max(num, curSum + num)
                maxSum = max(maxSum, curSum)
            return maxSum

        return first(nums)

    def spiralOrder(self, matrix: list) -> list:
        if not matrix:
            return []
        path = []
        rowBegin = 0
        rowEnd = len(matrix) - 1
        colBegin = 0
        colEnd = len(matrix[0]) - 1

        while rowBegin <= rowEnd and colBegin <= colEnd:
            for j in range(colBegin, colEnd + 1, 1):
                path.append(matrix[rowBegin][j])
            rowBegin += 1
            for i in range(rowBegin, rowEnd + 1, 1):
                path.append(matrix[i][colEnd])
            colEnd -= 1
            if rowBegin <= rowEnd:
                for j in range(colEnd, colBegin - 1, -1):
                    path.append(matrix[rowEnd][j])
            rowEnd -= 1
            if colBegin <= colEnd:
                for i in range(rowEnd, rowBegin - 1, -1):
                    path.append(matrix[i][colBegin])
            colBegin += 1

        return path

    def fourSum(self, nums: list, target: int) -> list:
        d = dict()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                sum2 = nums[i] + nums[j]
                if sum2 in d:
                    d[sum2].append((i, j))
                else:
                    d[sum2] = [(i, j)]
        result = set()
        for key in d:
            value = target - key
            if value in d:
                list1 = d[key]
                list2 = d[value]
                for (i, j) in list1:
                    for (k, l) in list2:
                        if i != k and i != l and j != k and j != l:
                            flist = [nums[i], nums[j], nums[k], nums[l]]
                            flist.sort()
                            result.add(tuple(flist))
        return list(result)

    def sortColors(self, nums: list) -> None:
        counts = [0] * 3
        for num in nums:
            counts[num] += 1
        index = 0
        for color, count in enumerate(counts):
            for i in range(count):
                nums[index] = color
                index += 1

    def largestNumber(self, nums: list) -> str:
        def _quicksort(nums: list) -> list:
            if len(nums) <= 1:  # базовый случай (массив с 1 или 0 элементов)
                return nums
            else:  # рекурсивный случай
                pivot = nums[
                    0]  # опорный элемент (для ускорения работы лучше выбирать рандомный элемент вместо первого)
                less = [i for i in nums[1:] if str(
                    pivot) + str(i) > str(i) + str(pivot)]
                greater = [i for i in nums[1:] if str(
                    pivot) + str(i) <= str(i) + str(pivot)]
                return _quicksort(less) + [pivot] + _quicksort(greater)

        result = ''.join(map(str, list(reversed(_quicksort(nums)))))
        return result if set(result) != {'0'} else '0'

    def insertionSortList(self, head: ListNode) -> ListNode:
        dummy = ListNode(0)
        current = dummy.next = head
        while current:
            print(current.val, end='->')
            current = current.next

    def romanToInt(self, s: str) -> int:
        roman_numerals = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }
        s = s.replace('IV', 'IIII')
        s = s.replace('IX', 'VIIII')
        s = s.replace('XL', 'XXXX')
        s = s.replace('XC', 'LXXXX')
        s = s.replace('CD', 'CCCC')
        s = s.replace('CM', 'DCCCC')
        count = 0
        print(s)
        for i in s:
            count += roman_numerals[i]
        return count

    @staticmethod
    def guess(n: int) -> int:
        my_number = randint(1, 10000)
        if n < my_number:
            return 1
        if n > my_number:
            return -1
        if n == my_number:
            return 0

    def guessNumber(self, n: int) -> int:
        l, r = 0, n
        while l <= r:
            mid = l + (r - l) // 2
            if self.guess(mid) == 0:
                return mid
            elif self.guess(mid) == 1:
                l = mid + 1
            else:
                r = mid - 1

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:  # проверка на пустое значение
            return head
        elif not head.next:
            return head
        cursor = head.next
        head.next = None
        last_node = self.reverseList(cursor)
        cursor.next = head
        return last_node

    def lengthOfLastWord(self, s: str) -> int:
        return len(s.split()[-1])

    def plusOne(self, digits: List[int]) -> List[int]:
        # Bad way of coding
        return list(map(int, list(str(int(''.join(map(str, digits))) + 1))))

    def mySqrt(self, x: int) -> int:  # Not done
        l, r = 0, x
        while l <= r:
            mid = l + (r - l) // 2
            if mid * mid <= x < (mid + 1) * (mid + 1):
                return mid
            elif x < mid * mid:
                r = mid - 1
            else:
                l = mid + 1

    def climbStairs(self, n: int) -> int:
        """
        You are climbing a staircase. It takes n steps to reach the top.
        Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
        :param n:
        :return:
        """
        fib1, fib2 = 1, 1
        for _ in range(n):
            fib2, fib1 = fib1 + fib2, fib2
        return fib1

    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cursor = head
        while cursor and cursor.next:
            if cursor.val == cursor.next.val:
                cursor.next = cursor.next.next
            else:
                cursor = cursor.next
        return head

    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        O(m + n)
        """
        while n > 0:
            if m <= 0 or nums2[n - 1] >= nums1[m - 1]:
                nums1[m + n - 1] = nums2[n - 1]
                n -= 1
            else:
                nums1[m + n - 1] = nums1[m - 1]
                m -= 1

    def singleNumber(self, nums: List[int]) -> int:
        """
        Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.
        You must implement a solution with a linear runtime complexity and use only constant extra space.
        """
        res = 0
        for num in nums:
            res ^= num
        return res

    # faster than 99.73%
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        """
        Given an array nums of n integers where nums[i] is in the range [1, n],
        return an array of all the integers in the range [1, n] that do not appear in nums.
        """
        return list(set(list(range(1, len(nums) + 1))) - set(nums))

    def hammingDistance(self, x: int, y: int) -> int:
        """
        The Hamming distance between two integers is the number of positions at which the corresponding
        bits are different.Given two integers x and y, return the Hamming distance between them.
        """
        xb, yb = f'{x:032b}', f'{y:032b}'
        return sum(i != j for i, j in zip(xb, yb))

    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        """
        Given the roots of two binary trees p and q, write a function to check if they are the same or not.
        Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.
        """
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

    def numIslands(self, grid: List[List[str]]) -> int:
        """
        Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water),
        return the number of islands. An island is surrounded by water and is formed by connecting
        adjacent lands horizontally or vertically. You may assume all four edges of the grid are all
        surrounded by water.
        """

        def dfs(grid, i, j):
            if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[i]) or grid[i][j] == '0':
                return

            grid[i][j] = '0'
            dfs(grid, i - 1, j)  # up
            dfs(grid, i + 1, j)  # down
            dfs(grid, i, j - 1)  # left
            dfs(grid, i, j + 1)  # right

        res = 0
        for i, iv in enumerate(grid):
            for j, jv in enumerate(grid[i]):
                if grid[i][j] == '1':
                    res += 1
                    dfs(grid, i, j)
        return res

    def maxDepth(self, root: Optional[TreeNode], res=0) -> int:
        """
        Given the root of a binary tree, return its maximum depth. A binary tree's maximum depth is the number
        of nodes along the longest path from the root node down to the farthest leaf node.
        """
        if not root:
            return res
        return max(self.maxDepth(root.left, res + 1), self.maxDepth(root.right, res + 1))

    def minDepth(self, root: Optional[TreeNode]) -> int:
        """
        Given a binary tree, find its minimum depth. The minimum depth is the number of nodes along
        the shortest path from the root node down to the nearest leaf node.
        Note: A leaf is a node with no children.
        """
        if not root:
            return 0
        queue = collections.deque([(root, 1)])  # element with it's depth level
        while queue:
            node, level = queue.popleft()
            if node:
                if not node.left and not node.right:
                    return level
                else:
                    queue.append((node.left, level + 1))
                    queue.append((node.right, level + 1))

    def fib(self, n: int) -> int:
        """
        The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci sequence,
        such that each number is the sum of the two preceding ones, starting from 0 and 1. That is,
        F(0) = 0, F(1) = 1
        F(n) = F(n - 1) + F(n - 2), for n > 1.
        Given n, calculate F(n).
        """
        if n > 1:
            return self.fib(n - 1) + self.fib(n - 2)
        elif n == 1:
            return 1
        else:
            return 0

    def reverseString(self, s: List[str]) -> None:  # O(1)
        """
        Write a function that reverses a string. The input string is given as an array of characters s.
        You must do this by modifying the input array in-place with O(1) extra memory.
        """
        lcur = 0
        rcur = len(s) - 1
        while lcur <= rcur:
            s[lcur], s[rcur] = s[rcur], s[lcur]
            lcur += 1
            rcur -= 1

    def isPowerOfFour(self, n: int) -> bool:
        """
        Given an integer n, return true if it is a power of four. Otherwise, return false.
        An integer n is a power of four, if there exists an integer x such that n == 4^x.
        """
        return n > 0 and math.log(n, 4).is_integer()

    def findPeakElement(self, nums: List[int]) -> int:
        """
        A peak element is an element that is strictly greater than its neighbors.
        Given an integer array nums, find a peak element, and return its index.
        If the array contains multiple peaks, return the index to any of the peaks.
        You may imagine that nums[-1] = nums[n] = -∞.
        You must write an algorithm that runs in O(log n) time.
        """
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < nums[mid + 1]:
                left = mid + 1  # go to the right
            else:
                right = mid  # go to the left
        return left

    def findPeakGrid(self, mat: List[List[int]]) -> List[int]:
        """
        A peak element in a 2D grid is an element that is strictly greater than all of its
        adjacent neighbors to the left, right, top, and bottom. Given a 0-indexed m x n matrix mat
        where no two adjacent cells are equal, find any peak element mat[i][j] and return the length 2 array [i,j].
        You may assume that the entire matrix is surrounded by an outer perimeter with the value -1 in each cell.
        You must write an algorithm that runs in O(m log(n)) or O(n log(m)) time.
        Attempt
        • Pick middle column j = m/2
        • Find global maximum on column j at (i, j)
        • Compare (i, j − 1),(i, j),(i, j + 1)
        • Pick left columns of (i, j − 1) > (i, j)
        • Similarly for right
        • (i, j) is a 2D-peak if neither condition holds ← WHY?
        • Solve the new problem with half the number of columns.
        • When you have a single column, find global maximum and you‘re done.
        """

        # if do binary search here it would work?
        def find_max_at_column(mat, column):
            m = i = j = 0
            for row in range(len(mat)):
                if mat[row][column] > m:
                    m = mat[row][column]
                    i, j = row, column
            return i, j

        midcolumn = len(mat[0]) // 2
        i, j = find_max_at_column(mat, midcolumn)
        while mat:
            if j > 0 and mat[i][j - 1] > mat[i][j]:
                j -= 1
            elif j < len(mat[i]) - 1 and mat[i][j + 1] > mat[i][j]:
                j += 1
            else:
                i, j = find_max_at_column(mat, j)
                return [i, j]

    # Merge sort (could be quicksort too!)
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Given the head of a linked list, return the list after sorting it in ascending order.
        """

        def merge(left, right):
            temp = dummy = ListNode()
            while left and right:
                if left.val < right.val:
                    temp.next = left
                    left = left.next
                else:
                    temp.next = right
                    right = right.next
                temp = temp.next
            if left:
                temp.next = left
            if right:
                temp.next = right
            return dummy.next

        def getMid(head):
            """
            return mid element of a ListNode
            """
            slow, fast = head, head.next
            while fast and fast.next:
                slow, fast = slow.next, fast.next.next
            return slow

        if not head or not head.next:
            return head

        # Divide list in a half
        left = head
        right = getMid(head)
        temp = right.next
        right.next = None
        right = temp

        left = self.sortList(left)
        right = self.sortList(right)
        return merge(left, right)

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """
        Given the root of a binary tree, check whether it is a mirror of itself
        (i.e., symmetric around its center).
        """

        def isSymetricNodes(l, r):
            if not l and not r:
                return True
            if not l or not r:
                return False
            if l.val != r.val:
                return False
            return isSymetricNodes(l.left, r.right) and isSymetricNodes(l.right, r.left)

        return isSymetricNodes(root.left, root.right)

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Given the root of a binary tree, return the level order traversal of its nodes' values.
        (i.e., from left to right, level by level).
        """
        queue = collections.deque([(root, 0)])
        answer = []
        while queue:
            node, level = queue.popleft()
            if node:
                if level > len(answer) - 1:
                    answer.append([node.val])
                else:
                    answer[level].append(node.val)
                if node.left:
                    queue.append((node.left, level + 1))
                if node.right:
                    queue.append((node.right, level + 1))
        return answer

    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Given the root of a binary tree, return the bottom-up level order traversal of its nodes' values.
        (i.e., from left to right, level by level from leaf to root).
        """
        queue = collections.deque([(root, 0)])
        answer = []
        while queue:
            node, level = queue.popleft()
            if node:
                if level > len(answer) - 1:
                    answer.append([node.val])
                else:
                    answer[level].append(node.val)
                if node.left:
                    queue.append((node.left, level + 1))
                if node.right:
                    queue.append((node.right, level + 1))
        return list(reversed(answer))

    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        """
        Given the root of a binary tree, return the average value of the nodes on each level in the form of an array.
        Answers within 10-5 of the actual answer will be accepted.
        """
        queue = collections.deque([(root, 0)])
        answer = []
        while queue:
            node, level = queue.popleft()
            if node:
                if level > len(answer) - 1:
                    answer.append([node.val])
                else:
                    answer[level].append(node.val)
                if node.left:
                    queue.append((node.left, level + 1))
                if node.right:
                    queue.append((node.right, level + 1))
        for i, v in enumerate(answer):
            answer[i] = sum(v) / len(v)
        return answer

    def solve(self, board: List[List[str]]) -> None:
        """
        Given an m x n matrix board containing 'X' and 'O', capture all regions
        that are 4-directionally surrounded by 'X'.
        A region is captured by flipping all 'O's into 'X's in that surrounded region.
        """

        def dfs(board, i, j):
            if 0 <= i < len(board) and 0 <= j < len(board[i]) and board[i][j] == 'O':
                board[i][j] = 'N'
                dfs(board, i - 1, j)  # up
                dfs(board, i + 1, j)  # down
                dfs(board, i, j - 1)  # left
                dfs(board, i, j + 1)  # right

        for i in range(len(board) - 1):
            dfs(board, i, 0)
            dfs(board, i, len(board[i]) - 1)
        for j in range(len(board[0])):
            dfs(board, 0, j)
            dfs(board, len(board) - 1, j)
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                if board[i][j] == 'N':
                    board[i][j] = 'O'

    def uniquePaths(self, m: int, n: int) -> int:
        """
        A robot is located at the top-left corner of a m x n grid.
        The robot can only move either down or right at any point in time. The robot is trying to reach
        the bottom-right corner of the grid.
        How many possible unique paths are there?
        m - rows
        n - columns
        """

        grid = [[1] * n] * m
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] = grid[i - 1][j] + grid[i][j - 1]
        return grid[m - 1][n - 1]

    def repeatedSubstringPattern(self, s: str) -> bool:
        """
        Given a string s, check if it can be constructed by taking a substring
        of it and appending multiple copies of the substring together.
        Basic idea:
        1. First char of input string is first char of repeated substring
        2. Last char of input string is last char of repeated substring
        3. Let S1 = S + S (where S in input string)
        4. Remove 1 and last char of S1. Let this be S2
        5. If S exists in S2 then return true else false
        6. Let i be index in S2 where S starts then repeated substring length i + 1 and repeated substring S[0: i+1]
        """
        if not s:
            return False

        ss = (s + s)[1:-1]
        return ss.find(s) != -1

    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """
        Given the head of a linked list, rotate the list to the right by k places.
        """
        if not head:
            return None

        lastElement = head
        length = 1

        while lastElement.next:
            lastElement = lastElement.next
            length += 1

        k = k % length
        lastElement.next = head
        tempNode = head

        for _ in range(length - k - 1):
            tempNode = tempNode.next

        answer = tempNode.next
        tempNode.next = None
        return answer

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """
        Given the root of a binary tree, return the inorder traversal of its nodes' values.
        """

        def helper(root: Optional[TreeNode], res: list) -> List[int]:
            if root is None:
                return root
            helper(root.left, res)
            res.append(root.val)
            helper(root.right, res)
            return res
        return helper(root, [])

    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        """
        Given an integer array nums where the elements are sorted in ascending order,
        convert it to a height-balanced binary search tree.
        A height-balanced binary tree is a binary tree in which the depth of the
        two subtrees of every node never differs by more than one.
        """
        def helper(start, end, nums):
            if start < end:
                mid = (start + end) // 2
                root = TreeNode(nums[mid])
                root.left = helper(start, mid, nums)
                root.right = helper(mid + 1, end, nums)
                return root

        return helper(0, len(nums), nums)

    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        """
        Given the root of a binary tree and an integer targetSum, return true if the tree
        has a root-to-leaf path such that adding up all the values along the path equals targetSum.
        A leaf is a node with no children.
        """
        if root is None:
            return False
        if root.left is None and root.right is None and targetSum == root.val:
            return True
        return self.hasPathSum(root.left, targetSum - root.val) or self.hasPathSum(root.right, targetSum - root.val)


if __name__ == '__main__':
    '''

    '''
    s = Solution()
