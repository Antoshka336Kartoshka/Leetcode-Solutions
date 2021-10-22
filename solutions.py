from typing import Optional, List
from random import randint


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
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
            s = (current_node_l1.val if current_node_l1 else 0) + (current_node_l2.val if current_node_l2 else 0)
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
        '''
        good solution(not mine)
        :return: 
        '''
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
        numerals = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
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
                c = 0 - (nums[a] + nums[b])  # c = 0 - (a + b); a + b + c = 0 -> a + b = -c
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

    def print_list_node(self, head: ListNode):
        cursor = head
        while cursor:
            print(cursor.val)
            cursor = cursor.next

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
        l = []
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
        columns = []
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
                less = [i for i in nums[1:] if str(pivot) + str(i) > str(i) + str(pivot)]
                greater = [i for i in nums[1:] if str(pivot) + str(i) <= str(i) + str(pivot)]
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
        s = s.replace('IV', 'IIII');
        s = s.replace('IX', 'VIIII')
        s = s.replace('XL', 'XXXX');
        s = s.replace('XC', 'LXXXX')
        s = s.replace('CD', 'CCCC');
        s = s.replace('CM', 'DCCCC')
        count = 0
        print(s)
        for i in s:
            count += roman_numerals[i]
        return count

    @staticmethod
    def guess(n: int) -> int:
        my_number = randint(1, 10000)
        if n < my_number: return 1
        if n > my_number: return -1
        if n == my_number: return 0

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
        return list(map(int, list(str(int(''.join(map(str, digits))) + 1))))  # Bad way of coding

    def mySqrt(self, x: int) -> int:  # Not done
        l, r = 0, x
        while l <= r:
            mid = l + (r-l)//2
            if mid * mid <= x < (mid+1)*(mid+1):
                return mid
            elif x < mid * mid:
                r = mid - 1
            else:
                l = mid + 1

    def climbStairs(self, n: int) -> int:
        '''
        You are climbing a staircase. It takes n steps to reach the top.
        Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
        :param n:
        :return:
        '''
        fib1, fib2 = 1, 1
        for _ in range(n):
            fib2, fib1 = fib1 + fib2, fib2
        return fib1






if __name__ == '__main__':

    s = Solution()

    print(s.climbStairs(1))
