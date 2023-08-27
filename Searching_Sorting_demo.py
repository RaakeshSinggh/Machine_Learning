#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
wt_list = [random.randint(30,70) for i in range(10)]


# In[2]:


wt_list


# In[3]:


sum(wt_list)


# In[4]:


len(wt_list)


# In[5]:


sum(wt_list)/len(wt_list)


# In[6]:


total_wt = 0
for w in wt_list:
    total_wt += w


# In[7]:


total_wt


# In[8]:


for w in wt_list:
    total_wt = 0
    total_wt += w


# In[9]:


total_wt


# In[10]:


wt_list[10]


# In[12]:


l = [random.randint(30,70) for  i in range(100)]


# In[13]:


no_to_be_searched = 42


# In[14]:


for i in l:
    if i == no_to_be_searched:
        print("Number is Present")
    else:
        print("Number is not Present")


# In[14]:


# Drawbacks of above code
# Very First Index should be returned.
# Execution should be stopped after first index is returned
# If element is not present, return -1


# In[16]:


found = 0
for idx, i in enumerate(l):
    if i == no_to_be_searched:
        print(l.index(i))
        found = 1
        break
    elif idx == len(l)-1 and found == 0:
        print(-1)
        
    


# In[17]:


no_to_be_searched = 80
found = 0
for idx, i in enumerate(l):
    if i == no_to_be_searched:
        print(l.index(i))
        found = 1
        break
    elif idx == len(l)-1 and found == 0:
        print(-1)


#  Linear Search

# In[19]:


def linear_search(l,n):
    for i in l:
        if i == n:
            return l.index(i)
    return -1


# In[20]:


linear_search(l,81)


# In[22]:


linear_search(l,42)


# In[23]:


l1 = [10,2,1,14]


# In[24]:


for idex,item in enumerate(l1):
    print(idex,item)


# In[25]:


for idex_item in enumerate(l1):
    print(idex_item)


#  Binary Search

# In[26]:


def binary_search(l,n):  # list should be sorted
    start_index = 0
    end_index = len(l)-1
    while  start_index <=  end_index:
        mid_index =  (start_index + end_index)//2
        if l[mid_index] == n:
            return mid_index
        elif l[mid_index] > n:
            end_index = mid_index - 1
        else:
            start_index = mid_index + 1
    return -1
        
            


# In[27]:


l_sorted = sorted(l)


# In[29]:


binary_search(l_sorted,81)


# In[31]:


binary_search(l_sorted,42)


# In[32]:


42 in l_sorted


# In[35]:


l_large = [random.randint(30,70) for i in range(10000)]
l_large_sorted = sorted(l_large)


# In[36]:


len(l_large)


# In[37]:


number_to_be_searched = [random.randint(30,70) for i in range(100)]


# In[38]:


import time
st_time = time.time()
for n in number_to_be_searched:
    o = linear_search(l_large,n)
print(time.time()-st_time)


# In[41]:


import time 
st_time = time.time()
for n in number_to_be_searched:
  o = linear_search(l_large_sorted, n)
print(time.time()-st_time)


# In[42]:


import time 
st_time = time.time()
for n in number_to_be_searched:
  o = binary_search(l_large_sorted, n)
print(time.time()-st_time)


# In[44]:


l.index(42)


# In[45]:


l.index(80)


# In[46]:


sorted(l, reverse = True)


# In[47]:


# selection sort
l = [random.randint(30,70) for i in range(10)]


# In[48]:


def selection_sort(l):
    size = len(l)
    for idx in range(size):
        min_idx = idx
        for j in range(idx+1, size):
            if l[j] < l[min_idx]:
                min_idx = j
        l[idx], l[min_idx] = l[min_idx], l[idx]
    return l
    


# In[49]:


l


# In[50]:


selection_sort(l)


# Recursion

# In[53]:


# factorial without recursion : fact(n) = n * (n-1) * (n-2) *.....*1
def fact_without_rec(n):
    mul = 1
    # while n > 1:
  #   mul *= n # mul = mul * n
  #   n -= 1   # n = n-1
    for i in range(2, n+1):
        mul = mul * i
    return mul


# In[55]:


fact_without_rec(7),720*7


# In[56]:


# Factorial with recursion : fact(n) = n * fact(n-1)
def fect_with_rec(n):
    if n <= 1:
        return 1
    else:
        return n * fect_with_rec(n-1)


# In[57]:


fect_with_rec(6)


# Merge Sort

# In[59]:


def merge(left_array, right_array):
    sorted_array = []
    size_left_array = len(left_array)
    size_right_array = len(right_array)
    i = 0
    j = 0
    while i < size_left_array and j < size_right_array:
        if left_array[i] < right_array[j]:
            sorted_array.append(left_array[i])
            i += 1
        else:
            sorted_array.append(right_array[j])
            j += 1
    if i < size_left_array:
        sorted_array.extend(left_array[i:])
    if j < size_right_array:
        sorted_array.extend(right_array[j:])
    return sorted_array


# In[60]:


def merge_sort(l):
    if len(l) <= 1:
        return l
    left_array = l[:len(l)//2]
    right_array = l[len(l)//2:]
    left_array = merge_sort(left_array)
    right_array = merge_sort(right_array)
    return merge(left_array,right_array)


# In[61]:


merge_sort([6, 5, 8, 9, 32, 21, 73, 3, 2, 1])


# In[62]:


help(sorted)


# Quick Sort

# In[65]:


def partition(l, s, e):
  i = s
  j = e - 1
  pivot = l[e]

  while i < j:
    while i < e and l[i] < pivot:
      i += 1
    while j > s and l[j] > pivot:
      j -= 1
    if i < j:
      l[i], l[j] = l[j], l[i]
  if l[i] > pivot:
    l[i], l[e] = l[e], l[i] 
  return i
            
        


# In[66]:


l = [2, 9, 8, 6, 5]
partition(l, 0, len(l)-1)


# In[67]:


def quick_sort(l,li,ri):
    if li < ri:
        position = partition(l,li,ri)
        quick_sort(l,li,position-1)
        quick_sort(l,position+1,ri)
    return l


# In[68]:


l = [2, 9, 8, 6, 5, 90, 32, 9, 30, 21]
quick_sort(l, 0, len(l)-1)


# Bubble Sort

# In[69]:


import random
l1 = [random.randint(10,80) for i in range(10)]


# In[70]:


l1


# In[71]:


def bubble_sort(l):
    size = len(l)
    for i in range(size-1):
        for j in range(0,size-i-1):
            if l[j] > l[j+1]:
                l[j],l[j+1] = l[j+1],l[j]
    return l


# In[72]:


l_new = bubble_sort(l1)


# In[73]:


l_new


# In[74]:


def bubble_sort_with_swap(l):
    size = len(l)
    for i in range(size-1):
        swap = 0
        for j in range(0,size-i-1):
            if l[j] > l[j+1]:
                swap = 1
                l[j],l[j+1] = l[j+1],l[j]
        if not swap:
            return l
    return l


# In[75]:


bubble_sort_with_swap([1, 2, 3, 4, 5])


# In[76]:


bubble_sort_with_swap([100, 21, 3, 4, 5])


# Insertion Sort

# In[77]:


def insertion_sort(l):
    size = len(l)
    for i in range(1,size):
        look_for = l[i]
        j = i-1
        while j >= 0:
            if l[j] > look_for:
                l[j],l[j+1] = l[j+1],l[j]
            else:
                break
            j -= 1
    return l
            
            


# In[78]:


insertion_sort([120, 110, 90, 100, 21, 3, 4, 5, 50])


#  Bucket Sort

# In[86]:


def bucket_sort(l,number_of_buckets):
    bucket = []
    for i in range(number_of_buckets):
        bucket.append([])
    for i in l:
        bucket_number = int(i*number_of_buckets)
        bucket[bucket_number].append(i)
    for b_index, b in enumerate(bucket):
        bucket[b_index] = insertion_sort(b)
        k = 0
        for i in range(number_of_buckets):
            for j in range(len(bucket[i])):
                l[k] = bucket[i][j]
                k += 1
    return l


# In[87]:


l = [.78, .17, .39, .26, .72, .94, .21, .12]


# In[88]:


bucket_sort(l, 10)


# In[89]:


bucket_sort(l, 10)


#  Binary Search Tree

# In[102]:


class BST:
  def __init__(self, data): # data or key
    self.data = data
    self.left = None
    self.right = None

  def insert(self, data):
    if self.data is None:
      self.data = data
      return
    if self.data >= data: # Root is found
      if self.left:
        self.left.insert(data)
      else:
        self.left = BST(data)
    else:
      if self.right:
        self.right.insert(data)
      else:
        self.right = BST(data)

  def inorder_traversal(self):
    if self.left:
      self.left.inorder_traversal()
    print(self.data)
    if self.right:
      self.right.inorder_traversal()


  def preorder_traversal(self):
    print(self.data)
    if self.left:
      self.left.preorder_traversal()
    if self.right:
      self.right.preorder_traversal()


  def postorder_traversal(self):
    if self.left:
      self.left.postorder_traversal()
    if self.right:
      self.right.postorder_traversal()
    print(self.data)


# In[103]:


bst = BST(8)


# In[104]:


bst.data, bst.left, bst.right


# In[105]:


l = [7, 13, 8, 10, 5, 6]


# In[106]:


[bst.insert(i) for i in l]


# In[107]:


bst.data, bst.left.data, bst.right.data


# In[108]:


bst.left.data, bst.left.left.data,  bst.left.right.data


# In[109]:


bst.right.data, bst.right.left.data,  bst.right.data


# In[110]:


bst.inorder_traversal()


# In[111]:


bst.preorder_traversal()


# In[112]:


bst.postorder_traversal()


# In[117]:


# Graph Traversal
# BFS, DFS
l = [1, 2, 3, 4]
e = l.pop()
e


# In[116]:


l = [1, 2, 3, 4]
e = l.pop(0)
e


# In[119]:


l = [1, 2, 3, 4]
while l:
    e = l.pop(0)
    print(e)


# In[120]:


graph = {
    'a':['b', 'c'],
    'b':['a', 'g'],
    'c':['a', 'd', 'e'],
    'd':['c', 'e', 'h'],
    'e':['c', 'd'],
    'g':['b', 'h'],
    'h':['d', 'g'],
}


# In[121]:


def bfs(graph,source):
    queue = []
    visited = []
    queue.append(source)
    visited.append(source)
    while queue:
        k = queue.pop(0)
        # print(k)
        for e in graph[k]:
            if e not in visited:
                queue.append(e)
                visited.append(e)
    return visited


# In[126]:


o = bfs(graph, 'a')


# In[127]:


o


# In[128]:


def dfs(graph, visited, source):
    if source not in visited:
        print(source)
        visited.add(source)
        for e in graph[source]:
            dfs(graph, visited, e)
        


# In[129]:


visited = set()
o = dfs(graph, visited, 'a')

