lst = [1]
old_id = id(lst)
lst.append(2)
old_id == id(lst)
# Return=bool: True
