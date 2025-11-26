obj = [1, 2, 3]
r1 = obj
r2 = r1
(id(obj) == id(r1), id(r1) == id(r2))
# Return=tuple: (True, True)
