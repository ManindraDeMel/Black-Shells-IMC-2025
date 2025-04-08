import itertools

toSnow = [1,0.7,1.95,1.34]
toPizza = [1.45,1,3.1,1.98]
toNug = [0.52,0.31,1,0.64]
toShell=[0.72,0.48,1.49,1]
matrix = [[1,0.7,1.95,1.34],[1.45,1,3.1,1.98],[0.52,0.31,1,0.64],[0.72,0.48,1.49,1]]
#SNOW,PIZZA,NUG,SHELL

#Start with Shell and end at shell length 4.
#First -> second -> third ->Shell
permutations = []
for a in range(4):
    for b in range(4):
        for c in range(4):
            for d in range(4):
                permutations.append([a, b, c, d])
finals = []
max_perm = []
for perm in permutations:
    (first,second,third,fourth) = perm 
    q2 = matrix[first][3]
    q3 = matrix[second][first]*q2
    q4 = matrix[third][second]*q3
    q5 = matrix[fourth][third]*q4 
    final = matrix[3][fourth]*q5 
    print(f"perm is {perm}, quantities are {q2,q3,q4,q5}, result is {final}")
    finals.append(final)
    if final >= max(finals):
        max_perm = perm 
    

print(max(finals))
print(max_perm)

