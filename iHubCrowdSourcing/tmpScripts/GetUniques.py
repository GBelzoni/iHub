fin =open('modeltree.txt')
lines = fin.readlines()
fin.close()

fout =open('unique.txt','w')

lines2 = [ line.lower() for line in lines] 
    
uniques = set(lines2)
print uniques

fout.writelines(uniques)