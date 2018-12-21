def getDependencyMap():
    remove_dependencies = {}
    remove_dependencies[1] = [-1]
    remove_dependencies[2] = [1]
    remove_dependencies[3] = [1,4]
    remove_dependencies[4] = [1,4]
    remove_dependencies[5] = [1,4]
    remove_dependencies[6] = [-1]
    remove_dependencies[7] = [0]
    remove_dependencies[8] = [13]
    remove_dependencies[9] = [1,3]
    remove_dependencies[10] = [1,4,7]
    remove_dependencies[11] = [1,4,7]
    
    return remove_dependencies
    
def getQuestionAnswerMap():
    q_a={}
    q_a[1]=[0,3]
    q_a[2]=[3,5]
    q_a[3]=[5,7]
    q_a[4]=[7,9]
    q_a[5]=[9,13]
    q_a[6]=[13,15]
    q_a[7]=[15,18]
    q_a[8]=[18,25]
    q_a[9]=[25,28]
    q_a[10]=[28,31]
    
    q_a[11]=[31,37]
    
    return q_a
    
