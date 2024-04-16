

isleaf = [0,1,0,0,1,1,0,1,1]
leafValue =   [0,0,1,1,1,1,2,2,2]
children_left =  [1,-1,3,4,-1,-1,7,-1,-1]
children_right = [2,-1,6,5,-1,-1,8,-1,-1]
feature = [2,-2,3,2,-2,-2,0,-2,-2]
threshold = [2.45,-2,1.75,4.45,-2,-2,6.35,-2,-2]
def predict(x, id):
    if(isleaf[id]):
        return leafValue[id]
    if(x[feature[id]] <= threshold[id]):
        next_id = children_left[id]
    else:
        next_id = children_right[id]
    return predict(x, next_id)

if __name__ == '__main__':
    x1 = [5.1, 3.5, 1.4, 0.2]
    x2 = [6.4, 3.5, 4.5, 1.2]
    x3 = [5.9, 3.0, 5.0, 1.8]
    res = predict(x3, 0)
    print(res)