
def run_v1NN():
    from v1NN import v1NN
    x = v1NN()
    perf = x.get_performance("3faces.list","params_feret.py")
   
    return perf