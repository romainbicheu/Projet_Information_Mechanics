import tensorflow as tf

def getWeights(model):
    weights = {}
    for layer in model.layers:
        print(f"Layer: {layer.name}, Type: {type(layer)}" + "==================================")
        if isinstance(layer, tf.keras.layers.Conv2D):
            weights[layer.name] = layer.get_weights()[0]  
    for i in range(len(weights)):
        print(f"Layer: {list(weights.keys())[i]}, Weights shape: {list(weights.values())[i].shape}") # {3,3,1,3} for 3 filters of size 3x3 with 1 input channel
        for j in range(weights[list(weights.keys())[i]].shape[-1]):
            print(f"Filter {j} weights:\n{weights[list(weights.keys())[i]][:,:,0,j]}\n")

    return weights